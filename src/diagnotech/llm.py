"""Language-model providers (spec section 8).

One interface, three real providers:

* ``LocalTransformersLLM`` – a locally downloaded multilingual model
  (default Qwen/Qwen2.5-1.5B-Instruct) served with transformers on CPU.
  The model is downloaded once into ``models/`` and kept loaded.
* ``OpenAICompatibleLLM`` – any OpenAI-compatible chat endpoint. This
  covers a local llama.cpp server (``llama-server`` exposes the same API
  on localhost) as well as a user-configured remote gateway. Disabled
  unless base URL, key, and model are configured.
* ``UnavailableLLM`` – explicit degradation: raises ``LLMUnavailable`` so
  the chat service can answer honestly instead of fabricating text.

``create_llm`` implements the ``auto`` policy: local first, then a
configured compatible endpoint, then unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMProvider:
    name = "base"

    def available(self) -> bool:
        return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailable("no language model provider is available")


class LocalTransformersLLM(LLMProvider):
    name = "local-transformers"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None

    def _model_dir(self) -> Path:
        safe = self.settings.llm_model.replace("/", "__")
        return self.settings.models_dir / safe

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise LLMUnavailable(self._load_error)
        try:
            import torch  # noqa: F401
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_dir = self._model_dir()
            if not model_dir.exists():
                snapshot_download(repo_id=self.settings.llm_model,
                                  local_dir=str(model_dir))
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype="auto")
            self._model.eval()
        except Exception as exc:
            self._load_error = f"local model failed to load: {exc}"
            logger.warning(self._load_error)
            raise LLMUnavailable(self._load_error) from exc

    def available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except LLMUnavailable:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        self._ensure_loaded()
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.llm_max_new_tokens,
                temperature=self.settings.llm_temperature if self.settings.llm_temperature > 0 else None,
                do_sample=self.settings.llm_temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAICompatibleLLM(LLMProvider):
    name = "openai-compatible"

    def __init__(self, settings: Settings):
        self.settings = settings

    def available(self) -> bool:
        s = self.settings
        return bool(s.openai_base_url and s.openai_api_key and s.openai_model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        s = self.settings
        if not self.available():
            raise LLMUnavailable("openai-compatible endpoint is not configured")
        url = s.openai_base_url.rstrip("/") + "/chat/completions"
        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {s.openai_api_key}"},
                json={"model": s.openai_model,
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "temperature": s.llm_temperature,
                      "max_tokens": s.llm_max_new_tokens},
                timeout=120.0)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise LLMUnavailable(f"openai-compatible endpoint failed: {exc}") from exc


class UnavailableLLM(LLMProvider):
    name = "unavailable"

    def available(self) -> bool:
        return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailable("no language model provider is configured or loadable")


def create_llm(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "local":
        return LocalTransformersLLM(settings)
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(settings)
    if provider == "none":
        return UnavailableLLM()
    # auto
    local = LocalTransformersLLM(settings)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return local
    except Exception:
        logger.warning("torch/transformers not installed; local LLM unavailable")
    compatible = OpenAICompatibleLLM(settings)
    if compatible.available():
        return compatible
    return local  # will raise LLMUnavailable with a precise reason when used
