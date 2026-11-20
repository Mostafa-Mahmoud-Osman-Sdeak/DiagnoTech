"""Central configuration for DiagnoTech.

Defaults live here; every value can be overridden with an environment
variable prefixed with ``DIAGNOTECH_`` (for example
``DIAGNOTECH_TOP_K=8``). No secrets are stored in this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DIAGNOTECH_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    db_path: Path = PROJECT_ROOT / "data" / "db" / "diagnotech.db"
    models_dir: Path = PROJECT_ROOT / "models"
    log_dir: Path = PROJECT_ROOT / "logs"
    frontend_dir: Path = PROJECT_ROOT / "frontend"

    # Retrieval
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))  # auto|sentence-transformers|tfidf
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.45))
    unanchored_threshold: float = field(default_factory=lambda: _env_float("UNANCHORED_THRESHOLD", 0.88))
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 900))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 150))

    # Generation
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))  # auto|local|openai-compatible|none
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    llm_max_new_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_NEW_TOKENS", 512))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", ""))

    # Trusted external research (spec section 9)
    external_research_enabled: bool = field(default_factory=lambda: _env_bool("EXTERNAL_RESEARCH", True))
    research_timeout_seconds: float = field(default_factory=lambda: _env_float("RESEARCH_TIMEOUT", 10.0))

    # Server
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.index_dir,
                     self.db_path.parent, self.models_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
