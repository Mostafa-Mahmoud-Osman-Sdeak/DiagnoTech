"""Conversation orchestration (spec sections 7 and 8).

Flow for every user message:
domain classification -> emergency / greeting / out-of-scope shortcuts ->
local retrieval -> (a) grounded generation from local evidence,
(b) optional trusted external research, or (c) the fixed unavailable
message. Both sides of the conversation are persisted with the answer
state and source references. The generator never runs without evidence.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import Settings
from .database import ConversationStore
from .domain import (EMERGENCY_MESSAGE, GREETING_MESSAGE, OUT_OF_SCOPE_MESSAGE,
                     classify)
from .llm import LLMProvider, LLMUnavailable
from .prompts import (DISCLAIMER, RETRY_INSTRUCTION, SYSTEM_PROMPT,
                      UNAVAILABLE_MESSAGE, build_external_prompt,
                      build_user_prompt)
from .research import WebResearcher
from .retrieval import Retriever

logger = logging.getLogger(__name__)

GENERATION_ERROR_MESSAGE = (
    "حدثت مشكلة أثناء توليد الإجابة، ولم أتمكن من إكمال الرد. "
    "حاول مرة أخرى بعد قليل."
)


def _bailed_out(text: str) -> bool:
    """True when the model replied with the unavailable phrase."""
    return UNAVAILABLE_MESSAGE.split("،")[0] in text and "غير متوفرة" in text


class ChatService:
    def __init__(self, settings: Settings, store: ConversationStore,
                 retriever: Retriever | None, llm: LLMProvider,
                 researcher: WebResearcher | None = None):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.researcher = researcher

    # ------------------------------------------------------------------
    def chat(self, conversation_id: str | None, message: str) -> dict[str, Any]:
        conversation = None
        if conversation_id:
            conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            conversation = self.store.create_conversation(title=message.strip()[:60])
            conversation_id = conversation["id"]
        self.store.add_message(conversation_id, "user", message)

        answer, state, sources = self._answer(message)
        self.store.add_message(conversation_id, "assistant", answer,
                               state=state, sources=sources)
        return {"conversation_id": conversation_id, "answer": answer,
                "state": state, "sources": sources}

    # ------------------------------------------------------------------
    def _answer(self, message: str) -> tuple[str, str, list[dict]]:
        domain = classify(message)
        if domain.is_emergency or domain.label == "emergency":
            return f"{EMERGENCY_MESSAGE}\n\n{DISCLAIMER}", "emergency", []
        if domain.label == "greeting":
            return GREETING_MESSAGE, "greeting", []
        if domain.label == "out_of_scope":
            return OUT_OF_SCOPE_MESSAGE, "out_of_scope", []

        if self.retriever is None:
            return ("قاعدة المعرفة غير مهيأة بعد. اطلب من المسؤول بناء الفهرس أولاً.",
                    "error", [])

        evidence = self.retriever.retrieve(message)
        if evidence:
            prompt = build_user_prompt(message, [e.evidence_block() for e in evidence])
            generated = self._generate(prompt)
            if generated is None:
                return GENERATION_ERROR_MESSAGE, "error", [e.source_ref() for e in evidence]
            if _bailed_out(generated):
                # The small local model sometimes refuses even when the
                # evidence is on-topic; retry once with a firmer
                # instruction before giving up to the unavailable state.
                generated = self._generate(prompt + RETRY_INSTRUCTION)
                if generated is None:
                    return GENERATION_ERROR_MESSAGE, "error", [e.source_ref() for e in evidence]
                if _bailed_out(generated):
                    return UNAVAILABLE_MESSAGE, "unavailable", []
            sources = [e.source_ref() for e in evidence]
            return f"{generated}\n\n{DISCLAIMER}", "known", sources

        # Unknown locally -> optional trusted external research (spec 9).
        if self.settings.external_research_enabled and self.researcher is not None:
            external = self._external_evidence(message)
            if external:
                prompt = build_external_prompt(
                    message, [e.evidence_block() for e in external])
                generated = self._generate(prompt)
                if generated:
                    sources = [e.source_ref() for e in external]
                    return f"{generated}\n\n{DISCLAIMER}", "external", sources

        return UNAVAILABLE_MESSAGE, "unavailable", []

    # ------------------------------------------------------------------
    def _generate(self, user_prompt: str) -> str | None:
        try:
            text = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            return text or None
        except LLMUnavailable as exc:
            logger.warning("generation unavailable: %s", exc)
            return None

    def _external_evidence(self, message: str):
        query = message
        translated = self._generate(
            "Translate the following Arabic medical question into a concise "
            f"English search query. Reply with the query only.\n\n{message}")
        if translated:
            query = translated.splitlines()[0].strip()
        return self.researcher.search(query)
