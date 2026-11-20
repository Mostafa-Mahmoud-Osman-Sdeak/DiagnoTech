"""FastAPI backend (spec section 12).

Endpoints: chat, conversation management, health, knowledge-base status,
and administrative actions (rebuild index, backup). The Arabic RTL
frontend is served as static files from the same application.
"""
from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat_service import ChatService
from .config import Settings, get_settings
from .database import ConversationStore
from .llm import create_llm
from .research import WebResearcher
from .services import build_retriever, ensure_index, index_exists

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class CreateConversationRequest(BaseModel):
    title: str = ""


class AppServices:
    """Lazily assembled, process-wide services. Models and the index are
    loaded once and kept in memory while the service runs (spec 13/14)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ConversationStore(settings.db_path)
        self._retriever = None
        self._retriever_loaded = False
        self._llm = None
        self._chat: ChatService | None = None
        self._lock = threading.Lock()

    @property
    def retriever(self):
        if not self._retriever_loaded:
            with self._lock:
                if not self._retriever_loaded:
                    self._retriever = build_retriever(self.settings)
                    self._retriever_loaded = True
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    @property
    def chat(self) -> ChatService:
        if self._chat is None:
            researcher = (WebResearcher(timeout=self.settings.research_timeout_seconds)
                          if self.settings.external_research_enabled else None)
            self._chat = ChatService(self.settings, self.store, self.retriever,
                                     self.llm, researcher)
        return self._chat

    def refresh_retriever(self) -> None:
        with self._lock:
            self._retriever = build_retriever(self.settings)
            self._retriever_loaded = True
            self._chat = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = AppServices(settings)
    app = FastAPI(title="DiagnoTech – Arabic Respiratory Medical Assistant")
    app.state.services = services

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(),
                "index_ready": index_exists(settings)}

    @app.get("/api/kb/status")
    def kb_status() -> dict[str, Any]:
        retriever = services.retriever
        manifest = {}
        if retriever is not None:
            manifest = retriever.store.manifest
        return {"index_ready": index_exists(settings),
                "chunks": len(retriever.store) if retriever else 0,
                "manifest": manifest,
                "llm_provider": services.llm.name,
                "embedder": retriever.embedder.name if retriever else None}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        return services.chat.chat(request.conversation_id, request.message)

    @app.post("/api/conversations")
    def create_conversation(request: CreateConversationRequest) -> dict[str, Any]:
        return services.store.create_conversation(request.title)

    @app.get("/api/conversations")
    def list_conversations() -> list[dict[str, Any]]:
        return services.store.list_conversations()

    @app.get("/api/conversations/{conversation_id}/messages")
    def get_messages(conversation_id: str) -> list[dict[str, Any]]:
        if services.store.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="المحادثة غير موجودة")
        return services.store.get_messages(conversation_id)

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not services.store.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="المحادثة غير موجودة")
        return {"deleted": True}

    @app.post("/api/admin/rebuild")
    def rebuild() -> dict[str, Any]:
        result = ensure_index(settings)
        services.refresh_retriever()
        return {"rebuilt": True, **result}

    @app.post("/api/admin/backup")
    def backup() -> dict[str, Any]:
        backup_dir = settings.project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = backup_dir / f"diagnotech-{stamp}.db"
        shutil.copy2(settings.db_path, target)
        return {"backup": str(target)}

    frontend = settings.frontend_dir
    if frontend.exists():
        app.mount("/static", StaticFiles(directory=str(frontend)), name="static")

        @app.get("/")
        def index_page() -> FileResponse:
            return FileResponse(str(frontend / "index.html"))

    return app


app = create_app()
