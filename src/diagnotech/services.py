"""Service assembly shared by the API, scripts, and tests.

``build_retriever`` only *opens* an existing persisted index — it never
rebuilds one (spec section 13). ``ensure_index`` is the explicit build
path used by the build script and the admin endpoint.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .chunking import chunk_records
from .config import Settings
from .embeddings import create_embedder
from .ingestion import load_processed
from .retrieval import Retriever
from .vector_store import build_index, open_store

logger = logging.getLogger(__name__)


def index_exists(settings: Settings) -> bool:
    manifest = settings.index_dir / "manifest.json"
    chunks = settings.index_dir / "chunks.jsonl"
    return manifest.exists() and chunks.exists()


def build_retriever(settings: Settings) -> Retriever | None:
    if not index_exists(settings):
        logger.warning("vector index not built yet")
        return None
    embedder = create_embedder(settings.embedding_backend, settings.embedding_model,
                               cache_dir=settings.models_dir, index_dir=settings.index_dir)
    store = open_store(settings.index_dir)
    if len(store) == 0:
        return None
    return Retriever(embedder, store, settings)


def ensure_index(settings: Settings) -> dict:
    """Run ingestion (if needed) and build the patient-facing index.

    Only patient-usable records (Mayo Clinic patient education) are
    indexed; see ingestion.py for the documented safety rule.
    """
    records = load_processed(settings)
    patient = [r for r in records if r.get("patient_usable")]
    chunks = chunk_records(patient, settings.chunk_max_chars, settings.chunk_overlap_chars)
    embedder = create_embedder(settings.embedding_backend, settings.embedding_model,
                               cache_dir=settings.models_dir, index_dir=settings.index_dir)
    store = build_index(chunks, embedder, settings.index_dir,
                        extra_manifest={"embedding_model": settings.embedding_model,
                                        "patient_records": len(patient)})
    return {"chunks": len(store), "patient_records": len(patient),
            "embedder": embedder.name, "store_backend": store.backend}
