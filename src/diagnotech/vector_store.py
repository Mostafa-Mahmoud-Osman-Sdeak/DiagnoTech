"""Persistent vector storage (spec section 6).

FAISS ``IndexFlatIP`` over L2-normalized vectors (exact cosine search —
the right choice at this corpus scale) with a NumPy fallback implementing
the same interface when FAISS is unavailable. The index, chunk metadata,
and a versioned manifest are persisted together and are never rebuilt on
application startup; ``build_index`` runs only when explicitly invoked or
when no index exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, TfidfEmbedder

INDEX_FILE = "index.faiss"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float


class _BaseStore:
    backend = "base"

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.chunks: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        raise NotImplementedError

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self.manifest["chunks"] = len(self.chunks)
        self.manifest["saved_at"] = datetime.now(timezone.utc).isoformat()
        (self.directory / MANIFEST_FILE).write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_chunks(self) -> None:
        path = self.directory / CHUNKS_FILE
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                self.chunks = [json.loads(line) for line in fh if line.strip()]
        manifest_path = self.directory / MANIFEST_FILE
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))


class FaissStore(_BaseStore):
    backend = "faiss"

    def __init__(self, directory: Path):
        super().__init__(directory)
        import faiss

        self._faiss = faiss
        self._index = None
        index_path = self.directory / INDEX_FILE
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._load_chunks()

    def build(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.asarray(vectors, dtype=np.float32))
        self.chunks = [c.to_dict() for c in chunks]
        self.manifest = dict(manifest)
        self.save()

    def save(self) -> None:
        super().save()
        if self._index is not None:
            self._faiss.write_index(self._index, str(self.directory / INDEX_FILE))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if self._index is None or len(self.chunks) == 0:
            return []
        scores, ids = self._index.search(
            np.asarray([query_vector], dtype=np.float32), min(top_k, len(self.chunks)))
        return [SearchHit(chunk=self.chunks[i], score=float(s))
                for s, i in zip(scores[0], ids[0]) if i >= 0]


class NumpyStore(_BaseStore):
    backend = "numpy"

    def __init__(self, directory: Path):
        super().__init__(directory)
        self._vectors: np.ndarray | None = None
        vectors_path = self.directory / VECTORS_FILE
        if vectors_path.exists():
            self._vectors = np.load(vectors_path)
            self._load_chunks()

    def build(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self.chunks = [c.to_dict() for c in chunks]
        self.manifest = dict(manifest)
        self.save()

    def save(self) -> None:
        super().save()
        if self._vectors is not None:
            np.save(self.directory / VECTORS_FILE, self._vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if self._vectors is None or len(self.chunks) == 0:
            return []
        scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in order]


def open_store(directory: Path):
    """Open an existing store (FAISS preferred) without building anything."""
    try:
        import faiss  # noqa: F401
        return FaissStore(directory)
    except Exception:
        return NumpyStore(directory)


def build_index(chunks: list[Chunk], embedder: Embedder, directory: Path,
                extra_manifest: dict | None = None):
    """Embed all chunks and persist a fresh store. Explicit action only."""
    texts = [c.text for c in chunks]
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit(texts)
        embedder.save(directory)
    vectors = embedder.encode_documents(texts)
    manifest = {
        "version": 1,
        "embedder": embedder.name,
        "dim": int(vectors.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(),
        **(extra_manifest or {}),
    }
    store = open_store(directory)
    # A store built by the other backend cannot be reused; construct matching class.
    if isinstance(embedder, TfidfEmbedder) and store.backend == "faiss":
        pass  # FAISS can hold TF-IDF vectors too; backend refers to the store, not the embedder.
    store.build(chunks, vectors, manifest)
    return store
