"""Embedding backends (spec section 6).

Two real backends behind one interface:

* ``SentenceTransformerEmbedder`` – a multilingual transformer model
  (default ``intfloat/multilingual-e5-small``) with strong Arabic support
  and cross-lingual retrieval (Arabic query -> English document). Loaded
  lazily and cached for the lifetime of the process.
* ``TfidfEmbedder`` – a scikit-learn TF-IDF fallback (word + character
  n-grams) that needs no model download. Used automatically when the
  transformer backend is unavailable, so the application still runs.

``auto`` mode tries the transformer first and falls back to TF-IDF.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder:
    name = "base"
    dim = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        self._model = SentenceTransformer(model_name, cache_folder=str(cache_dir) if cache_dir else None)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _prefix(self, texts: Sequence[str], kind: str) -> list[str]:
        if not self._is_e5:
            return list(texts)
        return [f"{kind}: {t}" for t in texts]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            self._prefix(texts, "passage"), batch_size=64,
            show_progress_bar=False, convert_to_numpy=True)
        return _normalize(vectors)

    def encode_query(self, text: str) -> np.ndarray:
        vectors = self._model.encode(
            self._prefix([text], "query"), show_progress_bar=False, convert_to_numpy=True)
        return _normalize(vectors)[0]


class TfidfEmbedder(Embedder):
    name = "tfidf"
    _MODEL_FILE = "tfidf_vectorizer.pkl"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1,
            analyzer="word", lowercase=True)
        self._fitted = False
        self.dim = 0

    def fit(self, texts: Sequence[str]) -> None:
        self._vectorizer.fit(list(texts))
        self._fitted = True
        self.dim = len(self._vectorizer.vocabulary_)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return _normalize(self._vectorizer.transform(list(texts)).toarray())

    def encode_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF embedder must be fitted on the corpus first")
        return _normalize(self._vectorizer.transform([text]).toarray())[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / self._MODEL_FILE).open("wb") as fh:
            pickle.dump(self._vectorizer, fh)

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder | None":
        path = directory / cls._MODEL_FILE
        if not path.exists():
            return None
        instance = cls.__new__(cls)
        with path.open("rb") as fh:
            instance._vectorizer = pickle.load(fh)
        instance._fitted = True
        instance.dim = len(instance._vectorizer.vocabulary_)
        return instance


def create_embedder(backend: str, model_name: str, cache_dir: Path | None = None,
                    index_dir: Path | None = None) -> Embedder:
    """Create an embedder according to the configured backend.

    In ``auto`` mode the transformer backend is preferred; any import or
    load failure falls back to TF-IDF with a logged warning.
    """
    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(model_name, cache_dir=cache_dir)
        except Exception as exc:  # model unavailable, no network, no torch...
            if backend == "sentence-transformers":
                raise
            logger.warning("Sentence-transformers unavailable (%s); falling back to TF-IDF", exc)
    if index_dir is not None:
        loaded = TfidfEmbedder.load(index_dir)
        if loaded is not None:
            return loaded
    return TfidfEmbedder()
