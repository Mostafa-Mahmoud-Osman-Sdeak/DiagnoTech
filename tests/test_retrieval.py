"""Retrieval over a tiny synthetic index using the TF-IDF backend and the
real FAISS store, so the test needs no model download."""
import numpy as np

from diagnotech.chunking import Chunk
from diagnotech.config import Settings
from diagnotech.embeddings import TfidfEmbedder
from diagnotech.retrieval import Retriever
from diagnotech.vector_store import build_index


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, record_id=cid, text=text,
                 metadata={"source": "Mayo Clinic", "disease_name": cid,
                           "category": "Symptoms", "question": text})


def _retriever(tmp_path) -> Retriever:
    chunks = [
        _chunk("asthma", "Question: What are the symptoms of asthma? Answer: Wheezing, coughing and shortness of breath are common asthma symptoms."),
        _chunk("pneumonia", "Question: What is pneumonia? Answer: Pneumonia is an infection that inflames the air sacs in the lungs, causing cough and fever."),
        _chunk("sinusitis", "Question: What is sinusitis? Answer: Sinusitis is inflammation of the sinuses causing facial pressure and nasal congestion."),
    ]
    embedder = TfidfEmbedder()
    store = build_index(chunks, embedder, tmp_path)
    settings = Settings()
    settings.similarity_threshold = 0.05
    # The toy TF-IDF score scale is not the e5 scale the unanchored gate
    # is calibrated for; disable the gate for these ranking tests.
    settings.unanchored_threshold = 0.0
    return Retriever(embedder, store, settings)


def test_retrieves_relevant_chunk(tmp_path):
    retriever = _retriever(tmp_path)
    hits = retriever.retrieve("asthma symptoms wheezing", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].record_id == "asthma"
    assert hits[0].source_ref()["source"] == "Mayo Clinic"


def test_threshold_filters_irrelevant(tmp_path):
    retriever = _retriever(tmp_path)
    hits = retriever.retrieve("quantum physics black holes", top_k=3, threshold=0.9)
    assert hits == []


def test_index_persists_and_reopens(tmp_path):
    retriever = _retriever(tmp_path)
    from diagnotech.vector_store import open_store
    reopened = open_store(tmp_path)
    assert len(reopened) == 3
    assert reopened.manifest["embedder"] == "tfidf"


# ---------------------------------------------------------------------------
# Stub embedder/store: deterministic scores for the gate and normalization
# tests (the real e5 score distribution is exercised by the eval scripts).
# ---------------------------------------------------------------------------
from diagnotech.vector_store import SearchHit


class _StubEmbedder:
    name = "stub"
    dim = 3

    def __init__(self):
        self.queries: list[str] = []

    def encode_query(self, text):
        self.queries.append(text)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts):
        raise NotImplementedError


class _StubStore:
    def __init__(self, hits: list[SearchHit]):
        self._hits = hits

    def __len__(self) -> int:
        return len(self._hits)

    def search(self, query_vector, top_k: int):
        return self._hits[:top_k]


def _stub_retriever(score: float, disease: str = "Asthma"):
    chunk = {"chunk_id": "c1", "record_id": "r1", "text": "evidence text",
             "metadata": {"source": "Mayo Clinic", "disease_name": disease,
                          "category": "Symptoms"}}
    embedder = _StubEmbedder()
    settings = Settings()  # production defaults: 0.45 / 0.88
    retriever = Retriever(embedder, _StubStore([SearchHit(chunk=chunk, score=score)]),
                          settings)
    return retriever, embedder


def test_query_is_normalized_before_embedding():
    retriever, embedder = _stub_retriever(0.90)
    hits = retriever.retrieve("ما هي أعراض الربو؟")
    assert hits
    (embedded,) = embedder.queries
    assert "أ" not in embedded          # hamza unified before embedding
    assert "اعراض" in embedded          # normalized Arabic question
    assert "Asthma" in embedded         # English expansion kept


def test_unanchored_gate_suppresses_moderate_scores():
    retriever, _ = _stub_retriever(0.84)
    # Pneumonia is not in the lexicon/KB: no disease is detected, so the
    # stricter unanchored gate (0.88) applies and 0.84 is suppressed,
    # even though it beats the plain 0.45 threshold.
    assert retriever.retrieve("ما هو الالتهاب الرئوي؟") == []


def test_anchored_question_keeps_normal_threshold():
    retriever, _ = _stub_retriever(0.84)
    # Same raw score, but the question names a known disease: anchored
    # path (0.45 + disease boost) returns the evidence.
    hits = retriever.retrieve("ما هي أعراض الربو؟")
    assert hits and hits[0].record_id == "r1"
