"""Retrieval engine (spec section 7).

Pipeline: normalize question -> embed -> vector search -> threshold ->
per-record de-duplication -> ranked evidence with provenance. Retrieval
never invents content: when nothing passes the threshold the caller gets
an empty list and must move to the next answer state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .arabic import normalize_arabic
from .config import Settings
from .diseases import detect_aspects, detect_diseases, disease_matches
from .embeddings import Embedder

# Score bonus for chunks whose disease matches the disease detected in
# the Arabic question (see diseases.py for why this anchoring is needed).
DISEASE_BOOST = 0.15


@dataclass
class RetrievedEvidence:
    chunk_id: str
    record_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_ref(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.metadata.get("source", ""),
            "source_url": self.metadata.get("source_url", ""),
            "disease_name": self.metadata.get("disease_name", ""),
            "category": self.metadata.get("category", ""),
            "score": round(self.score, 4),
        }

    def evidence_block(self) -> str:
        meta = self.metadata
        header = (f"المصدر: {meta.get('source', '')} | الموضوع: {meta.get('disease_name', '')} "
                  f"| التصنيف: {meta.get('category', '')}")
        return f"{header}\n{self.text}"


class Retriever:
    def __init__(self, embedder: Embedder, store, settings: Settings):
        self.embedder = embedder
        self.store = store
        self.settings = settings

    def retrieve(self, question: str, top_k: int | None = None,
                 threshold: float | None = None,
                 max_per_record: int = 2) -> list[RetrievedEvidence]:
        top_k = top_k or self.settings.top_k
        threshold = self.settings.similarity_threshold if threshold is None else threshold
        if len(self.store) == 0:
            return []
        detected = detect_diseases(question)
        aspects = detect_aspects(question)
        expansion = " ".join(detected + aspects)
        # Embed the normalized question (hamza/ة/ى variants unified) so
        # different spellings of the same question give stable vectors;
        # the English expansion is appended unchanged.
        normalized_question = normalize_arabic(question)
        expanded = f"{normalized_question} {expansion}" if expansion else normalized_question
        if not detected:
            # Unanchored gate: cross-lingual e5 cosine scores for
            # *unrelated* medical text still sit around 0.80-0.86, above
            # the anchored threshold, so a question with no detected
            # disease would retrieve irrelevant evidence. Require a much
            # higher score before trusting unanchored matches.
            threshold = max(threshold, self.settings.unanchored_threshold)
        query_vector = self.embedder.encode_query(expanded)
        hits = self.store.search(query_vector, top_k=top_k * 6)
        scored = []
        for hit in hits:
            score = hit.score
            if detected and disease_matches(
                    detected, hit.chunk.get("metadata", {}).get("disease_name", "")):
                score += DISEASE_BOOST
            scored.append((score, hit))
        scored.sort(key=lambda item: -item[0])
        results: list[RetrievedEvidence] = []
        per_record: dict[str, int] = {}
        for score, hit in scored:
            if score < threshold:
                continue
            record_id = hit.chunk.get("record_id", "")
            if per_record.get(record_id, 0) >= max_per_record:
                continue
            per_record[record_id] = per_record.get(record_id, 0) + 1
            results.append(RetrievedEvidence(
                chunk_id=hit.chunk.get("chunk_id", ""),
                record_id=record_id,
                text=hit.chunk.get("text", ""),
                score=score,
                metadata=hit.chunk.get("metadata", {})))
            if len(results) >= top_k:
                break
        return results
