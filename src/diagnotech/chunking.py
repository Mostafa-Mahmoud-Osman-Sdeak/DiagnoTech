"""Configurable chunking with metadata preservation (spec section 5).

Each chunk embeds the record's question together with a window of its
answer, because the questions are the strongest retrieval signal in this
FAQ-style dataset. Windows slide with overlap so facts near a boundary
are not lost. Chunk metadata carries the full provenance of the record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "record_id": self.record_id,
                "text": self.text, "metadata": self.metadata}


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):  # prefer breaking at a sentence/space boundary
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in windows if w]


def chunk_record(record: dict, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    header = f"Question: {question}\nAnswer: "
    budget = max(max_chars - len(header), 200)
    parts = _windows(answer, budget, overlap)
    metadata = {
        "record_id": record.get("record_id", ""),
        "disease_name": record.get("disease_name", ""),
        "category": record.get("category", ""),
        "category_ar": record.get("category_ar", ""),
        "source": record.get("source", ""),
        "source_url": record.get("source_url", ""),
        "question": question,
    }
    return [
        Chunk(chunk_id=f"{record.get('record_id', 'rec')}-{i}",
              record_id=str(record.get("record_id", "")),
              text=header + part,
              metadata=dict(metadata))
        for i, part in enumerate(parts)
    ]


def chunk_records(records: list[dict], max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(chunk_record(record, max_chars=max_chars, overlap=overlap))
    return chunks
