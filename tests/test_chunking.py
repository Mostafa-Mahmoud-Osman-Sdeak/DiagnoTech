from diagnotech.chunking import chunk_record


def _record(answer: str) -> dict:
    return {"record_id": "DT-1", "question": "What is asthma?",
            "answer": answer, "disease_name": "Asthma",
            "category": "Symptoms", "category_ar": "الأعراض",
            "source": "Mayo Clinic", "source_url": "https://example.org"}


def test_short_answer_single_chunk_with_question_header():
    chunks = chunk_record(_record("Asthma is a chronic condition."))
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Question: What is asthma?")
    assert chunks[0].metadata["source"] == "Mayo Clinic"
    assert chunks[0].chunk_id == "DT-1-0"


def test_long_answer_multiple_chunks_with_metadata():
    answer = " ".join(f"Sentence number {i} about asthma care and management."
                      for i in range(120))
    chunks = chunk_record(_record(answer), max_chars=400, overlap=60)
    assert len(chunks) > 3
    assert all(c.text.startswith("Question:") for c in chunks)
    assert all(c.metadata["disease_name"] == "Asthma" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"DT-1-{i}" for i in range(len(chunks))]
