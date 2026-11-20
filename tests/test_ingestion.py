import pandas as pd

from diagnotech.ingestion import deduplicate, is_respiratory, validate_and_flag

BASE = {
    "record_id": "DT-1", "category": "Symptoms", "category_ar": "الأعراض",
    "disease_name": "Asthma", "question": "What are the symptoms of asthma?",
    "answer": "Wheezing and coughing.", "source": "Mayo Clinic",
    "source_url": "https://example.org", "review_status": "",
}


def test_validate_flags_patient_usable_and_respiratory():
    df = pd.DataFrame([BASE, {**BASE, "record_id": "DT-2", "source": "Medical MCQ (train)",
                              "question": "What is the capital of France?",
                              "disease_name": "Geography", "answer": "Paris."}])
    stats = {}
    out = validate_and_flag(df, stats)
    assert bool(out.iloc[0]["patient_usable"]) is True
    assert bool(out.iloc[0]["respiratory"]) is True
    assert bool(out.iloc[1]["patient_usable"]) is False
    assert stats["dropped_missing_q_or_a"] == 0


def test_validate_drops_missing_answer():
    df = pd.DataFrame([{**BASE, "answer": ""}])
    stats = {}
    out = validate_and_flag(df, stats)
    assert len(out) == 0
    assert stats["dropped_missing_q_or_a"] == 1


def test_deduplicate_prefers_patient_usable_copy():
    mcq_copy = {**BASE, "record_id": "DT-9", "source": "Medical MCQ (train)"}
    df = pd.DataFrame([mcq_copy, BASE])
    stats = {}
    flagged = validate_and_flag(df, stats)
    out = deduplicate(flagged, stats)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "Mayo Clinic"
    assert stats["duplicates_removed"] == 1


def test_is_respiratory_keywords():
    assert is_respiratory("How is pneumonia treated?", "Pneumonia")
    assert not is_respiratory("What is diabetes?", "Diabetes")


def test_real_dataset_invariants():
    """Integration check against the shipped processed dataset."""
    from diagnotech.config import get_settings
    from diagnotech.ingestion import load_processed

    records = load_processed(get_settings())
    assert len(records) > 14_000
    patient = [r for r in records if r["patient_usable"]]
    assert len(patient) == 445
    assert all(r["source"] == "Mayo Clinic" for r in patient)
    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids))
