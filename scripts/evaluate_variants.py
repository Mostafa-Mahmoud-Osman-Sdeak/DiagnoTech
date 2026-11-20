#!/usr/bin/env python3
"""Production-behaviour evaluation with spelling variants.

The fixed evaluation (evaluate_retrieval.py) measures ranking with the
threshold disabled. This script measures what the user actually
experiences: it calls the retriever with the *production* threshold and
reports, for each question, whether any evidence comes back at all and
whether it belongs to the expected disease. The variant questions use
the spellings real users type: hamza dropped or changed, ة written ه,
ى instead of ي, dialect phrasing, and small typos.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnotech.config import get_settings
from diagnotech.diseases import disease_matches
from diagnotech.services import build_retriever

CASES = [
    # Original-style questions (control group)
    ("ما هي أعراض الربو؟", "Asthma"),
    ("ما هي أعراض السل؟", "Tuberculosis"),
    ("ما هي أعراض الإنفلونزا؟", "Influenza (flu)"),
    ("ما هي أعراض التهاب الجيوب الأنفية المزمن؟", "Chronic sinusitis"),
    ("ما هي أعراض الجلطة الرئوية؟", "Pulmonary embolism"),
    ("ازاي أقدر أمنع نوبات الربو؟", "Asthma attack"),
    # Spelling variants of the same intent
    ("ما هى أعراض الربو؟", "Asthma"),
    ("اعراض الربو ايه؟", "Asthma"),
    ("ماهي اعراض الانفلونزا؟", "Influenza (flu)"),
    ("إيه أعراض السل؟", "Tuberculosis"),
    ("التهاب الجيوب الانفيه المزمن اعراضه ايه؟", "Chronic sinusitis"),
    ("عايز اعرف اعراض سرطان الرئه", "Lung cancer"),
    ("ايه اعراض الجلطه الرئويه؟", "Pulmonary embolism"),
    ("اعراض الكورونا ايه؟", "Coronavirus disease 2019 (COVID-19)"),
    ("ازاى امنع نوبات الربو؟", "Asthma attack"),
    ("إلتهاب الجيوب الأنفية الحاد أسبابه إيه؟", "Acute sinusitis"),
    ("ما هى اعراض التليف الكيسى؟", "Cystic fibrosis"),
    ("اعراض انتفاخ الرئه ايه", "Emphysema"),
    ("اعراض الربوو ايه؟", "Asthma"),
]


def main() -> None:
    settings = get_settings()
    retriever = build_retriever(settings)
    if retriever is None:
        print("index not built; run scripts/build_index.py first")
        raise SystemExit(1)

    answered = correct = 0
    for question, disease in CASES:
        evidence = retriever.retrieve(question)  # production threshold
        ok_disease = any(
            disease_matches([disease], e.metadata.get("disease_name", ""))
            for e in evidence)
        top = f"{evidence[0].score:.3f}" if evidence else "-"
        if evidence:
            answered += 1
        if ok_disease:
            correct += 1
        status = "OK  " if ok_disease else ("WRONG" if evidence else "NONE")
        print(f"{status}  top={top}  {question}")
    print(json.dumps({
        "cases": len(CASES),
        "answered_rate": round(answered / len(CASES), 4),
        "correct_disease_rate": round(correct / len(CASES), 4),
        "threshold": settings.similarity_threshold,
        "embedder": retriever.embedder.name,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
