#!/usr/bin/env python3
"""Retrieval evaluation (spec section 15).

A fixed set of Arabic questions, each mapped to the set of Mayo records
that legitimately answer it (same disease, matching aspect phrase in the
record's English question — several records can be valid targets).
Targets are resolved dynamically from the processed data; cases whose
target set is empty are reported as skipped. Reports Recall@K and MRR.

Note: an earlier version of this set assumed Pneumonia / Common cold /
Acute bronchitis / Snoring records existed in the Mayo subset. They do
not — the knowledge base genuinely lacks them — so those cases were
replaced with diseases the data actually covers.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnotech.config import get_settings
from diagnotech.ingestion import load_processed
from diagnotech.services import build_retriever

CASES = [
    ("ازاي أقدر أمنع نوبات الربو؟", "Asthma attack", "be prevented"),
    ("ما هي أعراض الربو؟", "Asthma", "symptoms of asthma"),
    ("ما هي أسباب التهاب الجيوب الأنفية الحاد؟", "Acute sinusitis", "causes"),
    ("كيف يمكن الوقاية من التهاب الجيوب الأنفية الحاد؟", "Acute sinusitis", "be prevented"),
    ("ما هي أسباب مرض الانسداد الرئوي المزمن؟", "COPD", "causes"),
    ("ما هي أعراض انقطاع النفس أثناء النوم؟", "Sleep apnea", "symptoms"),
    ("ما هي أعراض السل؟", "Tuberculosis", "symptoms of tuberculosis"),
    ("ما هي أعراض سرطان الرئة؟", "Lung cancer", "symptoms"),
    ("ما هي أعراض الإنفلونزا؟", "Influenza (flu)", "symptoms"),
    ("ما هي أعراض السعال الديكي؟", "Whooping cough", "symptoms"),
    ("ما هي أعراض انتفاخ الرئة؟", "Emphysema", "symptoms"),
    ("ما هي أعراض التهاب الجيوب الأنفية المزمن؟", "Chronic sinusitis", "symptoms"),
    ("ما هي أعراض الجلطة الرئوية؟", "Pulmonary embolism", "symptoms"),
    ("ما هي أعراض كورونا؟", "Coronavirus disease 2019 (COVID-19)", "symptoms"),
    ("ما هي أعراض الخانوق عند الأطفال؟", "Croup", "symptoms"),
    ("ما هي أعراض التليف الكيسي؟", "Cystic fibrosis", "symptoms"),
    ("ما هي علامات التحذير في الربو؟", "Asthma", "warning signs"),
    ("ما هي أسباب الساركويد؟", "Sarcoidosis", "causes"),
]


def main() -> None:
    settings = get_settings()
    records = load_processed(settings)
    patient = [r for r in records if r.get("patient_usable")]

    retriever = build_retriever(settings)
    if retriever is None:
        print("index not built; run scripts/build_index.py first")
        raise SystemExit(1)

    k = settings.top_k
    evaluated, skipped = 0, 0
    recall_sum = mrr_sum = 0.0
    for arabic_query, disease, phrase in CASES:
        targets = {r["record_id"] for r in patient
                   if r["disease_name"].lower() == disease.lower()
                   and phrase.lower() in r["question"].lower()}
        if not targets:
            skipped += 1
            print(f"SKIP  {arabic_query}  (no target: {disease} / {phrase})")
            continue
        hits = retriever.retrieve(arabic_query, top_k=k, threshold=0.0)
        hit_ids = [h.record_id for h in hits]
        evaluated += 1
        ranks = [hit_ids.index(t) + 1 for t in targets if t in hit_ids]
        if ranks:
            rank = min(ranks)
            recall_sum += 1.0
            mrr_sum += 1.0 / rank
            print(f"HIT@{rank}  {arabic_query}")
        else:
            print(f"MISS  {arabic_query}  (top: {hit_ids[:3]})")
    if evaluated:
        print(json.dumps({
            "cases": evaluated, "skipped": skipped, "k": k,
            "recall@k": round(recall_sum / evaluated, 4),
            "mrr": round(mrr_sum / evaluated, 4),
            "embedder": retriever.embedder.name,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
