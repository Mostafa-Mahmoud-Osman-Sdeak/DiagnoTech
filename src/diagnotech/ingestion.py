"""Dataset ingestion and data engineering (spec section 4).

Pipeline: Raw -> Validation -> Cleaning -> Deduplication -> flags/metadata.
Original files in ``data/raw`` are never modified. Every stage is a separate
function and can be executed independently; ``run_ingestion`` orchestrates
them and writes processed outputs plus a statistics file.

Sources handled:
* ``DiagnoTech_Respiratory_Clean__1.xlsx`` – Arabic headers, includes
  source name and source URL columns (richest copy, treated as primary).
* ``data_chatbot_0_860u.csv`` – English headers, carries an unnamed
  ``needs_review`` flag column; contributes 23 records the XLSX lacks.

Patient-safety rule (documented in the development log): only Mayo Clinic
patient-education rows are flagged ``patient_usable``. Medical MCQ rows are
exam items whose answer options are absent from the data (thousands of
answers reference an option letter that is not in the file), so they are
kept in the processed dataset for completeness but excluded from the
patient-facing retrieval index.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .arabic import normalize_for_match
from .config import Settings, get_settings

CATEGORY_AR_TO_EN = {
    "مصطلحات طبية": "Medical Terms",
    "أمان الأدوية": "Medication Safety",
    "فحوصات طبية": "Medical Tests",
    "إرشادات الطوارئ": "Emergency Guidance",
    "الأعراض": "Symptoms",
    "الأسباب وعوامل الخطر": "Causes & Risk Factors",
    "معلومات عن المرض": "Disease Information",
    "إرشادات عامة": "General Guidance",
    "الوقاية": "Prevention",
}
CATEGORY_EN_TO_AR = {v: k for k, v in CATEGORY_AR_TO_EN.items()}

XLSX_COLUMNS = {
    "رقم السجل": "record_id",
    "التصنيف": "category_ar",
    "المرض / الموضوع": "disease_name",
    "السؤال": "question",
    "الإجابة": "answer",
    "المصدر": "source",
    "رابط المصدر": "source_url",
}

RESPIRATORY_KEYWORDS = (
    r"asthma|bronch|copd|pneumonia|sinus|lung|respirat|tuberculos|pulmon|cough|"
    r"emphysema|pleur|trachea|laryng|pharyng|rhinitis|apnea|influenza|cystic fibrosis|"
    r"bronchiolitis|whooping|sars|covid|mesothelioma|sarcoidosis|pneumothorax|"
    r"atelectasis|hay fever|tonsil|adenoid|nasal|nose|throat|chest|breath|snor|"
    r"sleep-disordered|upper respiratory|common cold|allergic"
)
_RESPIRATORY_RE = re.compile(RESPIRATORY_KEYWORDS, re.IGNORECASE)


def is_respiratory(question: str, disease_name: str) -> bool:
    return bool(_RESPIRATORY_RE.search(f"{question or ''} {disease_name or ''}"))


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl", dtype=str)
    df = df.rename(columns=XLSX_COLUMNS)
    df["category"] = df["category_ar"].map(CATEGORY_AR_TO_EN).fillna(df["category_ar"])
    df["review_status"] = ""
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: "review_status"})
    df["category_ar"] = df["category"].map(CATEGORY_EN_TO_AR).fillna("")
    df["source"] = ""
    df["source_url"] = ""
    return df


CANONICAL_COLUMNS = [
    "record_id", "category", "category_ar", "disease_name", "question",
    "answer", "source", "source_url", "review_status",
]


def merge_sources(xlsx: pd.DataFrame, csv: pd.DataFrame) -> pd.DataFrame:
    """Merge by record_id. XLSX wins for content/source; CSV adds the
    review flag and any records missing from the XLSX."""
    for df in (xlsx, csv):
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        for col in CANONICAL_COLUMNS:
            df[col] = df[col].map(_clean_str)
    base = xlsx.set_index("record_id")
    extra = csv.set_index("record_id")
    # review flag from CSV
    base["review_status"] = extra["review_status"].reindex(base.index).fillna("")
    only_csv = extra.loc[~extra.index.isin(base.index)]
    cols = [c for c in CANONICAL_COLUMNS if c != "record_id"]
    merged = pd.concat([base[cols], only_csv[cols]])
    merged.index.name = "record_id"
    return merged.reset_index()[CANONICAL_COLUMNS]


def validate_and_flag(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    before = len(df)
    df = df[(df["question"] != "") & (df["answer"] != "")].copy()
    stats["dropped_missing_q_or_a"] = before - len(df)
    df["patient_usable"] = df["source"].str.contains("Mayo", case=False, na=False)
    df["respiratory"] = [is_respiratory(q, d) for q, d in zip(df["question"], df["disease_name"])]
    return df


def deduplicate(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Remove exact duplicate Q+A pairs (normalized), preferring
    patient-usable records so the kept copy carries the best metadata."""
    df = df.sort_values(by="patient_usable", ascending=False, kind="stable")
    keys = (df["question"].map(normalize_for_match) + " ||| "
            + df["answer"].map(normalize_for_match))
    dup_mask = keys.duplicated()
    stats["duplicates_removed"] = int(dup_mask.sum())
    return df[~dup_mask].reset_index(drop=True)


def run_ingestion(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    stats: dict[str, Any] = {"sources": {}}

    xlsx_path = settings.raw_dir / "DiagnoTech_Respiratory_Clean__1.xlsx"
    csv_path = settings.raw_dir / "data_chatbot_0_860u.csv"
    xlsx = load_xlsx(xlsx_path)
    csv = load_csv(csv_path)
    stats["sources"]["xlsx_rows"] = len(xlsx)
    stats["sources"]["csv_rows"] = len(csv)

    merged = merge_sources(xlsx, csv)
    stats["merged_rows"] = len(merged)
    merged = validate_and_flag(merged, stats)
    merged = deduplicate(merged, stats)

    stats["final_rows"] = len(merged)
    stats["patient_usable_rows"] = int(merged["patient_usable"].sum())
    stats["respiratory_rows"] = int(merged["respiratory"].sum())
    stats["patient_usable_respiratory_rows"] = int(
        (merged["patient_usable"] & merged["respiratory"]).sum())
    stats["categories"] = merged["category"].value_counts().to_dict()
    stats["source_counts"] = merged["source"].replace("", "Unknown").value_counts().to_dict()

    out_jsonl = settings.processed_dir / "processed_records.jsonl"
    out_csv = settings.processed_dir / "processed_records.csv"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for record in merged.to_dict("records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    stats_path = settings.processed_dir / "ingestion_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def load_processed(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    path = settings.processed_dir / "processed_records.jsonl"
    if not path.exists():
        run_ingestion(settings)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    result = run_ingestion()
    print(json.dumps(result, ensure_ascii=False, indent=2))
