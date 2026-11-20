#!/usr/bin/env python3
"""Build (or rebuild) the patient-facing vector index.

Runs ingestion first if the processed dataset is missing, then embeds
all patient-usable records and persists the FAISS index. This is an
explicit administrative action; the server never rebuilds on startup.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnotech.config import get_settings
from diagnotech.ingestion import run_ingestion
from diagnotech.services import ensure_index


def main() -> None:
    settings = get_settings()
    stats = run_ingestion(settings)
    print("ingestion:", json.dumps(stats, ensure_ascii=False))
    result = ensure_index(settings)
    print("index:", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
