#!/usr/bin/env python3
"""Back up application data (SQLite database + index manifest + processed
dataset) into a timestamped folder under ``backups/``. Restore = copy the
files back to their original locations (see README)."""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnotech.config import get_settings


def main() -> None:
    settings = get_settings()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = settings.project_root / "backups" / stamp
    target.mkdir(parents=True, exist_ok=True)
    if settings.db_path.exists():
        shutil.copy2(settings.db_path, target / "diagnotech.db")
    for name in ("manifest.json", "chunks.jsonl"):
        source = settings.index_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
    stats = settings.processed_dir / "ingestion_stats.json"
    if stats.exists():
        shutil.copy2(stats, target / "ingestion_stats.json")
    print(f"backup written to {target}")


if __name__ == "__main__":
    main()
