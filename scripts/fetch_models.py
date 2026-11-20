#!/usr/bin/env python3
"""Download the embedding model and the local LLM into ``models/`` once,
so the application can run fully offline afterwards."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnotech.config import get_settings


def main() -> None:
    from huggingface_hub import snapshot_download

    settings = get_settings()
    print(f"embedding model: {settings.embedding_model}")
    snapshot_download(repo_id=settings.embedding_model,
                      cache_dir=str(settings.models_dir))
    safe = settings.llm_model.replace("/", "__")
    target = settings.models_dir / safe
    print(f"llm: {settings.llm_model} -> {target}")
    snapshot_download(repo_id=settings.llm_model, local_dir=str(target))
    print("done")


if __name__ == "__main__":
    main()
