"""Application entry point: ``python -m diagnotech.main``."""
from __future__ import annotations

import logging

import uvicorn

from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run() -> None:
    settings = get_settings()
    uvicorn.run("diagnotech.api:app", host=settings.host, port=settings.port,
                log_level="info")


if __name__ == "__main__":
    run()
