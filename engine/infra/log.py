from __future__ import annotations

import logging
import os
from typing import Optional


def _level_from_env(default: str = "INFO") -> int:
    lvl = os.environ.get("LOG_LEVEL", default).upper()
    return getattr(logging, lvl, logging.INFO)


def setup_basic_logging(level: Optional[int] = None) -> None:
    """Idempotent basic logging configuration.

    Respects LOG_LEVEL env (default INFO). Safe to call multiple times.
    """
    if getattr(setup_basic_logging, "_configured", False):  # type: ignore[attr-defined]
        return
    logging.basicConfig(
        level=(level if level is not None else _level_from_env()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    setattr(setup_basic_logging, "_configured", True)  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    setup_basic_logging()
    return logging.getLogger(name)
