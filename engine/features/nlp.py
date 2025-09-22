"""News/NLP-based scoring placeholder (LLM or classical sentiment)."""

from __future__ import annotations

from typing import Iterable


def score(news_items: Iterable[str]) -> float:
    """Return an uncalibrated score in [0, 1]. Placeholder returns 0.5."""
    _ = news_items
    return 0.5

