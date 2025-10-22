from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import numpy as np

from .providers import NewsItem


POSITIVE = {
    "beat",
    "beats",
    "beating",
    "surge",
    "soar",
    "soars",
    "record",
    "profit",
    "profits",
    "growth",
    "upbeat",
    "upgrade",
    "upgraded",
    "raises",
    "raise",
    "strong",
    "outperform",
    "buy",
    "bullish",
    "optimistic",
    "tailwind",
    "resilient",
}
NEGATIVE = {
    "miss",
    "misses",
    "slump",
    "slumps",
    "loss",
    "losses",
    "downgrade",
    "downgraded",
    "sell",
    "underperform",
    "cuts",
    "cut",
    "weak",
    "bearish",
    "cautious",
    "lawsuit",
    "investigation",
    "probe",
    "fraud",
    "guidance cut",
    "warns",
    "warning",
}


def _score_text(text: str) -> float:
    t = text.lower()
    # simple tokenization by non-letters
    tokens = re.split(r"[^a-z]+", t)
    pos = sum(tok in POSITIVE for tok in tokens)
    neg = sum(tok in NEGATIVE for tok in tokens)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1.0, pos + neg)
    return float(np.clip(score, -1.0, 1.0))


def score_news(
    items: Iterable[NewsItem],
) -> Tuple[float, int, List[Tuple[NewsItem, float]]]:
    """Return (average_sentiment, count, details[(item,score)])."""
    details: List[Tuple[NewsItem, float]] = []
    for it in items:
        s = _score_text((it.headline or "") + "\n" + (it.summary or ""))
        details.append((it, s))
    if not details:
        return 0.0, 0, []
    avg = float(np.mean([s for _, s in details]))
    return float(np.clip(avg, -1.0, 1.0)), len(details), details
