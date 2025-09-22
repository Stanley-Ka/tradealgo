"""Candlestick/pattern-based scoring.

Exposes a `score` function returning an uncalibrated probability-like score for a given symbol/bar.
"""

from __future__ import annotations

from typing import Mapping, Any


def score(ohlcv: Mapping[str, Any]) -> float:
    """Return an uncalibrated score in [0, 1].

    Placeholder: returns 0.5. Replace with real pattern logic (e.g., engulfing, doji, hammer).
    """
    _ = ohlcv
    return 0.5

