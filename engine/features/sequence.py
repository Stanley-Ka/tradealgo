"""Sequence model (RNN/LSTM/Transformer) scoring placeholder."""

from __future__ import annotations

from typing import Sequence as Seq, Mapping, Any


def score(series: Seq[Mapping[str, Any]]) -> float:
    """Return an uncalibrated score in [0, 1]. Placeholder returns 0.5."""
    _ = series
    return 0.5

