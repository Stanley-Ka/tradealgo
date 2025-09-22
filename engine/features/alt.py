"""Alt-data scoring placeholder (earnings calendar, options-implied, cross-sectional factors)."""

from __future__ import annotations

from typing import Mapping, Any


def score(features: Mapping[str, Any]) -> float:
    """Return an uncalibrated score in [0, 1]. Placeholder returns 0.5."""
    _ = features
    return 0.5

