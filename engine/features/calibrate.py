"""Probability calibration utilities (stubs)."""

from __future__ import annotations


def identity(p: float) -> float:
    """No-op calibration placeholder. Replace with Platt or isotonic."""
    return max(0.0, min(1.0, float(p)))

