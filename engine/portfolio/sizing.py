"""Position sizing utilities (stubs)."""

from __future__ import annotations


def size_from_probability(p: float, alpha: float = 1.0, w_max: float = 0.10) -> float:
    """Monotone map from probability to position weight.

    w = clip(alpha * (p - 0.5), -w_max, w_max)
    """
    x = alpha * (float(p) - 0.5)
    return max(-w_max, min(w_max, x))

