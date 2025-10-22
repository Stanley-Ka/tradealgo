"""Lightweight market-state classification utilities used by logging/training.

The helpers here derive coarse labels (trend/volatility/condition) from the
feature panel produced in `entry_loop` so that we can analyse specialist
performance by regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketState:
    trend: str
    volatility: str
    condition: str


def _safe_value(row: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        val = float(row.get(key, default))
    except Exception:
        val = default
    return val


def _clip_prob(val: float) -> float:
    if not np.isfinite(val):
        return float("nan")
    return float(max(0.0, min(1.0, val)))


def infer_trend_state(row: pd.Series) -> str:
    """Classify a coarse trend regime from momentum/return features."""

    mom_5_20 = _safe_value(row, "mom_sma_5_20")
    ret_20d = _safe_value(row, "ret_20d")
    ret_5d = _safe_value(row, "ret_5d")

    up_score = sum(v > 0 for v in (mom_5_20, ret_20d, ret_5d))
    down_score = sum(v < 0 for v in (mom_5_20, ret_20d, ret_5d))

    if up_score >= 2:
        return "uptrend"
    if down_score >= 2:
        return "downtrend"
    return "sideways"


def infer_vol_state(row: pd.Series) -> str:
    """Bucket volatility based on ATR percentage."""

    atr_pct = _safe_value(row, "atr_pct_14")
    if not np.isfinite(atr_pct):
        return "normal"
    if atr_pct >= 0.035:
        return "high_vol"
    if atr_pct <= 0.015:
        return "low_vol"
    return "normal"


def infer_condition_label(row: pd.Series, trend: str) -> str:
    """Aggregate broader condition label from specialist probabilities."""

    breakout = _clip_prob(_safe_value(row, "spec_breakout_prob"))
    meanrev = _clip_prob(_safe_value(row, "spec_meanrev_prob"))
    regime_vol = _clip_prob(_safe_value(row, "regime_vol"))

    if np.isfinite(breakout) and breakout >= 0.6:
        return "breakout"
    if np.isfinite(meanrev) and meanrev >= 0.6:
        return "reversion"
    if trend == "uptrend":
        return "uptrend"
    if trend == "downtrend":
        return "downtrend"
    if np.isfinite(regime_vol) and regime_vol >= 0.6:
        return "volatile_consolidation"
    return "consolidation"


def infer_market_state(row: pd.Series) -> MarketState:
    trend = infer_trend_state(row)
    vol = infer_vol_state(row)
    condition = infer_condition_label(row, trend)
    return MarketState(trend=trend, volatility=vol, condition=condition)


def state_as_dict(row: pd.Series) -> Dict[str, str]:
    state = infer_market_state(row)
    return {
        "trend_state": state.trend,
        "vol_state": state.volatility,
        "condition_label": state.condition,
    }
