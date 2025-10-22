from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_williams_r(
    df: pd.DataFrame,
    period: int = 14,
    params: dict | None = None,
) -> pd.Series:
    """Williams %R specialist normalized to [-1,1].

    Standard %R in [-100, 0]; map linearly to [-1, 1] via score = 1 + %R/50.
    """
    if isinstance(params, dict):
        period = int(params.get("period", period))
    h = df["adj_high"].astype(float)
    l = df["adj_low"].astype(float)
    c = df["adj_close"].astype(float)
    hh = h.rolling(period, min_periods=max(3, period // 2)).max()
    ll = l.rolling(period, min_periods=max(3, period // 2)).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        wpr = -100.0 * (hh - c) / (hh - ll)
    score = 1.0 + (wpr / 50.0)
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
