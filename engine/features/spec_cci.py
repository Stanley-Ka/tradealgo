from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_cci(
    df: pd.DataFrame,
    period: int = 20,
    params: dict | None = None,
) -> pd.Series:
    """Commodity Channel Index specialist in [-1,1].

    CCI = (TP - SMA(TP)) / (0.015 * MAD(TP)), score = clip(CCI/200, [-1,1]).
    """
    if isinstance(params, dict):
        period = int(params.get("period", period))
    h = df["adj_high"].astype(float)
    l = df["adj_low"].astype(float)
    c = df["adj_close"].astype(float)
    tp = (h + l + c) / 3.0
    sma = tp.rolling(period, min_periods=max(3, period // 2)).mean()

    # Rolling mean absolute deviation around SMA with NaN-safe handling
    def _mad(x: np.ndarray) -> float:
        xx = np.asarray(x, dtype=float)
        valid_vals = xx[np.isfinite(xx)]
        if valid_vals.size == 0:
            return float("nan")
        mean = float(valid_vals.mean())
        diffs = np.abs(valid_vals - mean)
        return float(diffs.mean()) if diffs.size else float("nan")

    mad = tp.rolling(period, min_periods=max(3, period // 2)).apply(_mad, raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (tp - sma) / (0.015 * mad)
    score = (cci / 200.0).replace([np.inf, -np.inf], 0.0)
    score = score.fillna(0.0).clip(-1.0, 1.0)
    return score
