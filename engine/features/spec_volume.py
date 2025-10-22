from __future__ import annotations

import numpy as np
import pandas as pd


def _money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c, v = df["adj_high"], df["adj_low"], df["adj_close"], df["adj_volume"]
    tp = (h + l + c) / 3.0
    mf = tp * v
    up = tp > tp.shift(1)
    pos_mf = mf.where(up, 0.0)
    neg_mf = mf.where(~up, 0.0)
    pos_sum = pos_mf.rolling(window, min_periods=max(2, window // 2)).sum()
    neg_sum = neg_mf.rolling(window, min_periods=max(2, window // 2)).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return mfi.fillna(50.0)


def compute_spec_flow(
    df: pd.DataFrame,
    window_obv: int = 20,
    window_mfi: int = 14,
    weights: dict | None = None,
) -> pd.Series:
    """Volume/flow specialist in [-1,1] using OBV slope and MFI.

    - OBV: cumulative signed volume; use normalized slope as signal.
    - MFI: normalized to [-1,1] around 50.
    """
    close, vol = df["adj_close"], df["adj_volume"]
    ret1 = close.pct_change(1)
    sign = np.sign(ret1.fillna(0.0))
    obv = (sign * vol.fillna(0.0)).cumsum()
    # Normalize OBV by rolling range to get a bounded slope proxy
    obv_roll = obv.rolling(window_obv, min_periods=max(3, window_obv // 2))
    obv_min = obv_roll.min()
    obv_max = obv_roll.max()
    obv_norm = (obv - obv_min) / (obv_max - obv_min).replace(0.0, np.nan)
    obv_norm = obv_norm.fillna(0.5)
    obv_slope = obv_norm.diff(window_obv // 2).fillna(0.0)
    obv_slope = obv_slope.clip(-1.0, 1.0)

    mfi = _money_flow_index(df, window=window_mfi)
    mfi_n = (mfi - 50.0) / 50.0  # [-1,1]

    w = {"obv": 0.6, "mfi": 0.4}
    if isinstance(weights, dict):
        w.update({k: float(v) for k, v in weights.items()})

    score = w["obv"] * obv_slope + w["mfi"] * mfi_n
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
