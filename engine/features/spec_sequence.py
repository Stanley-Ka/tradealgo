from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_sequence(df: pd.DataFrame, window: int = 10, params: dict | None = None) -> pd.Series:
    """Lightweight sequence-style score in [-1, 1].

    V0 proxy: recent return EMA divided by recent volatility (rolling).
    This approximates a short-horizon drift signal akin to a simple RNN trend.
    """
    if params and "window" in params:
        window = int(params["window"]) or window
    ret1 = df.get("ret_1d")
    if ret1 is None:
        ret1 = df["adj_close"].pct_change(1)
    ema = ret1.ewm(span=window, adjust=False, min_periods=max(2, window//2)).mean()
    vol = ret1.rolling(window, min_periods=max(2, window//2)).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ema / vol
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # gentle squash to [-1, 1]
    score = z.clip(-3, 3) / 3.0
    return score
