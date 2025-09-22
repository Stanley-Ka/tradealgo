from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_technicals(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Technical indicator composite in roughly [-1, 1].

    Uses baseline features if present; otherwise computes simple fallbacks.
    Ingredients:
    - SMA(5/20) crossover momentum (mom_sma_5_20)
    - 20D momentum (ret_20d)
    - Price z-score mean reversion guard (price_z_20)
    """
    out = pd.Series(0.0, index=df.index, dtype=float)

    if "mom_sma_5_20" in df:
        mom_x = df["mom_sma_5_20"].fillna(0.0)
    else:
        close = df["adj_close"]
        sma5 = close.rolling(5, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        mom_x = (sma5 / sma20 - 1.0).fillna(0.0)

    mom20 = df.get("ret_20d", df["adj_close"].pct_change(20)).fillna(0.0)
    z = df.get("price_z_20", (df["adj_close"] / df["adj_close"].rolling(20).mean() - 1.0)).fillna(0.0)

    # Core momentum score
    w = weights or {"mom_x": 0.6, "mom20": 0.4, "z_guard": -0.1}
    core = float(w.get("mom_x", 0.6)) * mom_x + float(w.get("mom20", 0.4)) * mom20

    # Mean reversion guard: penalize extreme overbought/oversold
    guard = (float(w.get("z_guard", -0.1)) * (z.clip(-3, 3)))

    score = core + guard
    # Normalize and clip to [-1, 1]
    score = score.replace([np.inf, -np.inf], 0.0)
    score = score.clip(-1.0, 1.0)
    return score
