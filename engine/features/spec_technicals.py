from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_technicals(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Technical indicator composite in roughly [-1, 1].

    Uses baseline features if present; otherwise computes simple fallbacks.
    Ingredients (with sensible defaults):
    - SMA(5/20) crossover momentum (mom_sma_5_20)
    - 20D momentum (ret_20d)
    - RSI(14) normalized to [-1,1]
    - MACD (12,26,9) signal gap
    - Bollinger Band width (20,2) as volatility guard
    - Price z-score mean reversion guard (price_z_20)
    """
    close = df["adj_close"]

    # Momentum crossover
    if "mom_sma_5_20" in df:
        mom_x = df["mom_sma_5_20"].astype(float).fillna(0.0)
    else:
        sma5 = close.rolling(5, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        mom_x = (sma5 / sma20 - 1.0).fillna(0.0)

    # 20D momentum
    mom20 = df.get("ret_20d", close.pct_change(20)).astype(float).fillna(0.0)

    # RSI(14) mapped to [-1,1]
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll = 14
    avg_gain = up.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
    avg_loss = down.ewm(alpha=1 / roll, adjust=False, min_periods=roll).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)  # neutral
    rsi_n = (rsi - 50.0) / 50.0  # 0 at 50, [-1,1] at 0/100

    # MACD (12,26,9): use gap to signal (positive bullish)
    ema12 = close.ewm(span=12, adjust=False, min_periods=6).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=13).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=5).mean()
    macd_gap = macd - macd_signal
    # Normalize MACD gap by price to avoid scale issues
    macd_n = (macd_gap / close.replace(0.0, np.nan)).fillna(0.0).clip(-0.1, 0.1) / 0.1

    # Bollinger Band width as volatility/overextension guard
    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std(ddof=0)
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    bb_width = (upper - lower) / ma20.replace(0.0, np.nan)
    bb_width = bb_width.replace([np.inf, -np.inf], np.nan)
    bb_guard = (
        -(bb_width.fillna(bb_width.median()).clip(0, 0.2) / 0.2) * 0.2
    )  # small negative when wide

    # Price z-score guard
    z = df.get("price_z_20", (close / ma20 - 1.0)).fillna(0.0)
    z_guard = -0.1 * z.clip(-3, 3)

    # Combine with weights (defaults chosen conservatively)
    w = {
        "mom_x": 0.35,
        "mom20": 0.20,
        "rsi": 0.15,
        "macd": 0.20,
        "bb_guard": 0.05,
        "z_guard": 0.05,
    }
    if isinstance(weights, dict):
        w.update({k: float(v) for k, v in weights.items()})

    score = (
        w["mom_x"] * mom_x
        + w["mom20"] * mom20
        + w["rsi"] * rsi_n
        + w["macd"] * macd_n
        + w["bb_guard"] * bb_guard
        + w["z_guard"] * z_guard
    )
    # Normalize gently and clip to [-1, 1]
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    score = score.clip(-1.0, 1.0)
    return score
