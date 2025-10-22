from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(max(1, period))
    avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_spec_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    params: dict | None = None,
) -> pd.Series:
    """Stochastic RSI specialist in [-1,1].

    Score = 2 * %K - 1, where %K = (RSI - RSImin) / (RSImax - RSImin), optionally smoothed.
    """
    if isinstance(params, dict):
        rsi_period = int(params.get("rsi_period", rsi_period))
        stoch_period = int(params.get("stoch_period", stoch_period))
        smooth_k = int(params.get("smooth_k", smooth_k))

    close = df["adj_close"].astype(float)
    rsi = _rsi(close, period=rsi_period)
    roll = rsi.rolling(stoch_period, min_periods=max(3, stoch_period // 2))
    rmin = roll.min()
    rmax = roll.max()
    stoch = (rsi - rmin) / (rmax - rmin).replace(0.0, np.nan)
    if smooth_k > 1:
        stoch = stoch.rolling(smooth_k, min_periods=1).mean()
    score = 2.0 * stoch.fillna(0.5) - 1.0
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
