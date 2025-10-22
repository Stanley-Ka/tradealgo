from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_adx(
    df: pd.DataFrame,
    period: int = 14,
    params: dict | None = None,
) -> pd.Series:
    """Directional Movement Index (ADX) with signed direction in [-1,1].

    Score = sign * (ADX/100), where sign = (DI+ - DI-) / (DI+ + DI-).
    Uses Wilder-style EWM smoothing (alpha=1/period).
    """
    if isinstance(params, dict):
        period = int(params.get("period", period))

    h = df["adj_high"].astype(float)
    l = df["adj_low"].astype(float)
    c = df["adj_close"].astype(float)

    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    up_move = h - prev_h
    down_move = prev_l - l
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(
        lower=0.0
    )
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(
        lower=0.0
    )

    tr1 = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing via EWM alpha=1/period
    alpha = 1.0 / float(max(1, period))
    tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = 100.0 * (plus_dm_s / tr_s)
        di_minus = 100.0 * (minus_dm_s / tr_s)
        dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)

    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    # Signed direction
    with np.errstate(divide="ignore", invalid="ignore"):
        sign = (di_plus - di_minus) / (di_plus + di_minus)
    score = sign * (adx / 100.0)
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
