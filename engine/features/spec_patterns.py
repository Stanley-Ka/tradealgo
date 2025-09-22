from __future__ import annotations

import numpy as np
import pandas as pd


def _bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c, p_o, p_c = df["adj_open"], df["adj_close"], df["adj_open"].shift(1), df["adj_close"].shift(1)
    prev_red = p_c < p_o
    cur_green = c > o
    engulf = (c >= p_o) & (o <= p_c)
    return (prev_red & cur_green & engulf).astype(float)


def _hammer(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["adj_open"], df["adj_high"], df["adj_low"], df["adj_close"]
    body = (c - o).abs()
    range_ = (h - l).replace(0, np.nan)
    lower_shadow = (np.minimum(o, c) - l).abs()
    upper_shadow = (h - np.maximum(o, c)).abs()
    # Hammer: small body near high, long lower shadow
    cond = (lower_shadow / range_ > 0.5) & (upper_shadow / range_ < 0.2) & (body / range_ < 0.3)
    return cond.fillna(False).astype(float)


def _inside_day_breakout(df: pd.DataFrame) -> pd.Series:
    # Yesterday range contains today; tomorrow close breakout (lead-lag proxy using today only):
    # We proxy a potential breakout with narrow range + close near range edge.
    h, l, c = df["adj_high"], df["adj_low"], df["adj_close"]
    rng = (h - l).replace(0, np.nan)
    narrow = (rng / rng.rolling(10, min_periods=5).mean()) < 0.7
    near_high = (h - c) / rng < 0.2
    near_low = (c - l) / rng < 0.2
    score = np.where(narrow & near_high, 1.0, np.where(narrow & near_low, -1.0, 0.0))
    return pd.Series(score, index=df.index, dtype=float)


def compute_spec_patterns(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Basic candlestick/structure pattern score in [-1, 1].

    Positive favors upside. Uses only same-bar OHLC and 1-bar lag info.
    """
    be = _bullish_engulfing(df)
    hm = _hammer(df)
    br = _inside_day_breakout(df)
    w = weights or {"engulfing": 0.5, "hammer": 0.3, "breakout": 0.2}
    raw = float(w.get("engulfing", 0.5)) * be + float(w.get("hammer", 0.3)) * hm + float(w.get("breakout", 0.2)) * br
    return raw.clip(-1.0, 1.0)
