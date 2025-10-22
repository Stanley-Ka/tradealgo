from __future__ import annotations

"""Intraday baseline features on minute/second bars.

Inputs per symbol: DataFrame with columns
- ts (datetime-like), open, high, low, close, volume

Outputs columns matching daily baseline expectations:
- date (timestamp), symbol, adj_open/high/low/close/adj_volume
- ret_1d (1-bar return), ret_5d (5-bar), ret_20d (20-bar)
- mom_sma_5_20, mom_20d (alias ret_20d), price_z_20, meanrev_20
- vol_z_20, atr_pct_14

Note: window sizes refer to bars (e.g., 20 = 20 minutes at 1m bars).
"""

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def compute_intraday_baseline(
    df: pd.DataFrame, symbol: str, price_col_map: dict | None = None
) -> pd.DataFrame:
    """Compute intraday baseline features on a single-symbol DataFrame.

    Expects columns: ts, open, high, low, close, volume.
    Returns a DataFrame with columns similar to daily baseline.
    """
    if price_col_map is None:
        price_col_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    req = {
        "ts",
        price_col_map["open"],
        price_col_map["high"],
        price_col_map["low"],
        price_col_map["close"],
        price_col_map["volume"],
    }
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy().sort_values("ts").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["ts"])  # intraday timestamp
    # Map adjusted columns to raw intraday
    o = out[price_col_map["open"]].astype(float)
    h = out[price_col_map["high"]].astype(float)
    l = out[price_col_map["low"]].astype(float)
    c = out[price_col_map["close"]].astype(float)
    v = out[price_col_map["volume"]].astype(float)
    (
        out["adj_open"],
        out["adj_high"],
        out["adj_low"],
        out["adj_close"],
        out["adj_volume"],
    ) = (o, h, l, c, v)
    # Returns over bars
    out["ret_1d"] = c.pct_change(1)
    out["ret_5d"] = c.pct_change(5)
    out["ret_20d"] = c.pct_change(20)
    # SMA crossover
    sma5 = c.rolling(5, min_periods=5).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    out["mom_sma_5_20"] = _safe_div(sma5, sma20) - 1.0
    out["mom_20d"] = out["ret_20d"]
    # Price z over 20 bars
    mean20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    out["price_z_20"] = _safe_div(c - mean20, std20)
    out["meanrev_20"] = -(c / mean20 - 1.0)
    # Volume z (log volume)
    lv = np.log(v.replace(0, np.nan))
    lv_mean20 = lv.rolling(20, min_periods=20).mean()
    lv_std20 = lv.rolling(20, min_periods=20).std(ddof=0)
    out["vol_z_20"] = _safe_div(lv - lv_mean20, lv_std20)
    # ATR% over 14 bars
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14.0, adjust=False).mean()
    out["atr_pct_14"] = _safe_div(atr14, c)
    # Simple VWAP distance over last 20 bars
    pv = c * v
    vwap20 = _safe_div(
        pv.rolling(20, min_periods=5).sum(), v.rolling(20, min_periods=5).sum()
    )
    out["vwap_dev_20"] = _safe_div(c - vwap20, vwap20)
    # Recent breakout flags
    out["breakout_high_20"] = (c > h.rolling(20, min_periods=5).max()).astype(float)
    out["breakout_low_20"] = (c < l.rolling(20, min_periods=5).min()).astype(float)
    # Relative volume surge vs 20-bar mean
    vma20 = v.rolling(20, min_periods=5).mean()
    out["vol_rel_20"] = _safe_div(v, vma20)
    # Hygiene
    for col in ("price_z_20", "vol_z_20"):
        out[col] = out[col].clip(-10, 10)
    out["symbol"] = symbol
    return out
