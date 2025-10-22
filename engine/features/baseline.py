from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def compute_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple, diverse features for one symbol's daily data.

    Expects columns: date, adj_open, adj_high, adj_low, adj_close, adj_volume.
    Returns a DataFrame with original date/symbol and added feature columns.
    """
    required = {
        "date",
        "symbol",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["adj_close"]
    high = out["adj_high"]
    low = out["adj_low"]
    volume = out["adj_volume"]
    open_ = out["adj_open"]

    # Returns
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    # Decompose daily move into overnight and intraday components
    prev_close = close.shift(1)
    out["ret_overnight"] = _safe_div(open_ - prev_close, prev_close)
    out["ret_intraday"] = _safe_div(close - open_, open_)

    # Moving averages and momentum
    sma5 = close.rolling(5, min_periods=5).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    out["mom_sma_5_20"] = _safe_div(sma5, sma20) - 1.0
    out["mom_20d"] = out["ret_20d"]

    # Price z-scores
    mean20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    out["price_z_20"] = _safe_div(close - mean20, std20)

    # Mean reversion proxy (deviation to 20D mean, sign-flipped)
    out["meanrev_20"] = -(close / mean20 - 1.0)

    # Volume z-score (log volume is often more stable)
    lv = np.log(volume.replace(0, np.nan))
    lv_mean20 = lv.rolling(20, min_periods=20).mean()
    lv_std20 = lv.rolling(20, min_periods=20).std(ddof=0)
    out["vol_z_20"] = _safe_div(lv - lv_mean20, lv_std20)

    # ATR(14) normalized
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14.0, adjust=False).mean()
    out["atr_pct_14"] = _safe_div(atr14, close)

    # Feature hygiene: clip extreme z-scores to reduce leakage risk
    for col in ("price_z_20", "vol_z_20"):
        out[col] = out[col].clip(-10, 10)

    return out
