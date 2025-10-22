from __future__ import annotations

"""Helpers to attach risk metrics (ADV, ATR%) for a target date.

Used by alert, entry, and paper tools to avoid duplication.
"""

import pandas as pd
import numpy as np


def attach_adv_atr(panel: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Compute ADV (20-bar dollar volume) and ATR% on the panel and return a
    DataFrame with rows for target_date containing symbol, adv20, atr_pct_14.
    Robustly fills atr_pct_14 via EWM true range per symbol and forward-fill.

    The implementation keeps memory usage low by trimming the panel to the
    lookback window we actually need and by avoiding deep copies of wide
    DataFrames.
    """
    target_ts = pd.Timestamp(target_date)

    required_cols = [
        "symbol",
        "date",
        "adj_close",
        "adj_volume",
        "adj_high",
        "adj_low",
    ]
    missing = [col for col in required_cols if col not in panel.columns]
    if missing:
        raise KeyError(f"attach_adv_atr missing required columns: {missing}")

    optional_cols: list[str] = []
    if "atr_pct_14" in panel.columns:
        optional_cols.append("atr_pct_14")

    # Use a small date window (120 calendar days) around the target date to
    # avoid holding the entire history in memory when we only need recent bars.
    lookback_days = 120
    panel_dates = pd.to_datetime(panel["date"], errors="coerce")
    cutoff = target_ts - pd.Timedelta(days=lookback_days)
    recent_mask = (panel_dates <= target_ts) & (panel_dates >= cutoff)
    if not recent_mask.any():
        return pd.DataFrame(columns=["symbol", "adv20", "atr_pct_14"])

    f = panel.loc[recent_mask, required_cols + optional_cols].copy()
    f["date"] = panel_dates.loc[recent_mask].values
    f = f.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    f["symbol"] = f["symbol"].astype(str).str.upper()
    # ADV
    f["dollar_vol"] = f["adj_close"].astype(float) * f["adj_volume"].astype(float)
    grouped = f.groupby("symbol")["dollar_vol"]
    adv = grouped.rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    counts = grouped.rolling(20, min_periods=1).count().reset_index(level=0, drop=True)
    # Require at least 5 observations; otherwise treat liquidity as missing.
    adv[counts < 5] = np.nan
    f["adv20"] = adv
    # ATR% robust fill
    need_atr_fill = ("atr_pct_14" not in f.columns) or f["atr_pct_14"].isna().any()
    if need_atr_fill:
        prev_close = f.groupby("symbol")["adj_close"].shift(1)
        tr1 = (f["adj_high"] - f["adj_low"]).abs()
        tr2 = (f["adj_high"] - prev_close).abs()
        tr3 = (f["adj_low"] - prev_close).abs()
        tr = tr1.where(tr1 >= tr2, tr2)
        tr = tr.where(tr >= tr3, tr3)
        # Fallback if high/low missing: absolute close change
        tr_close = (f["adj_close"] - prev_close).abs()
        tr = tr.fillna(tr_close)
        atr14 = tr.groupby(f["symbol"]).apply(
            lambda s: s.ewm(alpha=1 / 14.0, adjust=False).mean()
        )
        if isinstance(atr14.index, pd.MultiIndex):
            atr14.index = atr14.index.droplevel(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_pct = (atr14 / f["adj_close"]).replace([np.inf, -np.inf], np.nan)
        if "atr_pct_14" in f.columns:
            f["atr_pct_14"] = f["atr_pct_14"].fillna(atr_pct)
        else:
            f["atr_pct_14"] = atr_pct
    f["atr_pct_14"] = f.groupby("symbol")["atr_pct_14"].ffill()
    # Slice target date
    day = f[f["date"] == target_ts][["symbol", "adv20", "atr_pct_14"]].copy()
    return day


def attach_adv_atr_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute ADV20 (dollar volume) and ATR% for the entire panel efficiently.

    Returns a new DataFrame with added columns 'adv20' and 'atr_pct_14'. If
    'atr_pct_14' already exists, missing values will be filled using a robust
    EWM true range per symbol.

    This avoids repeated per-date computation when building datasets spanning
    long ranges (e.g., swing training datasets).
    """
    required_cols = [
        "symbol",
        "date",
        "adj_close",
        "adj_volume",
        "adj_high",
        "adj_low",
    ]
    missing = [col for col in required_cols if col not in panel.columns]
    if missing:
        raise KeyError(f"attach_adv_atr_panel missing required columns: {missing}")

    # Build column selection robustly (Python 3.10 safe)
    extra_cols: list[str] = ["atr_pct_14"] if "atr_pct_14" in panel.columns else []
    f = panel[required_cols + extra_cols].copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f["symbol"] = f["symbol"].astype(str).str.upper()
    f = f.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    f["dollar_vol"] = f["adj_close"].astype(float) * f["adj_volume"].astype(float)
    # ADV with minimum observations threshold to reduce noise
    grouped = f.groupby("symbol")["dollar_vol"]
    adv = grouped.rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    counts = grouped.rolling(20, min_periods=1).count().reset_index(level=0, drop=True)
    adv[counts < 5] = np.nan
    f["adv20"] = adv
    # ATR% robust fill per symbol if needed
    need_atr_fill = ("atr_pct_14" not in f.columns) or f["atr_pct_14"].isna().any()
    if need_atr_fill:
        prev_close = f.groupby("symbol")["adj_close"].shift(1)
        tr1 = (f["adj_high"] - f["adj_low"]).abs()
        tr2 = (f["adj_high"] - prev_close).abs()
        tr3 = (f["adj_low"] - prev_close).abs()
        tr = tr1.where(tr1 >= tr2, tr2)
        tr = tr.where(tr >= tr3, tr3)
        tr_close = (f["adj_close"] - prev_close).abs()
        tr = tr.fillna(tr_close)
        atr14 = tr.groupby(f["symbol"]).apply(
            lambda s: s.ewm(alpha=1 / 14.0, adjust=False).mean()
        )
        if isinstance(atr14.index, pd.MultiIndex):
            atr14.index = atr14.index.droplevel(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_pct = (atr14 / f["adj_close"]).replace([np.inf, -np.inf], np.nan)
        if "atr_pct_14" in f.columns:
            f["atr_pct_14"] = f["atr_pct_14"].fillna(atr_pct)
        else:
            f["atr_pct_14"] = atr_pct
    f["atr_pct_14"] = f.groupby("symbol")["atr_pct_14"].ffill()
    # Merge back onto the input to preserve any extra columns
    out = panel.copy()
    out = out.merge(
        f[["symbol", "date", "adv20", "atr_pct_14"]], on=["symbol", "date"], how="left"
    )
    return out
