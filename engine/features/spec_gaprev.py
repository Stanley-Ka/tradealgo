from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_gap_meanrev(
    df: pd.DataFrame,
    params: dict | None = None,
) -> pd.Series:
    """Gap mean‑reversion specialist in [-1,1].

    Idea: large down gaps tend to mean‑revert on average (and up gaps can fade).
    We form a signed score that is positive when a reversion is favored.

    Robustness improvements:
    - Use precomputed `ret_overnight` if present (from baseline features).
      This avoids incorrect cross‑symbol shifting when `df` only contains a
      single date with many symbols.
    - Otherwise, compute prev_close per symbol via groupby‑shift.

    Components:
    - Gap normalized by ATR%: g = (open/prev_close - 1) / max(atr_pct, eps)
      Score contribution = -clip(g, -cap, cap) / cap
    - Small guard with price z-score (prefer reversion when far from mean): -0.2*price_z_20

    Params (optional dict):
    - cap_sigma: float, cap for |normalized gap| before scaling to [-1,1] (default 2.0)
    - atr_floor: float, minimum ATR% to avoid explosion (default 0.005 i.e., 0.5%)
    - z_weight: float, weight for price z-score guard (default 0.2)
    """
    cap = float((params or {}).get("cap_sigma", 2.0))
    atr_floor = float((params or {}).get("atr_floor", 0.005))
    z_w = float((params or {}).get("z_weight", 0.2))

    o = df["adj_open"].astype(float)
    c = df["adj_close"].astype(float)

    # Prefer baseline overnight return if available
    if "ret_overnight" in df.columns:
        gap = df["ret_overnight"].astype(float)
    else:
        # Compute prev_close per symbol; if symbol column missing, fall back to plain shift
        if "symbol" in df.columns:
            prev_c = df.groupby("symbol")["adj_close"].shift(1)
        else:
            prev_c = c.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = o / prev_c - 1.0

    # ATR% if present, else robust EWM TR/close (per symbol)
    if "atr_pct_14" in df.columns:
        atrp = df["atr_pct_14"].astype(float).fillna(0.0)
    else:
        h = df["adj_high"].astype(float)
        l = df["adj_low"].astype(float)
        if "symbol" in df.columns:
            prev_c_for_tr = df.groupby("symbol")["adj_close"].shift(1)
        else:
            prev_c_for_tr = c.shift(1)
        tr = pd.concat(
            [(h - l).abs(), (h - prev_c_for_tr).abs(), (l - prev_c_for_tr).abs()],
            axis=1,
        ).max(axis=1)
        # EWM per symbol if possible
        if "symbol" in df.columns:
            atr = tr.groupby(df["symbol"]).apply(
                lambda s: s.ewm(alpha=1 / 14.0, adjust=False).mean()
            )
            if isinstance(atr.index, pd.MultiIndex):
                atr.index = atr.index.droplevel(0)
        else:
            atr = tr.ewm(alpha=1 / 14.0, adjust=False).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            atrp = (atr / c).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Normalize gap by ATR% with a floor
    atr_safe = atrp.copy()
    atr_safe[~np.isfinite(atr_safe) | (atr_safe < atr_floor)] = atr_floor
    g_norm = (gap / atr_safe).clip(-cap, cap) / cap
    score = -g_norm  # mean‑reversion: down gap -> positive

    # Add small guard from price z (prefer reversion when far from mean)
    if "price_z_20" in df.columns:
        z = df["price_z_20"].astype(float).clip(-3, 3)
        score = score + (-z_w * z / 3.0)

    # Hygiene
    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
