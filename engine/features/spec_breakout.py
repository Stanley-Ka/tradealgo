from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_breakout(
    df: pd.DataFrame,
    window: int = 20,
    atr_window: int = 14,
    params: dict | None = None,
) -> pd.Series:
    """Donchian/ATR breakout score in [-1, 1].

    Positive when price breaks/leans above recent range, negative below.
    Uses Donchian channels and ATR% as a volatility-scaled signal.

    Robustness improvements:
    - Compute Donchian channels and ATR fallback per symbol when a symbol column is present.
    """
    if isinstance(params, dict):
        window = int(params.get("window", window))
        atr_window = int(params.get("atr_window", atr_window))
    h, l, c = (
        df["adj_high"].astype(float),
        df["adj_low"].astype(float),
        df["adj_close"].astype(float),
    )

    minp = max(5, window // 2)
    if "symbol" in df.columns:
        sym = df["symbol"]
        try:
            from pandas.api.types import is_categorical_dtype  # type: ignore

            if is_categorical_dtype(sym):
                sym = sym.cat.remove_unused_categories()
        except Exception:
            pass
        don_high = (
            h.groupby(sym, observed=True)
            .rolling(window, min_periods=minp)
            .max()
            .reset_index(level=0, drop=True)
        )
        don_low = (
            l.groupby(sym, observed=True)
            .rolling(window, min_periods=minp)
            .min()
            .reset_index(level=0, drop=True)
        )
    else:
        don_high = h.rolling(window, min_periods=minp).max()
        don_low = l.rolling(window, min_periods=minp).min()
    rng = (don_high - don_low).replace(0, np.nan)

    # Position of close within channel [0,1]
    pos = (c - don_low) / rng
    pos = pos.clip(0.0, 1.0)

    # ATR proxy: use baseline atr_pct_14 if available, else EWM true range / close
    if "atr_pct_14" in df.columns:
        atr_pct = df["atr_pct_14"].astype(float)
    else:
        if "symbol" in df.columns:
            sym2 = df["symbol"]
            try:
                from pandas.api.types import is_categorical_dtype  # type: ignore

                if is_categorical_dtype(sym2):
                    sym2 = sym2.cat.remove_unused_categories()
            except Exception:
                pass
            prev_close = c.groupby(sym2, observed=True).shift(1)
        else:
            prev_close = c.shift(1)
        tr = pd.concat(
            [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
        ).max(axis=1)
        if "symbol" in df.columns:
            sym3 = df["symbol"]
            try:
                from pandas.api.types import is_categorical_dtype  # type: ignore

                if is_categorical_dtype(sym3):
                    sym3 = sym3.cat.remove_unused_categories()
            except Exception:
                pass
            atr = tr.groupby(sym3, observed=True).apply(
                lambda s: s.ewm(alpha=1 / float(atr_window), adjust=False).mean()
            )
            if isinstance(atr.index, pd.MultiIndex):
                atr.index = atr.index.droplevel(0)
        else:
            atr = tr.ewm(alpha=1 / float(atr_window), adjust=False).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_pct = (atr / c).replace([np.inf, -np.inf], np.nan)

    # Score: center at 0.5, amplify by ATR% up to a cap; guard extremes
    center = (pos - 0.5) * 2.0  # [-1,1]
    atr_amp = atr_pct.clip(0.0, 0.1) / 0.1  # [0,1] when atr% <= 10%
    score = center * (0.5 + 0.5 * atr_amp)  # modest boost in high-ATR regimes

    score = score.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return score.clip(-1.0, 1.0)
