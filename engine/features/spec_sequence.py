from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spec_sequence(
    df: pd.DataFrame, window: int = 10, params: dict | None = None
) -> pd.Series:
    """Lightweight sequence-style score in [-1, 1].

    V0 proxy: recent return EMA divided by recent volatility (rolling).
    This approximates a short-horizon drift signal akin to a simple RNN trend.

    Robustness improvements:
    - Compute prev/rolling per symbol when possible to avoid cross‑symbol leaks.
    - For `ret_kind='overnight'`, prefer precomputed `ret_overnight` if present.
    """
    if params and "window" in params:
        window = int(params["window"]) or window
    # Allow alternative return definitions for the short-horizon drift proxy
    ret_kind = (
        str(params.get("ret_kind", "close")) if isinstance(params, dict) else "close"
    )

    close = df["adj_close"].astype(float)
    open_ = df["adj_open"].astype(float)

    if ret_kind == "intraday":
        # (close / open - 1) — no history required
        ret1 = close / open_ - 1.0
    elif ret_kind == "overnight":
        if "ret_overnight" in df.columns:
            ret1 = df["ret_overnight"].astype(float)
        else:
            # (open / prev_close - 1) per symbol when available
            if "symbol" in df.columns:
                sym0 = df["symbol"]
                try:
                    from pandas.api.types import is_categorical_dtype  # type: ignore

                    if is_categorical_dtype(sym0):
                        sym0 = sym0.cat.remove_unused_categories()
                except Exception:
                    pass
                prev_c = df["adj_close"].groupby(sym0, observed=True).shift(1)
            else:
                prev_c = close.shift(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret1 = open_ / prev_c - 1.0
    else:
        # default: close-to-close
        ret1 = df.get("ret_1d")
        if ret1 is None:
            if "symbol" in df.columns:
                ret1 = close.groupby(df["symbol"]).pct_change(1)
            else:
                ret1 = close.pct_change(1)

    # EMA and rolling std per symbol (if available)
    minp = max(2, window // 2)
    if "symbol" in df.columns:
        # Avoid empty-category groups which can break rolling() on some pandas versions
        sym = df["symbol"]
        try:
            from pandas.api.types import is_categorical_dtype  # type: ignore

            if is_categorical_dtype(sym):
                sym = sym.cat.remove_unused_categories()
        except Exception:
            pass
        # Use observed=True to ignore categories with no rows
        ema = ret1.groupby(sym, observed=True).apply(
            lambda s: s.ewm(span=window, adjust=False, min_periods=minp).mean()
        )
        if isinstance(ema.index, pd.MultiIndex):
            ema.index = ema.index.droplevel(0)
        vol = (
            ret1.groupby(sym, observed=True)
            .rolling(window, min_periods=minp)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
    else:
        ema = ret1.ewm(span=window, adjust=False, min_periods=minp).mean()
        vol = ret1.rolling(window, min_periods=minp).std(ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = ema / vol
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # gentle squash to [-1, 1]
    score = z.clip(-3, 3) / 3.0
    return score
