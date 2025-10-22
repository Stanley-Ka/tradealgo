from __future__ import annotations

import pandas as pd

from .spec_patterns import compute_spec_patterns
from .spec_technicals import compute_spec_technicals
from .spec_sequence import compute_spec_sequence
from .spec_breakout import compute_spec_breakout
from .spec_volume import compute_spec_flow
from .spec_adx import compute_spec_adx
from .spec_stoch import compute_spec_stoch_rsi
from .spec_williams import compute_spec_williams_r
from .spec_cci import compute_spec_cci
from .spec_lstm import compute_spec_lstm
from .spec_news import compute_spec_news
from .spec_news import compute_spec_news_enhanced
from .spec_gaprev import compute_spec_gap_meanrev
from .spec_sector import compute_spec_sector
from .regime import compute_regime_features_daily


def compute_specialist_scores(
    df: pd.DataFrame,
    news_sentiment: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Compute four specialist raw scores and append as columns.

    Expects a DataFrame containing at least: date, symbol, adj_open/adj_high/adj_low/adj_close.
    If baseline features from engine/features/baseline.py exist, technical/sequence will use them.

    Emits columns:
    - spec_pattern   [-1,1]: candlestick/structure pattern score
    - spec_technical [-1,1]: indicator composite
    - spec_sequence  [-1,1]: recent drift/vol-adjusted score
    - spec_nlp       [-1,1]: news sentiment (0 if none provided)
    """
    out = df.copy()
    for col in ("date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close"):
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    p_params = (params or {}).get("patterns") if params else None
    t_params = (params or {}).get("technicals") if params else None
    s_params = (params or {}).get("sequence") if params else None
    b_params = (params or {}).get("breakout") if params else None
    v_params = (params or {}).get("flow") if params else None
    a_params = (params or {}).get("adx") if params else None
    sr_params = (params or {}).get("stoch_rsi") if params else None
    wr_params = (params or {}).get("williams_r") if params else None
    cci_params = (params or {}).get("cci") if params else None
    l_params = (params or {}).get("lstm") if params else None
    gr_params = (params or {}).get("gap_rev") if params else None
    n_params = (params or {}).get("news") if params else None
    sec_params = (params or {}).get("sector") if params else None
    # Regime features are global-by-date; compute once and merge if requested
    regime_params = (params or {}).get("regime") if params else None

    # Helper to check enabled flag (default True if not provided)
    def _enabled(p) -> bool:
        return not (isinstance(p, dict) and p.get("enabled") is False)

    if _enabled(p_params):
        out["spec_pattern"] = compute_spec_patterns(out, weights=p_params)
    if _enabled(t_params):
        out["spec_technical"] = compute_spec_technicals(out, weights=t_params)
    if _enabled(s_params):
        out["spec_sequence"] = compute_spec_sequence(
            out, params=s_params if isinstance(s_params, dict) else None
        )
    if _enabled(b_params):
        out["spec_breakout"] = compute_spec_breakout(
            out, params=b_params if isinstance(b_params, dict) else None
        )
    if _enabled(v_params):
        out["spec_flow"] = compute_spec_flow(
            out, weights=v_params if isinstance(v_params, dict) else None
        )
    if _enabled(a_params):
        out["spec_adx"] = compute_spec_adx(
            out, params=a_params if isinstance(a_params, dict) else None
        )
    if _enabled(sr_params):
        out["spec_stoch_rsi"] = compute_spec_stoch_rsi(
            out, params=sr_params if isinstance(sr_params, dict) else None
        )
    if _enabled(wr_params):
        out["spec_willr"] = compute_spec_williams_r(
            out, params=wr_params if isinstance(wr_params, dict) else None
        )
    if _enabled(cci_params):
        out["spec_cci"] = compute_spec_cci(
            out, params=cci_params if isinstance(cci_params, dict) else None
        )
    # LSTM requires a pre-trained model (optional). If none provided, returns zeros.
    if isinstance(l_params, dict) and l_params.get("enabled", False):
        mdl = l_params.get("model")
        out["spec_lstm"] = compute_spec_lstm(out, model=mdl, params=l_params)
    if _enabled(gr_params):
        out["spec_gaprev"] = compute_spec_gap_meanrev(
            out, params=gr_params if isinstance(gr_params, dict) else None
        )
    # News specialist: enhanced path supports provider weights, entity mapping, and earnings-only tone
    if isinstance(n_params, dict) and n_params.get("enhanced", False):
        nlp_dict = compute_spec_news_enhanced(out, news_sentiment, params=n_params)
        out["spec_nlp"] = nlp_dict.get("spec_nlp", pd.Series(0.0, index=out.index))
        if "spec_nlp_earnings" in nlp_dict:
            out["spec_nlp_earnings"] = nlp_dict["spec_nlp_earnings"]
    else:
        out["spec_nlp"] = compute_spec_news(
            out, news_sentiment, params=n_params if isinstance(n_params, dict) else None
        )

    # Sector specialist: relative strength within sector or cross-sectional
    if not (isinstance(sec_params, dict) and sec_params.get("enabled") is False):
        try:
            out["spec_sector"] = compute_spec_sector(
                out, params=sec_params if isinstance(sec_params, dict) else None
            )
        except Exception:
            # Fail-safe: if mapping/inputs missing, emit zeros
            out["spec_sector"] = 0.0

    # Optional regime features mixed into specialist table for downstream meta
    if isinstance(regime_params, dict) and regime_params.get("enabled", True):
        try:
            reg = compute_regime_features_daily(out)
            out = out.merge(reg, on="date", how="left")
        except Exception:
            pass

    # Ensure specialist outputs are finite; fill gaps with neutral scores
    for col in [c for c in out.columns if c.startswith("spec_")]:
        out[col] = out[col].fillna(0.0)

    return out
