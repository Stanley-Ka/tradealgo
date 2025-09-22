from __future__ import annotations

import pandas as pd

from .spec_patterns import compute_spec_patterns
from .spec_technicals import compute_spec_technicals
from .spec_sequence import compute_spec_sequence
from .spec_news import compute_spec_news


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

    out["spec_pattern"] = compute_spec_patterns(out, weights=p_params)
    out["spec_technical"] = compute_spec_technicals(out, weights=t_params)
    if isinstance(s_params, dict):
        out["spec_sequence"] = compute_spec_sequence(out, params=s_params)
    else:
        out["spec_sequence"] = compute_spec_sequence(out)
    out["spec_nlp"] = compute_spec_news(out, news_sentiment)
    return out
