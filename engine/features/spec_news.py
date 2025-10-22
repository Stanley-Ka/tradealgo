from __future__ import annotations

import math
import os
from typing import Dict, Optional

import pandas as pd


def load_sentiment(path: str) -> pd.DataFrame:
    """Load a sentiment file.

    Expected minimal columns: date, symbol(or ticker), sentiment in [-1,1].
    Optional columns are passed through if present: provider, weight, category, tag, entities, cusip.
    Supports CSV or Parquet.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    sym_col = cols.get("symbol") or cols.get("ticker")
    sent_col = cols.get("sentiment")
    if not date_col or not (sym_col or cols.get("entities")) or not sent_col:
        raise ValueError(
            "sentiment file must have columns: date, symbol(or entities), sentiment"
        )
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if sym_col:
        df[sym_col] = df[sym_col].astype(str).str.upper()
    df[sent_col] = df[sent_col].astype(float).clip(-1.0, 1.0)
    # Normalize core names
    df.rename(columns={date_col: "date", sent_col: "sentiment"}, inplace=True)
    if sym_col and sym_col != "symbol":
        df.rename(columns={sym_col: "symbol"}, inplace=True)
    return df


def compute_spec_news(
    df: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.Series:
    """Join or aggregate external sentiment to produce a specialist in [-1,1].

    Modes:
    - Default (no window): same-day join on (date,symbol) with zeros for missing.
    - Windowed (window_days>0): aggregate last N days up to date with optional
      exponential decay (half_life_days). If df has a single date (typical for
      daily prediction), apply accurate windowing. For multi-date frames (CV),
      falls back to same-day join to avoid heavy computation unless window_days
      is very small.
    Params:
      window_days: int (default 0)
      half_life_days: float (default 0 = uniform)
      agg: str in {mean, max} (default mean)
    """
    if sentiment is None or sentiment.empty:
        return pd.Series(0.0, index=df.index, dtype=float)

    window_days = int((params or {}).get("window_days", 0) or 0)
    half_life = float((params or {}).get("half_life_days", 0.0) or 0.0)
    agg = str((params or {}).get("agg", "mean"))

    left = df[["date", "symbol"]].copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left["symbol"] = left["symbol"].astype(str).str.upper()
    sent = sentiment.copy()
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    if "symbol" in sent.columns:
        sent["symbol"] = sent["symbol"].astype(str).str.upper()
    sent["sentiment"] = sent["sentiment"].astype(float).clip(-1.0, 1.0)

    # Fast path: no windowing
    if window_days <= 0:
        joined = left.merge(sent, on=["date", "symbol"], how="left")
        s = joined["sentiment"].fillna(0.0).astype(float).clip(-1.0, 1.0)
        s.index = df.index
        return s

    # If only one date in df (common for predict/alert), do exact window aggregation
    unique_dates = left["date"].unique()
    if len(unique_dates) == 1:
        tdate = unique_dates[0]
        start = pd.Timestamp(tdate) - pd.Timedelta(days=window_days)
        sub = sent[(sent["date"] >= start) & (sent["date"] <= tdate)]
        if sub.empty:
            out = pd.Series(0.0, index=df.index, dtype=float)
            return out

        def _agg_sym(sym: str) -> float:
            ss = sub[sub["symbol"] == sym]
            if ss.empty:
                return 0.0
            if half_life > 0:
                age = (tdate - ss["date"]).dt.days.astype(float)
                w = np.exp(-math.log(2.0) * (age / max(1e-6, half_life)))
            else:
                w = pd.Series(1.0, index=ss.index)
            if agg == "max":
                return float((ss["sentiment"] * w).max())
            # default: weighted mean
            den = float(w.sum()) if float(w.sum()) > 0 else 1.0
            return float((ss["sentiment"] * w).sum() / den)

        import numpy as np  # local to avoid hard dependency at import time

        syms = left["symbol"].astype(str).tolist()
        vals = [_agg_sym(s) for s in syms]
        out = pd.Series(vals, index=df.index, dtype=float).clip(-1.0, 1.0)
        return out

    # Fallback for multi-date frames: same-day join (performance-friendly during CV)
    joined = left.merge(sent, on=["date", "symbol"], how="left")
    s = joined["sentiment"].fillna(0.0).astype(float).clip(-1.0, 1.0)
    s.index = df.index
    return s


def compute_spec_news_enhanced(
    df: pd.DataFrame,
    sentiment: Optional[pd.DataFrame],
    params: Optional[dict] = None,
) -> Dict[str, pd.Series]:
    """Enhanced news specialist with provider weighting, entity resolution, and earnings-only tone.

    Returns a dict with keys:
      - spec_nlp: overall sentiment specialist [-1,1]
      - spec_nlp_earnings: earnings-tagged sentiment specialist [-1,1] (zeros if none)

    Supported params:
      - window_days: int, lookback window for decay aggregation
      - half_life_days: float, exponential decay half-life
      - agg: 'mean' or 'max'
      - provider_weights: {provider: weight}
      - entity_map_csv: optional CSV with columns [entity, symbol]
      - earnings_tags: list of strings to detect earnings news in 'category'/'tag' fields
    """
    out = {
        "spec_nlp": pd.Series(0.0, index=df.index, dtype=float),
        "spec_nlp_earnings": pd.Series(0.0, index=df.index, dtype=float),
    }
    if sentiment is None or sentiment.empty:
        return out

    p = params or {}
    provider_weights: dict[str, float] = {
        str(k).lower(): float(v) for k, v in (p.get("provider_weights") or {}).items()
    }
    earnings_tags = set(
        [str(x).lower() for x in (p.get("earnings_tags") or ["earnings"])]
    )

    sent = sentiment.copy()
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    # Resolve entities to symbols if provided
    if (
        "symbol" not in sent.columns
        and "entities" in sent.columns
        and isinstance(p.get("entity_map_csv"), str)
    ):
        try:
            m = pd.read_csv(p["entity_map_csv"])  # expects columns entity,symbol
            cols = {c.lower(): c for c in m.columns}
            ent_col = cols.get("entity")
            sym_col = cols.get("symbol") or cols.get("ticker")
            if ent_col and sym_col:
                m = m[[ent_col, sym_col]].rename(
                    columns={ent_col: "entity", sym_col: "symbol"}
                )
                m["entity"] = m["entity"].astype(str).str.lower()
                m["symbol"] = m["symbol"].astype(str).str.upper()
                # explode entities if list-like
                expl = sent.copy()
                expl["entities"] = expl["entities"].astype(str).str.lower()
                # simple split on commas/semicolons
                expl = expl.assign(
                    entity=expl["entities"].str.split(r"[,;]\s*", regex=True)
                ).explode("entity")
                sent = expl.merge(m, on="entity", how="left")
            else:
                sent["symbol"] = sent.get("symbol", "").astype(str).str.upper()
        except Exception:
            sent["symbol"] = sent.get("symbol", "").astype(str).str.upper()
    else:
        if "symbol" in sent.columns:
            sent["symbol"] = sent["symbol"].astype(str).str.upper()

    if "symbol" not in sent.columns:
        # cannot resolve, return zeros
        return out

    sent["sentiment"] = sent["sentiment"].astype(float).clip(-1.0, 1.0)

    # Apply provider weights if present
    if "provider" in sent.columns and provider_weights:
        w = sent["provider"].astype(str).str.lower().map(provider_weights).fillna(1.0)
        sent["sent_w"] = sent["sentiment"] * w
        sent["w"] = w
    else:
        sent["sent_w"] = sent["sentiment"]
        sent["w"] = 1.0

    # Identify earnings-tagged rows if possible
    earn_mask = pd.Series(False, index=sent.index)
    for col in ("category", "tag", "tags"):
        if col in sent.columns:
            earn_mask = earn_mask | sent[col].astype(str).str.lower().str.contains(
                "|".join(earnings_tags), regex=True
            )

    # Reuse windowing/decay aggregation per symbol
    base_params = {k: p.get(k) for k in ("window_days", "half_life_days", "agg")}

    def _aggregate(target: pd.DataFrame) -> pd.Series:
        return compute_spec_news(
            df, target[["date", "symbol", "sentiment"]], params=base_params
        )

    # Overall
    sent_overall = sent[["date", "symbol", "sentiment"]].copy()
    out["spec_nlp"] = _aggregate(sent_overall)

    # Earnings-only
    if earn_mask.any():
        sent_earn = sent.loc[earn_mask, ["date", "symbol", "sentiment"]]
        out["spec_nlp_earnings"] = _aggregate(sent_earn)
    return out
