from __future__ import annotations

import os
import pandas as pd


def load_sentiment(path: str) -> pd.DataFrame:
    """Load a sentiment file with columns: date, symbol, sentiment in [-1,1].

    Supports CSV or Parquet. Date is parsed to datetime (date-only granularity is ok).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "date" not in df.columns or "symbol" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("sentiment file must have columns: date, symbol, sentiment")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])  # tolerate date/time
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["sentiment"] = df["sentiment"].astype(float).clip(-1.0, 1.0)
    return df[["date", "symbol", "sentiment"]]


def compute_spec_news(df: pd.DataFrame, sentiment: pd.DataFrame | None = None) -> pd.Series:
    """Join external sentiment if provided; otherwise return zeros.

    The returned series is in [-1,1], aligned to df index on (date,symbol).
    """
    if sentiment is None or sentiment.empty:
        return pd.Series(0.0, index=df.index, dtype=float)
    left = df[["date", "symbol"]].copy()
    left["date"] = pd.to_datetime(left["date"])  # ensure dtype
    left["symbol"] = left["symbol"].astype(str).str.upper()
    joined = left.merge(sentiment, on=["date", "symbol"], how="left")
    s = joined["sentiment"].fillna(0.0).astype(float).clip(-1.0, 1.0)
    s.index = df.index
    return s

