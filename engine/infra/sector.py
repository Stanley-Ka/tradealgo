from __future__ import annotations

import os

import pandas as pd


def load_sector_map(sector_map: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sector_map, pd.DataFrame):
        df = sector_map.copy()
    else:
        if not os.path.exists(sector_map):
            raise FileNotFoundError(sector_map)
        df = pd.read_csv(sector_map)
    if "symbol" not in df.columns or "sector" not in df.columns:
        raise RuntimeError("sector map must have columns: symbol,sector")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df[["symbol", "sector"]]


def apply_sector_cap(
    picks: pd.DataFrame,
    sector_map: str | pd.DataFrame,
    cap: int,
    rank_col: str = "meta_prob",
) -> pd.DataFrame:
    """Apply a per-sector cap to a picks DataFrame.

    Expects 'symbol' and a ranking column (default 'meta_prob'). Returns a
    new DataFrame limited to at most 'cap' rows per sector, preserving order.
    """
    if cap is None or int(cap) <= 0 or picks.empty:
        return picks
    sm = load_sector_map(sector_map)
    df = picks.merge(sm, on="symbol", how="left")
    df["_sec"] = df["sector"].fillna("UNKNOWN")
    df = df.sort_values(rank_col, ascending=False)
    out = (
        df.groupby("_sec").head(int(cap)).drop(columns=["_sec"]).reset_index(drop=True)
    )
    return out
