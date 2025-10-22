from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def _read_sector_map(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # normalize expected columns
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("ric")
    sec_col = cols.get("sector") or cols.get("industry")
    if not sym_col or not sec_col:
        raise ValueError("sector map must have columns: symbol, sector")
    out = df[[sym_col, sec_col]].copy()
    out.columns = ["symbol", "sector"]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["sector"] = out["sector"].astype(str)
    return out


def compute_spec_sector(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.Series:
    """Compute a sector-relative strength specialist in [-1,1].

    Strategy:
    - If sector_map_csv provided, compute within-sector z-score of a momentum column per day.
    - Else, compute cross-sectional momentum z-score across entire universe per day.

    Params:
      sector_map_csv: path to CSV with columns [symbol, sector]
      mom_col: which column to rank/z-score (default 'mom_sma_5_20', fallback 'ret_20d')
      clip: clip z-score to [-clip, clip] and rescale to [-1,1] (default 2.0)
    """
    p = params or {}
    mom_col = str(p.get("mom_col", "mom_sma_5_20"))
    clip_val = float(p.get("clip", 2.0))

    use_col = (
        mom_col
        if mom_col in df.columns
        else ("ret_20d" if "ret_20d" in df.columns else None)
    )
    if use_col is None:
        # nothing to compute from; return zeros
        return pd.Series(0.0, index=df.index, dtype=float)

    left = df[["date", "symbol", use_col]].copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left["symbol"] = left["symbol"].astype(str).str.upper()

    # Attach sector if map available
    sec_map_path = p.get("sector_map_csv")
    if isinstance(sec_map_path, str) and sec_map_path.strip():
        try:
            smap = _read_sector_map(sec_map_path)
            left = left.merge(smap, on="symbol", how="left")
        except Exception:
            left["sector"] = None
    else:
        left["sector"] = None

    def _zscore(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=x.index)
        return (x - mu) / sd

    # Compute z-scores per day within sector if available, else across all names
    if left["sector"].notna().any():
        z = left.groupby(["date", "sector"], observed=True)[use_col].transform(_zscore)
    else:
        z = left.groupby("date")[use_col].transform(_zscore)

    # Map to [-1,1] with clipping
    z = z.clip(-clip_val, clip_val) / max(clip_val, 1e-6)
    z.index = df.index
    return z.astype(float)
