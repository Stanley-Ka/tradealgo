"""Merge multiple decision logs and optionally add recency-based sample weights.

Usage:
  python -m engine.models.merge_decision_logs \
    --inputs data/backtests/decision_log.csv --inputs data/paper/decision_log.csv \
    --out data/merged/decision_log_merged.csv \
    --dedupe keep_last --filter-selected --weight-mode exp --half-life-days 60 --normalize-weights

Notes:
  - Expects a column `date_decision` (YYYY-MM-DD) and `symbol`.
  - When `--filter-selected` is set, keeps only rows where `selected == True`.
  - Weight modes:
      * none: no `sample_weight` column added
      * exp:  weight = 0.5 ** ((max_date - date).days / half_life_days)
      * linear: weight = max(0, 1 - (max_date - date).days / window_days)
    Use `--normalize-weights` to rescale mean weight to 1.0.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge decision logs with optional recency weights"
    )
    p.add_argument(
        "--inputs",
        action="append",
        required=True,
        help="Input decision log CSV (repeat for multiple)",
    )
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    p.add_argument(
        "--dedupe", choices=["keep_last", "keep_first", "none"], default="keep_last"
    )
    p.add_argument(
        "--dedupe-keys",
        type=str,
        default="date_decision,symbol",
        help="Comma-separated key columns for dedupe",
    )
    p.add_argument(
        "--filter-selected",
        action="store_true",
        help="Keep only rows with selected==True if column exists",
    )
    p.add_argument(
        "--min-date", type=str, default="", help="Minimum date_decision (YYYY-MM-DD)"
    )
    p.add_argument(
        "--max-date", type=str, default="", help="Maximum date_decision (YYYY-MM-DD)"
    )
    p.add_argument("--weight-mode", choices=["none", "exp", "linear"], default="none")
    p.add_argument(
        "--half-life-days", type=float, default=60.0, help="Half-life for exp weights"
    )
    p.add_argument(
        "--window-days", type=float, default=120.0, help="Window for linear weights"
    )
    p.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Rescale average sample_weight to 1.0",
    )
    return p.parse_args(argv)


def _read_csv_safe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize date_decision to datetime
    if "date_decision" in df.columns:
        df["date_decision"] = pd.to_datetime(df["date_decision"])
    return df


def _merge_frames(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[merge] warning: missing input: {p}")
            continue
        try:
            dfs.append(_read_csv_safe(p))
        except Exception as e:
            print(f"[merge] warning: failed reading {p}: {e}")
    if not dfs:
        raise RuntimeError("No readable inputs")
    # Outer union of columns
    cols = set()
    for d in dfs:
        cols.update(d.columns)
    cols = list(cols)
    dfs = [d.reindex(columns=cols) for d in dfs]
    out = pd.concat(dfs, ignore_index=True)
    return out


def _apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.filter_selected and "selected" in out.columns:
        out = out[out["selected"] == True]  # noqa: E712
    if args.min_date:
        md = pd.to_datetime(args.min_date)
        if "date_decision" in out.columns:
            out = out[out["date_decision"] >= md]
    if args.max_date:
        xd = pd.to_datetime(args.max_date)
        if "date_decision" in out.columns:
            out = out[out["date_decision"] <= xd]
    return out


def _dedupe(df: pd.DataFrame, keys: List[str], mode: str) -> pd.DataFrame:
    if mode == "none":
        return df
    if not all(k in df.columns for k in keys):
        return df
    if mode == "keep_last":
        return df.sort_index().drop_duplicates(subset=keys, keep="last")
    return df.sort_index().drop_duplicates(subset=keys, keep="first")


def _add_weights(
    df: pd.DataFrame, mode: str, half_life: float, window: float, normalize: bool
) -> pd.DataFrame:
    out = df.copy()
    if mode == "none" or "date_decision" not in out.columns:
        return out
    dates = pd.to_datetime(out["date_decision"]).values
    maxd = pd.to_datetime(out["date_decision"]).max()
    dt_days = (
        (maxd - pd.to_datetime(out["date_decision"]).values)
        .astype("timedelta64[D]")
        .astype(float)
    )
    if mode == "exp":
        w = np.power(0.5, dt_days / max(1e-6, float(half_life)))
    else:
        w = 1.0 - (dt_days / max(1e-6, float(window)))
        w = np.clip(w, 0.0, 1.0)
    if normalize and np.isfinite(w).any():
        m = np.nanmean(w)
        if m > 0:
            w = w / m
    out["sample_weight"] = w
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    inputs = []
    for i in args.inputs:
        if "," in i:
            inputs.extend([p.strip() for p in i.split(",") if p.strip()])
        else:
            inputs.append(i)
    df = _merge_frames(inputs)
    df = _apply_filters(df, args)
    keys = [k.strip() for k in args.dedupe_keys.split(",") if k.strip()]
    df = _dedupe(df, keys, args.dedupe)
    df = _add_weights(
        df,
        args.weight_mode,
        float(args.half_life_days),
        float(args.window_days),
        bool(args.normalize_weights),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[merge] wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
