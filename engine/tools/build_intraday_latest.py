from __future__ import annotations

"""Build latest intraday features snapshot from per-symbol bars.

Assumes bars are stored per symbol under a root like:
  data/equities/polygon/intraday_1m/SYMBOL.parquet (or CSV)

Reads the last N bars per symbol, computes intraday baseline features with
windows in bars, and outputs a combined snapshot with the last row per symbol.
"""

import argparse
import os
from typing import List, Optional

import pandas as pd

from ..features.intraday_baseline import compute_intraday_baseline


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build latest intraday features snapshot")
    p.add_argument(
        "--bars-root",
        type=str,
        required=True,
        help="Root folder with intraday_{interval}/SYMBOL.parquet",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Bar interval folder suffix, e.g., 1m or 5m",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols (default: infer from files)",
    )
    p.add_argument(
        "--lookback-bars",
        type=int,
        default=200,
        help="Bars to read per symbol for features",
    )
    p.add_argument(
        "--out", type=str, required=True, help="Output Parquet path for latest snapshot"
    )
    return p.parse_args(argv)


def _infer_symbols(dir_path: str) -> List[str]:
    syms: List[str] = []
    for fn in os.listdir(dir_path):
        name, ext = os.path.splitext(fn)
        if ext.lower() in (".parquet", ".csv"):
            syms.append(name.upper())
    return syms


def _read_bars(path: str, n: int) -> pd.DataFrame:
    if not os.path.exists(path):
        # Try CSV fallback
        csv_path = os.path.splitext(path)[0] + ".csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(path)
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(path)
    # Keep last n
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])  # ensure dtype
        df = df.sort_values("ts").tail(int(n)).reset_index(drop=True)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    dir_path = os.path.join(args.bars_root, f"intraday_{args.interval}")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(dir_path)
    syms = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else _infer_symbols(dir_path)
    )
    if not syms:
        raise RuntimeError("No symbols found in bars directory")
    rows: List[pd.DataFrame] = []
    for sym in syms:
        path = os.path.join(dir_path, f"{sym}.parquet")
        try:
            bars = _read_bars(path, int(args.lookback_bars))
        except Exception:
            continue
        # Map columns
        col_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        # Compute intraday baseline features
        feat = compute_intraday_baseline(bars, symbol=sym, price_col_map=col_map)
        # Keep last row (latest bar)
        rows.append(feat.tail(1))
    if not rows:
        raise RuntimeError("No intraday features computed")
    snap = pd.concat(rows, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    snap.to_parquet(args.out, index=False)
    print(f"[intraday] latest snapshot rows={len(snap)} -> {args.out}")


if __name__ == "__main__":
    main()
