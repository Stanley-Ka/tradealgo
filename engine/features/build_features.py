"""Build baseline features dataset for a universe and date range.

Example:
  python -m engine.features.build_features --universe-file engine/data/universe/nasdaq100.example.txt \
      --start 2015-01-01 --end 2025-01-01 --out datasets/features_nasdaq100_1D.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
from tqdm import tqdm  # type: ignore

from ..data.load_panel import load_symbol
from ..data.store import storage_root
from .baseline import compute_baseline_features


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build baseline features for a universe")
    p.add_argument(
        "--universe-file",
        required=True,
        help="Path to a text file with one SYMBOL per line",
    )
    p.add_argument(
        "--provider",
        choices=["yahoo", "alphavantage", "polygon"],
        default="yahoo",
        help="Data provider for loading parquets",
    )
    p.add_argument(
        "--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD"
    )
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="End date YYYY-MM-DD (default=all available)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output Parquet path (default: storage_root/datasets/features_daily_1D.parquet)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    syms = read_universe(args.universe_file)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else None

    frames: List[pd.DataFrame] = []
    for sym in tqdm(syms, desc="features"):
        try:
            df = load_symbol(sym, start=start, end=end, provider=args.provider)
            # Keep only needed columns + labels if present
            cols_keep = [
                "date",
                "symbol",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                "adj_volume",
                # Pass through known labels if present
                *[c for c in ("fret_1d", "label_up_1d") if c in df.columns],
            ]
            df = df[cols_keep]
            feat = compute_baseline_features(df)
            frames.append(feat)
        except FileNotFoundError:
            continue
    if not frames:
        print("No data found for the given universe/date range.")
        return
    out = pd.concat(frames, axis=0, ignore_index=True)

    # Default output location
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "datasets"), exist_ok=True)
        out_path = os.path.join(root, "datasets", "features_daily_1D.parquet")

    # Ensure parent directory exists even when user passes a custom path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[features] rows={len(out)} symbols={len(syms)} -> {out_path}")


if __name__ == "__main__":
    main()
