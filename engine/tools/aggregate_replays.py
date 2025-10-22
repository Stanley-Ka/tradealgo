from __future__ import annotations

"""Aggregate replay trade summaries into a single dataset.

Scans `data/backtests/replays/<date>/<run>/summary.csv` (or `.json`) and
combines them into a consolidated table for further analysis or model training.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate replay backtest summaries")
    p.add_argument(
        "--root",
        type=str,
        default="data/backtests/replays",
        help="Root directory containing replay runs",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional CSV path for aggregated trades",
    )
    p.add_argument(
        "--out-parquet",
        type=str,
        default="",
        help="Optional Parquet path for aggregated trades",
    )
    p.add_argument(
        "--dedupe",
        choices=["none", "latest"],
        default="latest",
        help="How to deduplicate trades with the same (replay_date, symbol, entry_date)",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print aggregate PnL statistics to stdout",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _iter_runs(root: Path):
    if not root.exists():
        return
    for date_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        replay_date = date_dir.name
        for run_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
            summary_csv = run_dir / "summary.csv"
            summary_json = run_dir / "summary.json"
            if summary_csv.exists():
                yield replay_date, run_dir, "csv", summary_csv
            elif summary_json.exists():
                yield replay_date, run_dir, "json", summary_json


def _load_summary(kind: str, path: Path) -> pd.DataFrame:
    if kind == "csv":
        return pd.read_csv(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame()


def _dedupe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none" or df.empty:
        return df
    keys = ["replay_date", "symbol", "entry_date"]
    if not all(k in df.columns for k in keys):
        return df
    df = df.sort_values(["run_ts", "entry_date"], ascending=True).reset_index(drop=True)
    if mode == "latest":
        return df.drop_duplicates(subset=keys, keep="last")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    closed = (
        df[df["exit_reason"].astype(str) != "open"]
        if "exit_reason" in df.columns
        else df
    )
    trades = len(df)
    closed_trades = len(closed)
    pnl_total = (
        float(df.get("pnl_usd", 0).sum()) if "pnl_usd" in df.columns else float("nan")
    )
    win_rate = float("nan")
    if closed_trades and "pnl_usd" in closed.columns:
        wins = (closed["pnl_usd"] > 0).sum()
        win_rate = wins / closed_trades * 100.0
    print(
        f"[aggregate] trades={trades} closed={closed_trades} win_rate={win_rate:.1f}% pnl_total={pnl_total:.2f}"
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    root = Path(args.root)
    rows: list[pd.DataFrame] = []
    for replay_date, run_dir, kind, summary_path in _iter_runs(root):
        df = _load_summary(kind, summary_path)
        if df.empty:
            continue
        df = df.copy()
        df["replay_date"] = replay_date
        df["run_ts"] = run_dir.name
        df["source_dir"] = str(run_dir)
        if "entry_date" in df.columns:
            df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
        if "exit_date" in df.columns:
            df["exit_date"] = pd.to_datetime(df["exit_date"]).dt.strftime("%Y-%m-%d")
        rows.append(df)
    if not rows:
        print(f"[aggregate] no replay summaries found under {root}")
        return
    agg = pd.concat(rows, axis=0, ignore_index=True)
    agg = _dedupe(agg, args.dedupe)
    agg.sort_values(["replay_date", "run_ts", "symbol"], inplace=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        agg.to_csv(args.out_csv, index=False)
        print(f"[aggregate] wrote {len(agg)} rows -> {args.out_csv}")
    if args.out_parquet:
        os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
        agg.to_parquet(args.out_parquet, index=False)
        print(f"[aggregate] wrote {len(agg)} rows -> {args.out_parquet}")
    if args.print_summary:
        _print_summary(agg)


if __name__ == "__main__":
    main()
