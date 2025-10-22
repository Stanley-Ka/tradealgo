"""Summarize dataset and walk-forward outputs into compact JSON.

- Dataset summary: total rows, unique decision dates, last date, and per-horizon
  label distributions for label_up_{h}d and label_tp_before_sl_{h}d.
- Walk-forward summary: rows, last date, equity end, CAGR, Sharpe, MaxDD.

Usage:
  python -m engine.tools.summarize_outputs \
    --dataset data/datasets/swing_training_dataset.parquet \
    --walkforward data/backtests/walkforward/walkforward_results.parquet \
    --timeframes 3,7,14 --out-json data/reports/weekly_summary.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize dataset and walk-forward outputs"
    )
    p.add_argument(
        "--dataset", type=str, default="", help="Dataset CSV/Parquet with swing labels"
    )
    p.add_argument(
        "--walkforward", type=str, default="", help="Walk-forward daily results Parquet"
    )
    p.add_argument(
        "--timeframes",
        type=str,
        default="",
        help="Comma-separated horizons (e.g., 3,7,14). If empty, infer from columns",
    )
    p.add_argument(
        "--prev-count",
        type=int,
        default=None,
        help="Optional previous dataset total rows to compute new_rows",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Output JSON path (prints to stdout if empty)",
    )
    return p.parse_args(argv)


def _infer_timeframes(cols: List[str]) -> List[int]:
    out: List[int] = []
    for c in cols:
        if c.startswith("label_up_") and c.endswith("d"):
            try:
                n = int(c.split("label_up_")[-1].rstrip("d"))
                out.append(n)
            except Exception:
                continue
    return sorted(set(out))


def _dataset_summary(
    path: str, timeframes: List[int], prev_count: Optional[int]
) -> Dict:
    if not path:
        return {}
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    out: Dict = {"path": path, "rows": int(len(df))}
    if "date_decision" in df.columns:
        try:
            dd = pd.to_datetime(df["date_decision"])  # may already be date strings
            out["unique_dates"] = int(dd.dt.normalize().nunique())
            out["last_date"] = str(pd.to_datetime(dd).max().date())
        except Exception:
            out["unique_dates"] = int(df["date_decision"].astype(str).nunique())
            out["last_date"] = str(df["date_decision"].astype(str).max())
    # label distributions
    if not timeframes:
        timeframes = _infer_timeframes(list(df.columns))
    labels: Dict[str, Dict[str, int]] = {}
    for h in timeframes:
        for kind in ("label_up", "label_tp_before_sl"):
            col = f"{kind}_{h}d"
            if col in df.columns:
                ser = (
                    df[col].astype("Int8", errors="ignore")
                    if hasattr(df[col], "astype")
                    else df[col]
                )
                vc = pd.Series(ser).value_counts(dropna=True).to_dict()
                labels[col] = {
                    str(int(k)): int(v) for k, v in vc.items() if pd.notna(k)
                }
    out["labels"] = labels
    if prev_count is not None:
        try:
            out["new_rows"] = max(0, int(out["rows"]) - int(prev_count))
        except Exception:
            out["new_rows"] = None
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    peak = -np.inf
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return float(mdd)


def _walkforward_summary(path: str) -> Dict:
    if not path:
        return {}
    df = pd.read_parquet(path)
    out: Dict = {"path": path, "rows": int(len(df))}
    if "date" in df.columns:
        dts = pd.to_datetime(df["date"]).dt.normalize()
        out["last_date"] = str(dts.max().date())
    if "equity" in df.columns and "net_ret" in df.columns:
        eq = df["equity"].astype(float).values
        daily = df["net_ret"].astype(float).values
        out["equity_end"] = float(eq[-1]) if eq.size else None
        n = len(daily)
        ann = 252.0
        if n > 0:
            cagr = (eq[-1] ** (ann / n) - 1.0) if eq[-1] > 0 else float("nan")
            vol = float(np.std(daily)) * np.sqrt(ann)
            sharpe = (float(np.mean(daily)) * ann) / vol if vol > 0 else float("nan")
            mdd = _max_drawdown(eq)
            out.update(
                {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(mdd)}
            )
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    tfs: List[int] = []
    if args.timeframes:
        tfs = sorted(
            {int(x) for x in str(args.timeframes).split(",") if x.strip().isdigit()}
        )
    ds = _dataset_summary(args.dataset, tfs, args.prev_count)
    wf = _walkforward_summary(args.walkforward)
    out = {"dataset": ds, "walkforward": wf}
    js = json.dumps(out, indent=2)
    if args.out_json:
        import os

        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(js)
    else:
        print(js)


if __name__ == "__main__":
    main()
