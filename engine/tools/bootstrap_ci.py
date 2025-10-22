"""Block bootstrap confidence intervals for backtest daily results.

Inputs a CSV/Parquet of daily results (expects columns date, net_ret, equity)
and computes percentile confidence intervals for CAGR, max drawdown, and Sharpe
via block bootstrap on days.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block bootstrap CIs for daily results")
    p.add_argument(
        "--results",
        required=True,
        help="CSV/Parquet with columns date, net_ret, equity",
    )
    p.add_argument("--block-size", type=int, default=10, help="Block size in days")
    p.add_argument(
        "--samples", type=int, default=1000, help="Number of bootstrap samples"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out-csv",
        type=str,
        default="data/reports/bootstrap_ci.csv",
        help="Output CSV path",
    )
    return p.parse_args(argv)


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


def _metrics(daily: np.ndarray) -> Tuple[float, float, float]:
    n = len(daily)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    ann = 252.0
    equity = np.cumprod(1.0 + daily)
    cagr = equity[-1] ** (ann / n) - 1.0
    vol = np.std(daily) * np.sqrt(ann)
    sharpe = (np.mean(daily) * ann) / vol if vol > 0 else np.nan
    mdd = _max_drawdown(equity)
    return (float(cagr), float(mdd), float(sharpe))


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.results.lower().endswith(".csv"):
        df = pd.read_csv(args.results)
    else:
        df = pd.read_parquet(args.results)
    if "net_ret" not in df.columns:
        raise RuntimeError("results file must have 'net_ret'")
    r = df["net_ret"].astype(float).values
    n = len(r)
    if n == 0:
        raise RuntimeError("no rows in results")
    np.random.seed(int(args.seed))
    B = max(1, int(args.block_size))
    S = max(1, int(args.samples))
    out = []
    for _ in range(S):
        # Sample blocks with replacement
        idxs = []
        while len(idxs) < n:
            start = np.random.randint(0, n)
            end = min(n, start + B)
            idxs.extend(range(start, end))
        idxs = idxs[:n]
        sample = r[idxs]
        cagr, mdd, sharpe = _metrics(sample)
        out.append((cagr, mdd, sharpe))
    arr = np.array(out)

    def q(col, pct):
        return float(np.nanpercentile(arr[:, col], pct))

    res = pd.DataFrame(
        {
            "metric": ["CAGR", "MaxDD", "Sharpe"],
            "p05": [q(0, 5), q(1, 5), q(2, 5)],
            "p50": [q(0, 50), q(1, 50), q(2, 50)],
            "p95": [q(0, 95), q(1, 95), q(2, 95)],
        }
    )
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print(f"[bootstrap] wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
