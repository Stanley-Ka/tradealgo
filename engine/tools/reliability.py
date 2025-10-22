"""Reliability analysis: calibration tables for probabilities.

Supports two inputs:
- OOF parquet from run_cv (contains y_true and *_prob columns)
- Predictions + labels: join a predictions file with a features/labels file

Outputs a CSV with per-bin counts, mean predicted probability, empirical rate,
and overall ECE/MCE metrics.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute reliability tables and ECE/MCE")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--oof", type=str, help="OOF parquet from run_cv with y_true and *_prob cols"
    )
    src.add_argument(
        "--pred", type=str, help="Predictions parquet/CSV with date,symbol,prob col"
    )
    p.add_argument(
        "--features",
        type=str,
        default="",
        help="Features parquet with labels (required if --pred is used)",
    )
    p.add_argument(
        "--label-col",
        type=str,
        default="y_true",
        help="Label column (or features label name)",
    )
    p.add_argument(
        "--prob-col",
        type=str,
        default="meta_prob",
        help="Probability column (when using --pred)",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width bins for probability",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="data/reports/reliability.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--all-specialists",
        action="store_true",
        help="When --oof is used, compute per-specialist reliability for all *_prob cols",
    )
    return p.parse_args(argv)


def _reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.clip(p.astype(float), 0.0, 1.0)
    y = y.astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges, right=True)
    idx[idx == 0] = 1
    idx[idx > bins] = bins
    rows = []
    for b in range(1, bins + 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "count": 0, "p_mean": np.nan, "y_rate": np.nan})
        else:
            rows.append(
                {
                    "bin": b,
                    "count": n,
                    "p_mean": float(p[m].mean()),
                    "y_rate": float(y[m].mean()),
                }
            )
    tbl = pd.DataFrame(rows)
    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    valid = tbl.dropna(subset=["p_mean", "y_rate"])
    total = int(valid["count"].sum())
    if total > 0:
        w = valid["count"].values / total
        ece = float(np.sum(w * np.abs(valid["y_rate"].values - valid["p_mean"].values)))
        mce = float(np.max(np.abs(valid["y_rate"].values - valid["p_mean"].values)))
    else:
        ece, mce = float("nan"), float("nan")
    tbl.attrs["ECE"] = ece
    tbl.attrs["MCE"] = mce
    return tbl


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    if args.oof:
        df = pd.read_parquet(args.oof)
        if args.label_col not in df.columns:
            raise RuntimeError(f"Label column '{args.label_col}' not found in OOF")
        y = df[args.label_col].astype(int).values
        cols: List[str]
        if args.all_specialists:
            cols = sorted(
                [c for c in df.columns if isinstance(c, str) and c.endswith("_prob")]
            )
        else:
            cols = ["meta_prob"] if "meta_prob" in df.columns else []
            if not cols:
                raise RuntimeError(
                    "Specify --all-specialists or ensure meta_prob present in OOF"
                )
        out_rows = []
        for c in cols:
            p = df[c].astype(float).values
            tbl = _reliability_table(y, p, bins=int(args.bins))
            ece, mce = tbl.attrs.get("ECE", np.nan), tbl.attrs.get("MCE", np.nan)
            tbl["metric"] = c
            out_rows.append(tbl)
            print(f"{c}: ECE={ece:.4f} MCE={mce:.4f}")
        out = pd.concat(out_rows, axis=0, ignore_index=True)
        out.to_csv(args.out_csv, index=False)
        print(f"[reliability] wrote -> {args.out_csv}")
        return

    # predictions + labels
    if not (args.pred and args.features):
        raise RuntimeError("Provide --pred and --features when not using --oof")
    if args.pred.lower().endswith(".csv"):
        pred = pd.read_csv(args.pred)
    else:
        pred = pd.read_parquet(args.pred)
    feat = pd.read_parquet(args.features, columns=["date", "symbol", args.label_col])
    for col in ("date",):
        pred[col] = pd.to_datetime(pred[col])
        feat[col] = pd.to_datetime(feat[col])
    merged = pred.merge(feat, on=["date", "symbol"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between predictions and features/labels")
    y = merged[args.label_col].astype(int).values
    p = merged[args.prob_col].astype(float).values
    tbl = _reliability_table(y, p, bins=int(args.bins))
    ece, mce = tbl.attrs.get("ECE", np.nan), tbl.attrs.get("MCE", np.nan)
    print(f"meta_prob: ECE={ece:.4f} MCE={mce:.4f}")
    tbl["metric"] = args.prob_col
    tbl.to_csv(args.out_csv, index=False)
    print(f"[reliability] wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
