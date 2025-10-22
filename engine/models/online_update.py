"""Online meta calibrator from decision logs.

Reads a decision log CSV (from paper_trader/backtests) and fits a calibrator
mapping `meta_prob` -> P(y=1) where y is derived from realized forward returns.

Usage:
  python -m engine.models.online_update \
    --decision-log data/paper/decision_log.csv \
    --out-calibrator data/models/meta_calibrator.pkl \
    --kind platt --label-threshold 0.0

You can then pass the calibrator to predictors via:
  --meta-calibrator-pkl data/models/meta_calibrator.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.metrics import roc_auc_score, brier_score_loss  # type: ignore

from ..data.store import storage_root


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a meta-level calibrator from decision logs"
    )
    p.add_argument(
        "--decision-log",
        required=True,
        help="CSV with columns meta_prob and fret_1d_next",
    )
    p.add_argument(
        "--out-calibrator",
        type=str,
        default="",
        help="Where to save the calibrator pickle",
    )
    p.add_argument("--kind", choices=["platt", "isotonic"], default="platt")
    p.add_argument(
        "--label-threshold",
        type=float,
        default=0.0,
        help="Label y=1 if fret_1d_next > threshold",
    )
    p.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum rows required to fit calibrator",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.decision_log):
        raise FileNotFoundError(args.decision_log)
    df = pd.read_csv(args.decision_log)
    required = ["meta_prob", "fret_1d_next"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column in decision log: {c}")
    x = df["meta_prob"].astype(float).values
    y = (df["fret_1d_next"].astype(float).values > float(args.label_threshold)).astype(
        int
    )
    # Drop NaNs
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    if len(x) < int(args.min_rows):
        raise RuntimeError(
            f"Not enough rows to fit calibrator: {len(x)} < {args.min_rows}"
        )
    if args.kind == "platt":
        mdl = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        mdl.fit(x.reshape(-1, 1), y)
        p = mdl.predict_proba(x.reshape(-1, 1))[:, 1]
    else:
        mdl = IsotonicRegression(out_of_bounds="clip")
        mdl.fit(x, y)
        p = mdl.transform(x)
    # Metrics
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    try:
        brier = brier_score_loss(y, p)
    except Exception:
        brier = float("nan")
    print(
        f"[online] fitted meta calibrator: kind={args.kind} rows={len(x)} AUC={auc:.4f} Brier={brier:.6f}"
    )
    out_path = args.out_calibrator.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "models"), exist_ok=True)
        out_path = os.path.join(root, "models", "meta_calibrator.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": mdl, "kind": args.kind}, f)
    print(f"[online] saved calibrator -> {out_path}")


if __name__ == "__main__":
    main()
