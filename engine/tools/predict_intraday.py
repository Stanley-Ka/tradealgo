from __future__ import annotations

"""Predict latest picks from an intraday snapshot (no alerts).

Reads an intraday features snapshot (one row per symbol), computes specialists,
applies calibrators and meta model, and prints top-K. Optionally writes CSV.
"""

import argparse
import pickle
from typing import Dict, Optional, List

import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..models.calib_utils import (
    load_spec_calibrators as load_cals,
    apply_calibrator as apply_cal,
    naive_prob_map as naive_map,
    apply_meta_calibrator as apply_meta,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict from intraday snapshot")
    p.add_argument(
        "--intraday-features",
        required=True,
        help="Intraday snapshot parquet (one row per symbol)",
    )
    p.add_argument("--model-pkl", required=True, help="Meta model pickle")
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators (from run_cv)",
    )
    p.add_argument(
        "--oof",
        type=str,
        default="",
        help="OOF parquet to fit calibrators if pkl missing",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-level calibrator",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-csv", type=str, default="")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    snap = pd.read_parquet(args.intraday_features)
    snap["symbol"] = snap["symbol"].astype(str).str.upper()
    # Compute specialists
    specs = compute_specialist_scores(snap)
    # Calibrators
    calibrators: Dict[str, object] = load_cals(
        calibrators_pkl=args.calibrators_pkl or None,
        oof_path=args.oof or None,
        kind="platt",
    )
    prob_cols: List[str] = []
    for sc in [
        c for c in specs.columns if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = specs[sc].astype(float).values
        prob = (
            apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else naive_map(raw)
        )
        specs[f"{sc}_prob"] = prob
        prob_cols.append(f"{sc}_prob")
    # Meta
    with open(args.model_pkl, "rb") as fpk:
        meta = pickle.load(fpk)
    clf = meta.get("model")
    feat_names = meta.get("features") or prob_cols
    scaler = meta.get("scaler")
    for col in feat_names:
        if col not in specs.columns:
            specs[col] = 0.5
    X = specs[feat_names].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    meta_prob = (
        clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)
    )
    if args.meta_calibrator_pkl:
        try:
            meta_prob = apply_meta(args.meta_calibrator_pkl, meta_prob)
        except Exception:
            pass
    picks = specs[["symbol"]].copy()
    picks["meta_prob"] = meta_prob
    picks = (
        picks.sort_values("meta_prob", ascending=False)
        .head(int(args.top_k))
        .reset_index(drop=True)
    )
    print("[intraday-predict] top-{}".format(int(args.top_k)))
    print(picks.to_string(index=False))
    if args.out_csv:
        import os

        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        picks.to_csv(args.out_csv, index=False)
        print(f"[intraday-predict] saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
