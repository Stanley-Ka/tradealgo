from __future__ import annotations

"""Fit a rolling meta calibrator from the decision log (optionally per regime)."""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from ..features.regime import compute_regime_features_daily


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a rolling meta calibrator from decision log"
    )
    p.add_argument("--decision-log", required=True)
    p.add_argument(
        "--features",
        type=str,
        default="",
        help="Features parquet for regime features (optional)",
    )
    p.add_argument(
        "--out", required=True, help="Output pickle path for calibrator payload"
    )
    p.add_argument("--kind", choices=["isotonic", "platt"], default="isotonic")
    p.add_argument("--window-days", type=int, default=60)
    p.add_argument(
        "--per-regime", choices=["none", "regime_vol", "regime_risk"], default="none"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    log = pd.read_csv(args.decision_log)
    if (
        "date_decision" not in log.columns
        or "meta_prob" not in log.columns
        or "fret_1d_next" not in log.columns
    ):
        raise RuntimeError(
            "decision log must contain date_decision, meta_prob, fret_1d_next"
        )
    log["date_decision"] = pd.to_datetime(log["date_decision"])  # date-like
    end = log["date_decision"].max()
    start = end - pd.Timedelta(days=int(args.window_days))
    win = log[log["date_decision"] >= start].copy()
    if win.empty:
        raise RuntimeError("no rows in window to fit calibrator")
    p = win["meta_prob"].astype(float).values
    y = (win["fret_1d_next"].astype(float) > 0).astype(int).values
    if args.per_regime != "none" and args.features and os.path.exists(args.features):
        cols = ["date", "symbol"]
        try:
            f = pd.read_parquet(args.features, columns=cols)
        except Exception:
            f = pd.read_parquet(args.features)
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        reg = compute_regime_features_daily(f)
        reg.rename(columns={"date": "date_decision"}, inplace=True)
        win = win.merge(
            reg[["date_decision", args.per_regime]], on="date_decision", how="left"
        )
        rv = win[args.per_regime].astype(float).values
        qs = np.quantile(rv[np.isfinite(rv)], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        models = []
        if args.kind == "platt":
            from sklearn.linear_model import LogisticRegression  # type: ignore
        else:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
        for i in range(3):
            lo, hi = qs[i], qs[i + 1]
            mask = (rv >= lo) & (rv <= hi)
            if args.kind == "platt":
                m = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                m.fit(p[mask].reshape(-1, 1), y[mask])
            else:
                m = IsotonicRegression(out_of_bounds="clip")
                m.fit(p[mask], y[mask])
            models.append(m)
        payload = {
            "by_regime": {
                "feature": args.per_regime,
                "bins": qs.tolist(),
                "models": models,
            }
        }
    else:
        if args.kind == "platt":
            from sklearn.linear_model import LogisticRegression  # type: ignore

            m = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
            m.fit(p.reshape(-1, 1), y)
            payload = {"model": m, "kind": "platt"}
        else:
            from sklearn.isotonic import IsotonicRegression  # type: ignore

            m = IsotonicRegression(out_of_bounds="clip")
            m.fit(p, y)
            payload = {"model": m, "kind": "isotonic"}
    import pickle

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fpk:
        pickle.dump(payload, fpk)
    print(f"[meta-cal] wrote calibrator -> {args.out}")


if __name__ == "__main__":
    main()
