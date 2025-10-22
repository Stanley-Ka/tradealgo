"""Online meta-learner refit from decision logs with specialist *_prob features.

Reads a decision log CSV (from paper_trader/backtests with specialist probs logged)
and trains a meta-learner that maps specialist probabilities -> P(up) labels derived
from realized next-day returns.

Usage (batch logistic regression):
  python -m engine.models.online_meta_refit \
    --decision-log data/paper/decision_log.csv \
    --out-model data/models/meta_online.pkl \
    --label-threshold 0.0 --algo logreg --C 1.0

Usage (online SGD, time-ordered by date_decision):
  python -m engine.models.online_meta_refit \
    --decision-log data/paper/decision_log.csv \
    --out-model data/models/meta_online.pkl \
    --algo sgd --epochs 2 --alpha 0.0005 --shuffle
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier  # type: ignore
from sklearn.metrics import roc_auc_score, brier_score_loss  # type: ignore

from ..data.store import storage_root


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online/meta refit of meta-learner from decision logs"
    )
    p.add_argument(
        "--decision-log",
        required=True,
        help="CSV decision log with specialist *_prob columns and fret_1d_next",
    )
    p.add_argument(
        "--out-model",
        type=str,
        default="",
        help="Where to save the refit meta model pickle",
    )
    p.add_argument(
        "--label-threshold",
        type=float,
        default=0.0,
        help="Label y=1 if fret_1d_next > threshold",
    )
    p.add_argument(
        "--min-rows", type=int, default=500, help="Minimum rows required to fit"
    )
    p.add_argument("--algo", choices=["logreg", "sgd"], default="logreg")
    # logreg
    p.add_argument(
        "--C", type=float, default=1.0, help="Inverse regularization strength (logreg)"
    )
    p.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Use class_weight='balanced' (logreg)",
    )
    # sgd (online)
    p.add_argument("--epochs", type=int, default=1, help="Epochs over data (sgd)")
    p.add_argument("--alpha", type=float, default=0.0001, help="L2 strength (sgd)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle each epoch (sgd)")
    # feature detection
    p.add_argument(
        "--feature-prefix", type=str, default="spec_", help="Column prefix for features"
    )
    p.add_argument(
        "--feature-suffix", type=str, default="_prob", help="Column suffix for features"
    )
    p.add_argument(
        "--date-col",
        type=str,
        default="date_decision",
        help="Date column to order by for online SGD",
    )
    p.add_argument("--random-seed", type=int, default=42)
    # sample weights
    p.add_argument(
        "--sample-weights-col",
        type=str,
        default="sample_weight",
        help="Optional column with per-row sample weights",
    )
    return p.parse_args(argv)


def _detect_features(df: pd.DataFrame, prefix: str, suffix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.decision_log):
        raise FileNotFoundError(args.decision_log)
    df = pd.read_csv(args.decision_log)
    if "fret_1d_next" not in df.columns:
        raise RuntimeError("decision log must include fret_1d_next column")
    feat_cols = _detect_features(df, args.feature_prefix, args.feature_suffix)
    if not feat_cols:
        raise RuntimeError(
            f"No features found with prefix={args.feature_prefix} and suffix={args.feature_suffix}"
        )
    X = df[feat_cols].astype(float).values
    y = (df["fret_1d_next"].astype(float).values > float(args.label_threshold)).astype(
        int
    )
    sw = None
    if args.sample_weights_col in df.columns:
        try:
            sw = df[args.sample_weights_col].astype(float).values
        except Exception:
            sw = None
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if sw is not None:
        ok = ok & np.isfinite(sw)
    X, y = X[ok], y[ok]
    if sw is not None:
        sw = sw[ok]
    if len(X) < int(args.min_rows):
        raise RuntimeError(f"Not enough rows to fit: {len(X)} < {args.min_rows}")
    # Seed
    try:
        np.random.seed(int(args.random_seed))
    except Exception:
        pass

    if args.algo == "logreg":
        clf = LogisticRegression(
            solver="lbfgs",
            C=float(args.C),
            max_iter=1000,
            class_weight=("balanced" if args.class_weight_balanced else None),
        )
        clf.fit(X, y, sample_weight=sw)
        p = clf.predict_proba(X)[:, 1]
    else:
        # Online SGD in chronological order if date_col present
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(args.alpha),
            learning_rate="optimal",
            random_state=int(args.random_seed),
            average=True,
        )
        order = np.arange(len(X))
        if args.date_col in df.columns:
            # stable sort by date then as-is
            try:
                d = pd.to_datetime(df.loc[ok, args.date_col]).values
                order = np.argsort(d, kind="mergesort")
            except Exception:
                pass
        X_ord, y_ord = X[order], y[order]
        classes = np.array([0, 1], dtype=int)
        for ep in range(int(args.epochs)):
            if args.shuffle:
                idx = np.random.permutation(len(X_ord))
            else:
                idx = np.arange(len(X_ord))
            X_ep, y_ep = X_ord[idx], y_ord[idx]
            # one pass partial_fit
            if sw is not None:
                sw_ep = sw[order][idx]
                clf.partial_fit(X_ep, y_ep, classes=classes, sample_weight=sw_ep)
            else:
                clf.partial_fit(X_ep, y_ep, classes=classes)
        # probabilities via decision function + logistic link approximation
        try:
            p = clf.predict_proba(X)[:, 1]
        except Exception:
            # fallback using decision_function
            z = clf.decision_function(X)
            p = 1 / (1 + np.exp(-z))

    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    try:
        brier = brier_score_loss(y, p)
    except Exception:
        brier = float("nan")
    print(
        f"[online-meta] fitted meta model: algo={args.algo} rows={len(X)} AUC={auc:.4f} Brier={brier:.6f}"
    )

    out_path = args.out_model.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "models"), exist_ok=True)
        out_path = os.path.join(root, "models", "meta_online.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            {"model": clf, "features": feat_cols, "source": "online_meta_refit"}, f
        )
    print(f"[online-meta] saved -> {out_path}")


if __name__ == "__main__":
    main()
