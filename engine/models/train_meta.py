"""Train a meta-learner on OOF specialist probabilities and evaluate on held-out folds.

Usage examples:
  # Train on all folds except the last one, test on the last year
  python -m engine.models.train_meta \
    --oof data/datasets/oof_specialists.parquet \
    --train-folds all-but-last:1 --test-folds last:1 \
    --out data/datasets/meta_predictions.parquet --model-out data/models/meta_lr.pkl

  # Explicit fold lists: train on Y2018..Y2022, test on Y2023
  python -m engine.models.train_meta \
    --oof data/datasets/oof_specialists.parquet \
    --train-folds Y2018,Y2019,Y2020,Y2021,Y2022 --test-folds Y2023
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore

from ..data.store import storage_root


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta-learner on OOF specialist probabilities")
    p.add_argument("--oof", required=True, help="Path to OOF parquet from run_cv")
    p.add_argument("--train-folds", type=str, default="all-but-last:1", help="Fold selection for training")
    p.add_argument("--test-folds", type=str, default="last:1", help="Fold selection for testing")
    p.add_argument("--out", type=str, default="", help="Output parquet for meta predictions on test folds")
    p.add_argument("--model-out", type=str, default="", help="Where to save the trained meta model (pickle)")
    return p.parse_args(argv)


def list_folds(df: pd.DataFrame) -> List[str]:
    return sorted(df["fold"].astype(str).unique())


def select_folds(all_folds: Sequence[str], spec: str) -> List[str]:
    spec = spec.strip()
    if spec.startswith("last:"):
        n = int(spec.split(":", 1)[1])
        return list(all_folds[-n:])
    if spec.startswith("all-but-last:"):
        n = int(spec.split(":", 1)[1])
        return list(all_folds[:-n]) if n > 0 else list(all_folds)
    if spec == "all":
        return list(all_folds)
    # Comma-separated list
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    return [f for f in all_folds if f in wanted]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    df = pd.read_parquet(args.oof)
    df["fold"] = df["fold"].astype(str)
    folds = list_folds(df)
    train_folds = select_folds(folds, args.train_folds)
    test_folds = select_folds(folds, args.test_folds)
    if not train_folds or not test_folds:
        raise RuntimeError("Empty train/test fold selection")
    if set(train_folds).intersection(test_folds):
        raise RuntimeError("Train and test folds must be disjoint")

    # Features: all calibrated specialist columns ending with _prob
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    if not prob_cols:
        raise RuntimeError("No *_prob columns found in OOF file")

    X_tr = df.loc[df["fold"].isin(train_folds), prob_cols].values
    y_tr = df.loc[df["fold"].isin(train_folds), "y_true"].astype(int).values
    X_te = df.loc[df["fold"].isin(test_folds), prob_cols].values
    y_te = df.loc[df["fold"].isin(test_folds), "y_true"].astype(int).values

    # Logistic regression meta-learner with regularization
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    auc_tr = roc_auc_score(y_tr, p_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    print(f"Meta AUC train={auc_tr:.3f} test={auc_te:.3f}")

    # Save predictions on test folds
    pred = df.loc[df["fold"].isin(test_folds), ["date", "symbol", "y_true", "fold"]].copy()
    pred["meta_prob"] = p_te
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "datasets"), exist_ok=True)
        out_path = os.path.join(root, "datasets", "meta_predictions.parquet")
    pred.to_parquet(out_path, index=False)
    print(f"[meta] test predictions -> {out_path}")

    # Save model
    model_path = args.model_out.strip()
    if not model_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "models"), exist_ok=True)
        model_path = os.path.join(root, "models", "meta_lr.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "features": prob_cols, "folds": {"train": train_folds, "test": test_folds}}, f)
    print(f"[meta] model saved -> {model_path}")


if __name__ == "__main__":
    main()

