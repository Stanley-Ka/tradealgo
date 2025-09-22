"""Time-based CV + calibration for simple specialists.

Loads a features parquet, derives raw specialist scores, performs rolling yearly
splits, calibrates each specialist per fold (Platt or isotonic), and writes
OOF predictions to a parquet file.

Example:
  python -m engine.models.run_cv --features data/datasets/features_daily_1D.parquet \
    --label label_up_1d --calibration platt --out data/datasets/oof_specialists.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # type: ignore

from ..data.store import storage_root
from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_news_sentiment
from .calibration import fit_isotonic, fit_platt
from .cv import rolling_year_splits


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rolling yearly CV + calibration for specialists")
    p.add_argument("--features", required=True, help="Path to features parquet (from build_features)")
    p.add_argument("--label", type=str, default="label_up_1d", help="Binary label column name (Int8/0/1)")
    p.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--start", type=str, default="", help="Start date filter YYYY-MM-DD")
    p.add_argument("--end", type=str, default="", help="End date filter YYYY-MM-DD")
    p.add_argument("--cv-scheme", choices=["rolling_year", "time_kfold"], default="rolling_year", help="Cross-validation scheme")
    p.add_argument("--kfolds", type=int, default=5, help="Number of folds for time_kfold")
    p.add_argument("--purge-days", type=int, default=0, help="Purge window around validation (time_kfold)")
    p.add_argument("--embargo-days", type=int, default=0, help="Embargo period after validation (time_kfold)")
    p.add_argument("--out", type=str, default="", help="Output parquet for OOF predictions")
    p.add_argument("--news-sentiment", type=str, default="", help="Optional path to news sentiment CSV/Parquet")
    p.add_argument("--mlflow", action="store_true", help="Log CV metrics to MLflow if available")
    p.add_argument("--mlflow-experiment", type=str, default="research-cv", help="MLflow experiment name")
    p.add_argument("--calibrators-out", type=str, default="", help="Optional path to save fitted per-specialist calibrators (pickle)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)]

    # Filter rows with labels present
    if args.label not in df.columns:
        raise RuntimeError(f"Label column not found: {args.label}. Rebuild features with labels present.")
    lab = df[args.label].astype("Int8").astype(int)
    mask = lab.isin([0, 1])
    df = df[mask].reset_index(drop=True)
    lab = lab[mask].reset_index(drop=True)

    # Compute specialist raw scores
    news_df = None
    if args.news_sentiment:
        try:
            news_df = load_news_sentiment(args.news_sentiment)
        except Exception as e:
            print(f"[cv] failed to load news sentiment: {e}; proceeding with zeros")
            news_df = None
    scores_df = compute_specialist_scores(df, news_sentiment=news_df)
    spec_cols = [c for c in scores_df.columns if c.startswith("spec_")]

    if args.cv_scheme == "rolling_year":
        folds = rolling_year_splits(df["date"], min_train_years=3)
    else:
        from .cv import time_kfold_purged
        folds = time_kfold_purged(
            df["date"], n_splits=int(args.kfolds), purge_days=int(args.purge_days), embargo_days=int(args.embargo_days)
        )
    rows: List[pd.DataFrame] = []
    fold_metrics: List[tuple[str, dict]] = []
    print(f"Folds: {[f.name for f in folds]}")
    for fold in folds:
        tr_idx, va_idx = fold.train_idx, fold.val_idx
        y_tr = lab.iloc[tr_idx].values
        y_va = lab.iloc[va_idx].values
        part = pd.DataFrame({
            "date": df["date"].iloc[va_idx].values,
            "symbol": df["symbol"].iloc[va_idx].values,
            "y_true": y_va,
            "fold": fold.name,
        })
        for sc in spec_cols:
            sc_tr = scores_df[sc].iloc[tr_idx].values
            sc_va = scores_df[sc].iloc[va_idx].values
            # Calibration fit on train
            if args.calibration == "platt":
                calib = fit_platt(sc_tr, y_tr)
            else:
                calib = fit_isotonic(sc_tr, y_tr)
            prob_va = calib.apply(np.asarray(sc_va, dtype=float))
            # Quality check (AUC)
            try:
                auc = roc_auc_score(y_va, prob_va)
            except Exception:
                auc = float("nan")
            print(f"{fold.name} {sc}: AUC={auc:.3f}")
            part[f"{sc}_raw"] = sc_va
            part[f"{sc}_prob"] = prob_va
            fold_metrics.append((fold.name, {f"AUC_{sc}": float(auc)}))
        rows.append(part)

    oof = pd.concat(rows, axis=0, ignore_index=True)
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "datasets"), exist_ok=True)
        out_path = os.path.join(root, "datasets", "oof_specialists.parquet")
    oof.to_parquet(out_path, index=False)
    print(f"[oof] rows={len(oof)} -> {out_path}")

    # Optionally fit and save global calibrators on aggregated OOF raw->y_true
    if args.calibrators_out:
        try:
            import pickle  # noqa: WPS433
            calib_models: dict[str, object] = {}
            for sc in spec_cols:
                raw_col = f"{sc}_raw"
                if raw_col not in oof.columns:
                    continue
                x = oof[raw_col].astype(float).values
                y = oof["y_true"].astype(int).values
                if args.calibration == "platt":
                    from sklearn.linear_model import LogisticRegression  # type: ignore

                    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                    lr.fit(x.reshape(-1, 1), y)
                    calib_models[sc] = lr
                else:
                    from sklearn.isotonic import IsotonicRegression  # type: ignore

                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(x, y)
                    calib_models[sc] = iso
            payload = {"kind": args.calibration, "models": calib_models, "spec_cols": spec_cols}
            with open(args.calibrators_out, "wb") as f:
                pickle.dump(payload, f)
            print(f"[cv] saved calibrators -> {args.calibrators_out}")
        except Exception as e:
            print(f"[cv] saving calibrators failed: {e}")

    # Optional MLflow logging
    if args.mlflow:
        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name=f"cv_{args.cv_scheme}"):
                mlflow.log_params(
                    {
                        "cv_scheme": args.cv_scheme,
                        "kfolds": args.kfolds,
                        "purge_days": args.purge_days,
                        "embargo_days": args.embargo_days,
                        "calibration": args.calibration,
                        "label": args.label,
                    }
                )
                # Aggregate metrics per fold
                # Flatten and log mean AUC per specialist
                per_spec: dict[str, list[float]] = {}
                for _, md in fold_metrics:
                    for k, v in md.items():
                        per_spec.setdefault(k, []).append(v)
                for k, arr in per_spec.items():
                    vals = [float(x) for x in arr if isinstance(x, (int, float))]
                    if vals:
                        mlflow.log_metric(f"mean_{k}", float(np.mean(vals)))
                # Save OOF as artifact
                mlflow.log_artifact(out_path)
                print("[cv] MLflow logging complete.")
        except Exception as e:
            print(f"[cv] MLflow logging skipped/failed: {e}")


if __name__ == "__main__":
    main()
