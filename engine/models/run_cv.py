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
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # type: ignore

from ..data.store import storage_root
from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_news_sentiment
from .calibration import fit_isotonic, fit_platt
from .calib_utils import naive_prob_map as _naive
from .cv import rolling_year_splits


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rolling yearly CV + calibration for specialists"
    )
    p.add_argument(
        "--features",
        required=True,
        help="Path to features parquet (from build_features)",
    )
    p.add_argument(
        "--label",
        type=str,
        default="label_up_1d",
        help="Binary label column name (Int8/0/1)",
    )
    p.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--start", type=str, default="", help="Start date filter YYYY-MM-DD")
    p.add_argument("--end", type=str, default="", help="End date filter YYYY-MM-DD")
    p.add_argument(
        "--cv-scheme",
        choices=["rolling_year", "time_kfold"],
        default="rolling_year",
        help="Cross-validation scheme",
    )
    p.add_argument(
        "--kfolds", type=int, default=5, help="Number of folds for time_kfold"
    )
    p.add_argument(
        "--purge-days",
        type=int,
        default=0,
        help="Purge window around validation (time_kfold)",
    )
    p.add_argument(
        "--embargo-days",
        type=int,
        default=0,
        help="Embargo period after validation (time_kfold)",
    )
    p.add_argument(
        "--out", type=str, default="", help="Output parquet for OOF predictions"
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional path to news sentiment CSV/Parquet",
    )
    p.add_argument(
        "--mlflow", action="store_true", help="Log CV metrics to MLflow if available"
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default="research-cv",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--calibrators-out",
        type=str,
        default="",
        help="Optional path to save fitted per-specialist calibrators (pickle)",
    )
    p.add_argument(
        "--spec-config",
        type=str,
        default="",
        help="Optional YAML with specialist params (weights/windows)",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (where applicable)",
    )
    p.add_argument(
        "--drift-report",
        type=str,
        default="",
        help="Optional CSV to write simple train/val drift metrics per fold",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Seed numpy to keep any stochastic parts stable (future algorithms)
    try:
        import numpy as _np  # local alias to avoid shadowing

        _np.random.seed(int(args.random_seed))
    except Exception:
        pass
    # Memory-friendly read: load only the columns needed by specialists + label
    cols_try = {
        "date",
        "symbol",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
        # common baseline features used by specialists
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "mom_sma_5_20",
        "price_z_20",
        "meanrev_20",
        "vol_z_20",
        "atr_pct_14",
        args.label,
    }
    try:
        df = pd.read_parquet(args.features, columns=[c for c in cols_try if c])
    except Exception:
        # Fallback to full read if columns projection fails
        df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    # Reduce memory by compressing symbol dtype
    try:
        df["symbol"] = df["symbol"].astype(str).str.upper().astype("category")
    except Exception:
        pass
    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)]

    # Filter rows with valid binary labels (robust to NA/float/bool)
    if args.label not in df.columns:
        raise RuntimeError(
            f"Label column not found: {args.label}. Rebuild features with labels present."
        )
    # Coerce to numeric; invalid parses -> NaN
    lab_series = pd.to_numeric(df[args.label], errors="coerce")
    # Accept 0/1 whether float or int; drop everything else (including NaN)
    valid_mask = lab_series.isin([0, 1, 0.0, 1.0])
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(
            f"[cv] dropping {dropped} rows with missing/invalid labels in '{args.label}'"
        )
    df = df[valid_mask].reset_index(drop=True)
    lab = lab_series[valid_mask].astype(int).reset_index(drop=True)

    # Compute specialist raw scores
    news_df = None
    if args.news_sentiment:
        try:
            news_df = load_news_sentiment(args.news_sentiment)
        except Exception as e:
            print(f"[cv] failed to load news sentiment: {e}; proceeding with zeros")
            news_df = None
    spec_params = {}
    if args.spec_config:
        try:
            from ..infra.yaml_config import load_yaml_config

            cfg = load_yaml_config(args.spec_config)
            spec_params = cfg.get("specialists", cfg)
        except Exception as e:
            print(f"[cv] warning: could not load spec config: {e}")
    scores_df = compute_specialist_scores(
        df, news_sentiment=news_df, params=spec_params
    )
    spec_cols = [c for c in scores_df.columns if c.startswith("spec_")]
    # Regime columns (global-by-date) may be present in scores_df
    regime_cols = [c for c in ("regime_vol", "regime_risk") if c in scores_df.columns]

    if args.cv_scheme == "rolling_year":
        folds = rolling_year_splits(df["date"], min_train_years=3)
    else:
        from .cv import time_kfold_purged

        folds = time_kfold_purged(
            df["date"],
            n_splits=int(args.kfolds),
            purge_days=int(args.purge_days),
            embargo_days=int(args.embargo_days),
        )
    rows: List[pd.DataFrame] = []
    fold_metrics: List[tuple[str, dict]] = []
    drift_rows: List[dict] = []
    print(f"Folds: {[f.name for f in folds]}")
    for fold in folds:
        tr_idx, va_idx = fold.train_idx, fold.val_idx
        y_tr = lab.iloc[tr_idx].values
        y_va = lab.iloc[va_idx].values
        part = pd.DataFrame(
            {
                "date": df["date"].iloc[va_idx].values,
                "symbol": df["symbol"].iloc[va_idx].values,
                "y_true": y_va,
                "fold": fold.name,
            }
        )
        # Attach regime features on validation rows
        for rc in regime_cols:
            part[rc] = scores_df[rc].iloc[va_idx].values
        for sc in spec_cols:
            sc_tr = scores_df[sc].iloc[tr_idx].astype(float).values
            sc_va = scores_df[sc].iloc[va_idx].astype(float).values
            # Safe fit: drop NaNs on train; if insufficient data or single class, fallback to naive mapping
            mask = np.isfinite(sc_tr)
            xfit = sc_tr[mask]
            yfit = y_tr[mask]
            calib = None
            try:
                if len(xfit) > 0 and len(np.unique(yfit)) > 1:
                    if args.calibration == "platt":
                        calib = fit_platt(xfit, yfit)
                    else:
                        calib = fit_isotonic(xfit, yfit)
            except Exception:
                calib = None
            # Apply to validation; fill NaNs with 0.0 before mapping
            sc_va_filled = np.where(np.isfinite(sc_va), sc_va, 0.0)
            if calib is not None and hasattr(calib, "apply"):
                prob_va = calib.apply(np.asarray(sc_va_filled, dtype=float))
            else:
                prob_va = _naive(np.asarray(sc_va_filled, dtype=float))
            # Quality check (AUC)
            try:
                auc = roc_auc_score(y_va, prob_va)
            except Exception:
                auc = float("nan")
            print(f"{fold.name} {sc}: AUC={auc:.3f}")
            part[f"{sc}_raw"] = sc_va
            part[f"{sc}_prob"] = prob_va
            fold_metrics.append((fold.name, {f"AUC_{sc}": float(auc)}))
            # Simple drift metric (PSI) between train and val for this specialist raw score
            try:
                psi = _psi(sc_tr.astype(float), sc_va.astype(float))
                drift_rows.append(
                    {"fold": fold.name, "spec": sc, "psi_raw": float(psi)}
                )
            except Exception:
                pass
        rows.append(part)

    oof = pd.concat(rows, axis=0, ignore_index=True)
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "datasets"), exist_ok=True)
        out_path = os.path.join(root, "datasets", "oof_specialists.parquet")
    oof.to_parquet(out_path, index=False)
    print(f"[oof] rows={len(oof)} -> {out_path}")

    # Write drift report if requested
    if args.drift_report and drift_rows:
        try:
            pd.DataFrame(drift_rows).to_csv(args.drift_report, index=False)
            print(f"[cv] drift report -> {args.drift_report}")
        except Exception as e:
            print(f"[cv] failed to write drift report: {e}")

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
                mask = np.isfinite(x)
                x = x[mask]
                y = y[mask]
                if len(x) == 0 or len(np.unique(y)) <= 1:
                    print(
                        f"[cv] skipping calibrator for {sc}: insufficient finite samples"
                    )
                    continue
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
            payload = {
                "kind": args.calibration,
                "models": calib_models,
                "spec_cols": spec_cols,
            }
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


def _psi(train_scores: np.ndarray, val_scores: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using train quantile bins on raw scores."""
    train_scores = np.asarray(train_scores, dtype=float)
    val_scores = np.asarray(val_scores, dtype=float)
    # Handle degenerate
    if len(train_scores) == 0 or len(val_scores) == 0:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(train_scores, qs))
    if len(edges) < 2:
        return 0.0
    # Bin counts → proportions with small floor to avoid div/0
    tr_hist, _ = np.histogram(train_scores, bins=edges)
    va_hist, _ = np.histogram(val_scores, bins=edges)
    tr_p = np.maximum(tr_hist / max(1, tr_hist.sum()), 1e-6)
    va_p = np.maximum(va_hist / max(1, va_hist.sum()), 1e-6)
    psi = np.sum((va_p - tr_p) * np.log(va_p / tr_p))
    return float(psi)


if __name__ == "__main__":
    main()
