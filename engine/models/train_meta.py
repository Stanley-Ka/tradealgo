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
from sklearn.metrics import roc_auc_score, brier_score_loss  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from ..data.store import storage_root


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train meta-learner on OOF specialist probabilities"
    )
    p.add_argument("--oof", required=True, help="Path to OOF parquet from run_cv")
    p.add_argument(
        "--train-folds",
        type=str,
        default="all-but-last:1",
        help="Fold selection for training",
    )
    p.add_argument(
        "--test-folds", type=str, default="last:1", help="Fold selection for testing"
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output parquet for meta predictions on test folds",
    )
    p.add_argument(
        "--model-out",
        type=str,
        default="",
        help="Where to save the trained meta model (pickle)",
    )
    p.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--select-top-specs",
        type=int,
        default=0,
        help="If >0, keep only top-K specialists by AUC on train folds",
    )
    p.add_argument(
        "--min-auc",
        type=float,
        default=0.0,
        help="If >0, drop specialists with train AUC below this threshold",
    )
    # Meta model options
    p.add_argument(
        "--model",
        choices=["lr", "hgb"],
        default="lr",
        help="Meta model: logistic regression (lr) or HistGradientBoosting (hgb)",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression",
    )
    p.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting for logistic regression meta model",
    )
    # HistGradientBoosting hyperparams (used when --model=hgb)
    p.add_argument(
        "--hgb-learning-rate", type=float, default=0.05, help="HGB learning rate"
    )
    p.add_argument(
        "--hgb-max-depth",
        type=int,
        default=None,
        help="HGB max depth (None for unlimited)",
    )
    p.add_argument(
        "--hgb-max-iter", type=int, default=400, help="HGB iterations (trees)"
    )
    p.add_argument("--hgb-l2", type=float, default=0.0, help="HGB L2 regularization")
    # Optional meta-level calibration on top of meta predictions
    p.add_argument(
        "--meta-calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Fit calibrator on train folds and apply to test preds",
    )
    p.add_argument(
        "--meta-calibrator-out",
        type=str,
        default="",
        help="Where to save fitted meta calibrator (pickle)",
    )
    p.add_argument(
        "--replace-prob-with-calibrated",
        action="store_true",
        help="Overwrite meta_prob with calibrated values in output parquet",
    )
    # Per-regime meta calibration (fit separate calibrators by regime feature if present in OOF)
    p.add_argument(
        "--meta-calibration-per-regime",
        choices=["none", "regime_vol", "regime_risk"],
        default="none",
    )
    # Feature engineering options
    p.add_argument(
        "--add-odds",
        action="store_true",
        help="Add odds/logit transforms of *_prob features",
    )
    p.add_argument(
        "--add-interactions",
        action="store_true",
        help="Add interaction terms with regime features if present",
    )
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
    # Seed numpy for reproducibility (future algorithms may rely on RNG)
    try:
        import numpy as _np  # noqa: WPS433

        _np.random.seed(int(args.random_seed))
    except Exception:
        pass
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

    # Optional feature selection by train-fold AUC per specialist
    if args.select_top_specs > 0 or args.min_auc > 0:
        aucs: list[tuple[str, float]] = []
        sub = df.loc[df["fold"].isin(train_folds)].copy()
        y_tr_fs = sub["y_true"].astype(int).values
        for col in prob_cols:
            try:
                auc = roc_auc_score(y_tr_fs, sub[col].astype(float).values)
            except Exception:
                auc = float("nan")
            aucs.append((col, float(auc)))
        # Filter by min_auc
        if args.min_auc > 0:
            prob_cols = [c for c, a in aucs if a >= args.min_auc]
        # Keep top-K by AUC if requested
        if args.select_top_specs > 0:
            aucs_sorted = sorted(
                aucs, key=lambda x: (-(x[1] if x[1] == x[1] else -1))
            )  # NaNs last
            prob_cols = [c for c, _ in aucs_sorted[: args.select_top_specs]]
        print(f"[meta] Using {len(prob_cols)} specialist probs: {prob_cols}")

    # Start with probability features
    feat_cols = list(prob_cols)
    # Optional regime features saved by CV
    regime_cols = [c for c in ("regime_vol", "regime_risk") if c in df.columns]
    feat_cols += regime_cols

    # Optionally add odds/logit transforms
    if args.add_odds:
        for c in prob_cols:
            p_clip = df[c].astype(float).clip(1e-6, 1 - 1e-6)
            df[f"{c}_odds"] = (p_clip / (1 - p_clip)).astype(float)
            df[f"{c}_logit"] = np.log(df[f"{c}_odds"].astype(float))
        feat_cols += [f"{c}_odds" for c in prob_cols] + [
            f"{c}_logit" for c in prob_cols
        ]

    # Optionally add interactions prob x regime
    if args.add_interactions and regime_cols:
        for rc in regime_cols:
            for pc in prob_cols:
                name = f"{pc}__x__{rc}"
                df[name] = df[pc].astype(float) * df[rc].astype(float)
                feat_cols.append(name)

    X_tr = df.loc[df["fold"].isin(train_folds), feat_cols].values
    y_tr = df.loc[df["fold"].isin(train_folds), "y_true"].astype(int).values
    X_te = df.loc[df["fold"].isin(test_folds), feat_cols].values
    y_te = df.loc[df["fold"].isin(test_folds), "y_true"].astype(int).values

    # Train meta model
    scaler: StandardScaler | None = None
    if args.model == "hgb":
        # Strong, non-linear tree-based meta; robust to interactions
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

        clf = HistGradientBoostingClassifier(
            learning_rate=float(args.hgb_learning_rate),
            max_depth=(
                None if args.hgb_max_depth in (None, 0) else int(args.hgb_max_depth)
            ),
            max_iter=int(args.hgb_max_iter),
            l2_regularization=float(args.hgb_l2),
            early_stopping=True,
            random_state=int(args.random_seed),
        )
        clf.fit(X_tr, y_tr)
    else:
        cw = None if str(args.class_weight).lower() == "none" else "balanced"
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(
            solver="lbfgs", C=float(args.C), max_iter=1000, class_weight=cw
        )
        clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    auc_tr = roc_auc_score(y_tr, p_tr) if len(np.unique(y_tr)) > 1 else float("nan")
    auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
    try:
        brier_tr = brier_score_loss(y_tr, p_tr)
        brier_te = brier_score_loss(y_te, p_te)
    except Exception:
        brier_tr = float("nan")
        brier_te = float("nan")
    print(
        f"Meta AUC train={auc_tr:.3f} test={auc_te:.3f} | Brier train={brier_tr:.4f} test={brier_te:.4f}"
    )
    # Coefficients insight
    # Model diagnostics
    try:
        if args.model == "lr":
            coef = clf.coef_.ravel()  # type: ignore[attr-defined]
            top = sorted(
                zip(feat_cols, coef), key=lambda x: abs(float(x[1])), reverse=True
            )[:10]
            print("[meta] Top weights:")
            for name, w in top:
                print(f"  {name}: {w:+.4f}")
        else:
            print(
                "[meta] HGB model trained; feature importances via SHAP recommended off-line."
            )
    except Exception:
        pass

    # Optional meta-level calibration
    meta_calib = None
    meta_calib_payload = None
    if args.meta_calibration != "none":
        try:
            if args.meta_calibration == "platt":
                calib = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                calib.fit(p_tr.reshape(-1, 1), y_tr)
                p_tr_cal = calib.predict_proba(p_tr.reshape(-1, 1))[:, 1]
                p_te_cal = calib.predict_proba(p_te.reshape(-1, 1))[:, 1]
                meta_calib = calib
            else:
                from sklearn.isotonic import IsotonicRegression  # type: ignore

                calib = IsotonicRegression(out_of_bounds="clip")
                calib.fit(p_tr, y_tr)
                p_tr_cal = calib.transform(p_tr)
                p_te_cal = calib.transform(p_te)
                meta_calib = calib
            auc_tr_cal = (
                roc_auc_score(y_tr, p_tr_cal)
                if len(np.unique(y_tr)) > 1
                else float("nan")
            )
            auc_te_cal = (
                roc_auc_score(y_te, p_te_cal)
                if len(np.unique(y_te)) > 1
                else float("nan")
            )
            try:
                brier_tr_cal = brier_score_loss(y_tr, p_tr_cal)
                brier_te_cal = brier_score_loss(y_te, p_te_cal)
            except Exception:
                brier_tr_cal = float("nan")
                brier_te_cal = float("nan")
            print(
                f"[meta] Calibrated AUC train={auc_tr_cal:.3f} test={auc_te_cal:.3f} "
                f"Brier train={brier_tr_cal:.4f} test={brier_te_cal:.4f} kind={args.meta_calibration}"
            )
        except Exception as e:
            print(f"[meta] warning: meta calibration failed: {e}")
            meta_calib = None

    # Optional per-regime calibrators
    if (
        meta_calib is not None
        and args.meta_calibration_per_regime in ("regime_vol", "regime_risk")
        and any(c in df.columns for c in ("regime_vol", "regime_risk"))
    ):
        try:
            feat_name = args.meta_calibration_per_regime
            # Use train folds to define bins
            rv_tr = df.loc[df["fold"].isin(train_folds), feat_name].astype(float).values
            qs = np.quantile(rv_tr, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
            models = []
            if args.meta_calibration != "platt":
                from sklearn.isotonic import IsotonicRegression  # type: ignore
            for i in range(3):
                lo, hi = qs[i], qs[i + 1]
                mask_tr = (rv_tr >= lo) & (rv_tr <= hi)
                if args.meta_calibration == "platt":
                    m = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                    m.fit(p_tr[mask_tr].reshape(-1, 1), y_tr[mask_tr])
                else:
                    m = IsotonicRegression(out_of_bounds="clip")
                    m.fit(p_tr[mask_tr], y_tr[mask_tr])
                models.append(m)
            meta_calib_payload = {
                "by_regime": {
                    "feature": feat_name,
                    "bins": qs.tolist(),
                    "models": models,
                }
            }
            print(
                f"[meta] Built per-regime meta calibrators on {feat_name} with bins={qs}"
            )
        except Exception as e:
            print(f"[meta] warning: per-regime calibration failed: {e}")

    # Save predictions on test folds
    pred = df.loc[
        df["fold"].isin(test_folds), ["date", "symbol", "y_true", "fold"]
    ].copy()
    pred["meta_prob"] = p_te
    if meta_calib is not None:
        try:
            if args.meta_calibration == "platt":
                pred["meta_prob_cal"] = meta_calib.predict_proba(p_te.reshape(-1, 1))[
                    :, 1
                ]
            else:
                pred["meta_prob_cal"] = meta_calib.transform(p_te)
            if args.replace_prob_with_calibrated:
                pred["meta_prob"] = pred["meta_prob_cal"]
        except Exception:
            pass
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
    payload = {
        "model": clf,
        "features": feat_cols,
        "folds": {"train": train_folds, "test": test_folds},
        "meta_model": str(args.model),
    }
    if scaler is not None:
        payload["scaler"] = scaler
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[meta] model saved -> {model_path}")

    # Save meta calibrator if fitted
    if (
        meta_calib is not None or meta_calib_payload is not None
    ) and args.meta_calibrator_out:
        try:
            with open(args.meta_calibrator_out, "wb") as f:
                if meta_calib_payload is not None:
                    pickle.dump(meta_calib_payload, f)
                else:
                    pickle.dump({"model": meta_calib, "kind": args.meta_calibration}, f)
            print(f"[meta] meta calibrator saved -> {args.meta_calibrator_out}")
        except Exception as e:
            print(f"[meta] warning: failed to save meta calibrator: {e}")


if __name__ == "__main__":
    main()
