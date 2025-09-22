"""Predict daily picks using the trained meta-learner.

Steps:
- Load features parquet and select a date (default: latest available).
- Compute V0 specialist raw scores (patterns/technicals/sequence/news).
- If an OOF parquet is provided, fit calibrators per specialist and apply to today's scores.
  Otherwise, fall back to a naive mapping from raw [-1,1] -> prob [0,1].
- Load the saved meta model (from train_meta.py) and compute meta probabilities.
- Output a ranked list and optional CSV.

Usage:
  python -m engine.tools.predict_daily \
    --features data/datasets/features_daily_1D.parquet \
    --model-pkl data/models/meta_lr.pkl \
    --oof data/datasets/oof_specialists.parquet \
    --news-sentiment data/datasets/dummy_sentiment.parquet \
    --top-k 20 --out-csv data/signals/picks.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment
from ..infra.yaml_config import load_yaml_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict daily top-K using meta-learner")
    p.add_argument("--config", type=str, default="", help="YAML config with defaults/thresholds")
    p.add_argument("--features", required=False, default="", help="Features parquet with date,symbol,OHLCV & baseline")
    p.add_argument("--model-pkl", required=False, default="", help="Trained meta model pickle from train_meta.py")
    p.add_argument("--oof", type=str, default="", help="OOF parquet to fit calibrators (from run_cv)")
    p.add_argument("--calibrators-pkl", type=str, default="", help="Pickle with per-specialist calibrators (from run_cv --calibrators-out)")
    p.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--news-sentiment", type=str, default="", help="Optional sentiment file (CSV/Parquet)")
    p.add_argument("--date", type=str, default="", help="Target date YYYY-MM-DD (default: latest)")
    p.add_argument("--top-k", type=int, default=20, help="Number of symbols to output")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV export path")
    return p.parse_args(argv)


def _fit_calibrators_from_oof(oof: pd.DataFrame, kind: str) -> Dict[str, object]:
    """Fit a calibrator per specialist from OOF raw->prob pairs.

    Returns a dict: {spec_name: fitted_model}
    """
    calibrators: Dict[str, object] = {}
    specs = sorted({c.split("_raw")[0] for c in oof.columns if c.endswith("_raw")})
    for spec in specs:
        x = oof[f"{spec}_raw"].astype(float).values
        y = oof["y_true"].astype(int).values
        if kind == "platt":
            lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
            lr.fit(x.reshape(-1, 1), y)
            calibrators[spec] = lr
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x, y)
            calibrators[spec] = iso
    return calibrators


def _apply_calibrator(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    if hasattr(model, "transform"):
        return model.transform(x)
    # fallback
    return (x + 1.0) * 0.5


def _naive_prob_map(x: np.ndarray) -> np.ndarray:
    # Map [-1,1] to [0,1]
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Load config and fill defaults
    cfg = {}
    if args.config:
        try:
            cfg = load_yaml_config(args.config)
        except Exception as e:
            print(f"[predict] warning: failed to load YAML config: {e}")
    features_path = args.features or cfg.get("paths", {}).get("features", "")
    model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model", "")
    if not features_path or not model_path:
        raise RuntimeError("Provide --features and --model-pkl or set them in YAML under paths.features/meta_model")

    f = pd.read_parquet(features_path)
    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()
    target_date = pd.Timestamp(args.date) if args.date else f["date"].max()
    day = f[f["date"] == target_date].copy()
    if day.empty:
        raise RuntimeError(f"No rows for date {target_date.date()} in features")

    news_df = None
    cfg_news = cfg.get("paths", {}).get("news_sentiment", "")
    if args.news_sentiment or cfg_news:
        try:
            news_df = load_sentiment(args.news_sentiment or cfg_news)
        except Exception as e:
            print(f"[predict] warning: failed loading sentiment: {e}")

    # Compute raw specialist scores
    specs = compute_specialist_scores(day, news_sentiment=news_df, params=cfg.get("sensitivity", {}))

    # Calibrate scores to probabilities
    prob_cols: List[str] = []
    oof = None
    calibrators: Dict[str, object] = {}
    # Prefer pre-saved calibrators if provided
    calib_pkl = args.calibrators_pkl or cfg.get("calibration", {}).get("calibrators_pkl", "")
    if calib_pkl:
        try:
            with open(calib_pkl, "rb") as fpk:
                payload = pickle.load(fpk)
            calibrators = payload.get("models", {})
        except Exception as e:
            print(f"[predict] warning: failed to load calibrators: {e}")
    elif args.oof or cfg.get("paths", {}).get("oof"):
        try:
            oof = pd.read_parquet(args.oof or cfg.get("paths", {}).get("oof"))
            kind = cfg.get("calibration", {}).get("kind", args.calibration)
            calibrators = _fit_calibrators_from_oof(oof, kind)
        except Exception as e:
            print(f"[predict] warning: failed to load/fit calibrators from OOF: {e}")

    for sc in [c for c in specs.columns if c.startswith("spec_")]:
        raw = specs[sc].astype(float).values
        if calibrators:
            model = calibrators.get(sc)
            if model is not None:
                prob = _apply_calibrator(model, raw)
            else:
                prob = _naive_prob_map(raw)
        else:
            prob = _naive_prob_map(raw)
        specs[f"{sc}_prob"] = prob
        prob_cols.append(f"{sc}_prob")

    # Load trained meta model
    with open(model_path, "rb") as fpk:
        meta = pickle.load(fpk)
    clf = meta.get("model")
    feature_names = meta.get("features")
    if not feature_names:
        feature_names = prob_cols  # fallback to all available *_prob

    # Prepare X in the expected column order
    for col in feature_names:
        if col not in specs.columns:
            specs[col] = 0.5  # neutral
    X = specs[feature_names].values
    meta_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)

    picks = specs[["symbol"]].copy()
    picks["date"] = target_date
    picks["meta_prob"] = meta_prob
    top_k = int(cfg.get("alert", {}).get("top_k", args.top_k))
    picks = picks.sort_values("meta_prob", ascending=False).head(top_k).reset_index(drop=True)

    print(f"[predict] {target_date.date()} top-{top_k}")
    print(picks.to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        picks.to_csv(args.out_csv, index=False)
        print(f"[predict] saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
