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
from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment
from ..infra.yaml_config import load_yaml_config
from ..models.calib_utils import (
    fit_per_specialist_calibrators_from_oof as fit_calibs,
    apply_calibrator as apply_cal,
    naive_prob_map as naive_map,
    load_spec_calibrators as load_cals,
    apply_meta_calibrator as apply_meta,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict daily top-K using meta-learner")
    p.add_argument(
        "--config", type=str, default="", help="YAML config with defaults/thresholds"
    )
    p.add_argument(
        "--features",
        required=False,
        default="",
        help="Features parquet with date,symbol,OHLCV & baseline",
    )
    p.add_argument(
        "--model-pkl",
        required=False,
        default="",
        help="Trained meta model pickle from train_meta.py",
    )
    p.add_argument(
        "--oof",
        type=str,
        default="",
        help="OOF parquet to fit calibrators (from run_cv)",
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Pickle with per-specialist calibrators (from run_cv --calibrators-out)",
    )
    p.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file (CSV/Parquet)",
    )
    p.add_argument(
        "--date", type=str, default="", help="Target date YYYY-MM-DD (default: latest)"
    )
    p.add_argument("--top-k", type=int, default=20, help="Number of symbols to output")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV export path")
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-level calibrator pickle to adjust meta_prob",
    )
    return p.parse_args(argv)


def _fit_calibrators_from_oof(oof: pd.DataFrame, kind: str) -> Dict[str, object]:
    return fit_calibs(oof, kind)


def _apply_calibrator(model: object, x: np.ndarray) -> np.ndarray:
    # Keep name for backward compatibility in other modules that import it
    if hasattr(model, "predict_proba") or hasattr(model, "predict"):
        return apply_cal(model, x)
    if hasattr(model, "transform"):
        return model.transform(x)
    return naive_map(x)


def _naive_prob_map(x: np.ndarray) -> np.ndarray:
    return naive_map(x)


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
        raise RuntimeError(
            "Provide --features and --model-pkl or set them in YAML under paths.features/meta_model"
        )

    # Load features efficiently. If a target date is provided, filter at read time.
    target_date: pd.Timestamp | None = pd.Timestamp(args.date) if args.date else None
    day: pd.DataFrame
    f: pd.DataFrame
    if target_date is not None:
        wanted_cols = [
            "date",
            "symbol",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_volume",
        ]
        # Try pyarrow dataset filter for minimal IO; fall back to pandas if unavailable
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.dataset as ds  # type: ignore
            import pyarrow.compute as pc  # type: ignore

            dataset = ds.dataset(features_path)
            tbl = dataset.to_table(
                filter=pc.equal(pc.field("date"), pa.scalar(pd.Timestamp(target_date))),
                columns=wanted_cols,
            )
            day = tbl.to_pandas(types_mapper=None)
            # Build a tiny parent frame to satisfy later code paths expecting `f`
            f = day.copy()
        except Exception:
            # Fallback: read full file but only necessary columns, then filter
            try:
                f = pd.read_parquet(features_path, columns=wanted_cols)
            except Exception:
                f = pd.read_parquet(features_path)[wanted_cols]
            f["date"] = pd.to_datetime(f["date"])
            day = f[f["date"] == target_date].copy()
    else:
        # No date provided: load file then choose the latest available
        f = pd.read_parquet(features_path)
        f["date"] = pd.to_datetime(f["date"])
        target_date = f["date"].max()
        day = f[f["date"] == target_date].copy()

    # Normalize symbol dtype
    f["symbol"] = f["symbol"].astype(str).str.upper()
    day["symbol"] = day["symbol"].astype(str).str.upper()
    # Symbols as categorical to reduce memory
    f["symbol"] = f["symbol"].astype("category")
    day["symbol"] = day["symbol"].astype("category")
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
    specs = compute_specialist_scores(
        day, news_sentiment=news_df, params=cfg.get("sensitivity", {})
    )

    # Calibrate scores to probabilities
    prob_cols: List[str] = []
    oof = None
    calibrators: Dict[str, object] = load_cals(
        calibrators_pkl=(
            args.calibrators_pkl
            or cfg.get("calibration", {}).get("calibrators_pkl", "")
        )
        or None,
        oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
        kind=cfg.get("calibration", {}).get("kind", args.calibration),
    )

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
    scaler = meta.get("scaler")
    if not feature_names:
        feature_names = prob_cols  # fallback to all available *_prob

    # Prepare X in the expected column order
    for col in feature_names:
        if col not in specs.columns:
            specs[col] = 0.5  # neutral
    X = specs[feature_names].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    meta_prob = (
        clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)
    )
    # Optional meta-level calibration
    meta_calib_pkl = args.meta_calibrator_pkl or cfg.get("calibration", {}).get(
        "meta_calibrator_pkl", ""
    )
    if meta_calib_pkl:
        try:
            meta_prob = apply_meta(meta_calib_pkl, meta_prob)
        except Exception as e:
            print(f"[predict] warning: failed to apply meta calibrator: {e}")

    picks = specs[["symbol"]].copy()
    picks["date"] = target_date
    picks["meta_prob"] = meta_prob
    # Deterministic tiebreakers to reduce jitter vs trade_alert
    # Try to attach ADV/ATR for rank stability if panel available
    try:
        from ..infra.feature_join import attach_adv_atr as _attach

        day_adv = _attach(
            f if "date" in f.columns and "symbol" in f.columns else day, target_date
        )
        picks = picks.merge(day_adv, on="symbol", how="left")
    except Exception:
        pass
    rank_cols = ["meta_prob"]
    ascend = [False]
    if "adv20" in picks.columns:
        rank_cols.append("adv20")
        ascend.append(False)
    rank_cols.append("symbol")
    ascend.append(True)
    top_k = int(cfg.get("alert", {}).get("top_k", args.top_k))
    picks = (
        picks.sort_values(rank_cols, ascending=ascend)
        .head(top_k)
        .reset_index(drop=True)
    )

    print(f"[predict] {target_date.date()} top-{top_k}")
    print(picks.to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        picks.to_csv(args.out_csv, index=False)
        print(f"[predict] saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
