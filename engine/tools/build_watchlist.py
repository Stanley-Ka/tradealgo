from __future__ import annotations

"""Build a daily watchlist of symbols by meta probability.

Inputs:
- Features parquet (daily or intraday snapshot) with date,symbol and baseline columns
- Either a meta predictions parquet (paths.meta) with date,symbol,meta_prob
  or a model pickle (paths.meta_model) plus specialists config to compute meta_prob.

Outputs:
- Text file with one SYMBOL per line (uppercased), top-N by meta_prob for the target date.

Usage:
  python -m engine.tools.build_watchlist \
    --features data/datasets/features_daily_1D.parquet \
    --model-pkl data/models/meta_lr.pkl \
    --oof data/datasets/oof_specialists.parquet \
    --out engine/data/universe/watchlist.txt --top-k 500 --min-price 1 --min-adv-usd 5e6
"""

import argparse
import os
from typing import Optional, List

import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..models.calib_utils import (
    load_spec_calibrators as load_cals,
    apply_calibrator as apply_cal,
    naive_prob_map as naive_map,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily watchlist by meta_prob")
    p.add_argument(
        "--config", type=str, default="", help="YAML preset/config with default paths"
    )
    p.add_argument("--features", required=False, default="")
    p.add_argument(
        "--pred",
        type=str,
        default="",
        help="Optional predictions parquet with date,symbol,meta_prob",
    )
    p.add_argument(
        "--model-pkl", type=str, default="", help="Meta model pickle if --pred missing"
    )
    p.add_argument(
        "--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)"
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Pickle with per-specialist calibrators (optional)",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file for spec_nlp",
    )
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="Target date YYYY-MM-DD (default latest in features)",
    )
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--min-adv-usd", type=float, default=None)
    p.add_argument("--bucket-adv", action="store_true", help="Diversify by ADV buckets")
    p.add_argument(
        "--buckets", type=int, default=3, help="Number of ADV buckets when --bucket-adv"
    )
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _read_universe(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    price_bounds: tuple[Optional[float], Optional[float]],
    min_adv: Optional[float],
) -> pd.DataFrame:
    day = df[df["date"] == target_date].copy()
    pmin, pmax = price_bounds
    prc = day.get("adj_close", day.get("close"))
    if pmin is not None:
        mask = prc >= float(pmin)
        day = day.loc[mask]
    if pmax is not None:
        mask = prc <= float(pmax)
        day = day.loc[mask]
    if min_adv is not None and "adv20" in day.columns:
        day = day.loc[day["adv20"] >= float(min_adv)]
    return day


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Resolve features path: prefer CLI; else, from config paths.features
    feat_path = args.features
    if (not feat_path) and args.config:
        try:
            from ..infra.yaml_config import load_yaml_config

            cfg = load_yaml_config(args.config)
            feat_path = (
                cfg.get("paths", {}).get("features", "")
                if isinstance(cfg.get("paths"), dict)
                else ""
            )
        except Exception:
            feat_path = ""
    if not feat_path:
        raise RuntimeError(
            "Provide --features or set paths.features in YAML via --config"
        )
    f = pd.read_parquet(feat_path)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["symbol"] = f["symbol"].astype(str).str.upper()
    all_dates = sorted(f["date"].unique())
    if not all_dates:
        raise RuntimeError("features has no dates")
    target_date = pd.Timestamp(args.date) if args.date else all_dates[-1]
    day = _read_universe(
        f, target_date, (args.min_price, args.max_price), args.min_adv_usd
    )
    if day.empty:
        print(f"[watchlist] no rows on {target_date.date()} after filters")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").close()
        return

    probs_df = None
    if args.pred:
        try:
            p = pd.read_parquet(args.pred)
            p["date"] = pd.to_datetime(p["date"]).dt.normalize()
            probs_df = p[p["date"] == target_date][["symbol", "meta_prob"]].copy()
        except Exception:
            probs_df = None
    if probs_df is None:
        # Compute specialist scores and meta from model
        if not args.model_pkl:
            raise RuntimeError("Provide --pred or --model-pkl to build watchlist")
        sent = None
        if args.news_sentiment:
            try:
                sent = load_sentiment_file(args.news_sentiment)
            except Exception:
                sent = None
        specs = compute_specialist_scores(day, news_sentiment=sent, params={})
        calibrators = load_cals(
            calibrators_pkl=args.calibrators_pkl or None,
            oof_path=args.oof or None,
            kind="platt",
        )
        prob_cols: list[str] = []
        for sc in [
            c
            for c in specs.columns
            if c.startswith("spec_") and not c.endswith("_prob")
        ]:
            raw = specs[sc].astype(float).values
            prob = (
                apply_cal(calibrators.get(sc), raw)
                if (calibrators and sc in calibrators)
                else naive_map(raw)
            )
            specs[f"{sc}_prob"] = prob
            prob_cols.append(f"{sc}_prob")
        import pickle

        with open(args.model_pkl, "rb") as fpk:
            payload = pickle.load(fpk)
        clf = payload.get("model")
        feature_names = payload.get("features") or prob_cols
        scaler = payload.get("scaler")
        for col in feature_names:
            if col not in specs.columns:
                specs[col] = 0.5
        X = specs[feature_names].values
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                pass
        meta_prob = (
            clf.predict_proba(X)[:, 1]
            if hasattr(clf, "predict_proba")
            else clf.predict(X)
        )
        probs_df = pd.DataFrame(
            {"symbol": specs["symbol"].values, "meta_prob": meta_prob}
        )

    merged = day[["symbol"]].merge(probs_df, on="symbol", how="left")
    if (
        bool(getattr(args, "bucket_adv", False))
        and "adv20" in day.columns
        and int(args.buckets) > 1
    ):
        # Build ADV buckets and take top per bucket equally
        k = int(args.top_k)
        b = int(args.buckets)
        merged = merged.merge(day[["symbol", "adv20"]], on="symbol", how="left")
        try:
            q = pd.qcut(
                merged["adv20"].rank(method="first"), b, labels=False, duplicates="drop"
            )
        except Exception:
            q = None
        if q is not None:
            merged["_bucket"] = q
            per = max(1, k // b)
            parts = []
            for i in sorted(merged["_bucket"].dropna().unique().astype(int).tolist()):
                part = (
                    merged[merged["_bucket"] == i]
                    .sort_values("meta_prob", ascending=False)
                    .head(per)
                )
                parts.append(part)
            wl = pd.concat(parts, ignore_index=True).dropna(subset=["symbol"]).head(k)
        else:
            wl = (
                merged.sort_values("meta_prob", ascending=False)
                .head(int(args.top_k))
                .dropna(subset=["symbol"])
            )
    else:
        wl = (
            merged.sort_values("meta_prob", ascending=False)
            .head(int(args.top_k))
            .dropna(subset=["symbol"])
        )
    syms = wl["symbol"].astype(str).str.upper().drop_duplicates().tolist()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(s + "\n")
    print(f"[watchlist] wrote {len(syms)} symbols -> {args.out}")


if __name__ == "__main__":
    main()
