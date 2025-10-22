"""Build a supervised learning dataset for swing trading.

For each trading day in a date range, select top-K candidates (swing preset)
and record realized outcomes at multiple horizons (e.g., 3/7/14 days) as
returns and binary labels (up/down vs entry reference price).

The script is designed to be idempotent and resume-friendly: if the output
CSV/Parquet exists and --resume is set, it skips dates already processed.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..infra.yaml_config import load_yaml_config
from ..infra.styles import resolve_style
from ..infra.feature_join import attach_adv_atr_panel
from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..models.calib_utils import (
    load_spec_calibrators as _load_cal,
    apply_calibrator as _apply_cal,
    naive_prob_map as _naive,
    apply_meta_calibrator as _apply_meta,
)


def _read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build swing training dataset with multi-horizon labels"
    )
    p.add_argument(
        "--features",
        required=True,
        help="Features parquet with daily bars and baseline features",
    )
    p.add_argument(
        "--model-pkl", required=True, help="Meta model pickle to compute meta_prob"
    )
    p.add_argument(
        "--universe-file", required=True, help="Universe file (one SYMBOL per line)"
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/datasets/swing_training_dataset.csv",
        help="Output CSV/Parquet path",
    )
    p.add_argument(
        "--timeframes",
        type=str,
        default="3,7,14",
        help="Comma-separated horizons in trading days (e.g., 3,7,14)",
    )
    p.add_argument(
        "--entry-price",
        choices=["close", "open"],
        default="close",
        help="Reference entry price for labels",
    )
    p.add_argument(
        "--top-k", type=int, default=20, help="Top-K picks per day to record"
    )
    p.add_argument(
        "--start", type=str, default="", help="Start date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--end", type=str, default="", help="End date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML with defaults (overridden by flags)",
    )
    p.add_argument(
        "--style",
        type=str,
        default="",
        help="Preset style (swing_aggressive|swing_conservative)",
    )
    p.add_argument(
        "--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)"
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators pickle (optional)",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Meta-level calibrator (optional)",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file for spec_nlp",
    )
    # Risk gates / filters
    p.add_argument("--min-adv-usd", type=float, default=1e7)
    p.add_argument("--max-atr-pct", type=float, default=0.05)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    # Behavior
    p.add_argument(
        "--resume", action="store_true", help="Skip dates already present in the output"
    )
    p.add_argument(
        "--require-all-horizons",
        action="store_true",
        help="Drop rows with missing labels on any horizon",
    )
    # Optional hit-before-stop labels within horizon windows
    p.add_argument(
        "--tp-pct",
        type=float,
        default=0.02,
        help="Take-profit threshold as simple return (e.g., 0.02 for +2%)",
    )
    p.add_argument(
        "--sl-pct",
        type=float,
        default=0.03,
        help="Stop-loss threshold as simple return (e.g., 0.03 for -3%)",
    )
    p.add_argument(
        "--add-hit-before-stop",
        action="store_true",
        help="Add labels for TP hit before SL within each horizon",
    )
    return p.parse_args(argv)


def _coerce_timeframes(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return sorted(set([t for t in out if t > 0]))


def _prepare_forward_returns(f: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    # Attach future close for each horizon per symbol; we’ll use it to compute entry-referenced returns
    f = f.copy()
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    for h in horizons:
        col = f"_close_fut_{h}d"
        f[col] = f.groupby("symbol")["adj_close"].shift(-h)
    return f


def _load_meta(model_path: str):
    import pickle

    with open(model_path, "rb") as fpk:
        meta = pickle.load(fpk)
    clf = meta.get("model")
    feature_names = meta.get("features")
    scaler = meta.get("scaler")
    return clf, feature_names, scaler


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg_path = args.config
    if args.style:
        maybe = resolve_style(args.style)
        if maybe:
            cfg_path = maybe
    cfg = load_yaml_config(cfg_path) if cfg_path else {}

    # Load features (filter columns and dates early to reduce memory)
    wanted = [
        "date",
        "symbol",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
        # common baseline features used by specialists
        "ret_1d",
        "fret_1d",
        "ret_5d",
        "ret_20d",
        "mom_sma_5_20",
        "price_z_20",
        "meanrev_20",
        "vol_z_20",
        "atr_pct_14",
    ]
    try:
        f = pd.read_parquet(args.features, columns=wanted)
    except Exception:
        f = pd.read_parquet(args.features)
    f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None)
    f["symbol"] = f["symbol"].astype(str).str.upper()
    # Filter date range early
    if args.start:
        f = f[f["date"] >= pd.Timestamp(args.start)]
    if args.end:
        f = f[f["date"] <= pd.Timestamp(args.end)]
    # Universe filtering
    uni = set(_read_universe(args.universe_file))
    f = f[f["symbol"].isin(uni)].copy()
    # Ensure forward 1D return available (for path-based labels)
    if "fret_1d" not in f.columns:
        f = f.sort_values(["symbol", "date"])  # ensure order
        f["fret_1d"] = f.groupby("symbol")["adj_close"].pct_change(-1) * (-1.0)
    # Precompute forward closes for horizons
    horizons = _coerce_timeframes(args.timeframes)
    f = _prepare_forward_returns(f, horizons)
    # Compute risk metrics across the panel once (ADV/ATR)
    f = attach_adv_atr_panel(f)
    # Dates to process
    all_dates = sorted(f["date"].unique().tolist())

    # Resume support: skip dates already present in output
    processed: set[pd.Timestamp] = set()
    out_path = args.out
    is_csv = out_path.lower().endswith(".csv")
    if args.resume and os.path.exists(out_path):
        try:
            if is_csv:
                prev = pd.read_csv(out_path)
            else:
                prev = pd.read_parquet(out_path)
            if "date_decision" in prev.columns:
                processed = set(
                    pd.to_datetime(prev["date_decision"])
                    .dt.normalize()
                    .unique()
                    .tolist()
                )
        except Exception:
            processed = set()

    # Load calibrators (fit from OOF if provided)
    calibrators = _load_cal(
        calibrators_pkl=(
            args.calibrators_pkl
            or cfg.get("calibration", {}).get("calibrators_pkl", "")
        )
        or None,
        oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
        kind=cfg.get("calibration", {}).get("kind", "platt"),
    )
    # Load meta model and optional meta-calibrator
    clf, feature_names, scaler = _load_meta(args.model_pkl)
    meta_cal_path = args.meta_calibrator_pkl or cfg.get("calibration", {}).get(
        "meta_calibrator_pkl", ""
    )

    # Optional sentiment
    news_df = None
    if args.news_sentiment or cfg.get("paths", {}).get("news_sentiment"):
        try:
            news_df = load_sentiment_file(
                args.news_sentiment or cfg.get("paths", {}).get("news_sentiment")
            )
        except Exception:
            news_df = None

    # Compute specialists and calibrate for the entire subset (vectorized)
    specs_all = compute_specialist_scores(
        f, news_sentiment=news_df, params=cfg.get("specialists", {})
    )
    # Per-specialist probs
    for sc in [
        c
        for c in specs_all.columns
        if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = specs_all[sc].astype(float).values
        prob = (
            _apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else _naive(raw)
        )
        specs_all[f"{sc}_prob"] = prob
    # Meta probability for all rows
    cols = feature_names or [c for c in specs_all.columns if c.endswith("_prob")]
    for col in cols:
        if col not in specs_all.columns:
            specs_all[col] = 0.5
    X = specs_all[cols].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    if hasattr(clf, "predict_proba"):
        mp = clf.predict_proba(X)[:, 1]
    else:
        mp = clf.predict(X)
    mp = _apply_meta(meta_cal_path, mp)
    specs_all["meta_prob"] = mp
    # Risk gates across panel
    min_adv = float(args.min_adv_usd)
    max_atr = float(args.max_atr_pct)
    liq = specs_all["adv20"].notna() & (specs_all["adv20"] >= min_adv)
    atr_ok = (specs_all["atr_pct_14"] <= max_atr) | specs_all["atr_pct_14"].isna()
    gate = liq & atr_ok
    # Price gating on target date
    if args.min_price is not None or args.max_price is not None:
        pcol = "adj_close" if args.entry_price == "close" else "adj_open"
        try:
            if args.min_price is not None:
                gate = gate & (specs_all[pcol] >= float(args.min_price))
            if args.max_price is not None:
                gate = gate & (specs_all[pcol] <= float(args.max_price))
        except Exception:
            pass
    gated = specs_all[gate].copy()
    # For each date, take top-K by meta_prob
    gated.sort_values(["date", "meta_prob"], ascending=[True, False], inplace=True)
    # Drop duplicates per date/symbol to be safe
    gated = gated.drop_duplicates(["date", "symbol"], keep="first")
    # If resuming, drop dates already processed
    if processed:
        gated = gated[~gated["date"].dt.normalize().isin(list(processed))]
    # Top-K per date via groupby+head
    topk = (
        gated.groupby("date", as_index=False, sort=True, group_keys=False)
        .head(int(args.top_k))
        .copy()
    )
    if topk.empty:
        print("[swing-ds] no rows after gating; nothing to write")
        return
    # Join forward closes for horizons on same date
    fut_cols = {h: f"_close_fut_{h}d" for h in horizons}
    # Build map for the date's base frame
    base = f[["date", "symbol", *fut_cols.values()]].copy()
    merged = topk.merge(base, on=["date", "symbol"], how="left")
    entry_ref = "adj_close" if args.entry_price == "close" else "adj_open"
    for h in horizons:
        fut = merged[fut_cols[h]].astype(float)
        ref = merged[entry_ref].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (fut / ref) - 1.0
        merged[f"ret_{h}d"] = r.replace([np.inf, -np.inf], np.nan)
        merged[f"label_up_{h}d"] = (merged[f"ret_{h}d"].astype(float) > 0.0).astype(
            "Int8"
        )
    if args.require_all_horizons:
        ok = np.ones(len(merged), dtype=bool)
        for h in horizons:
            ok &= merged[f"ret_{h}d"].notna().values
        merged = merged[ok].copy()
        if merged.empty:
            print("[swing-ds] no rows with complete horizons; nothing to write")
            return
    # Final selection of columns
    merged.insert(0, "date_decision", merged["date"].dt.date)
    merged["entry_ref_price"] = merged[entry_ref].astype(float)
    keep = [
        "symbol",
        "date_decision",
        "meta_prob",
        "adv20",
        "atr_pct_14",
        "entry_ref_price",
    ]
    keep += [c for c in merged.columns if c.startswith("spec_") and c.endswith("_prob")]
    for h in horizons:
        keep += [f"ret_{h}d", f"label_up_{h}d"]
    # Optional: TP-before-SL labels per horizon using path of daily returns
    if bool(args.add_hit_before_stop):
        # Build per-symbol forward daily returns map
        f_sorted = f.sort_values(["symbol", "date"]).reset_index(drop=True)
        grp = {sym: df.reset_index(drop=True) for sym, df in f_sorted.groupby("symbol")}

        def _time_to_hits(
            sym: str, dt: pd.Timestamp, n: int, tp: float, sl: float
        ) -> tuple[float, float, float]:
            g = grp.get(sym)
            if g is None:
                return float("nan"), float("nan"), float("nan")
            idx = g.index[g["date"] == dt]
            if idx.size == 0:
                prev = g.index[g["date"] < dt]
                if prev.size == 0:
                    return float("nan"), float("nan"), float("nan")
                start = int(prev.max())
            else:
                start = int(idx.max())
            seq = g.loc[start + 1 : start + int(n), "fret_1d"].astype(float).values
            if seq.size == 0:
                return float("nan"), float("nan"), float("nan")
            path = np.cumprod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0
            t_tp = next(
                (i for i, v in enumerate(path, start=1) if v >= float(tp)), float("nan")
            )
            t_sl = next(
                (i for i, v in enumerate(path, start=1) if v <= -float(sl)),
                float("nan"),
            )
            # winner label: 1 if tp hit before sl within horizon
            if not np.isnan(t_tp) and (np.isnan(t_sl) or t_tp < t_sl):
                win = 1.0
            else:
                win = 0.0
            return float(t_tp), float(t_sl), float(win)

        for h in horizons:
            tps: List[float] = []
            sls: List[float] = []
            wins: List[int] = []
            # iterate rows of merged for this horizon
            for _, rrow in merged.iterrows():
                sym = str(rrow["symbol"]).upper()
                dt = pd.Timestamp(rrow["date"]) if "date" in rrow else pd.Timestamp(merged.loc[_, "date_decision"])  # type: ignore[index]
                # merged has both date and date_decision; prefer date if present
                dte = pd.Timestamp(
                    rrow.get("date", pd.Timestamp(rrow.get("date_decision")))
                )
                t_tp, t_sl, win = _time_to_hits(
                    sym, dte, int(h), float(args.tp_pct), float(args.sl_pct)
                )
                tps.append(t_tp)
                sls.append(t_sl)
                wins.append(int(win))
            merged[f"t_hit_tp_{h}d"] = tps
            merged[f"t_hit_sl_{h}d"] = sls
            merged[f"label_tp_before_sl_{h}d"] = pd.Series(wins, dtype="Int8")
            keep += [f"t_hit_tp_{h}d", f"t_hit_sl_{h}d", f"label_tp_before_sl_{h}d"]
    out_df = merged[[c for c in keep if c in merged.columns]].copy()
    # Write once (append if resume)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and args.resume:
        if is_csv:
            out_df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            prev = pd.read_parquet(out_path)
            pd.concat([prev, out_df], axis=0, ignore_index=True).to_parquet(
                out_path, index=False
            )
    else:
        if is_csv:
            out_df.to_csv(out_path, index=False)
        else:
            out_df.to_parquet(out_path, index=False)
    print(f"[swing-ds] wrote rows={len(out_df)} -> {out_path}")


if __name__ == "__main__":
    main()
