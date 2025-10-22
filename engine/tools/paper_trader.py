"""Paper trader: step forward one day using meta probabilities.

Logic:
- On target decision date D (default: latest in features), select top-K symbols by
  meta probability (from --pred for D, or compute via --model-pkl), apply risk gates.
- Rebalance to equal-weights on the picked set subject to turnover cap.
- Compute portfolio return on next trading day D+1 using features['fret_1d'] and
  update equity and ledger. Persist weights as the new holdings.

This is a return-based paper trader (weights, not shares) to keep the prototype simple.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..infra.yaml_config import load_yaml_config
from ..infra.feature_join import attach_adv_atr
from ..infra.env import load_env_files
from ..models.calib_utils import (
    fit_per_specialist_calibrators_from_oof as fit_calibs,
    apply_calibrator as apply_cal,
    naive_prob_map as naive_map,
    load_spec_calibrators as load_cals,
    apply_meta_calibrator as apply_meta_cal,
)
from ..portfolio.scenario_tracker import (
    SCENARIO_FEATURE_COLUMNS,
    ScenarioClassifier,
    ScenarioPerformanceTracker,
)
from ..features.regime import compute_regime_features_daily


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Return-based paper trader (top-K, equal-weight)"
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML with defaults (paths, specialists, risk) — overridden by --style",
    )
    p.add_argument(
        "--style",
        type=str,
        default="",
        help="Preset style (e.g., swing_aggressive, swing_conservative)",
    )
    p.add_argument(
        "--features", required=True, help="Features parquet with date,symbol, fret_1d"
    )
    p.add_argument(
        "--pred",
        type=str,
        default="",
        help="Predictions parquet with date,symbol,meta_prob (optional)",
    )
    p.add_argument(
        "--model-pkl",
        type=str,
        default="",
        help="Meta model pickle to compute probs if --pred missing",
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
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-level calibrator pickle to adjust meta_prob",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file for spec_nlp",
    )
    p.add_argument(
        "--universe-file",
        type=str,
        required=True,
        help="Universe text file (one SYMBOL per line)",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--turnover-cap", type=float, default=None)
    p.add_argument("--state-dir", type=str, default="data/paper")
    p.add_argument(
        "--date", type=str, default="", help="Decision date YYYY-MM-DD (default latest)"
    )
    p.add_argument(
        "--fallback-prev-date",
        action="store_true",
        help="If --date has no next bar, fall back to the previous date that does",
    )
    p.add_argument(
        "--run-to-end",
        action="store_true",
        help="Iterate over all remaining dates until the end of features (requires --pred)",
    )
    # Decision logging
    p.add_argument(
        "--decision-log-csv",
        type=str,
        default="",
        help="Optional CSV to append per-decision symbol diagnostics",
    )
    p.add_argument(
        "--realized-lookahead",
        type=int,
        default=5,
        help="Cumulative forward days for realized return in log",
    )
    p.add_argument(
        "--realized-target-pct",
        type=float,
        default=0.01,
        help="Target pct for hit within lookahead (e.g., 0.01)",
    )
    p.add_argument(
        "--log-specialists",
        action="store_true",
        help="Recompute specialist raw/prob for logging (slower)",
    )
    p.add_argument(
        "--log-all-candidates",
        action="store_true",
        help="Log all gated candidates, not only selected picks",
    )
    p.add_argument(
        "--trade-learning-csv",
        type=str,
        default="",
        help="Optional CSV with per-trade scenario summary for downstream learning",
    )
    # Dynamic specialist weighting
    p.add_argument(
        "--use-specialist-weights",
        action="store_true",
        help="Compute meta_prob as weighted average of per-specialist probabilities",
    )
    p.add_argument(
        "--specialist-weights-yaml",
        type=str,
        default="",
        help="YAML with mapping of specialist prob columns to weights (e.g., spec_technical_prob: 1.0)",
    )
    p.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help=">0.5 => buy consensus; <=0.5 => sell consensus (for logging)",
    )
    # Risk gates
    p.add_argument("--min-adv-usd", type=float, default=1e7)
    p.add_argument("--max-atr-pct", type=float, default=0.05)
    p.add_argument("--earnings-file", type=str, default="")
    p.add_argument("--earnings-blackout", type=int, default=2)
    # Sector/correlation and volatility-aware sizing
    p.add_argument(
        "--sector-map-csv",
        type=str,
        default="",
        help="CSV with columns symbol,sector for sector caps",
    )
    p.add_argument(
        "--sector-cap",
        type=float,
        default=None,
        help="Max portfolio weight per sector (e.g., 0.30)",
    )
    p.add_argument(
        "--vol-lookback",
        type=int,
        default=20,
        help="Lookback days for per-name vol estimate (rolling std of fret_1d)",
    )
    p.add_argument(
        "--kelly-cap",
        type=float,
        default=None,
        help="Cap for Kelly-style fraction tilt based on meta_prob edge",
    )
    # Notifications
    # Separate webhook for trade executions if desired
    p.add_argument(
        "--discord-webhook",
        type=str,
        default=(
            os.environ.get(
                "DISCORD_TRADES_WEBHOOK_URL", os.environ.get("DISCORD_WEBHOOK_URL", "")
            )
        ),
    )
    p.add_argument("--initial-equity", type=float, default=1_000_000.0)
    p.add_argument(
        "--max-name-weight",
        type=float,
        default=None,
        help="Optional cap on per-name target weight (e.g., 0.10)",
    )
    # Conviction/ATR-based sizing (optional)
    p.add_argument(
        "--conviction-risk-sizing",
        action="store_true",
        help="Use conviction- & ATR-based sizing instead of equal-weight",
    )
    p.add_argument(
        "--risk-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="Risk sizing mode for conviction sizing",
    )
    p.add_argument(
        "--risk-pct",
        type=float,
        default=0.005,
        help="Fixed risk fraction per name (e.g., 0.005 for 0.5%)",
    )
    p.add_argument(
        "--risk-min-pct", type=float, default=None, help="Min risk fraction (auto)"
    )
    p.add_argument(
        "--risk-max-pct", type=float, default=None, help="Max risk fraction (auto)"
    )
    p.add_argument(
        "--risk-curve",
        choices=["linear", "quadratic"],
        default="linear",
        help="Conviction mapping curve",
    )
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=1.0,
        help="Stop distance in ATR× for sizing denominator",
    )
    # Optional gradient-based weight optimization
    p.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Use projected gradient to choose weights vs equal-weight",
    )
    p.add_argument(
        "--opt-steps", type=int, default=50, help="Gradient steps for optimization"
    )
    p.add_argument(
        "--opt-lr", type=float, default=0.5, help="Learning rate for optimization"
    )
    p.add_argument(
        "--opt-l2-turnover",
        type=float,
        default=5.0,
        help="L2 penalty coefficient on turnover (approx bps influence)",
    )
    return p.parse_args(argv)


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def _fit_calibrators_from_oof(oof: pd.DataFrame, kind: str = "platt") -> dict:
    return fit_calibs(oof, kind)


def _apply_calibrator(model: object, x: np.ndarray) -> np.ndarray:
    return apply_cal(model, x)


def _naive_prob_map(x: np.ndarray) -> np.ndarray:
    return naive_map(x)


def main(argv: Optional[List[str]] = None) -> None:
    # Ensure env vars from scripts/api.env are available for direct runs
    load_env_files()
    args = parse_args(argv)
    cfg = load_yaml_config(args.config) if args.config else {}
    # Resolve features from CLI or YAML
    if (not args.features) and isinstance(cfg.get("paths"), dict):
        try:
            args.features = cfg["paths"].get("features", args.features)
        except Exception:
            pass
    # Merge paper config defaults for dynamic specialist weights
    try:
        if isinstance(cfg.get("paper", {}), dict):
            if not args.specialist_weights_yaml and cfg["paper"].get(
                "specialist_weights_yaml"
            ):
                args.specialist_weights_yaml = str(cfg["paper"]["specialist_weights_yaml"])  # type: ignore
            if not bool(args.use_specialist_weights) and bool(
                cfg["paper"].get("use_specialist_weights", False)
            ):
                args.use_specialist_weights = True  # type: ignore
            if not args.decision_log_csv and cfg["paper"].get("decision_log_csv"):
                args.decision_log_csv = str(cfg["paper"]["decision_log_csv"])  # type: ignore
    except Exception:
        pass

    features_path = args.features
    f = pd.read_parquet(features_path)
    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()

    uni = set(read_universe(args.universe_file))
    # Paper state paths (load early to choose decision date properly)
    os.makedirs(args.state_dir, exist_ok=True)
    pos_path = os.path.join(args.state_dir, "weights.parquet")
    led_path = os.path.join(args.state_dir, "ledger.parquet")
    led = pd.DataFrame(
        columns=["date", "gross_ret", "turnover", "cost", "net_ret", "equity", "names"]
    )
    if os.path.exists(led_path):
        led = pd.read_parquet(led_path)
        led["date"] = pd.to_datetime(led["date"])  # ensure dtype

    all_dates = sorted(f["date"].unique())
    # Choose decision date:
    if args.date:
        target_date = pd.Timestamp(args.date)
        # If the requested date is the last available (no next day), optionally fall back
        try:
            idx = all_dates.index(target_date)
        except ValueError:
            raise RuntimeError(
                f"Requested --date {target_date.date()} not present in features"
            )
        if idx == len(all_dates) - 1:
            if args.fallback_prev_date:
                if len(all_dates) < 2:
                    raise RuntimeError("Not enough dates in features to fall back.")
                target_date = pd.Timestamp(all_dates[-2])
                print(
                    f"[paper] --date has no next day; falling back to {target_date.date()}"
                )
            else:
                raise RuntimeError(
                    "No next date available in features for paper step. Use --fallback-prev-date or choose an earlier --date."
                )
    else:
        if not led.empty:
            last_step = pd.Timestamp(led["date"].max())
            # Continue stepping from last_step if there is a later date
            later_exists = any(d > last_step for d in all_dates)
            if not later_exists:
                print(
                    f"[paper] No further dates available to step. Last step at {last_step.date()}."
                )
                return
            target_date = last_step
        else:
            # First step: use latest date that has a next day
            if len(all_dates) < 2:
                raise RuntimeError("Not enough dates in features to step.")
            target_date = pd.Timestamp(all_dates[-2])

    # Build risk gates metrics on panel
    f_sorted = f.sort_values(["symbol", "date"]).copy()
    f_sorted["dollar_vol"] = f_sorted["adj_close"] * f_sorted["adj_volume"]
    f_sorted["adv20"] = (
        f_sorted.groupby("symbol")["dollar_vol"]
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # ATR% robust fill
    need_atr_fill = ("atr_pct_14" not in f_sorted.columns) or f_sorted[
        "atr_pct_14"
    ].isna().any()
    if need_atr_fill:
        prev_close = f_sorted.groupby("symbol")["adj_close"].shift(1)
        tr1 = (f_sorted["adj_high"] - f_sorted["adj_low"]).abs()
        tr2 = (f_sorted["adj_high"] - prev_close).abs()
        tr3 = (f_sorted["adj_low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr_close = (f_sorted["adj_close"] - prev_close).abs()
        tr = tr.fillna(tr_close)
        atr14 = tr.groupby(f_sorted["symbol"]).apply(
            lambda s: s.ewm(alpha=1 / 14.0, adjust=False).mean()
        )
        atr14.index = (
            atr14.index.droplevel(0)
            if isinstance(atr14.index, pd.MultiIndex)
            else atr14.index
        )
        atr_pct = (atr14 / f_sorted["adj_close"]).replace([np.inf, -np.inf], np.nan)
        if "atr_pct_14" in f_sorted.columns:
            f_sorted["atr_pct_14"] = f_sorted["atr_pct_14"].fillna(atr_pct)
        else:
            f_sorted["atr_pct_14"] = atr_pct
    f_sorted["atr_pct_14"] = f_sorted.groupby("symbol")["atr_pct_14"].ffill()
    # Per-name daily return volatility estimate for sizing
    try:
        lb = int(max(5, args.vol_lookback))
    except Exception:
        lb = 20
    f_sorted["_sigma"] = (
        f_sorted.groupby("symbol")["fret_1d"]
        .rolling(lb, min_periods=max(5, lb // 2))
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    # Sector map (optional)
    sector_map = None
    if args.sector_map_csv and os.path.exists(args.sector_map_csv):
        try:
            sm = pd.read_csv(args.sector_map_csv)
            sm["symbol"] = sm["symbol"].astype(str).str.upper()
            sector_map = sm.set_index("symbol")["sector"].to_dict()
        except Exception:
            sector_map = None

    # Slice decision day rows in universe (with optional fallback to most recent overlap)
    day = f[(f["date"] == target_date) & (f["symbol"].isin(uni))].copy()
    if day.empty:
        # Try fallback to the most recent date with any overlap AND a next trading date available
        try:
            from pandas import Timestamp as _Ts  # type: ignore

            all_days = sorted(f["date"].unique())
            valid_for_decision = set(
                all_days[:-1]
            )  # require a next-day bar for realized return
            cand = (
                f[f["symbol"].isin(uni)]
                .groupby("date")["symbol"]
                .nunique()
                .sort_index()
            )
            cand = cand[(cand > 0) & (cand.index.isin(list(valid_for_decision)))]
            if not cand.empty and bool(args.fallback_prev_date):
                fb_date = _Ts(cand.index[-1])
                if fb_date != target_date:
                    print(
                        f"[paper] no overlap on {target_date.date()}; falling back to {fb_date.date()}"
                    )
                    target_date = fb_date
                    day = f[(f["date"] == target_date) & (f["symbol"].isin(uni))].copy()
        except Exception:
            pass
    if day.empty:
        raise RuntimeError(f"No rows for universe on date {target_date.date()}")
    day_adv = attach_adv_atr(f_sorted, target_date).rename(
        columns={"atr_pct_14": "atr_pct_14_from_panel"}
    )

    # Compute or load meta probabilities for target_date
    probs_df = None
    # Optional meta-level calibrator
    meta_calib = None
    if args.meta_calibrator_pkl:
        try:
            import pickle as _pickle

            with open(args.meta_calibrator_pkl, "rb") as _fpk:
                payload = _pickle.load(_fpk)
            meta_calib = payload.get("model", payload)
        except Exception as e:
            print(f"[paper] warning: failed to load meta calibrator: {e}")
    # Recompute specialists when using dynamic specialist weights or if no predictions provided
    force_compute_specs = bool(args.use_specialist_weights)
    if args.pred and not force_compute_specs:
        p = pd.read_parquet(args.pred)
        p["date"] = pd.to_datetime(p["date"])
        # For single-step, filter by target_date; for run-to-end we'll filter per-iteration
        if not args.run_to_end:
            probs_df = p[p["date"] == target_date][["symbol", "meta_prob"]].copy()
            if meta_calib is not None and not probs_df.empty:
                try:
                    v = probs_df["meta_prob"].astype(float).values
                    if hasattr(meta_calib, "predict_proba"):
                        v = meta_calib.predict_proba(v.reshape(-1, 1))[:, 1]
                    elif hasattr(meta_calib, "transform"):
                        v = meta_calib.transform(v)
                    probs_df["meta_prob"] = v
                except Exception:
                    pass
    else:
        if args.run_to_end and not force_compute_specs:
            raise RuntimeError(
                "--run-to-end requires --pred containing meta_prob for all dates (unless --use-specialist-weights)"
            )
        model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model", "")
        if not model_path and not args.use_specialist_weights:
            raise RuntimeError(
                "Provide --pred or --model-pkl (or set in YAML), or enable --use-specialist-weights"
            )
        sentiment = None
        if args.news_sentiment or cfg.get("paths", {}).get("news_sentiment"):
            try:
                sentiment = load_sentiment_file(
                    args.news_sentiment or cfg.get("paths", {}).get("news_sentiment")
                )
            except Exception:
                sentiment = None
        spec_params = cfg.get("specialists", {})
        specs = compute_specialist_scores(
            day, news_sentiment=sentiment, params=spec_params
        )
        # Calibrators
        calibrators = load_cals(
            calibrators_pkl=(
                args.calibrators_pkl
                or cfg.get("calibration", {}).get("calibrators_pkl", "")
            )
            or None,
            oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
            kind=cfg.get("calibration", {}).get("kind", "platt"),
        )

        prob_cols: List[str] = []
        for sc in [c for c in specs.columns if c.startswith("spec_")]:
            raw = specs[sc].astype(float).values
            if calibrators and sc in calibrators:
                prob = _apply_calibrator(calibrators[sc], raw)
            else:
                prob = _naive_prob_map(raw)
            specs[f"{sc}_prob"] = prob
            prob_cols.append(f"{sc}_prob")
        # Compute meta probability either via dynamic specialist weights or model
        if args.use_specialist_weights:
            # Load weights YAML; default from config paper.specialist_weights_yaml if not provided
            import yaml  # type: ignore

            weights_path = (
                args.specialist_weights_yaml
                or (
                    cfg.get("paper", {}).get("specialist_weights_yaml")
                    if isinstance(cfg.get("paper", {}), dict)
                    else ""
                )
                or ""
            )
            wmap: dict[str, float] = {}
            if weights_path and os.path.exists(weights_path):
                try:
                    with open(weights_path, "r", encoding="utf-8") as fy:
                        data_y = yaml.safe_load(fy) or {}
                    if isinstance(data_y, dict):
                        wmap = {str(k): float(v) for k, v in data_y.items()}
                except Exception as e:
                    print(
                        f"[paper] warning: failed to load specialist weights from {weights_path}: {e}"
                    )
            # Fallback to equal weights if empty
            if not wmap:
                wmap = {c: 1.0 for c in prob_cols}
            # Build aligned weight vector over available prob columns
            wvec = []
            for c in prob_cols:
                wvec.append(float(wmap.get(c, wmap.get(c.replace("_prob", ""), 0.0))))
            w = pd.Series(wvec, index=prob_cols)
            s = float(w.sum())
            if s <= 0:
                w = pd.Series(1.0 / max(1, len(prob_cols)), index=prob_cols)
            else:
                w = w / s
            meta_prob = (specs[prob_cols] * w.values).sum(axis=1).astype(float).values
            probs_df = pd.DataFrame(
                {"symbol": specs["symbol"].values, "meta_prob": meta_prob}
            )
        else:
            import pickle

            with open(model_path, "rb") as fpk:
                meta = pickle.load(fpk)
            clf = meta.get("model")
            feature_names = meta.get("features") or prob_cols
            scaler = meta.get("scaler")
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
            # Apply optional meta-level calibration (supports per-regime payload)
            if meta_calib is not None:
                try:
                    # Compute regime value for the decision date (single value per date)
                    try:
                        reg = compute_regime_features_daily(f)
                        reg_day = (
                            float(
                                reg.loc[
                                    reg["date"] == target_date,
                                    "regime_vol"
                                    if "regime_vol" in reg.columns
                                    else "regime_risk",
                                ].iloc[0]
                            )
                            if not reg.empty
                            else None
                        )
                    except Exception:
                        reg_day = None
                    rv = np.full_like(
                        meta_prob,
                        reg_day if reg_day is not None else np.nan,
                        dtype=float,
                    )
                    meta_prob = apply_meta_cal(meta_calib, meta_prob, regime_vals=rv)
                except Exception as e:
                    print(f"[paper] warning: meta calibration failed: {e}")
            probs_df = pd.DataFrame(
                {"symbol": specs["symbol"].values, "meta_prob": meta_prob}
            )

    # Apply risk gates
    specs = (
        day[["symbol"]].merge(probs_df, on="symbol", how="left")
        if probs_df is not None
        else day[["symbol"]]
    )
    specs = specs.merge(day_adv, on="symbol", how="left")
    specs = specs.drop_duplicates(subset="symbol", keep="first").reset_index(drop=True)
    if "atr_pct_14_from_panel" in specs:
        if "atr_pct_14" in specs.columns:
            specs["atr_pct_14"] = specs["atr_pct_14"].fillna(
                specs["atr_pct_14_from_panel"]
            )
        else:
            specs["atr_pct_14"] = specs["atr_pct_14_from_panel"]
        specs = specs.drop(columns=["atr_pct_14_from_panel"])
    min_adv = float(cfg.get("risk", {}).get("min_adv_usd", args.min_adv_usd))
    max_atr = float(cfg.get("risk", {}).get("max_atr_pct", args.max_atr_pct))
    earn_blk = int(cfg.get("risk", {}).get("earnings_blackout", args.earnings_blackout))
    atr_missing_ok = bool(cfg.get("risk", {}).get("atr_missing_ok", True))
    liq = specs["adv20"].notna() & (specs["adv20"] >= min_adv)
    atr = (specs["atr_pct_14"] <= max_atr) | (
        specs["atr_pct_14"].isna() if atr_missing_ok else False
    )
    earn = pd.Series(True, index=specs.index)
    if args.earnings_file:
        try:
            earn_df = (
                pd.read_parquet(args.earnings_file)
                if args.earnings_file.lower().endswith(".parquet")
                else pd.read_csv(args.earnings_file)
            )
            earn_df["date"], earn_df["symbol"] = (
                pd.to_datetime(earn_df["date"]),
                earn_df["symbol"].astype(str).str.upper(),
            )
            blk = earn_df.copy()
            blk["start"], blk["end"] = blk["date"] - pd.Timedelta(days=earn_blk), blk[
                "date"
            ] + pd.Timedelta(days=earn_blk)
            in_blk = (
                blk[(blk["start"] <= target_date) & (blk["end"] >= target_date)][
                    "symbol"
                ]
                .unique()
                .tolist()
            )
            earn = ~specs["symbol"].isin(in_blk)
        except Exception:
            pass
    specs["liq_ok"] = liq.astype(int)
    specs["atr_ok"] = atr.astype(int)
    specs["earn_ok"] = earn.astype(int)
    specs["_risk_ok"] = (liq & atr & earn).astype(int)

    picks = specs.loc[specs["_risk_ok"], ["symbol", "meta_prob"]].sort_values(
        "meta_prob", ascending=False
    )
    picks = picks.head(int(cfg.get("top_k", args.top_k))).reset_index(drop=True)

    # Next trading date helper
    def _next_date(cur: pd.Timestamp) -> Optional[pd.Timestamp]:
        later = [d for d in all_dates if d > cur]
        return later[0] if later else None

    next_date = _next_date(target_date)
    if next_date is None:
        raise RuntimeError("No next date available in features for paper step")

    # Load previous weights and equity
    if os.path.exists(pos_path):
        prev = pd.read_parquet(pos_path)
        if not prev.empty:
            prev = prev.drop_duplicates(subset="symbol", keep="last")
        prev_w = (
            prev.set_index("symbol")["weight"]
            if not prev.empty
            else pd.Series(dtype=float)
        )
    else:
        prev_w = pd.Series(dtype=float)
    last_equity = (
        float(led["equity"].iloc[-1]) if not led.empty else float(args.initial_equity)
    )

    # Equal-weight or optimized target weights
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        # Project onto the probability simplex {w>=0, sum w=1}
        if v.size == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            theta = 0.0
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1) / float(rho + 1)
        w = np.maximum(v - theta, 0)
        s = w.sum()
        return w / s if s > 0 else w

    if len(picks) == 0:
        w_target = pd.Series(dtype=float)
    else:
        # Sizing: optimize, conviction/ATR sizing, or equal-weight
        if args.optimize_weights:
            # Optimize: maximize sum(w * score) - lambda * ||w - prev||^2 subject to simplex
            # Use score ~ (meta_prob - 0.5) as a simple proxy for expected return
            scores = picks["meta_prob"].astype(float).values - 0.5
            w0 = np.full(len(picks), 1.0 / len(picks), dtype=float)
            prev_w_vec = prev_w.reindex(picks["symbol"].values).fillna(0.0).values
            lam = float(args.opt_l2_turnover)
            lr = float(args.opt_lr)
            w_vec = w0.copy()
            for _ in range(int(args.opt_steps)):
                grad = scores - 2.0 * lam * (w_vec - prev_w_vec)
                w_vec = w_vec + lr * grad
                w_vec = _project_simplex(w_vec)
            w_target = pd.Series(w_vec, index=picks["symbol"].values)
        elif (
            bool(args.conviction_risk_sizing)
            or str(cfg.get("risk", {}).get("sizing_mode", "")).lower()
            == "conviction_atr"
        ):
            # Conviction- and ATR-based sizing: w ∝ risk_frac(meta_prob) / (stop_mult * ATR%)
            stop_mult = float(
                cfg.get("risk", {}).get("stop_atr_mult", args.stop_atr_mult)
            )

            # Determine per-name risk fraction
            def _risk_frac(p: float) -> float:
                mode = str(cfg.get("risk", {}).get("risk_mode", args.risk_mode)).lower()
                if mode == "auto":
                    rmin = cfg.get("risk", {}).get("min_risk_pct", args.risk_min_pct)
                    rmax = cfg.get("risk", {}).get("max_risk_pct", args.risk_max_pct)
                    rmin = float(rmin if rmin is not None else 0.002)
                    rmax = float(rmax if rmax is not None else 0.006)
                    conv_n = max(0.0, min(1.0, (float(p) - 0.5) / 0.5))
                    curve = str(cfg.get("risk", {}).get("risk_curve", args.risk_curve))
                    if curve == "quadratic":
                        conv_n = conv_n * conv_n
                    return float(rmin + (rmax - rmin) * conv_n)
                return float(cfg.get("risk", {}).get("risk_pct", args.risk_pct))

            # Gather ATR% for picks
            atr_map = (
                specs.set_index("symbol")["atr_pct_14"]
                .reindex(picks["symbol"])
                .astype(float)
            )
            den = stop_mult * atr_map
            # Replace non-finite/zero with median of finite; if still invalid, fallback to equal-weight
            finite = den.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                w_target = pd.Series(1.0 / len(picks), index=picks["symbol"].values)
            else:
                med = float(finite.median())
                den = den.replace([np.inf, -np.inf], np.nan).fillna(med)
                den = den.clip(lower=1e-6)
                risks = picks["meta_prob"].astype(float).apply(_risk_frac).values
                score = risks / den.values
                score = np.clip(score, 0.0, None)
                s = float(score.sum())
                if s <= 0:
                    w_target = pd.Series(1.0 / len(picks), index=picks["symbol"].values)
                else:
                    w_target = pd.Series(score / s, index=picks["symbol"].values)
        else:
            w_target = pd.Series(1.0 / len(picks), index=picks["symbol"].values)

        # Apply per-name weight cap if provided (from args or YAML risk)
        w_cap = args.max_name_weight
        if w_cap is None:
            try:
                w_cap = (
                    float(cfg.get("risk", {}).get("max_name_weight"))
                    if cfg.get("risk", {})
                    else None
                )
            except Exception:
                w_cap = None
        if w_cap is not None and float(w_cap) > 0:
            w_cap = float(w_cap)
            w_target = w_target.clip(upper=w_cap)
            s = float(w_target.sum())
            if s > 0:
                w_target = w_target / s

        # Volatility targeting tilt (risk parity style)
        try:
            sig_map = (
                f_sorted[f_sorted["date"] == target_date]
                .set_index("symbol")["_sigma"]
                .to_dict()
            )
            sig = w_target.index.to_series().map(sig_map).astype(float)
            med = (
                float(sig.replace([np.inf, -np.inf], np.nan).median())
                if len(sig)
                else 0.02
            )
            sig = sig.replace([np.inf, -np.inf], np.nan).fillna(
                med if med > 0 else 0.02
            )
            inv = 1.0 / sig.replace(0.0, np.nan)
            inv = inv.fillna(inv.median() if hasattr(inv, "median") else 1.0)
            vw = w_target * inv
            s = float(vw.sum())
            if s > 0:
                w_target = vw / s
        except Exception:
            pass

        # Kelly-style tilt
        try:
            if args.kelly_cap is not None and float(args.kelly_cap) >= 0:
                prob = (
                    picks.set_index("symbol")["meta_prob"]
                    .reindex(w_target.index)
                    .astype(float)
                    .fillna(0.5)
                )
                edge = (2.0 * prob - 1.0).clip(lower=0.0)
                cap = float(args.kelly_cap)
                if cap > 0:
                    edge = edge.clip(0.0, cap)
                kw = w_target * edge
                s = float(kw.sum())
                if s > 0:
                    w_target = kw / s
        except Exception:
            pass

        # Sector cap projection if provided
        def _project_sector_cap(w: pd.Series, cap: float) -> pd.Series:
            if sector_map is None:
                return w
            s = w.copy()
            for _ in range(5):
                dfw = s.reset_index()
                dfw.columns = ["symbol", "w"]
                dfw["sector"] = dfw["symbol"].map(sector_map).fillna("UNKNOWN")
                sec_sum = dfw.groupby("sector")["w"].sum()
                over = sec_sum[sec_sum > cap]
                if over.empty:
                    break
                for sec, tot in over.items():
                    mask = dfw["sector"] == sec
                    if tot > 0:
                        dfw.loc[mask, "w"] *= cap / float(tot)
                s = dfw.set_index("symbol")["w"]
                s = s / max(1e-12, float(s.sum()))
            return s

        try:
            if args.sector_cap is not None and float(args.sector_cap) > 0:
                w_target = _project_sector_cap(w_target, float(args.sector_cap))
        except Exception:
            pass

    # Turnover and cap
    aligned_prev = prev_w.reindex(w_target.index).fillna(0.0)
    delta = w_target - aligned_prev
    turnover = float(delta.abs().sum())
    if args.turnover_cap is not None and turnover > float(args.turnover_cap):
        alpha = max(0.0, min(1.0, float(args.turnover_cap) / max(1e-12, turnover)))
        w_target = aligned_prev + alpha * delta
        s = float(w_target.sum())
        if s > 0:
            w_target = w_target / s
        turnover = float((w_target - aligned_prev).abs().sum())

    scenario_classifier = ScenarioClassifier()
    scenario_tracker = ScenarioPerformanceTracker(scenario_classifier)
    scenario_cols_all = ["symbol"]
    scenario_cols_all.extend([c for c in SCENARIO_FEATURE_COLUMNS if c in f.columns])
    scenario_cols_all.extend(
        [c for c in f.columns if isinstance(c, str) and c.startswith("spec_")]
    )
    seen_cols = set()
    scenario_cols_all = [
        c for c in scenario_cols_all if not (c in seen_cols or seen_cols.add(c))
    ]

    def _write_trade_learning_log() -> None:
        if not args.trade_learning_csv:
            return
        try:
            trades_df = scenario_tracker.completed_trades_frame()
            if trades_df.empty:
                print("[paper] scenario trade log: no completed trades to write")
                return
            dir_name = os.path.dirname(args.trade_learning_csv) or "."
            os.makedirs(dir_name, exist_ok=True)
            trades_df.to_csv(args.trade_learning_csv, index=False)
            print(
                f"[paper] scenario trade log -> {args.trade_learning_csv} rows={len(trades_df)}"
            )
        except Exception as e:
            print(f"[paper] warning: failed to write scenario trade log: {e}")

    # Helper: decision log writer
    def _append_decision_log(
        decision_date: pd.Timestamp,
        next_date_val: pd.Timestamp,
        chosen: pd.DataFrame,
        r_map_series: pd.Series | None,
        w_prev_series: pd.Series,
        w_tgt_series: pd.Series,
        specs_all: pd.DataFrame,
    ) -> None:
        if not args.decision_log_csv:
            return
        try:
            os.makedirs(os.path.dirname(args.decision_log_csv), exist_ok=True)
        except Exception:
            pass
        # Build a wide frame for logging
        cols = ["symbol", "meta_prob", "adv20", "atr_pct_14"]
        base = specs_all.copy()
        for c in cols:
            if c not in base.columns:
                base[c] = np.nan
        base = base[cols].drop_duplicates("symbol")
        base["selected"] = base["symbol"].isin(w_tgt_series.index)
        base["w_prev"] = base["symbol"].map(w_prev_series.to_dict()).fillna(0.0)
        base["w_tgt"] = base["symbol"].map(w_tgt_series.to_dict()).fillna(0.0)
        base["dw"] = (base["w_tgt"] - base["w_prev"]).abs()
        if r_map_series is not None:
            base["fret_1d_next"] = (
                base["symbol"].map(r_map_series.to_dict()).fillna(0.0)
            )
            base["gross_contrib"] = base["w_tgt"] * base["fret_1d_next"]
        else:
            base["fret_1d_next"] = np.nan
            base["gross_contrib"] = np.nan
        cost_per_unit = float(args.cost_bps) / 1e4
        base["cost_contrib"] = cost_per_unit * base["dw"]
        base["net_contrib"] = base["gross_contrib"] - base["cost_contrib"]
        # Risk diagnostics if available from specs_all
        for name in ["_ok", "_risk_ok", "liq_ok", "atr_ok", "earn_ok"]:
            if name in specs_all.columns:
                m = specs_all.set_index("symbol")[name]
                base[name] = base["symbol"].map(m.to_dict())
        base.insert(0, "date_decision", pd.Timestamp(decision_date).date())
        base.insert(1, "date_next", pd.Timestamp(next_date_val).date())
        # Optionally compute specialist scores for logging
        if args.log_specialists:
            try:
                day_local = f[
                    (f["date"] == decision_date) & (f["symbol"].isin(base["symbol"]))
                ].copy()
                sentiment = None
                if args.news_sentiment:
                    try:
                        sentiment = load_sentiment_file(args.news_sentiment)
                    except Exception:
                        sentiment = None
                spec_params = cfg.get("specialists", {})
                det = compute_specialist_scores(
                    day_local, news_sentiment=sentiment, params=spec_params
                )
                # Compute per-specialist probabilities for logging
                # Prepare calibrators if available
                cals = {}
                try:
                    if args.calibrators_pkl or cfg.get("calibration", {}).get(
                        "calibrators_pkl"
                    ):
                        import pickle as _pkl

                        with open(
                            args.calibrators_pkl
                            or cfg.get("calibration", {}).get("calibrators_pkl"),
                            "rb",
                        ) as fpk:
                            payload = _pkl.load(fpk)
                        cals = payload.get("models", {})
                    elif args.oof or cfg.get("paths", {}).get("oof"):
                        oof = pd.read_parquet(
                            args.oof or cfg.get("paths", {}).get("oof")
                        )
                        cals = _fit_calibrators_from_oof(
                            oof, kind=cfg.get("calibration", {}).get("kind", "platt")
                        )
                except Exception:
                    cals = {}
                for sc in [
                    c
                    for c in det.columns
                    if c.startswith("spec_") and not c.endswith("_prob")
                ]:
                    rawv = det[sc].astype(float).values
                    if sc in cals:
                        pv = _apply_calibrator(cals[sc], rawv)
                    else:
                        pv = _naive_prob_map(rawv)
                    det[f"{sc}_prob"] = pv
                # Attach raw and prob specialist columns
                attach_cols = [c for c in det.columns if c.startswith("spec_")]
                det = det[["symbol"] + attach_cols].drop_duplicates("symbol")
                base = base.merge(det, on="symbol", how="left")
                # Compute consensus metrics from specialist probabilities
                prob_cols_local = [
                    c
                    for c in base.columns
                    if c.startswith("spec_") and c.endswith("_prob")
                ]
                if prob_cols_local:
                    base["spec_consensus_prob"] = base[prob_cols_local].mean(axis=1)
                    thr = float(args.consensus_threshold)
                    base["spec_consensus"] = (base["spec_consensus_prob"] > thr).astype(
                        int
                    )
                    base["spec_positive_count"] = (
                        base[prob_cols_local].gt(thr).sum(axis=1)
                    )
            except Exception as e:
                print(f"[paper] warning: specialist logging failed: {e}")
        # If only selected wanted, filter
        if not args.log_all_candidates:
            base = base[base["selected"]]
        # Realized outcomes (requires fret_1d)
        if "fret_1d" in f.columns:
            f_sorted = f.sort_values(["symbol", "date"]).reset_index(drop=True)
            groups = {
                sym: df.reset_index(drop=True) for sym, df in f_sorted.groupby("symbol")
            }
            look = int(max(1, args.realized_lookahead))

            def _cum_ret(sym: str, dt: pd.Timestamp, n: int) -> float:
                g = groups.get(sym)
                if g is None:
                    return float("nan")
                idx = g.index[g["date"] == dt]
                if idx.size == 0:
                    prev = g.index[g["date"] < dt]
                    if prev.size == 0:
                        return float("nan")
                    start = int(prev.max())
                else:
                    start = int(idx.max())
                seq = g.loc[start + 1 : start + int(n), "fret_1d"].astype(float).values
                if seq.size == 0:
                    return float("nan")
                return float(np.prod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0)

            for N in (1, 3, 5):
                base[f"ret_{N}d"] = [
                    _cum_ret(str(s), decision_date, N) for s in base["symbol"].values
                ]
            base[f"cum_ret_{look}d"] = [
                _cum_ret(str(s), decision_date, look) for s in base["symbol"].values
            ]

            def _t_hit(sym: str, dt: pd.Timestamp, target: float, n: int) -> float:
                g = groups.get(sym)
                if g is None:
                    return float("nan")
                idx = g.index[g["date"] == dt]
                if idx.size == 0:
                    prev = g.index[g["date"] < dt]
                    if prev.size == 0:
                        return float("nan")
                    start = int(prev.max())
                else:
                    start = int(idx.max())
                seq = g.loc[start + 1 : start + int(n), "fret_1d"].astype(float).values
                if seq.size == 0:
                    return float("nan")
                path = np.cumprod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0
                hit_idx = next(
                    (i for i, v in enumerate(path, start=1) if v >= float(target)), None
                )
                return float(hit_idx) if hit_idx is not None else float("nan")

            base[f"t_hit_{look}d"] = [
                _t_hit(str(s), decision_date, float(args.realized_target_pct), look)
                for s in base["symbol"].values
            ]
            base[f"hit_{look}d"] = base[f"t_hit_{look}d"].apply(
                lambda x: 1 if pd.notna(x) else 0
            )
        try:
            feat_cols_today = [c for c in scenario_cols_all if c in f.columns]
            feat_slice = f.loc[f["date"] == decision_date, feat_cols_today].copy()
            if "symbol" in feat_slice.columns:
                feat_slice["symbol"] = feat_slice["symbol"].astype(str)
            base["symbol"] = base["symbol"].astype(str)
            base = scenario_tracker.process_day(decision_date, base, feat_slice)
        except Exception as e:
            print(f"[paper] warning: failed to enrich scenario log: {e}")
        # Append or create CSV
        try:
            existing = None
            if os.path.exists(args.decision_log_csv):
                try:
                    existing = pd.read_csv(args.decision_log_csv, low_memory=False)
                except Exception as e_read:
                    print(
                        f"[paper] warning: decision log load failed, rebuilding file: {e_read}"
                    )
                    existing = None
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, base], ignore_index=True, sort=False)
                combined = combined.drop_duplicates(
                    subset=["date_decision", "symbol"], keep="last"
                )
                combined.to_csv(args.decision_log_csv, index=False)
            else:
                base.to_csv(args.decision_log_csv, index=False)
            print(
                f"[paper] appended decision log -> {args.decision_log_csv} rows={len(base)}"
            )
        except Exception as e:
            print(f"[paper] warning: failed to write decision log: {e}")

    # Run-to-end mode accumulates multiple steps; otherwise do one step
    rows_out = []
    # Preload predictions once if provided (for run-to-end)
    preloaded_pred = None
    if args.pred:
        try:
            preloaded_pred = pd.read_parquet(args.pred)
            preloaded_pred["date"] = pd.to_datetime(preloaded_pred["date"])
        except Exception:
            preloaded_pred = None

    def _step_once(
        decision_date: pd.Timestamp, w_prev: pd.Series, eq_prev: float
    ) -> tuple[pd.Timestamp, pd.DataFrame, float, float, float, pd.Series, pd.Series]:
        # rebuild picks for decision_date using predictions
        nonlocal probs_df
        local_probs = probs_df
        if bool(args.use_specialist_weights):
            # Recompute specialists for this date and build weighted consensus
            day_loc2 = f[(f["date"] == decision_date) & (f["symbol"].isin(uni))].copy()
            spec_params2 = cfg.get("specialists", {})
            det2 = compute_specialist_scores(
                day_loc2, news_sentiment=None, params=spec_params2
            )
            cals2 = load_cals(
                calibrators_pkl=(
                    args.calibrators_pkl
                    or cfg.get("calibration", {}).get("calibrators_pkl", "")
                )
                or None,
                oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
                kind=cfg.get("calibration", {}).get("kind", "platt"),
            )
            prob_cols2: list[str] = []
            for sc in [c for c in det2.columns if c.startswith("spec_")]:
                rawv = det2[sc].astype(float).values
                if cals2 and sc in cals2:
                    pv = _apply_calibrator(cals2[sc], rawv)
                else:
                    pv = _naive_prob_map(rawv)
                det2[f"{sc}_prob"] = pv
                prob_cols2.append(f"{sc}_prob")
            # Load weights
            import yaml as _yaml  # type: ignore

            wpath = (
                args.specialist_weights_yaml
                or (
                    cfg.get("paper", {}).get("specialist_weights_yaml")
                    if isinstance(cfg.get("paper", {}), dict)
                    else ""
                )
                or ""
            )
            wmap: dict[str, float] = {}
            if wpath and os.path.exists(wpath):
                try:
                    with open(wpath, "r", encoding="utf-8") as fy:
                        wy = _yaml.safe_load(fy) or {}
                    if isinstance(wy, dict):
                        wmap = {str(k): float(v) for k, v in wy.items()}
                except Exception:
                    wmap = {}
            if not wmap:
                wmap = {c: 1.0 for c in prob_cols2}
            wvec = [
                float(wmap.get(c, wmap.get(c.replace("_prob", ""), 0.0)))
                for c in prob_cols2
            ]
            wser = pd.Series(wvec, index=prob_cols2)
            ss = float(wser.sum())
            wser = (
                (wser / ss)
                if ss > 0
                else pd.Series(1.0 / max(1, len(prob_cols2)), index=prob_cols2)
            )
            det2["meta_prob"] = (det2[prob_cols2] * wser.values).sum(axis=1)
            local_probs = det2[["symbol", "meta_prob"]]
        elif local_probs is None and args.pred:
            # If we deferred filtering, use preloaded df if available
            if preloaded_pred is not None:
                pp = preloaded_pred
            else:
                pp = pd.read_parquet(args.pred)
                pp["date"] = pd.to_datetime(pp["date"])
            local_probs = pp[pp["date"] == decision_date][["symbol", "meta_prob"]]
            # Apply optional meta calibration to parquet slice
            if meta_calib is not None and not local_probs.empty:
                try:
                    v = local_probs["meta_prob"].astype(float).values
                    if hasattr(meta_calib, "predict_proba"):
                        v = meta_calib.predict_proba(v.reshape(-1, 1))[:, 1]
                    elif hasattr(meta_calib, "transform"):
                        v = meta_calib.transform(v)
                    local_probs["meta_prob"] = v
                except Exception:
                    pass
        day_local = f[(f["date"] == decision_date) & (f["symbol"].isin(uni))].copy()
        adv_local = f_sorted[f_sorted["date"] == decision_date][
            ["symbol", "adv20", "atr_pct_14"]
        ].rename(columns={"atr_pct_14": "atr_pct_14_from_panel"})
        specs_local = (
            day_local[["symbol"]].merge(local_probs, on="symbol", how="left")
            if local_probs is not None
            else day_local[["symbol"]]
        )
        specs_local = specs_local.merge(adv_local, on="symbol", how="left")
        if "atr_pct_14_from_panel" in specs_local:
            if "atr_pct_14" in specs_local.columns:
                specs_local["atr_pct_14"] = specs_local["atr_pct_14"].fillna(
                    specs_local["atr_pct_14_from_panel"]
                )
            else:
                specs_local["atr_pct_14"] = specs_local["atr_pct_14_from_panel"]
            specs_local = specs_local.drop(columns=["atr_pct_14_from_panel"])
        liq_l = specs_local["adv20"].notna() & (specs_local["adv20"] >= min_adv)
        atr_l = (specs_local["atr_pct_14"] <= max_atr) | (
            specs_local["atr_pct_14"].isna() if atr_missing_ok else False
        )
        specs_local["liq_ok"], specs_local["atr_ok"] = liq_l.astype(int), atr_l.astype(
            int
        )
        specs_local["_ok"] = liq_l & atr_l
        picks_local = (
            specs_local.loc[specs_local["_ok"], ["symbol", "meta_prob", "atr_pct_14"]]
            .sort_values("meta_prob", ascending=False)
            .head(int(cfg.get("top_k", args.top_k)))
        )
        # Sizing inside run-to-end
        if len(picks_local) == 0:
            w_tgt = pd.Series(dtype=float)
        elif args.optimize_weights:
            scr = picks_local["meta_prob"].astype(float).values - 0.5
            w0 = np.full(len(picks_local), 1.0 / len(picks_local), dtype=float)
            prev_w_vec = w_prev.reindex(picks_local["symbol"].values).fillna(0.0).values
            lam = float(args.opt_l2_turnover)
            lr = float(args.opt_lr)
            w_vec = w0.copy()
            for _ in range(int(args.opt_steps)):
                grad = scr - 2.0 * lam * (w_vec - prev_w_vec)
                w_vec = _project_simplex(w_vec + lr * grad)
            w_tgt = pd.Series(w_vec, index=picks_local["symbol"].values)
        elif (
            bool(args.conviction_risk_sizing)
            or str(cfg.get("risk", {}).get("sizing_mode", "")).lower()
            == "conviction_atr"
        ):
            stop_mult = float(
                cfg.get("risk", {}).get("stop_atr_mult", args.stop_atr_mult)
            )

            def _risk_frac(p: float) -> float:
                mode = str(cfg.get("risk", {}).get("risk_mode", args.risk_mode)).lower()
                if mode == "auto":
                    rmin = cfg.get("risk", {}).get("min_risk_pct", args.risk_min_pct)
                    rmax = cfg.get("risk", {}).get("max_risk_pct", args.risk_max_pct)
                    rmin = float(rmin if rmin is not None else 0.002)
                    rmax = float(rmax if rmax is not None else 0.006)
                    conv_n = max(0.0, min(1.0, (float(p) - 0.5) / 0.5))
                    curve = str(cfg.get("risk", {}).get("risk_curve", args.risk_curve))
                    if curve == "quadratic":
                        conv_n = conv_n * conv_n
                    return float(rmin + (rmax - rmin) * conv_n)
                return float(cfg.get("risk", {}).get("risk_pct", args.risk_pct))

            den = stop_mult * picks_local["atr_pct_14"].astype(float).values
            den = np.where(~np.isfinite(den) | (den <= 0), np.nan, den)
            if np.all(np.isnan(den)):
                w_tgt = pd.Series(
                    1.0 / len(picks_local), index=picks_local["symbol"].values
                )
            else:
                med = float(np.nanmedian(den))
                den = np.where(np.isnan(den), med, den)
                risks = picks_local["meta_prob"].astype(float).apply(_risk_frac).values
                score = np.clip(risks / den, 0.0, None)
                s = float(score.sum())
                w_tgt = pd.Series(
                    (score / s) if s > 0 else np.full(len(score), 1.0 / len(score)),
                    index=picks_local["symbol"].values,
                )
        else:
            w_tgt = pd.Series(
                1.0 / len(picks_local), index=picks_local["symbol"].values
            )
        # Volatility targeting tilt
        try:
            sig_map_l = (
                f_sorted[f_sorted["date"] == decision_date]
                .set_index("symbol")["_sigma"]
                .to_dict()
            )
            sig_l = w_tgt.index.to_series().map(sig_map_l).astype(float)
            med_l = (
                float(sig_l.replace([np.inf, -np.inf], np.nan).median())
                if len(sig_l)
                else 0.02
            )
            sig_l = sig_l.replace([np.inf, -np.inf], np.nan).fillna(
                med_l if med_l > 0 else 0.02
            )
            inv_l = 1.0 / sig_l.replace(0.0, np.nan)
            inv_l = inv_l.fillna(inv_l.median() if hasattr(inv_l, "median") else 1.0)
            vw_l = w_tgt * inv_l
            s_l = float(vw_l.sum())
            if s_l > 0:
                w_tgt = vw_l / s_l
        except Exception:
            pass
        # Kelly tilt
        try:
            if args.kelly_cap is not None and float(args.kelly_cap) >= 0:
                prob_l = (
                    picks_local.set_index("symbol")["meta_prob"]
                    .reindex(w_tgt.index)
                    .astype(float)
                    .fillna(0.5)
                )
                edge_l = (2.0 * prob_l - 1.0).clip(lower=0.0)
                cap_l = float(args.kelly_cap)
                if cap_l > 0:
                    edge_l = edge_l.clip(0.0, cap_l)
                kw_l = w_tgt * edge_l
                s2 = float(kw_l.sum())
                if s2 > 0:
                    w_tgt = kw_l / s2
        except Exception:
            pass
        # Sector cap
        try:
            if args.sector_cap is not None and float(args.sector_cap) > 0:
                # reuse helper defined above
                def _project_sector_cap_l(w: pd.Series, cap: float) -> pd.Series:
                    if sector_map is None:
                        return w
                    s = w.copy()
                    for _ in range(5):
                        dfw = s.reset_index()
                        dfw.columns = ["symbol", "w"]
                        dfw["sector"] = dfw["symbol"].map(sector_map).fillna("UNKNOWN")
                        sec_sum = dfw.groupby("sector")["w"].sum()
                        over = sec_sum[sec_sum > cap]
                        if over.empty:
                            break
                        for sec, tot in over.items():
                            mask = dfw["sector"] == sec
                            if tot > 0:
                                dfw.loc[mask, "w"] *= cap / float(tot)
                        s = dfw.set_index("symbol")["w"]
                        s = s / max(1e-12, float(s.sum()))
                    return s

                w_tgt = _project_sector_cap_l(w_tgt, float(args.sector_cap))
        except Exception:
            pass
        aligned_prev_l = w_prev.reindex(w_tgt.index).fillna(0.0)
        delta_l = w_tgt - aligned_prev_l
        turn_l = float(delta_l.abs().sum())
        if args.turnover_cap is not None and turn_l > float(args.turnover_cap):
            alpha = max(0.0, min(1.0, float(args.turnover_cap) / max(1e-12, turn_l)))
            w_tgt = aligned_prev_l + alpha * delta_l
            s = float(w_tgt.sum())
            if s > 0:
                w_tgt = w_tgt / s
            turn_l = float((w_tgt - aligned_prev_l).abs().sum())
        dec_rows_l = f[(f["date"] == decision_date) & (f["symbol"].isin(w_tgt.index))]
        # Ensure unique symbol -> return mapping for this date (dedupe by first occurrence)
        dec_rows_l = dec_rows_l.drop_duplicates(subset="symbol", keep="first")
        r_map_l = (
            dec_rows_l.set_index("symbol")["fret_1d"].reindex(w_tgt.index).fillna(0.0)
        )
        gross_l = float((w_tgt * r_map_l).sum())
        cost_l = (float(args.cost_bps) / 1e4) * turn_l
        net_l = gross_l - cost_l
        eq_l = eq_prev * (1.0 + net_l)
        nx = _next_date(decision_date)
        if nx is None:
            raise StopIteration
        return nx, picks_local, gross_l, cost_l, net_l, w_tgt, r_map_l

    if args.run_to_end:
        cur_dec = target_date
        cur_w = prev_w
        cur_eq = last_equity
        while True:
            try:
                (
                    nx_date,
                    picks_local,
                    gross_ret,
                    cost,
                    net_ret,
                    w_target,
                    r_map,
                ) = _step_once(cur_dec, cur_w, cur_eq)
            except StopIteration:
                break
            # Persist per step
            out_pos = pd.DataFrame(
                {"symbol": w_target.index, "weight": w_target.values}
            ).sort_values("symbol")
            out_pos = out_pos.drop_duplicates(subset="symbol", keep="last")
            out_pos.to_parquet(pos_path, index=False)
            new_row = pd.DataFrame(
                {
                    "date": [nx_date],
                    "gross_ret": [gross_ret],
                    "turnover": [
                        float(
                            (w_target - cur_w.reindex(w_target.index).fillna(0.0))
                            .abs()
                            .sum()
                        )
                    ],
                    "cost": [cost],
                    "net_ret": [net_ret],
                    "equity": [cur_eq * (1.0 + net_ret)],
                    "names": [int((w_target > 0).sum())],
                }
            )
            led = (
                pd.concat([led, new_row], ignore_index=True)
                if not led.empty
                else new_row
            )
            # Decision log for this step
            specs_local_full = f[(f["date"] == cur_dec) & (f["symbol"].isin(uni))][
                ["symbol"]
            ].merge(picks_local, on="symbol", how="left")
            _append_decision_log(
                cur_dec, nx_date, picks_local, r_map, cur_w, w_target, specs_local_full
            )
            cur_w = w_target
            cur_eq = float(new_row["equity"].iloc[-1])
            cur_dec = nx_date
        led.to_parquet(led_path, index=False)
        _write_trade_learning_log()
        print(
            f"[paper] Run-to-end completed. Steps={len(led)} Final equity=${cur_eq:,.2f}"
        )
        return
    else:
        # Single-step path
        dec_rows = f[(f["date"] == target_date) & (f["symbol"].isin(w_target.index))]
        dec_rows = dec_rows.drop_duplicates(subset="symbol", keep="first")
        if "fret_1d" not in f.columns:
            raise RuntimeError("features must include 'fret_1d' forward return")
        r_map = (
            dec_rows.set_index("symbol")["fret_1d"].reindex(w_target.index).fillna(0.0)
        )
        gross_ret = float((w_target * r_map).sum())
        cost = (float(args.cost_bps) / 1e4) * turnover
        net_ret = gross_ret - cost
        equity = last_equity * (1.0 + net_ret)
        # Decision log for single step
        specs_full = specs.copy()
        _append_decision_log(
            target_date, next_date, picks, r_map, prev_w, w_target, specs_full
        )

    # Persist state
    out_pos = pd.DataFrame(
        {"symbol": w_target.index, "weight": w_target.values}
    ).sort_values("symbol")
    out_pos = out_pos.drop_duplicates(subset="symbol", keep="last")
    out_pos.to_parquet(pos_path, index=False)
    new_row = pd.DataFrame(
        {
            "date": [next_date],
            "gross_ret": [gross_ret],
            "turnover": [turnover],
            "cost": [cost],
            "net_ret": [net_ret],
            "equity": [equity],
            "names": [int((w_target > 0).sum())],
        }
    )
    led = pd.concat([led, new_row], ignore_index=True) if not led.empty else new_row
    led.to_parquet(led_path, index=False)
    _write_trade_learning_log()

    print(
        f"[paper] Decision date={target_date.date()} next_date={next_date.date()} picks={len(w_target)} turnover={turnover:.3f}"
    )
    print(
        f"[paper] gross_ret={gross_ret:.5f} cost={cost:.5f} net_ret={net_ret:.5f} equity={equity:,.2f}"
    )
    if len(w_target):
        show = pd.DataFrame(
            {
                "symbol": w_target.index,
                "weight": w_target.values,
                "ret_next": r_map.values,
            }
        )
        if show["symbol"].duplicated().any():
            show = show.groupby("symbol", as_index=False).agg(
                weight=("weight", "sum"),
                ret_next=("ret_next", "mean"),
            )
        show = show.sort_values("weight", ascending=False)
        print("[paper] holdings (top 10):")
        print(show.head(10).to_string(index=False))

    # Optional Discord notification
    # Prefer YAML 'paper.discord_webhook' (or 'alert.discord_webhook') over env/CLI default
    hook_cfg = (
        (
            cfg.get("paper", {}).get("discord_webhook")
            if isinstance(cfg.get("paper", {}), dict)
            else ""
        )
        or (
            cfg.get("alert", {}).get("discord_webhook")
            if isinstance(cfg.get("alert", {}), dict)
            else ""
        )
        or ""
    )
    notif_hook = hook_cfg or args.discord_webhook
    if notif_hook:
        try:
            from ..infra.notify import send_discord

            tail = notif_hook[-6:]
            pct = 100.0 * net_ret
            msg_lines = [
                f"Paper Trade — {str(next_date.date())}",
                f"Net: {pct:+.2f}%  Equity: ${equity:,.0f}",
                f"Turnover: {turnover:.2f}  Names: {int((w_target > 0).sum())}",
            ]
            if len(w_target):
                top_lines = []
                for _, r in show.head(5).iterrows():
                    top_lines.append(
                        f"- {r['symbol']}: w={100*float(r['weight']):.1f}%  r_next={100*float(r['ret_next']):+.2f}%"
                    )
                msg_lines.append("Top holdings:")
                msg_lines.extend(top_lines)
            msg = "\n".join(msg_lines)
            print(f"[paper] sending to Discord webhook ...{tail}")
            send_discord(notif_hook, f"@everyone\n{msg}")
            print("[paper] notification sent.")
        except Exception as e:
            print(f"[paper] notification failed: {e}")


if __name__ == "__main__":
    main()
