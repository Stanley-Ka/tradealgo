"""Trade alert generator: find top setup in a universe and fetch recent news sentiment.

Usage:
  python -m engine.tools.trade_alert \
    --features data/datasets/features_daily_1D.parquet \
    --model-pkl data/models/meta_lr.pkl \
    --oof data/datasets/oof_specialists.parquet \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --provider finnhub --from-days 3 --top-k 1

Requires FINNHUB_API_KEY in env for news provider (if provider=finnhub).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from typing import List, Optional

import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..news.providers import fetch_news
from ..news.sentiment import score_news
from ..infra.yaml_config import load_yaml_config
from ..infra.styles import resolve_style, normalize_plan
from ..infra.env import load_env_files
from ..infra.log import get_logger
from ..infra.feature_join import attach_adv_atr
from ..infra.reason import consensus_for_symbol, expected_return_and_horizon
from ..infra.sector import apply_sector_cap as _sector_cap
from ..models.calib_utils import apply_meta_calibrator as _apply_meta

log = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trade alert with top setup + news sentiment"
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML config with defaults/thresholds (overridden by --style)",
    )
    p.add_argument(
        "--style",
        type=str,
        default="",
        help="Preset style (e.g., swing_aggressive, swing_conservative)",
    )
    p.add_argument(
        "--features",
        required=False,
        default="",
        help="Features parquet with date,symbol,baseline features",
    )
    p.add_argument(
        "--model-pkl",
        required=False,
        default="",
        help="Trained meta model pickle (from train_meta.py)",
    )
    p.add_argument(
        "--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)"
    )
    p.add_argument(
        "--universe-file",
        type=str,
        default="engine/data/universe/nasdaq100.example.txt",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional precomputed sentiment file",
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Pickle with per-specialist calibrators (from run_cv)",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-level calibrator to transform meta_prob",
    )
    p.add_argument(
        "--provider",
        type=str,
        default="polygon",
        help="News provider key (polygon|finnhub|none)",
    )
    p.add_argument(
        "--from-days", type=int, default=3, help="News lookback window in days"
    )
    p.add_argument(
        "--top-k", type=int, default=1, help="How many top candidates to fetch news for"
    )
    p.add_argument(
        "--explore-prob",
        type=float,
        default=0.0,
        help="Probability to replace last pick with a random from next-N",
    )
    p.add_argument(
        "--explore-topn",
        type=int,
        default=10,
        help="Pool size for exploration (next-N after top-K)",
    )
    p.add_argument(
        "--exclude-file",
        type=str,
        default="",
        help="Optional file with symbols to exclude (one per line)",
    )
    # Price range filters (applied on target date)
    p.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="If set, require price >= min on target date",
    )
    p.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="If set, require price <= max on target date",
    )
    # Risk gates
    p.add_argument(
        "--min-adv-usd",
        type=float,
        default=1e7,
        help="Minimum 20D average dollar volume",
    )
    p.add_argument(
        "--max-atr-pct",
        type=float,
        default=0.05,
        help="Maximum ATR(14)/price to limit volatility",
    )
    p.add_argument(
        "--earnings-file",
        type=str,
        default="",
        help="Optional earnings calendar CSV/Parquet (date,symbol)",
    )
    p.add_argument(
        "--earnings-blackout",
        type=int,
        default=2,
        help="Do not alert within +/- N days of earnings",
    )
    # Sizing suggestions for Discord
    p.add_argument(
        "--account-equity",
        type=float,
        default=100000.0,
        help="Account equity for sizing suggestions",
    )
    p.add_argument(
        "--risk-pct",
        type=float,
        default=0.005,
        help="Per-trade risk fraction (e.g., 0.005 for 0.5%)",
    )
    # Dynamic risk sizing
    p.add_argument(
        "--risk-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="fixed: use --risk-pct; auto: choose risk between min/max based on conviction",
    )
    p.add_argument(
        "--risk-min-pct",
        type=float,
        default=None,
        help="Min risk fraction for auto mode (e.g., 0.002 for 0.2%)",
    )
    p.add_argument(
        "--risk-max-pct",
        type=float,
        default=None,
        help="Max risk fraction for auto mode (e.g., 0.006 for 0.6%)",
    )
    p.add_argument(
        "--risk-curve",
        choices=["linear", "quadratic", "sqrt"],
        default="linear",
        help="Conviction mapping curve for auto risk sizing",
    )
    p.add_argument(
        "--risk-base-prob",
        type=float,
        default=None,
        help="Base probability for zero-conviction (default 0.5)",
    )
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=1.0,
        help="Stop distance in ATR% multiples",
    )
    # Notional caps
    p.add_argument(
        "--max-name-weight",
        type=float,
        default=None,
        help="Max fraction of equity per name (e.g., 0.10 for 10%)",
    )
    p.add_argument(
        "--max-position-notional",
        type=float,
        default=None,
        help="Absolute $ cap per position",
    )
    p.add_argument(
        "--entry-price",
        choices=["close", "open"],
        default="close",
        help="Reference price for sizing (close/open of target date)",
    )
    p.add_argument(
        "--price-source",
        choices=["feature", "live"],
        default=None,
        help="Use feature price or fetch latest live price for sizing/message (CLI overrides YAML)",
    )
    p.add_argument(
        "--live-provider",
        choices=["yahoo", "polygon"],
        default=None,
        help="Provider for live price if price-source=live (CLI overrides YAML)",
    )
    p.add_argument(
        "--polygon-plan",
        choices=["auto", "starter", "pro", "enterprise"],
        default=None,
        help="(Deprecated: prefer --plan) Internal polygon plan hint",
    )
    p.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Provider plan tier (basic|starter|developer|advanced)",
    )
    p.add_argument(
        "--debug-risk",
        action="store_true",
        help="Print per-symbol risk gate diagnostics",
    )
    # Notifications (Discord/Slack)
    # Prefer dedicated alerts webhook if set; fall back to generic
    p.add_argument(
        "--discord-webhook",
        type=str,
        default=(
            os.environ.get(
                "DISCORD_ALERTS_WEBHOOK_URL", os.environ.get("DISCORD_WEBHOOK_URL", "")
            )
        ),
    )
    p.add_argument(
        "--slack-webhook", type=str, default=os.environ.get("SLACK_WEBHOOK_URL", "")
    )
    p.add_argument(
        "--date", type=str, default="", help="Target date YYYY-MM-DD (default latest)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build message but do not send to Discord",
    )
    p.add_argument(
        "--print-specialist-probs",
        action="store_true",
        help="Print per-specialist probability columns for picked symbols",
    )
    p.add_argument(
        "--alert-log-csv",
        type=str,
        default="",
        help="Append a CSV row per run with top picks and diagnostics",
    )
    p.add_argument(
        "--heartbeat-on-empty",
        action="store_true",
        help="If no candidates, send a small heartbeat to Discord",
    )
    p.add_argument(
        "--alert-kind",
        choices=["pre-market", "intraday"],
        default="intraday",
        help="Template variant used for Discord messaging",
    )
    p.add_argument(
        "--alert-category",
        choices=["watchlist", "explore"],
        default="watchlist",
        help="Classifier for recommendations logging (refined watchlist vs exploratory scan)",
    )
    p.add_argument(
        "--recommendations-csv",
        type=str,
        default="",
        help="Optional CSV to append sent recommendations for downstream consumers",
    )
    p.add_argument(
        "--min-repeat-mins",
        type=int,
        default=60,
        help="Throttle duplicate alerts: if the exact same symbol set was alerted within N minutes, skip send",
    )
    p.add_argument(
        "--dedupe-per-day",
        action="store_true",
        help="When recommendations CSV is set, skip symbols already alerted today",
    )
    # Optional sector cap
    p.add_argument(
        "--sector-map-csv",
        type=str,
        default="",
        help="CSV with columns symbol,sector for diversification",
    )
    p.add_argument(
        "--sector-cap",
        type=int,
        default=0,
        help="If >0, limit top-K per sector to this number",
    )
    # Intraday mix (optional): blend daily meta with intraday snapshot heuristic
    p.add_argument(
        "--mix-intraday",
        type=float,
        default=0.0,
        help="Weight in [0,1] for intraday blend; 0 disables",
    )
    p.add_argument(
        "--intraday-features",
        type=str,
        default="",
        help="Latest intraday features snapshot parquet",
    )
    p.add_argument(
        "--mix-k",
        type=float,
        default=2.0,
        help="Logistic slope for converting intraday score to prob",
    )
    return p.parse_args(argv)


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def main(argv: Optional[List[str]] = None) -> None:
    from ..models.calib_utils import (
        fit_per_specialist_calibrators_from_oof as _fit_calibrators_from_oof,
        apply_calibrator as _apply_calibrator,
        naive_prob_map as _naive_prob_map,
    )
    import pickle

    # Ensure env vars (e.g., POLYGON_API_KEY) are available even for direct runs
    load_env_files()
    args = parse_args(argv)

    # Load config and features (prefer --style if provided)
    cfg = {}
    cfg_path = args.config
    if args.style:
        maybe = resolve_style(args.style)
        if maybe:
            cfg_path = maybe
    if cfg_path:
        try:
            cfg = load_yaml_config(cfg_path)
        except Exception as e:
            log.warning("failed to load YAML config: %s", e)
    features_path = args.features or cfg.get("paths", {}).get("features", "")
    model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model", "")
    if not features_path or not model_path:
        raise RuntimeError(
            "Provide --features and --model-pkl or set them in YAML under paths.features/meta_model"
        )
    f = pd.read_parquet(features_path)
    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()
    target_date = pd.Timestamp(args.date) if args.date else f["date"].max()
    uni_file = (
        args.universe_file
        or cfg.get("alert", {}).get("universe_file", "")
        or "engine/data/universe/nasdaq100.example.txt"
    )
    uni = set(read_universe(uni_file))
    day = f[(f["date"] == target_date) & (f["symbol"].isin(uni))].copy()
    if day.empty:
        raise RuntimeError(f"No rows for universe on date {target_date.date()}")

    # Optional precomputed sentiment file to include as a specialist
    sentiment_file = None
    cfg_news = cfg.get("paths", {}).get("news_sentiment", "")
    if args.news_sentiment or cfg_news:
        try:
            sentiment_file = load_sentiment_file(args.news_sentiment or cfg_news)
        except Exception as e:
            print(f"[alert] warning: failed to load sentiment file: {e}")

    # Compute specialists and calibrate
    spec_params = cfg.get("specialists", {})
    specs = compute_specialist_scores(
        day, news_sentiment=sentiment_file, params=spec_params
    )
    # Load OOF for calibrators if provided
    calibrators = {}
    calib_pkl = args.calibrators_pkl or cfg.get("calibration", {}).get(
        "calibrators_pkl", ""
    )
    if calib_pkl:
        try:
            import pickle

            with open(calib_pkl, "rb") as fpk:
                payload = pickle.load(fpk)
            calibrators = payload.get("models", {})
        except Exception as e:
            print(f"[alert] warning: could not load calibrators: {e}")
    elif args.oof or cfg.get("paths", {}).get("oof"):
        try:
            oof = pd.read_parquet(args.oof or cfg.get("paths", {}).get("oof"))
            kind = cfg.get("calibration", {}).get("kind", "platt")
            calibrators = _fit_calibrators_from_oof(oof, kind)
        except Exception as e:
            print(f"[alert] warning: could not fit calibrators: {e}")

    prob_cols: List[str] = []
    for sc in [c for c in specs.columns if c.startswith("spec_")]:
        raw = specs[sc].astype(float).values
        if calibrators and sc in calibrators:
            prob = _apply_calibrator(calibrators[sc], raw)
        else:
            prob = _naive_prob_map(raw)
        specs[f"{sc}_prob"] = prob
        prob_cols.append(f"{sc}_prob")

    # Load meta model and compute meta probabilities
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
        clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)
    )
    # Optional meta-level calibration from args or YAML
    meta_calib_path = args.meta_calibrator_pkl or cfg.get("calibration", {}).get(
        "meta_calibrator_pkl", ""
    )
    if meta_calib_path:
        meta_prob = _apply_meta(meta_calib_path, meta_prob)
    specs["meta_prob"] = meta_prob

    # Risk gates: liquidity (ADV), volatility (ATR pct), earnings blackout
    # Compute 20D ADV on the panel and join today rows
    # Risk thresholds from config or flags
    min_adv_usd = float(cfg.get("risk", {}).get("min_adv_usd", args.min_adv_usd))
    max_atr_pct = float(cfg.get("risk", {}).get("max_atr_pct", args.max_atr_pct))
    earnings_blackout = int(
        cfg.get("risk", {}).get("earnings_blackout", args.earnings_blackout)
    )

    day_adv = attach_adv_atr(f, target_date).rename(
        columns={"atr_pct_14": "atr_pct_14_from_panel"}
    )
    specs = specs.merge(day_adv, on="symbol", how="left")
    if "atr_pct_14" in specs.columns:
        specs["atr_pct_14"] = specs["atr_pct_14"].fillna(specs["atr_pct_14_from_panel"])
    else:
        specs["atr_pct_14"] = specs["atr_pct_14_from_panel"]
    specs = specs.drop(columns=["atr_pct_14_from_panel"])
    # Optional price range gating using feature prices on target_date
    # CLI overrides YAML; if CLI not set, use YAML alert.min_price/max_price
    try:
        min_price_cfg = cfg.get("alert", {}).get("min_price", None)
        max_price_cfg = cfg.get("alert", {}).get("max_price", None)
    except Exception:
        min_price_cfg, max_price_cfg = None, None
    min_price = args.min_price if args.min_price is not None else min_price_cfg
    max_price = args.max_price if args.max_price is not None else max_price_cfg
    if (min_price is not None) or (max_price is not None):
        try:
            pcol = "adj_close" if args.entry_price == "close" else "adj_open"
            if min_price is not None:
                specs = specs[specs[pcol] >= float(min_price)]
            if max_price is not None:
                specs = specs[specs[pcol] <= float(max_price)]
        except Exception:
            pass
    # Drop any duplicate symbols (e.g., if present in features multiple times for the date)
    specs = specs.drop_duplicates(subset="symbol", keep="first").reset_index(drop=True)
    # Strict gating: require non-null metrics and enforce thresholds
    adv_series = (
        specs["adv20"]
        if "adv20" in specs.columns
        else pd.Series(index=specs.index, dtype=float)
    )
    atr_series = (
        specs["atr_pct_14"]
        if "atr_pct_14" in specs.columns
        else pd.Series(index=specs.index, dtype=float)
    )
    liq_mask = adv_series.notna() & (adv_series >= min_adv_usd)
    # Allow configuration to treat missing ATR as pass (useful when latest bar lacks high/low)
    atr_missing_ok = bool(cfg.get("risk", {}).get("atr_missing_ok", True))
    atr_mask = (atr_series <= max_atr_pct) | (
        atr_series.isna() if atr_missing_ok else False
    )
    earn_mask = pd.Series(True, index=specs.index)
    if args.earnings_file:
        try:
            if args.earnings_file.lower().endswith(".parquet"):
                earn = pd.read_parquet(args.earnings_file)
            else:
                earn = pd.read_csv(args.earnings_file)
            earn["date"] = pd.to_datetime(earn["date"])  # expected columns: date,symbol
            earn["symbol"] = earn["symbol"].astype(str).str.upper()
            # Build blackout windows
            blk = earn.copy()
            blk["start"] = blk["date"] - pd.Timedelta(days=earnings_blackout)
            blk["end"] = blk["date"] + pd.Timedelta(days=earnings_blackout)
            # For today's date, mark symbols in blackout
            in_blk = (
                blk[(blk["start"] <= target_date) & (blk["end"] >= target_date)][
                    "symbol"
                ]
                .unique()
                .tolist()
            )
            earn_mask = ~specs["symbol"].isin(in_blk)
        except Exception as e:
            log.warning("failed to apply earnings blackout: %s", e)

    specs["_risk_ok"] = liq_mask & atr_mask & earn_mask

    # Diagnostics: show gate pass counts
    total = len(specs)
    n_liq = int(liq_mask.sum())
    n_atr = int(atr_mask.sum())
    n_earn = int(earn_mask.sum()) if isinstance(earn_mask, pd.Series) else total
    n_ok = int(specs["_risk_ok"].sum())
    log.info(
        "Risk gates: total=%d liquidity_ok=%d atr_ok=%d earnings_ok=%d passed_all=%d",
        total,
        n_liq,
        n_atr,
        n_earn,
        n_ok,
    )
    if args.debug_risk:
        dbg = specs.copy()
        dbg["liq_ok"] = liq_mask
        dbg["atr_ok"] = atr_mask
        dbg["earn_ok"] = earn_mask if isinstance(earn_mask, pd.Series) else True

        def _why(row):
            rs = []
            if not row["liq_ok"]:
                rs.append("liquidity")
            if not row["atr_ok"]:
                rs.append("atr")
            if not row["earn_ok"]:
                rs.append("earnings")
            return ",".join(rs) if rs else "ok"

        dbg["risk_reason"] = dbg.apply(_why, axis=1)
        cols = [
            "symbol",
            "meta_prob",
            "adv20",
            "atr_pct_14",
            "liq_ok",
            "atr_ok",
            "earn_ok",
            "risk_reason",
        ]
        print("[alert] Risk debug (top 10 by meta):")
        print(
            dbg.sort_values("meta_prob", ascending=False)[cols]
            .head(10)
            .to_string(index=False)
        )

    # Intraday mix (optional): blend daily meta_prob with intraday heuristic
    w_mix = float(args.mix_intraday or 0.0)
    if w_mix > 0.0 and args.intraday_features:
        try:
            snap = pd.read_parquet(args.intraday_features)
            snap["symbol"] = snap["symbol"].astype(str).str.upper()

            def _score_row(r: pd.Series) -> float:
                vwap = float(r.get("vwap_dev_20", 0.0))
                brk = float(r.get("breakout_high_20", 0.0))
                volr = float(r.get("vol_rel_20", 1.0))
                atrp = float(r.get("atr_pct_14", 0.0))
                s = 0.0
                s += 2.0 * max(0.0, vwap)
                s += 0.5 * brk
                s += 0.2 * max(0.0, min(volr - 1.0, 1.0))
                s += 0.5 * min(atrp, 0.05)
                return float(s)

            snap_scores = (
                snap.drop_duplicates("symbol")
                .set_index("symbol")
                .apply(_score_row, axis=1)
            )
            k = float(args.mix_k or 2.0)
            intr_prob = 1.0 / (1.0 + np.exp(-k * snap_scores))
            intr_prob = intr_prob.rename("intraday_prob")
            specs = specs.merge(intr_prob.reset_index(), on="symbol", how="left")
            specs["intraday_prob"] = specs["intraday_prob"].fillna(0.5)
            specs["meta_prob_mix"] = (1.0 - w_mix) * specs["meta_prob"].astype(
                float
            ) + w_mix * specs["intraday_prob"].astype(float)
        except Exception as e:
            log.warning("intraday mix failed: %s", e)
            specs["meta_prob_mix"] = specs["meta_prob"].astype(float)
    else:
        specs["meta_prob_mix"] = specs["meta_prob"].astype(float)

    # Rank by mixed meta probability and select top-K after risk gates
    top_k = int(cfg.get("alert", {}).get("top_k", args.top_k))
    for col in ("adv20", "atr_pct_14"):
        if col not in specs.columns:
            specs[col] = np.nan
    base_cols = ["symbol", "meta_prob", "meta_prob_mix", "adv20", "atr_pct_14"]
    if "intraday_prob" in specs.columns:
        base_cols.append("intraday_prob")
    picks = specs.loc[specs["_risk_ok"], base_cols].copy()
    # Optional sector cap diversification
    if args.sector_map_csv and int(args.sector_cap) > 0 and not picks.empty:
        try:
            picks = _sector_cap(
                picks, args.sector_map_csv, int(args.sector_cap), rank_col="meta_prob"
            )
        except Exception as e:
            log.warning("sector cap failed: %s", e)
    # Apply exclude list (cooldown) if provided
    if args.exclude_file and os.path.exists(args.exclude_file):
        try:
            ex = [
                ln.strip().upper()
                for ln in open(args.exclude_file, "r", encoding="utf-8")
                .read()
                .splitlines()
                if ln.strip()
            ]
            if ex:
                picks = picks[~picks["symbol"].isin(ex)]
        except Exception:
            pass
    rank_col = "meta_prob_mix" if "meta_prob_mix" in picks.columns else "meta_prob"
    # Deterministic tiebreakers to reduce jitter and avoid alpha-bias on symbol
    sort_cols = [rank_col]
    asc = [False]
    if "adv20" in picks.columns:
        sort_cols.append("adv20")
        asc.append(False)
    if "atr_pct_14" in picks.columns:
        sort_cols.append("atr_pct_14")
        asc.append(True)  # prefer lower ATR%% when scores tie
    sort_cols.append("symbol")
    asc.append(True)
    sorted_all = picks.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    picks = sorted_all.head(top_k).copy()
    # Exploration: occasionally replace last pick with a random from next-N
    try:
        p_exp = float(args.explore_prob or 0.0)
        pool_n = int(args.explore_topn or 10)
    except Exception:
        p_exp, pool_n = 0.0, 10
    if top_k > 0 and p_exp > 0.0 and len(sorted_all) > top_k:
        import random as _rnd

        if _rnd.random() < p_exp:
            start = top_k
            end = min(len(sorted_all), top_k + max(1, pool_n))
            pool = sorted_all.iloc[start:end]
            if not pool.empty:
                repl = pool.sample(1, random_state=None)
                picks.iloc[-1:] = repl.values

    try:
        today_d = pd.Timestamp.utcnow().normalize().date()
        stale_days = (today_d - target_date.date()).days
    except Exception:
        stale_days = 0
    stale_note = f" (stale {stale_days}d)" if stale_days > 2 else ""
    print(
        f"[alert] Date: {target_date.date()}{stale_note}  Universe: {args.universe_file}"
    )
    print("[alert] Top candidates:")
    print(picks.to_string(index=False))
    if args.print_specialist_probs and not picks.empty:
        spec_cols = [c for c in specs.columns if c.endswith("_prob")]
        if spec_cols:
            det = specs.merge(picks[["symbol"]], on="symbol", how="inner")
            cols = ["symbol"] + sorted(spec_cols)
            print("[alert] Specialist probabilities for picks:")
            try:
                print(det[cols].drop_duplicates("symbol").to_string(index=False))
            except Exception:
                print(
                    det[cols].drop_duplicates("symbol").head(50).to_string(index=False)
                )
    if picks.empty:
        # Help user understand gating by showing top-5 by meta and their metrics
        dbg = specs[["symbol", "meta_prob", "adv20", "atr_pct_14", "_risk_ok"]].copy()
        dbg = dbg.sort_values("meta_prob", ascending=False).head(5)
        print("[alert] No candidates after gates. Top-5 by meta with metrics:")
        print(dbg.to_string(index=False))
        # Optional heartbeat message
        if (
            (not args.dry_run)
            and args.heartbeat_on_empty
            and (args.discord_webhook or cfg.get("alert", {}).get("discord_webhook"))
        ):
            try:
                from ..infra.notify import send_discord

                hook = args.discord_webhook or cfg.get("alert", {}).get(
                    "discord_webhook"
                )
                now_s = (
                    pd.Timestamp.now(tz="UTC")
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M %Z")
                )
                universe_name = os.path.basename(args.universe_file)
                msg = f"Heartbeat: no candidates at {now_s}\nUniverse: {universe_name} • top_k={top_k} • provider={provider}"
                send_discord(hook, msg)
                log.info("heartbeat sent to Discord")
            except Exception as e:
                log.warning("heartbeat failed: %s", e)

    # For each candidate, fetch recent news and compute sentiment
    lookback = int(
        args.from_days or cfg.get("alert", {}).get("news_lookback_days", args.from_days)
    )
    # CLI flag overrides YAML for provider
    provider = str(args.provider or cfg.get("alert", {}).get("news_provider", "none"))
    # Limit news window to recent days if features are stale
    try:
        _today_d = pd.Timestamp.utcnow().normalize().date()
        stale_days = (_today_d - target_date.date()).days
    except Exception:
        _today_d = pd.Timestamp.utcnow().date()
        stale_days = 0
    if stale_days > 2:
        start_dt = (_today_d - pd.Timedelta(days=int(max(1, lookback)))).isoformat()
        end_dt = _today_d.isoformat()
    else:
        start_dt = (
            (pd.Timestamp(target_date) - pd.Timedelta(days=int(max(1, lookback))))
            .date()
            .isoformat()
        )
        end_dt = max(pd.Timestamp(target_date).date().isoformat(), _today_d.isoformat())
    notify_lines = []
    compact_lines = []
    # Collect per-symbol details for diagnostics and downstream consumers
    _ref_price_map: dict[str, float] = {}
    _stop_price_map: dict[str, float] = {}
    _risk_frac_map: dict[str, float] = {}
    _shares_map: dict[str, int] = {}
    _notional_map: dict[str, float] = {}
    _price_note_map: dict[str, str] = {}
    reasons: dict[str, str] = {}
    exp_ret_pct: dict[str, float] = {}
    horizons: dict[str, str] = {}
    news_avg_map: dict[str, float] = {}
    news_count_map: dict[str, int] = {}
    order_msg_map: dict[str, str] = {}
    for _, row in picks.iterrows():
        sym = row["symbol"]
        # Use ASCII arrow for Windows console compatibility
        log.info("News for %s (%s -> %s)", sym, start_dt, end_dt)
        if provider.lower() == "none":
            avg, cnt, details = 0.0, 0, []
        else:
            try:
                items = fetch_news(sym, start=start_dt, end=end_dt, provider=provider)
            except Exception as e:
                log.warning("news fetch failed for %s: %s", sym, e)
                avg, cnt, details = 0.0, 0, []
            else:
                avg, cnt, details = score_news(items)
        log.info("sentiment=%+.3f from %d articles", avg, cnt)
        for it, s in details[:5]:
            print(
                f"  - {s:+.2f} {it.date.date()} {it.source}: {it.headline[:120]}{'...' if len(it.headline)>120 else ''}"
            )
        if details:
            print(f"  More: {details[0][0].url}")
        # Build sizing suggestion
        # Prefer unadjusted close/open if available; fall back to adjusted
        base_col = "close" if args.entry_price == "close" else "open"
        adj_col = "adj_close" if args.entry_price == "close" else "adj_open"
        prc_series = day.loc[day["symbol"] == sym]
        if (
            not prc_series.empty
            and base_col in prc_series.columns
            and pd.notna(prc_series.iloc[0][base_col])
        ):
            ref_price = float(prc_series.iloc[0][base_col])
            price_note = f"feature {base_col}"
        else:
            prc_val = prc_series.get(adj_col, pd.Series([], dtype=float))
            ref_price = float(prc_val.iloc[0]) if not prc_val.empty else float("nan")
            price_note = f"feature {adj_col}"

        # Optional live price override (and auto-fallback if features are stale)
        # CLI overrides YAML: if flag not set, fall back to YAML, then default
        price_source_cfg = str(
            (args.price_source or cfg.get("alert", {}).get("price_source") or "feature")
        ).lower()
        # Auto-detect stale features and prefer live price
        try:
            days_stale = (
                pd.Timestamp.utcnow().normalize()
                - pd.Timestamp(target_date).normalize()
            ).days
        except Exception:
            days_stale = 0
        price_source = price_source_cfg
        if price_source == "feature" and days_stale > 2:
            price_source = "live"
        if price_source == "live":
            # CLI overrides YAML for live provider (order of attempts ignores this; note is for preference only)
            live_provider = str(
                (
                    args.live_provider
                    or cfg.get("alert", {}).get("live_provider")
                    or "yahoo"
                )
            ).lower()
            # Accept friendly --plan then explicit polygon plan hints from args/cfg
            friendly = normalize_plan(args.plan)
            polygon_plan = str(
                (
                    friendly
                    or args.polygon_plan
                    or cfg.get("alert", {}).get("polygon_plan")
                    or (cfg.get("provider", {}) or {}).get("polygon_plan")
                    or "auto"
                )
            ).lower()
            allow_polygon_live = polygon_plan not in ("starter",)
            try:
                got_live = False
                # Prefer Polygon first: last trade
                from ..infra.http import HttpClient, HttpConfig

                api_key = os.environ.get("POLYGON_API_KEY", "")
                client = HttpClient(HttpConfig(requests_per_second=5.0, timeout=10.0))
                if api_key and allow_polygon_live:
                    try:
                        url = f"https://api.polygon.io/v2/last/trade/{sym}"
                        data = client.get_json(url, params={"apiKey": api_key}) or {}
                        p = (data.get("results", {}) or {}).get("p")
                        if p:
                            ref_price = float(p)
                            got_live = True
                            price_note = "polygon (live trade)"
                    except Exception:
                        # NOT_AUTHORIZED or network error; continue to fallbacks
                        pass
                # Next: Polygon snapshot (single-ticker) — offers lastTrade/lastQuote/day
                if (not got_live) and api_key and allow_polygon_live:
                    try:
                        snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{sym}"
                        s = client.get_json(snap_url, params={"apiKey": api_key}) or {}
                        tkr = (s.get("ticker") or {}) if isinstance(s, dict) else {}
                        # Prefer lastTrade price, then lastQuote (mid/ask), then day close if present
                        lt = (
                            (tkr.get("lastTrade") or {})
                            if isinstance(tkr, dict)
                            else {}
                        )
                        lq = (
                            (tkr.get("lastQuote") or {})
                            if isinstance(tkr, dict)
                            else {}
                        )
                        snap_day = (
                            (tkr.get("day") or {}) if isinstance(tkr, dict) else {}
                        )
                        cand = None
                        # Common fields: lastTrade.p (price)
                        if isinstance(lt, dict) and lt.get("p"):
                            cand = float(lt.get("p"))
                            price_note = "polygon (snapshot trade)"
                        # Some payloads use upper-case keys; be defensive
                        if cand is None and isinstance(lt, dict) and lt.get("P"):
                            cand = float(lt.get("P"))
                            price_note = "polygon (snapshot trade)"
                        # Fallback to lastQuote price if available
                        if (
                            cand is None
                            and isinstance(lq, dict)
                            and (lq.get("p") or lq.get("P"))
                        ):
                            cand = float(lq.get("p") or lq.get("P"))
                            price_note = "polygon (snapshot quote)"
                        # As a last resort from snapshot, use today's aggregated close if present
                        if (
                            cand is None
                            and isinstance(snap_day, dict)
                            and snap_day.get("c")
                        ):
                            cand = float(snap_day.get("c"))
                            price_note = "polygon (snapshot day c)"
                        if cand is not None:
                            ref_price = float(cand)
                            got_live = True
                    except Exception:
                        pass
                # Yahoo fallback regardless of live_provider
                if not got_live:
                    try:
                        import yfinance as yf  # type: ignore

                        tk = yf.Ticker(sym)
                        info = getattr(tk, "fast_info", None)
                        if info and getattr(info, "last_price", None) is not None:
                            ref_price = float(info.last_price)
                            got_live = True
                            price_note = "yahoo (live)"
                        else:
                            hist = tk.history(period="1d")
                            if not hist.empty and float(hist["Close"].iloc[-1]) > 0:
                                ref_price = float(hist["Close"].iloc[-1])
                                got_live = True
                                price_note = "yahoo (live)"
                    except Exception:
                        pass
                # Fallback: Yahoo public quote endpoint
                if not got_live:
                    try:
                        import requests as _rq

                        q = _rq.get(
                            "https://query1.finance.yahoo.com/v7/finance/quote",
                            params={"symbols": sym},
                            timeout=6,
                        )
                        js = q.json() if q.ok else {}
                        rmp = (
                            ((js or {}).get("quoteResponse", {}) or {}).get(
                                "result", []
                            )
                            or [{}]
                        )[0].get("regularMarketPrice")
                        if rmp:
                            ref_price = float(rmp)
                            got_live = True
                            price_note = "yahoo (quote)"
                    except Exception:
                        pass
                # Last resort: Polygon previous close (adjusted)
                if api_key and not got_live:
                    try:
                        prev_url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev"
                        data2 = (
                            client.get_json(
                                prev_url, params={"adjusted": "true", "apiKey": api_key}
                            )
                            or {}
                        )
                        res = data2.get("results") or []
                        if res:
                            c = res[0].get("c")
                            if c:
                                ref_price = float(c)
                                got_live = True
                                price_note = "polygon (prev close)"
                    except Exception:
                        pass
            except Exception:
                pass
            # Annotate if this was an auto-fallback due to stale features
            if price_source_cfg != "live" and days_stale > 2:
                price_note += f" • stale features {pd.Timestamp(target_date).date()}"
        atr_pct = float(row.get("atr_pct_14", float("nan")))
        stop_mult = float(cfg.get("risk", {}).get("stop_atr_mult", args.stop_atr_mult))
        # Determine risk fraction (fixed or auto based on conviction)
        risk_mode = str(cfg.get("risk", {}).get("risk_mode", args.risk_mode)).lower()
        if risk_mode == "auto":
            rmin = cfg.get("risk", {}).get("min_risk_pct", args.risk_min_pct)
            rmax = cfg.get("risk", {}).get("max_risk_pct", args.risk_max_pct)
            # sensible defaults if not provided
            rmin = float(rmin if rmin is not None else 0.002)  # 0.2%
            rmax = float(rmax if rmax is not None else 0.006)  # 0.6%
            conv = float(row.get("meta_prob", 0.5))
            base_p = cfg.get("risk", {}).get("base_prob", args.risk_base_prob)
            base_p = float(base_p if base_p is not None else 0.5)
            denom = max(1e-6, 1.0 - base_p)
            conv_n = max(
                0.0, min(1.0, (conv - base_p) / denom)
            )  # 0 at base_p, 1 at 1.0
            curve = str(cfg.get("risk", {}).get("risk_curve", args.risk_curve))
            if curve == "quadratic":
                conv_n = conv_n * conv_n
            elif curve == "sqrt":
                conv_n = conv_n**0.5
            risk_frac = rmin + (rmax - rmin) * conv_n
        else:
            risk_frac = float(cfg.get("risk", {}).get("risk_pct", args.risk_pct))
        acct = float(cfg.get("risk", {}).get("account_equity", args.account_equity))
        if (
            np.isfinite(ref_price)
            and ref_price > 0
            and np.isfinite(atr_pct)
            and atr_pct > 0
        ):
            stop_price = ref_price * (1.0 - stop_mult * atr_pct)
            per_share_risk = max(1e-6, ref_price - stop_price)
            shares = int(np.floor((acct * risk_frac) / per_share_risk))
            # Optional notional caps: max position notional or max weight of equity
            try:
                notional_cap = float(
                    (cfg.get("risk", {}).get("max_position_notional") if cfg else None)
                    or (
                        args.max_position_notional
                        if args.max_position_notional is not None
                        else 0.0
                    )
                    or 0.0
                )
            except Exception:
                notional_cap = 0.0
            try:
                w_cap = float(
                    (cfg.get("risk", {}).get("max_name_weight") if cfg else None)
                    or (
                        args.max_name_weight
                        if args.max_name_weight is not None
                        else (cfg.get("sizing", {}).get("w_max") if cfg else 0.0)
                    )
                    or 0.0
                )
            except Exception:
                w_cap = 0.0
            cap_notional = 0.0
            if notional_cap and notional_cap > 0:
                cap_notional = max(cap_notional, float(notional_cap))
            if w_cap and w_cap > 0 and acct > 0:
                cap_notional = max(cap_notional, float(w_cap) * acct)
            if cap_notional and cap_notional > 0 and ref_price > 0:
                shares_cap = int(np.floor(cap_notional / ref_price))
                shares = min(shares, shares_cap)
            notional = shares * ref_price
            weight = notional / acct if acct > 0 else 0.0
            # Persist details for diagnostics and compact formatting
            _ref_price_map[sym] = float(ref_price)
            _stop_price_map[sym] = float(stop_price)
            _risk_frac_map[sym] = float(risk_frac)
            _shares_map[sym] = int(max(0, shares))
            _notional_map[sym] = float(notional)
            _price_note_map[sym] = str(price_note)
            order_msg = (
                (
                    f"Adding {shares} shares of ${sym} @ ${ref_price:.2f}\n"
                    f"Stop loss @ ${stop_price:.2f}\n"
                    f"Risking {100*risk_frac:.1f}% • Notional ${notional:,.0f} ({100*weight:.1f}% of equity)\n"
                    f"Price source: {price_note}"
                )
                if shares > 0
                else (
                    f"Adding 0 shares of ${sym} (risk too small)\n"
                    f"Ref ${ref_price:.2f} Stop @{stop_mult:.1f}×ATR -> ${stop_price:.2f}\n"
                    f"Price source: {price_note}"
                )
            )
        else:
            order_msg = f"${sym}: price/ATR unavailable for sizing"
        # Reasoning
        top_specs_txt = consensus_for_symbol(specs, sym)
        rsec = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
        base_p = rsec.get("base_prob", args.risk_base_prob)
        k_scale = rsec.get("expected_k", None)
        cut1 = rsec.get("horizon_cut1", 0.55)
        cut2 = rsec.get("horizon_cut2", 0.60)
        cap_mult = rsec.get("expected_cap_mult", 2.0)
        exp_pct, horizon = expected_return_and_horizon(
            float(row.get("meta_prob", 0.5)),
            atr_pct,
            base_p,
            k_scale=k_scale,
            cap_mult=float(cap_mult),
            cut1=float(cut1),
            cut2=float(cut2),
        )
        mix_note = ""
        if w_mix > 0.0 and "intraday_prob" in picks.columns:
            try:
                mix_note = f" • intraday={float(row.get('intraday_prob', np.nan)):.3f} w={w_mix:.2f}"
            except Exception:
                mix_note = f" • intraday w={w_mix:.2f}"
        reasons[sym] = (
            f"Consensus: meta={float(row.get('meta_prob', 0.5)):.3f} via [{top_specs_txt}] • "
            f"news={avg:+.2f} ({cnt} arts){mix_note} • gates ok (adv ${row.get('adv20', float('nan')):,.0f}, atr {100*atr_pct:.2f}%)"
        )
        exp_ret_pct[sym] = float(exp_pct)
        horizons[sym] = str(horizon)

        mp = float(row.get("meta_prob", 0.5))
        mpm = float(row.get("meta_prob_mix", mp))
        summary = f"{sym}: meta_prob={mp:.3f}{(' mix->'+format(mpm,'.3f')) if w_mix>0 else ''} adv20=${row.get('adv20', float('nan')):,.0f} atr%={100*row.get('atr_pct_14', float('nan')):.2f} news={avg:+.2f} ({cnt} arts)"
        notify_lines.append(summary + "\n" + order_msg)
        news_avg_map[sym] = float(avg)
        news_count_map[sym] = int(cnt)
        order_msg_map[sym] = order_msg
        # Compact one-liner for Discord
        # Price delta vs last alert (if diagnostics CSV provided)
        delta_txt = ""
        if args.alert_log_csv and os.path.exists(args.alert_log_csv):
            try:
                _df_prev = pd.read_csv(args.alert_log_csv)
                _df_prev = _df_prev[_df_prev["symbol"].astype(str).str.upper() == sym]
                if not _df_prev.empty and "ref_price" in _df_prev.columns:
                    prev_p = float(_df_prev.iloc[-1]["ref_price"])
                    if (
                        np.isfinite(prev_p)
                        and prev_p > 0
                        and np.isfinite(ref_price)
                        and ref_price > 0
                    ):
                        d = (ref_price / prev_p) - 1.0
                        delta_txt = f" • Δ{100*d:+.2f}%"
            except Exception:
                delta_txt = ""
        compact = (
            (
                f"${sym} • p={mpm:.3f}"
                f" • **Entry ${ref_price:.2f}**"
                f" • **Stop ${stop_price:.2f}**"
                f" • Risk {100*risk_frac:.1f}% • Sz {shares} (${notional:,.0f})"
                f" • {price_note}{delta_txt}"
            )
            if np.isfinite(ref_price)
            and "stop_price" in locals()
            and np.isfinite(stop_price)
            else (f"${sym} • p={mpm:.3f} • pricing unavailable")
        )
        compact_lines.append(compact)

    # Optional Discord notification
    msg = None
    alert_kind = str(args.alert_kind or "intraday")
    alert_category = str(args.alert_category or "watchlist")
    if not picks.empty:
        picks = picks.copy()
        picks["trade_date"] = target_date.date()
        picks["reason"] = picks["symbol"].map(lambda s: reasons.get(str(s), ""))
        picks["expected_ret_pct"] = picks["symbol"].map(
            lambda s: exp_ret_pct.get(str(s), float("nan"))
        )
        picks["horizon"] = picks["symbol"].map(lambda s: horizons.get(str(s), ""))
        picks["news_score"] = picks["symbol"].map(
            lambda s: news_avg_map.get(str(s), float("nan"))
        )
        picks["news_count"] = picks["symbol"].map(
            lambda s: news_count_map.get(str(s), 0)
        )
        picks["order_note"] = picks["symbol"].map(
            lambda s: order_msg_map.get(str(s), "")
        )
        for col_name, mapping in (
            ("ref_price", _ref_price_map),
            ("stop_price", _stop_price_map),
            ("risk_frac", _risk_frac_map),
            ("shares", _shares_map),
            ("notional", _notional_map),
        ):
            picks[col_name] = picks["symbol"].map(
                lambda s: mapping.get(str(s), float("nan"))
            )
        picks["price_note"] = picks["symbol"].map(
            lambda s: _price_note_map.get(str(s), "")
        )
        picks["alert_kind"] = alert_kind
        picks["alert_category"] = alert_category

        rec_path = str(args.recommendations_csv or "")
        dedupe_today = bool(args.dedupe_per_day and rec_path)
        already_sent: set[str] = set()
        if dedupe_today and os.path.exists(rec_path):
            try:
                rec_df = pd.read_csv(rec_path)
                if not rec_df.empty and "alert_date" in rec_df.columns:
                    dates = pd.to_datetime(
                        rec_df["alert_date"], errors="coerce"
                    ).dt.date
                    today = pd.Timestamp.utcnow().date()
                    already_sent = set(
                        rec_df.loc[dates == today, "symbol"].astype(str).str.upper()
                    )
            except Exception as exc:
                log.warning("recommendations dedupe load failed: %s", exc)
        if already_sent:
            kept_rows = []
            new_notify: list[str] = []
            new_compact: list[str] = []
            skipped: list[str] = []
            for idx, row in picks.iterrows():
                sym_up = str(row["symbol"]).upper()
                if sym_up in already_sent:
                    skipped.append(sym_up)
                    continue
                kept_rows.append(idx)
                if idx < len(notify_lines):
                    new_notify.append(notify_lines[idx])
                if idx < len(compact_lines):
                    new_compact.append(compact_lines[idx])
            if skipped:
                log.info(
                    "skipping already-alerted symbols: %s",
                    ", ".join(sorted(set(skipped))),
                )
            picks = picks.loc[kept_rows].reset_index(drop=True)
            notify_lines = new_notify
            compact_lines = new_compact
        if rec_path and not picks.empty:
            try:
                now_ts = pd.Timestamp.utcnow().astimezone()
                rec_cols = [
                    "alert_ts",
                    "alert_date",
                    "trade_date",
                    "alert_kind",
                    "alert_category",
                    "symbol",
                    "meta_prob",
                    "meta_prob_mix",
                    "adv20",
                    "atr_pct_14",
                    "ref_price",
                    "stop_price",
                    "reason",
                    "order_note",
                    "news_score",
                    "news_count",
                    "expected_ret_pct",
                    "horizon",
                    "risk_frac",
                    "shares",
                    "notional",
                    "price_note",
                ]
                rec_df = picks.copy()
                rec_df.insert(0, "alert_ts", now_ts.isoformat())
                rec_df.insert(1, "alert_date", now_ts.date())
                rec_payload = rec_df[
                    [c for c in rec_cols if c in rec_df.columns]
                ].copy()
                if "alert_date" in rec_payload.columns:
                    rec_payload["alert_date"] = rec_payload["alert_date"].astype(str)
                if "trade_date" in rec_payload.columns:
                    rec_payload["trade_date"] = rec_payload["trade_date"].astype(str)
                if "symbol" in rec_payload.columns:
                    rec_payload["symbol"] = (
                        rec_payload["symbol"].astype(str).str.upper()
                    )
                # Normalize any datetime columns to naive ISO strings to avoid tz_convert issues.
                for col in rec_payload.columns:
                    series = rec_payload[col]
                    if is_datetime64tz_dtype(series):
                        rec_payload[col] = series.dt.tz_convert("UTC").dt.tz_localize(
                            None
                        )
                        series = rec_payload[col]
                    if is_datetime64_dtype(series):
                        rec_payload[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
                os.makedirs(os.path.dirname(rec_path), exist_ok=True)
                header = not os.path.exists(rec_path)
                rec_payload.to_csv(rec_path, mode="a", index=False, header=header)
            except Exception as exc:
                log.warning("failed to append recommendations: %s", exc)

    # Duplicate throttle: if the exact same symbol set was alerted very recently, skip sending
    try:
        repeat_mins = int(args.min_repeat_mins or 0)
    except Exception:
        repeat_mins = 0
    if repeat_mins > 0 and args.recommendations_csv and not picks.empty:
        try:
            rec_path = str(args.recommendations_csv)
            if os.path.exists(rec_path):
                rec = pd.read_csv(rec_path)
                if (
                    not rec.empty
                    and "alert_ts" in rec.columns
                    and "symbol" in rec.columns
                ):
                    # Parse timestamps and filter to recent window
                    rts = pd.to_datetime(rec["alert_ts"], errors="coerce")
                    now_ = pd.Timestamp.utcnow().tz_localize(None)
                    recent = rec[
                        (now_ - rts.dt.tz_localize(None))
                        <= pd.Timedelta(minutes=repeat_mins)
                    ]
                    # Compare against the most recent alert event (same category) in window
                    if not recent.empty:
                        recent = recent.copy()
                        recent_syms = set(
                            recent[recent["alert_category"] == args.alert_category][
                                "symbol"
                            ]
                            .astype(str)
                            .str.upper()
                        )
                        cur_syms = set(picks["symbol"].astype(str).str.upper())
                        if (
                            cur_syms
                            and cur_syms == recent_syms
                            and len(cur_syms) == int(len(recent_syms))
                        ):
                            log.info(
                                "duplicate alert skipped (same symbols within %d minutes)",
                                repeat_mins,
                            )
                            # Still append diagnostics for observability
                            if args.alert_log_csv:
                                try:
                                    import os as _os
                                    from datetime import datetime as _dt

                                    ts2 = _dt.now().astimezone()
                                    diag2 = picks.copy()
                                    diag2.insert(0, "time", ts2.strftime("%H:%M"))
                                    diag2.insert(0, "date", target_date.date())
                                    _os.makedirs(
                                        _os.path.dirname(args.alert_log_csv),
                                        exist_ok=True,
                                    )
                                    if _os.path.exists(args.alert_log_csv):
                                        diag2.to_csv(
                                            args.alert_log_csv,
                                            mode="a",
                                            header=False,
                                            index=False,
                                        )
                                    else:
                                        diag2.to_csv(args.alert_log_csv, index=False)
                                except Exception:
                                    pass
                            return picks
        except Exception:
            pass

    msg = ""
    header = (
        "Pre-market Watchlist Alert"
        if alert_kind == "pre-market"
        else "Intraday Event Alert"
    )
    if alert_category == "explore":
        header = f"{header} • Explore"
    if notify_lines:
        msg = header + "\n" + "\n\n".join(notify_lines)
    elif compact_lines:
        msg = header + "\n" + "\n".join(compact_lines)
    if args.dry_run and msg:
        log.info("DRY-RUN MESSAGE:\n%s", msg)
        # Append diagnostics log if requested
        if args.alert_log_csv:
            try:
                import os as _os
                from datetime import datetime as _dt

                ts = _dt.now().astimezone()
                # Keep a compact diagnostics table: date,time,symbol,meta_prob,adv20,atr_pct_14
                diag = picks.copy()
                diag.insert(0, "time", ts.strftime("%H:%M"))
                diag.insert(0, "date", target_date.date())
                # attach reasoning fields if available
                if reasons:
                    diag["reason"] = diag["symbol"].map(
                        lambda s: reasons.get(str(s), "")
                    )
                if exp_ret_pct:
                    diag["expected_ret_pct"] = diag["symbol"].map(
                        lambda s: exp_ret_pct.get(str(s), float("nan"))
                    )
                if horizons:
                    diag["horizon"] = diag["symbol"].map(
                        lambda s: horizons.get(str(s), "")
                    )
                _os.makedirs(_os.path.dirname(args.alert_log_csv), exist_ok=True)
                if _os.path.exists(args.alert_log_csv):
                    diag.to_csv(args.alert_log_csv, mode="a", header=False, index=False)
                else:
                    diag.to_csv(args.alert_log_csv, index=False)
                log.info(
                    "appended alert diagnostics -> %s rows=%d",
                    args.alert_log_csv,
                    len(diag),
                )
            except Exception as e:
                log.warning("alert diagnostics failed: %s", e)
        return picks
    # Send Discord/Slack if configured
    if msg and (args.discord_webhook or cfg.get("alert", {}).get("discord_webhook")):
        try:
            from ..infra.notify import send_discord

            hook = args.discord_webhook or cfg.get("alert", {}).get("discord_webhook")
            # Mask webhook for logs (show last 6 chars)
            tail = hook[-6:] if isinstance(hook, str) else ""
            log.info("sending to Discord webhook ...%s", tail)
            send_discord(hook, f"@everyone\n{msg}")
            log.info("Discord notification sent.")
        except Exception as e:
            log.warning("Discord notification failed: %s", e)
    # Append diagnostics after send as well
    if args.alert_log_csv:
        try:
            import os as _os
            from datetime import datetime as _dt

            ts = _dt.now().astimezone()
            diag = picks.copy()
            # Attach pricing/sizing diagnostics for delta computation on next runs
            for col_name, m in (
                ("ref_price", _ref_price_map),
                ("stop_price", _stop_price_map),
                ("risk_frac", _risk_frac_map),
                ("shares", _shares_map),
                ("notional", _notional_map),
                ("price_note", _price_note_map),
            ):
                try:
                    if col_name not in diag.columns:
                        diag[col_name] = diag["symbol"].map(
                            lambda s: m.get(str(s), float("nan"))
                        )
                except Exception:
                    pass
            diag.insert(0, "time", ts.strftime("%H:%M"))
            diag.insert(0, "date", target_date.date())
            if reasons:
                diag["reason"] = diag["symbol"].map(lambda s: reasons.get(str(s), ""))
            if exp_ret_pct:
                diag["expected_ret_pct"] = diag["symbol"].map(
                    lambda s: exp_ret_pct.get(str(s), float("nan"))
                )
            if horizons:
                diag["horizon"] = diag["symbol"].map(lambda s: horizons.get(str(s), ""))
            _os.makedirs(_os.path.dirname(args.alert_log_csv), exist_ok=True)
            if _os.path.exists(args.alert_log_csv):
                diag.to_csv(args.alert_log_csv, mode="a", header=False, index=False)
            else:
                diag.to_csv(args.alert_log_csv, index=False)
            log.info(
                "appended alert diagnostics -> %s rows=%d",
                args.alert_log_csv,
                len(diag),
            )
        except Exception as e:
            log.warning("alert diagnostics failed: %s", e)
    if msg and (args.slack_webhook or cfg.get("alert", {}).get("slack_webhook")):
        try:
            from ..infra.notify import send_slack

            hook = args.slack_webhook or cfg.get("alert", {}).get("slack_webhook")
            tail = hook[-6:] if isinstance(hook, str) else ""
            log.info("sending to Slack webhook ...%s", tail)
            send_slack(hook, msg)
            log.info("Slack notification sent.")
        except Exception as e:
            log.warning("Slack notification failed: %s", e)

    return picks


if __name__ == "__main__":
    main()
