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
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..news.providers import fetch_news
from ..news.sentiment import score_news
from ..infra.yaml_config import load_yaml_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade alert with top setup + news sentiment")
    p.add_argument("--config", type=str, default="", help="YAML config with defaults/thresholds")
    p.add_argument("--features", required=False, default="", help="Features parquet with date,symbol,baseline features")
    p.add_argument("--model-pkl", required=False, default="", help="Trained meta model pickle (from train_meta.py)")
    p.add_argument("--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)")
    p.add_argument("--universe-file", type=str, default="engine/data/universe/nasdaq100.example.txt")
    p.add_argument("--news-sentiment", type=str, default="", help="Optional precomputed sentiment file")
    p.add_argument("--calibrators-pkl", type=str, default="", help="Pickle with per-specialist calibrators (from run_cv)")
    p.add_argument("--provider", type=str, default="finnhub", help="News provider key")
    p.add_argument("--from-days", type=int, default=3, help="News lookback window in days")
    p.add_argument("--top-k", type=int, default=1, help="How many top candidates to fetch news for")
    # Risk gates
    p.add_argument("--min-adv-usd", type=float, default=1e7, help="Minimum 20D average dollar volume")
    p.add_argument("--max-atr-pct", type=float, default=0.05, help="Maximum ATR(14)/price to limit volatility")
    p.add_argument("--earnings-file", type=str, default="", help="Optional earnings calendar CSV/Parquet (date,symbol)")
    p.add_argument("--earnings-blackout", type=int, default=2, help="Do not alert within +/- N days of earnings")
    # Notifications
    p.add_argument("--slack-webhook", type=str, default=os.environ.get("SLACK_WEBHOOK_URL", ""))
    p.add_argument("--discord-webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", ""))
    p.add_argument("--date", type=str, default="", help="Target date YYYY-MM-DD (default latest)")
    return p.parse_args(argv)


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f.readlines() if ln.strip() and not ln.startswith("#")]


def main(argv: Optional[List[str]] = None) -> None:
    from .predict_daily import parse_args as pred_parse, main as pred_main  # reuse calibrators + meta
    from .predict_daily import _fit_calibrators_from_oof, _apply_calibrator, _naive_prob_map
    import pickle
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    args = parse_args(argv)

    # Load config and features
    cfg = {}
    if args.config:
        try:
            cfg = load_yaml_config(args.config)
        except Exception as e:
            print(f"[alert] warning: failed to load YAML config: {e}")
    features_path = args.features or cfg.get("paths", {}).get("features", "")
    model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model", "")
    if not features_path or not model_path:
        raise RuntimeError("Provide --features and --model-pkl or set them in YAML under paths.features/meta_model")
    f = pd.read_parquet(features_path)
    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()
    target_date = pd.Timestamp(args.date) if args.date else f["date"].max()
    uni_file = args.universe_file or cfg.get("alert", {}).get("universe_file", "") or "engine/data/universe/nasdaq100.example.txt"
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
    specs = compute_specialist_scores(day, news_sentiment=sentiment_file)
    # Load OOF for calibrators if provided
    calibrators = {}
    calib_pkl = args.calibrators_pkl or cfg.get("calibration", {}).get("calibrators_pkl", "")
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
    for col in feature_names:
        if col not in specs.columns:
            specs[col] = 0.5
    X = specs[feature_names].values
    meta_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)
    specs["meta_prob"] = meta_prob

    # Risk gates: liquidity (ADV), volatility (ATR pct), earnings blackout
    # Compute 20D ADV on the panel and join today rows
    # Risk thresholds from config or flags
    min_adv_usd = float(cfg.get("risk", {}).get("min_adv_usd", args.min_adv_usd))
    max_atr_pct = float(cfg.get("risk", {}).get("max_atr_pct", args.max_atr_pct))
    earnings_blackout = int(cfg.get("risk", {}).get("earnings_blackout", args.earnings_blackout))

    f_sorted = f.sort_values(["symbol", "date"]).copy()
    f_sorted["dollar_vol"] = f_sorted["adj_close"] * f_sorted["adj_volume"]
    f_sorted["adv20"] = f_sorted.groupby("symbol")["dollar_vol"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    day_adv = f_sorted[f_sorted["date"] == target_date][["symbol", "adv20", "atr_pct_14"]]
    specs = specs.merge(day_adv, on="symbol", how="left")
    liq_mask = (specs["adv20"].fillna(0.0) >= min_adv_usd)
    atr_mask = (specs.get("atr_pct_14", pd.Series(0.0, index=specs.index)).fillna(0.0) <= max_atr_pct)
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
            in_blk = blk[(blk["start"] <= target_date) & (blk["end"] >= target_date)]["symbol"].unique().tolist()
            earn_mask = ~specs["symbol"].isin(in_blk)
        except Exception as e:
            print(f"[alert] warning: failed to apply earnings blackout: {e}")

    specs["_risk_ok"] = (liq_mask & atr_mask & earn_mask)

    # Rank by meta probability and select top-K after risk gates
    top_k = int(cfg.get("alert", {}).get("top_k", args.top_k))
    picks = specs.loc[specs["_risk_ok"], ["symbol", "meta_prob", "adv20", "atr_pct_14"]].copy()
    picks = picks.sort_values("meta_prob", ascending=False).head(top_k).reset_index(drop=True)

    print(f"[alert] Date: {target_date.date()}  Universe: {args.universe_file}")
    print("[alert] Top candidates:")
    print(picks.to_string(index=False))

    # For each candidate, fetch recent news and compute sentiment
    lookback = int(cfg.get("alert", {}).get("news_lookback_days", args.from_days))
    provider = str(cfg.get("alert", {}).get("news_provider", args.provider))
    start_dt = (pd.Timestamp(target_date) - pd.Timedelta(days=int(max(1, lookback)))).date().isoformat()
    end_dt = pd.Timestamp(target_date).date().isoformat()
    notify_lines = []
    for _, row in picks.iterrows():
        sym = row["symbol"]
        print(f"\n[alert] News for {sym} ({start_dt} → {end_dt})")
        try:
            items = fetch_news(sym, start=start_dt, end=end_dt, provider=provider)
        except Exception as e:
            print(f"[alert] news fetch failed for {sym}: {e}")
            continue
        avg, cnt, details = score_news(items)
        print(f"[alert] sentiment={avg:.3f} from {cnt} articles")
        for it, s in details[:5]:
            print(f"  - {s:+.2f} {it.date.date()} {it.source}: {it.headline[:120]}{'...' if len(it.headline)>120 else ''}")
        if details:
            print(f"  More: {details[0][0].url}")
        notify_lines.append(f"{sym}: meta_prob={row['meta_prob']:.3f} adv20=${row.get('adv20', float('nan')):,.0f} atr%={100*row.get('atr_pct_14', float('nan')):.2f} news={avg:+.2f} ({cnt} arts)")

    # Optional Slack/Discord notification
    msg = None
    if notify_lines:
        msg = "Trade Alert\n" + "\n".join(notify_lines)
    # Prefer Discord per user; Slack optional
    if msg and (args.discord_webhook or cfg.get("alert", {}).get("discord_webhook")):
        try:
            from ..infra.notify import send_discord
            send_discord(args.discord_webhook or cfg.get("alert", {}).get("discord_webhook"), msg)
            print("[alert] notification sent.")
        except Exception as e:
            print(f"[alert] notification failed: {e}")


if __name__ == "__main__":
    main()
