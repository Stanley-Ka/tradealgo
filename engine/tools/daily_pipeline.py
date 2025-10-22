"""Daily pipeline: update FinBERT sentiment -> predict picks -> send trade alert.

This reads a YAML config for defaults and allows CLI overrides. Typical use is to
schedule this script to run once per day before the market opens.

Example YAML (minimal):
  paths:
    features: data/datasets/features_daily_1D.parquet
    meta_model: data/models/meta_lr.pkl
    calibrators_pkl: data/models/spec_calibrators.pkl
    sentiment_out: data/datasets/sentiment_finbert.parquet
    picks_csv: data/signals/picks.csv
  finbert:
    universe_file: engine/data/universe/nasdaq100.example.txt
    days: 3
    model: yiyanghkust/finbert-tone
    provider: finnhub
  predict:
    top_k: 20
  alert:
    universe_file: engine/data/universe/nasdaq100.example.txt
    top_k: 5
    discord_webhook: ${DISCORD_WEBHOOK_URL}

Usage:
  python -m engine.tools.daily_pipeline --config engine/config.example.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from ..infra.yaml_config import load_yaml_config
from .update_daily_sentiment import main as sentiment_update_main
from .predict_daily import main as predict_daily_main
from .trade_alert import main as trade_alert_main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily sentiment->predict->alert pipeline")
    p.add_argument("--config", required=True, help="YAML with paths and step options")
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="Target date YYYY-MM-DD for prediction/alert (default latest)",
    )
    # Skips
    p.add_argument("--skip-sentiment", action="store_true")
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--skip-alert", action="store_true")
    return p.parse_args(argv)


def _cfg_get(cfg: dict, path: List[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    # 1) Update sentiment (FinBERT)
    if not args.skip_sentiment:
        print("[daily] Step 1/3: Updating FinBERT sentiment…")
        uni = (
            _cfg_get(cfg, ["finbert", "universe_file"])
            or _cfg_get(cfg, ["paths", "alert_universe"])
            or _cfg_get(cfg, ["paths", "universe_file"])
            or ""
        )
        sent_out = (
            _cfg_get(cfg, ["paths", "sentiment_out"])
            or "data/datasets/sentiment_finbert.parquet"
        )
        days = str(_cfg_get(cfg, ["finbert", "days"], 3))
        model = _cfg_get(cfg, ["finbert", "model"], "yiyanghkust/finbert-tone")
        provider = _cfg_get(cfg, ["finbert", "provider"], "finnhub")
        # Pick provider-specific env var for token if available
        if str(provider).lower() == "polygon":
            token = os.environ.get("POLYGON_API_KEY", "")
        else:
            token = os.environ.get("FINNHUB_API_KEY", "")
        skip_rl = bool(_cfg_get(cfg, ["finbert", "skip_on_rate_limit"], False))
        if not uni:
            raise RuntimeError(
                "finbert.universe_file (or paths.alert_universe) missing in YAML"
            )
        try:
            sentiment_update_main(
                [
                    "--universe-file",
                    uni,
                    "--days",
                    str(days),
                    "--provider",
                    provider,
                    *(["--token", token] if token else []),
                    "--model",
                    model,
                    "--out",
                    sent_out,
                    *(["--skip-on-rate-limit"] if skip_rl else []),
                ]
            )
        except Exception as e:
            if any(x in str(e) for x in ("429", "Too Many Requests")):
                print("[daily] Step 1/3: Rate limited; skipping sentiment update.")
            else:
                raise
    else:
        print("[daily] Step 1/3: Skipped sentiment update.")

    # 2) Predict daily picks
    if not args.skip_predict:
        print("[daily] Step 2/3: Predicting daily picks…")
        features = _cfg_get(cfg, ["paths", "features"]) or ""
        model_pkl = _cfg_get(cfg, ["paths", "meta_model"]) or ""
        picks_csv = _cfg_get(cfg, ["paths", "picks_csv"], "data/signals/picks.csv")
        sent_out = _cfg_get(cfg, ["paths", "sentiment_out"]) or ""
        oof = _cfg_get(cfg, ["paths", "oof"]) or ""
        calibs = (
            _cfg_get(cfg, ["calibration", "calibrators_pkl"])
            or _cfg_get(cfg, ["paths", "calibrators_pkl"])
            or ""
        )
        meta_calib = _cfg_get(cfg, ["calibration", "meta_calibrator_pkl"], "")
        top_k = str(_cfg_get(cfg, ["predict", "top_k"], 20))
        if not (features and model_pkl):
            raise RuntimeError(
                "paths.features and paths.meta_model required in YAML for prediction"
            )
        pd_args = [
            "--features",
            features,
            "--model-pkl",
            model_pkl,
            "--top-k",
            top_k,
            "--out-csv",
            picks_csv,
        ]
        if args.date:
            pd_args += ["--date", args.date]
        if sent_out:
            pd_args += ["--news-sentiment", sent_out]
        if calibs:
            pd_args += ["--calibrators-pkl", calibs]
        elif oof:
            pd_args += ["--oof", oof]
        if meta_calib:
            pd_args += ["--meta-calibrator-pkl", meta_calib]
        predict_daily_main(pd_args)
    else:
        print("[daily] Step 2/3: Skipped prediction.")

    # 3) Trade alert (Discord-friendly)
    if not args.skip_alert:
        print("[daily] Step 3/3: Sending trade alert…")
        features = _cfg_get(cfg, ["paths", "features"]) or ""
        # Optional: include intraday snapshot for mixing (keep daily features for risk metrics)
        use_intraday = bool(_cfg_get(cfg, ["alert", "use_intraday_features"], False))
        intraday_feat = _cfg_get(cfg, ["paths", "intraday_features"], "")
        mix_w = _cfg_get(cfg, ["alert", "mix_intraday_weight"], None)
        model_pkl = _cfg_get(cfg, ["paths", "meta_model"]) or ""
        uni = (
            _cfg_get(cfg, ["alert", "universe_file"])
            or _cfg_get(cfg, ["paths", "alert_universe"])
            or ""
        )
        sent_out = _cfg_get(cfg, ["paths", "sentiment_out"]) or ""
        calibs = (
            _cfg_get(cfg, ["calibration", "calibrators_pkl"])
            or _cfg_get(cfg, ["paths", "calibrators_pkl"])
            or ""
        )
        oof = _cfg_get(cfg, ["paths", "oof"]) or ""
        top_k = str(_cfg_get(cfg, ["alert", "top_k"], 5))
        webhook = _cfg_get(
            cfg, ["alert", "discord_webhook"], os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
        price_source = _cfg_get(cfg, ["alert", "price_source"], "")
        live_provider = _cfg_get(cfg, ["alert", "live_provider"], "")
        polygon_plan = _cfg_get(cfg, ["alert", "polygon_plan"], "")
        # If features are stale and no explicit live setting, force live for sizing to avoid split-adjusted prices
        try:
            import pandas as _pd

            _tmp = _pd.read_parquet(features, columns=["date"]).assign(
                date=lambda d: _pd.to_datetime(d["date"])
            )
            _last = _tmp["date"].max()
            _stale_days = int(
                (
                    _pd.Timestamp.utcnow().normalize()
                    - _pd.Timestamp(_last).normalize()
                ).days
            )
            auto_enable_intraday = False
            if (
                not price_source or str(price_source).lower() == "feature"
            ) and _stale_days > 2:
                price_source = "live"
                if not live_provider:
                    live_provider = "yahoo"
                print(
                    f"[daily] features are stale (last={_last.date()}, {_stale_days}d); using live prices for alert sizing"
                )
                # If an intraday snapshot is configured and present, auto-enable mixing to get fresher ranking
                try:
                    if intraday_feat and os.path.exists(intraday_feat):
                        auto_enable_intraday = True
                except Exception:
                    auto_enable_intraday = False
        except Exception:
            pass

        # Optionally block if features are too old and no intraday snapshot configured
        allow_stale = bool(_cfg_get(cfg, ["alert", "allow_stale_features"], False))
        try:
            if _stale_days > 30 and not (use_intraday and intraday_feat):
                if not allow_stale:
                    raise RuntimeError(
                        f"Features are too stale for alerts (last={_last.date()}, {_stale_days}d). "
                        "Build newer features or set alert.use_intraday_features with paths.intraday_features, "
                        "or set alert.allow_stale_features: true to override."
                    )
        except NameError:
            pass
        # Risk fields to override CLI explicitly
        r = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
        acct_equity = r.get("account_equity")
        risk_mode = r.get("risk_mode")
        risk_pct = r.get("risk_pct")
        min_risk = r.get("min_risk_pct")
        max_risk = r.get("max_risk_pct")
        risk_curve = r.get("risk_curve")
        stop_mult = r.get("stop_atr_mult")
        max_w = r.get("max_name_weight")
        max_notional = r.get("max_position_notional")
        base_prob = r.get("base_prob")
        if not (features and model_pkl and uni):
            raise RuntimeError(
                "paths.features, paths.meta_model, and alert.universe_file required in YAML for alert"
            )
        # News provider: prefer YAML alert.news_provider, default to polygon
        news_provider = str(_cfg_get(cfg, ["alert", "news_provider"], "polygon"))
        ta_args = [
            "--features",
            features,
            "--model-pkl",
            model_pkl,
            "--universe-file",
            uni,
            "--provider",
            news_provider,
            "--top-k",
            top_k,
        ]
        if price_source:
            ta_args += ["--price-source", str(price_source)]
        if live_provider:
            ta_args += ["--live-provider", str(live_provider)]
        if polygon_plan:
            ta_args += ["--polygon-plan", str(polygon_plan)]
        if (
            use_intraday or "auto_enable_intraday" in locals() and auto_enable_intraday
        ) and intraday_feat:
            ta_args += ["--intraday-features", intraday_feat]
            if mix_w is None:
                ta_args += ["--mix-intraday", "1.0"]
            else:
                ta_args += ["--mix-intraday", str(mix_w)]
        # Forward risk args explicitly (take precedence if provided)
        if acct_equity is not None:
            ta_args += ["--account-equity", str(acct_equity)]
        if risk_mode:
            ta_args += ["--risk-mode", str(risk_mode)]
        if risk_pct is not None:
            ta_args += ["--risk-pct", str(risk_pct)]
        if min_risk is not None:
            ta_args += ["--risk-min-pct", str(min_risk)]
        if max_risk is not None:
            ta_args += ["--risk-max-pct", str(max_risk)]
        if risk_curve:
            ta_args += ["--risk-curve", str(risk_curve)]
        if stop_mult is not None:
            ta_args += ["--stop-atr-mult", str(stop_mult)]
        if max_w is not None:
            ta_args += ["--max-name-weight", str(max_w)]
        if max_notional is not None:
            ta_args += ["--max-position-notional", str(max_notional)]
        if base_prob is not None:
            ta_args += ["--risk-base-prob", str(base_prob)]
        if args.date:
            ta_args += ["--date", args.date]
        if sent_out:
            ta_args += ["--news-sentiment", sent_out]
        if calibs:
            ta_args += ["--calibrators-pkl", calibs]
        elif oof:
            ta_args += ["--oof", oof]
        if webhook:
            ta_args += ["--discord-webhook", webhook]
        trade_alert_main(ta_args)
    else:
        print("[daily] Step 3/3: Skipped alert.")


if __name__ == "__main__":
    main()
