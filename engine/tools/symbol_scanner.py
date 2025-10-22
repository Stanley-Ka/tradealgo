from __future__ import annotations

"""Ad-hoc scanner for a user-provided list of symbols.

For each symbol the script:
- Computes specialist probabilities and the current meta probability
- Evaluates liquidity/volatility gates
- Provides an expected return / horizon estimate using existing heuristics
- Surfaces the top contributing specialists and recent meta-prob trend

Example usage:
  python -m engine.tools.symbol_scanner --symbols AAPL,MSFT,AMZN \
      --config engine/config.research.yaml --history-days 10
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..infra.env import load_env_files
from ..infra.feature_join import attach_adv_atr
from ..infra.reason import consensus_for_symbol, expected_return_and_horizon
from ..infra.styles import resolve_style
from ..infra.yaml_config import load_yaml_config
from ..models.calib_utils import (
    apply_calibrator as apply_cal,
    apply_meta_calibrator as apply_meta_cal,
    load_spec_calibrators as load_calibrators,
    naive_prob_map as naive_map,
)


@dataclass
class ScanResult:
    symbol: str
    meta_prob: float
    meta_prob_mix: Optional[float]
    adv20: Optional[float]
    atr_pct: Optional[float]
    price: Optional[float]
    expected_ret_pct: Optional[float]
    expected_horizon: Optional[str]
    top_specialists: List[tuple[str, float]]
    risk_details: Dict[str, bool]
    suggestions: List[str]
    upcoming_earnings: Optional[str]
    history: pd.DataFrame


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan a list of symbols using the current engine models"
    )
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of tickers (e.g., AAPL,MSFT)",
    )
    p.add_argument("--config", default="engine/config.research.yaml")
    p.add_argument(
        "--features",
        default="",
        help="Override features parquet (defaults to config paths.features)",
    )
    p.add_argument("--model-pkl", default="", help="Override meta model pickle")
    p.add_argument(
        "--calibrators-pkl", default="", help="Override specialist calibrators pickle"
    )
    p.add_argument("--oof", default="", help="OOF parquet for calibrators (fallback)")
    p.add_argument(
        "--meta-calibrator-pkl", default="", help="Optional meta calibrator pickle"
    )
    p.add_argument(
        "--news-sentiment",
        default="",
        help="Override sentiment parquet/CSV for spec_nlp",
    )
    p.add_argument(
        "--earnings-file",
        default="",
        help="Optional earnings parquet/CSV to surface upcoming events",
    )
    p.add_argument(
        "--date", default="", help="Target date YYYY-MM-DD (default latest in features)"
    )
    p.add_argument(
        "--history-days",
        type=int,
        default=15,
        help="Days of history to show for meta probability trend",
    )
    p.add_argument(
        "--top-specialists",
        type=int,
        default=5,
        help="How many specialist probabilities to list",
    )
    return p.parse_args(argv)


def _coerce_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _compute_specialist_probs(
    df: pd.DataFrame,
    spec_params: Dict,
    calibrators,
) -> pd.DataFrame:
    specs = compute_specialist_scores(df, news_sentiment=None, params=spec_params)
    prob_cols: List[str] = []
    for sc in [
        c for c in specs.columns if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = specs[sc].astype(float).values
        prob = (
            apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else naive_map(raw)
        )
        specs[f"{sc}_prob"] = prob
        prob_cols.append(f"{sc}_prob")
    return specs


def _compute_meta_prob(
    specs: pd.DataFrame, model_path: str, meta_cal_path: Optional[str]
) -> pd.DataFrame:
    import pickle

    with open(model_path, "rb") as fpk:
        payload = pickle.load(fpk)
    clf = payload.get("model")
    feature_names = payload.get("features") or [
        c for c in specs.columns if c.endswith("_prob")
    ]
    # Ensure odds/logit/interactions if required by the model
    for base in list(feature_names):
        if base not in specs.columns and base.endswith("_prob"):
            specs[base] = 0.5
        if base.endswith("_prob"):
            root = base
            odds_col = f"{root}_odds"
            logit_col = f"{root}_logit"
            if odds_col in feature_names and odds_col not in specs.columns:
                pclip = pd.Series(specs[root]).astype(float).clip(1e-6, 1 - 1e-6)
                specs[odds_col] = (pclip / (1 - pclip)).astype(float)
            if logit_col in feature_names and logit_col not in specs.columns:
                if odds_col in specs.columns:
                    specs[logit_col] = np.log(pd.Series(specs[odds_col]).astype(float))
                else:
                    pclip = pd.Series(specs[root]).astype(float).clip(1e-6, 1 - 1e-6)
                    specs[logit_col] = np.log((pclip / (1 - pclip)).astype(float))
    # Fill any missing columns expected by the model
    for col in feature_names:
        if col not in specs.columns:
            if (
                col in ("regime_vol", "regime_risk")
                or col.endswith("_odds")
                or col.endswith("_logit")
                or "__x__" in col
            ):
                specs[col] = 0.0
            else:
                specs[col] = 0.5
    # Interaction terms
    for col in feature_names:
        if "__x__" in col and col not in specs.columns:
            a, b = col.split("__x__", 1)
            if a in specs.columns and b in specs.columns:
                try:
                    specs[col] = specs[a].astype(float) * specs[b].astype(float)
                except Exception:
                    specs[col] = 0.0
            else:
                specs[col] = 0.0
    X = specs[feature_names].values
    meta_prob = (
        clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X)
    )
    meta_prob = apply_meta_cal(meta_cal_path, meta_prob)
    specs["meta_prob"] = meta_prob
    return specs


def _load_earnings_map(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["symbol", "date"])
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _summarise_specialists(row: pd.Series, top_n: int) -> List[tuple[str, float]]:
    pairs: List[tuple[str, float]] = []
    for col in row.index:
        if col.startswith("spec_") and col.endswith("_prob"):
            pairs.append(
                (col.replace("spec_", "").replace("_prob", ""), float(row[col]))
            )
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def _build_suggestions(
    row: pd.Series, risk_ok: Dict[str, bool], hist: pd.DataFrame
) -> List[str]:
    suggestions: List[str] = []
    p = float(row.get("meta_prob", 0.5))
    if p >= 0.62:
        suggestions.append("High conviction setup (meta ≥ 0.62)")
    elif p >= 0.57:
        suggestions.append("Moderate conviction (meta in high-50s)")
    else:
        suggestions.append("Signal still developing (meta < 0.57)")
    if not risk_ok.get("liq", True):
        suggestions.append("Fails liquidity gate (ADV below minimum)")
    if not risk_ok.get("atr", True):
        suggestions.append("Fails volatility gate (ATR% too high)")
    if not risk_ok.get("earn", True):
        suggestions.append("Inside earnings blackout window")
    # Trend insight
    if not hist.empty:
        latest = float(hist.iloc[-1]["meta_prob"])
        prev = float(hist.iloc[-min(len(hist), 3) :]["meta_prob"].mean())
        delta = latest - prev
        if abs(delta) >= 0.02:
            direction = "improving" if delta > 0 else "softening"
            suggestions.append(
                f"Meta trend {direction} over last {min(len(hist), 3)} days (Δ {delta:+.3f})"
            )
    return suggestions


def scan_symbols(args: argparse.Namespace) -> None:
    load_env_files()
    symbols = _coerce_symbols(args.symbols)
    if not symbols:
        raise ValueError("No symbols parsed from --symbols")

    cfg_path = args.config
    if cfg_path and os.path.exists(cfg_path):
        cfg = load_yaml_config(cfg_path)
    else:
        resolved = resolve_style(cfg_path)
        cfg = load_yaml_config(resolved) if resolved else {}

    features_path = args.features or cfg.get("paths", {}).get("features", "")
    if not features_path:
        raise RuntimeError(
            "Features path missing; provide --features or set paths.features in config"
        )
    model_path = args.model_pkl or cfg.get("paths", {}).get("meta_model", "")
    if not model_path:
        raise RuntimeError(
            "Meta model path missing; provide --model-pkl or set paths.meta_model in config"
        )
    calibrators_path = args.calibrators_pkl or cfg.get("calibration", {}).get(
        "calibrators_pkl", ""
    )
    oof_path = args.oof or cfg.get("paths", {}).get("oof", "")
    meta_cal_path = args.meta_calibrator_pkl or cfg.get("calibration", {}).get(
        "meta_calibrator_pkl", ""
    )

    feats = _load_features(features_path)
    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = feats["date"].max()
    if pd.isna(target_date):
        raise RuntimeError("No dates present in features parquet")

    latest_rows = feats[
        (feats["date"] == target_date) & (feats["symbol"].isin(symbols))
    ].copy()
    missing = sorted(set(symbols) - set(latest_rows["symbol"].unique()))
    if missing:
        print(f"[scanner] missing data on {target_date.date()}: {', '.join(missing)}")
    if latest_rows.empty:
        raise RuntimeError("No feature rows for requested symbols on target date")

    # News sentiment (optional)
    sentiment_file = args.news_sentiment or cfg.get("paths", {}).get(
        "news_sentiment", ""
    )
    sentiment = None
    if sentiment_file:
        try:
            sentiment = load_sentiment_file(sentiment_file)
        except Exception:
            sentiment = None

    spec_params = (
        cfg.get("specialists", {})
        if isinstance(cfg.get("specialists", {}), dict)
        else {}
    )
    calibrators = load_calibrators(
        calibrators_pkl=calibrators_path or None,
        oof_path=oof_path or None,
        kind=cfg.get("calibration", {}).get("kind", "platt"),
    )

    day_specs = compute_specialist_scores(
        latest_rows, news_sentiment=sentiment, params=spec_params
    )
    for sc in [
        c
        for c in day_specs.columns
        if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = day_specs[sc].astype(float).values
        day_specs[f"{sc}_prob"] = (
            apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else naive_map(raw)
        )

    day_specs = _compute_meta_prob(day_specs, model_path, meta_cal_path or None)

    # Attach liquidity/volatility metrics
    adv_atr = attach_adv_atr(feats, target_date).rename(
        columns={"atr_pct_14": "atr_pct_panel"}
    )
    day_specs = day_specs.merge(adv_atr, on="symbol", how="left")
    if "atr_pct_14" not in day_specs.columns or day_specs["atr_pct_14"].isna().any():
        day_specs["atr_pct_14"] = day_specs.get("atr_pct_panel")
    day_specs = day_specs.drop(
        columns=[c for c in ("atr_pct_panel",) if c in day_specs.columns]
    )

    # Price, adv, etc should come from latest_rows (same index)
    base_cols = ["symbol", "adj_close"]
    optional_cols = ["adv20", "atr_pct_14"]
    for col in optional_cols:
        if col in latest_rows.columns:
            base_cols.append(col)
    day_specs = day_specs.merge(
        latest_rows[base_cols], on="symbol", how="left", suffixes=("", "_feat")
    )
    if "adv20" not in day_specs or day_specs["adv20"].isna().all():
        if "adv20_feat" in day_specs.columns:
            day_specs["adv20"] = day_specs["adv20_feat"]
        else:
            day_specs["adv20"] = np.nan
    if "atr_pct_14" not in day_specs or day_specs["atr_pct_14"].isna().all():
        if "atr_pct_14_feat" in day_specs.columns:
            day_specs["atr_pct_14"] = day_specs["atr_pct_14_feat"]
    drop_cols = [c for c in day_specs.columns if c.endswith("_feat")]
    if drop_cols:
        day_specs = day_specs.drop(columns=drop_cols)

    # Earnings lookup
    earn_df = _load_earnings_map(args.earnings_file)
    earn_lookup = {}
    if not earn_df.empty:
        earn_lookup = {
            sym: earn_df[earn_df["symbol"] == sym]["date"].min()
            for sym in day_specs["symbol"]
        }

    results: List[ScanResult] = []
    risk_min_adv = float(cfg.get("risk", {}).get("min_adv_usd", 0.0))
    risk_max_atr = float(cfg.get("risk", {}).get("max_atr_pct", 1.0))
    earnings_blackout = int(cfg.get("risk", {}).get("earnings_blackout", 0))

    # Build history subset for trend (all symbols, last N days)
    hist_start = target_date - pd.Timedelta(days=max(args.history_days, 1))
    hist_rows = feats[
        (feats["symbol"].isin(symbols)) & (feats["date"] >= hist_start)
    ].copy()
    hist_specs = compute_specialist_scores(
        hist_rows, news_sentiment=sentiment, params=spec_params
    )
    for sc in [
        c
        for c in hist_specs.columns
        if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = hist_specs[sc].astype(float).values
        hist_specs[f"{sc}_prob"] = (
            apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else naive_map(raw)
        )
    hist_specs = _compute_meta_prob(hist_specs, model_path, meta_cal_path or None)

    for sym in symbols:
        row = day_specs[day_specs["symbol"] == sym]
        if row.empty:
            continue
        row = row.iloc[0]
        adv = float(row.get("adv20", np.nan)) if not pd.isna(row.get("adv20")) else None
        atr_pct = (
            float(row.get("atr_pct_14", np.nan))
            if not pd.isna(row.get("atr_pct_14"))
            else None
        )
        price = (
            float(row.get("adj_close", np.nan))
            if not pd.isna(row.get("adj_close"))
            else None
        )
        meta_prob = float(row.get("meta_prob", 0.5))
        meta_mix = (
            float(row.get("meta_prob_mix", meta_prob))
            if "meta_prob_mix" in row and not pd.isna(row.get("meta_prob_mix"))
            else None
        )

        enr = None
        if earn_lookup:
            next_e = earn_lookup.get(sym)
            if isinstance(next_e, (pd.Timestamp,)):
                next_e = next_e.date()
            if next_e:
                delt = (pd.Timestamp(next_e) - target_date).days
                if -earnings_blackout <= delt <= earnings_blackout:
                    enr = f"Within blackout window ({delt}d)"
                elif delt >= 0:
                    enr = f"{delt} days until report"
                else:
                    enr = f"reported {abs(delt)} days ago"

        exp_ret_pct = None
        exp_horizon = None
        try:
            adv_safe = float(adv) if adv is not None else np.nan
            atr_safe = float(atr_pct) if atr_pct is not None else np.nan
            exp_pct, horizon = expected_return_and_horizon(
                meta_prob,
                atr_safe,
                base_prob=cfg.get("risk", {}).get("base_prob"),
                k_scale=cfg.get("risk", {}).get("expected_k"),
            )
            exp_ret_pct = float(exp_pct) if np.isfinite(exp_pct) else None
            exp_horizon = str(horizon)
        except Exception:
            exp_ret_pct, exp_horizon = None, None

        top_specs = _summarise_specialists(row, args.top_specialists)
        risk_details = {
            "liq": adv is None or adv >= risk_min_adv,
            "atr": atr_pct is None or atr_pct <= risk_max_atr,
            "earn": enr is None or not enr.startswith("Within"),
        }
        hist = hist_specs[hist_specs["symbol"] == sym][
            ["date", "meta_prob"]
        ].sort_values("date")
        suggestions = _build_suggestions(row, risk_details, hist)

        results.append(
            ScanResult(
                symbol=sym,
                meta_prob=meta_prob,
                meta_prob_mix=meta_mix,
                adv20=adv,
                atr_pct=atr_pct,
                price=price,
                expected_ret_pct=exp_ret_pct,
                expected_horizon=exp_horizon,
                top_specialists=top_specs,
                risk_details=risk_details,
                suggestions=suggestions,
                upcoming_earnings=enr,
                history=hist,
            )
        )

    if not results:
        print("[scanner] No symbols to report.")
        return

    print(
        f"Symbol scan for {target_date.date()} (model: {os.path.basename(model_path)})"
    )
    for res in results:
        print("-" * 72)
        print(
            f"{res.symbol} – meta={res.meta_prob:.3f}"
            + (f" mix={res.meta_prob_mix:.3f}" if res.meta_prob_mix is not None else "")
        )
        if res.price is not None:
            print(f"  Price ${res.price:,.2f}")
        if res.adv20 is not None:
            print(f"  ADV20 ${res.adv20:,.0f}")
        if res.atr_pct is not None:
            print(f"  ATR%% {res.atr_pct*100:.2f}")
        if res.expected_ret_pct is not None:
            print(
                f"  Expected return {res.expected_ret_pct:.2f}% horizon {res.expected_horizon}"
            )
        if res.upcoming_earnings:
            print(f"  Earnings: {res.upcoming_earnings}")
        print("  Specialists:")
        for name, prob in res.top_specialists:
            print(f"    {name}: {prob:.3f}")
        try:
            consensus = consensus_for_symbol(day_specs, res.symbol)
            if consensus:
                print(f"  Consensus: {consensus}")
        except Exception:
            pass
        print("  Risk gates:")
        for key, ok in res.risk_details.items():
            note = "ok" if ok else "check"
            print(f"    {key}: {note}")
        if res.history.shape[0] >= 2:
            tail = res.history.tail(min(5, len(res.history)))
            tail_txt = ", ".join(
                [
                    f"{d.strftime('%m-%d')}: {v:.3f}"
                    for d, v in zip(tail["date"], tail["meta_prob"])
                ]
            )
            print(f"  Recent meta trend: {tail_txt}")
        if res.suggestions:
            print("  Suggestions:")
            for msg in res.suggestions:
                print(f"    - {msg}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    scan_symbols(args)


if __name__ == "__main__":
    main()
