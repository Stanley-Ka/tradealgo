"""Entry loop (daily proxy): select entries and append to positions.

This tool reuses the daily feature panel to simulate an intraday entry loop.
It computes specialist probabilities, meta probability, applies risk gates
and (optionally) a meta_prob threshold with confirmations (hysteresis),
applies an optional sector cap, sizes positions via conviction+ATR,
and appends new entries to positions CSV.

Intended to be called periodically (e.g., via real_time_alert or scheduler).
When intraday features are available, this tool can be switched to use the
intraday latest-bar features without changing its interface.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..infra.yaml_config import load_yaml_config
from ..infra.feature_join import attach_adv_atr
from ..infra.sector import apply_sector_cap
from ..models.calib_utils import (
    load_spec_calibrators as _load_cal,
    apply_meta_calibrator as _apply_meta,
)
from ..infra.reason import consensus_for_symbol, expected_return_and_horizon
from ..infra.market_state import state_as_dict
from ..infra.env import load_env_files


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select entries and append to positions CSV"
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML with paths/specialists/risk (overridden by --style)",
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
        help="Features parquet with daily rows",
    )
    p.add_argument(
        "--intraday-features",
        type=str,
        default="",
        help="Intraday latest features snapshot parquet (overrides --features if set)",
    )
    p.add_argument("--model-pkl", required=True, help="Meta model pickle path")
    p.add_argument(
        "--universe-file", required=True, help="Universe file (one SYMBOL per line)"
    )
    p.add_argument(
        "--oof", type=str, default="", help="OOF parquet to fit calibrators (optional)"
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators (optional)",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-calibrator for meta_prob",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file for spec_nlp",
    )
    p.add_argument("--top-k", type=int, default=5, help="Max entries per run")
    p.add_argument(
        "--entry-threshold",
        type=float,
        default=0.0,
        help="If >0, require meta_prob >= threshold",
    )
    p.add_argument(
        "--confirmations",
        type=int,
        default=1,
        help="Consecutive confirmations above threshold before entry",
    )
    p.add_argument(
        "--state-csv",
        type=str,
        default="data/state/entry_state.csv",
        help="Path to persist confirmations state",
    )
    # Sector cap
    p.add_argument(
        "--sector-map-csv", type=str, default="", help="CSV with columns symbol,sector"
    )
    p.add_argument(
        "--sector-cap",
        type=int,
        default=0,
        help="If >0, limit entries per sector to this number",
    )
    # Corporate actions gating (optional, powered by Polygon dividends CSV)
    p.add_argument(
        "--dividends-csv",
        type=str,
        default="",
        help="Optional dividends CSV (symbol,ex_dividend_date,...) to blackout around ex-div",
    )
    p.add_argument(
        "--exdiv-blackout-days",
        type=int,
        default=0,
        help="If >0, exclude symbols within +/- N days of an ex-dividend date",
    )
    # Positions & sizing
    p.add_argument("--positions-csv", type=str, default="data/paper/positions.csv")
    p.add_argument("--entry-price", choices=["close", "open"], default="close")
    p.add_argument("--price-source", choices=["feature", "live"], default="feature")
    p.add_argument("--live-provider", choices=["yahoo", "polygon"], default="yahoo")
    p.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Provider plan tier (basic|starter|developer|advanced)",
    )
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
    p.add_argument("--account-equity", type=float, default=100000.0)
    p.add_argument("--risk-mode", choices=["fixed", "auto"], default="auto")
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--risk-min-pct", type=float, default=0.002)
    p.add_argument("--risk-max-pct", type=float, default=0.006)
    p.add_argument(
        "--risk-curve", choices=["linear", "quadratic", "sqrt"], default="quadratic"
    )
    p.add_argument("--risk-base-prob", type=float, default=None)
    # Notional caps
    p.add_argument("--max-name-weight", type=float, default=None)
    p.add_argument("--max-position-notional", type=float, default=None)
    p.add_argument("--stop-atr-mult", type=float, default=1.0)
    # Risk gates
    p.add_argument("--min-adv-usd", type=float, default=1e7)
    p.add_argument("--max-atr-pct", type=float, default=0.05)
    p.add_argument(
        "--date", type=str, default="", help="Target date YYYY-MM-DD (default latest)"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--decision-log-csv",
        type=str,
        default="",
        help="Append per-run entries diagnostics",
    )
    # Notifications (optional)
    p.add_argument(
        "--discord-webhook",
        type=str,
        default=(
            os.environ.get(
                "DISCORD_TRADES_WEBHOOK_URL", os.environ.get("DISCORD_WEBHOOK_URL", "")
            )
        ),
        help="Discord webhook for trade entries",
    )
    p.add_argument(
        "--send-discord",
        action="store_true",
        help="Send Discord message for new entries",
    )
    # Realized outcome logging
    p.add_argument(
        "--realized-lookahead",
        type=int,
        default=5,
        help="Cumulative forward days for realized return",
    )
    p.add_argument(
        "--realized-target-pct",
        type=float,
        default=0.01,
        help="Target pct for hit within lookahead (e.g., 0.01=+1%)",
    )
    # Portfolio-level caps (long-only gross)
    p.add_argument(
        "--max-gross",
        type=float,
        default=None,
        help="Max gross exposure as fraction of equity (e.g., 1.0)",
    )
    return p.parse_args(argv)


def _read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def _load_state(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        return {
            str(r.symbol).upper(): int(r.get("consec", 0)) for _, r in df.iterrows()
        }
    except Exception:
        return {}


def _save_state(path: str, counts: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({"symbol": list(counts.keys()), "consec": list(counts.values())})
    df.to_csv(path, index=False)


def _risk_fraction(meta_prob: float, args: argparse.Namespace) -> float:
    # Robust to missing attributes in tests or external callers
    mode = getattr(args, "risk_mode", "auto")
    if mode == "fixed":
        return float(getattr(args, "risk_pct", 0.005))
    rmin = float(getattr(args, "risk_min_pct", 0.002))
    rmax = float(getattr(args, "risk_max_pct", 0.006))
    base_p = getattr(args, "risk_base_prob", None)
    base_p = float(base_p if base_p is not None else 0.5)
    denom = max(1e-6, 1.0 - base_p)
    conv_n = max(0.0, min(1.0, (float(meta_prob) - base_p) / denom))
    curve = getattr(args, "risk_curve", "linear")
    if curve == "quadratic":
        conv_n = conv_n * conv_n
    elif curve == "sqrt":
        conv_n = conv_n**0.5
    return float(rmin + (rmax - rmin) * conv_n)


def main(argv: Optional[List[str]] = None) -> None:
    from ..models.calib_utils import (
        apply_calibrator as _apply_cal,
        naive_prob_map as _naive,
    )
    import pickle

    # Load env for direct runs (e.g., POLYGON_API_KEY from scripts/api.env)
    from ..infra.styles import resolve_style, normalize_plan

    load_env_files()
    args = parse_args(argv)
    # Best-effort single-run guard to avoid duplicate concurrent entries
    lock_path = os.path.join("data", "entries", ".entry_loop.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_acquired = False
    # Acquire lock (remove if stale > 180s)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        lock_acquired = True
    except FileExistsError:
        try:
            mtime = os.path.getmtime(lock_path)
        except Exception:
            mtime = 0.0
        import time as _t

        if (_t.time() - float(mtime)) > 180.0:
            try:
                os.remove(lock_path)
            except Exception:
                pass
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                lock_acquired = True
            except Exception:
                pass
    if not lock_acquired:
        print("[entry] another entry_loop is running; skipping")
        return
    # Ensure lock is removed on process exit
    try:
        import atexit

        def _cleanup_lock(path: str = lock_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        atexit.register(_cleanup_lock)
    except Exception:
        pass

    cfg_path = args.config
    if args.style:
        maybe = resolve_style(args.style)
        if maybe:
            cfg_path = maybe
    cfg = load_yaml_config(cfg_path) if cfg_path else {}

    feat_path = args.intraday_features or args.features
    if not feat_path:
        raise RuntimeError("Provide --features or --intraday-features")
    f = pd.read_parquet(feat_path)
    f["date"] = pd.to_datetime(f["date"])  # ensure dtype
    f["symbol"] = f["symbol"].astype(str).str.upper()
    target_date = pd.Timestamp(args.date) if args.date else f["date"].max()

    uni = set(_read_universe(args.universe_file))
    day = f[(f["date"] == target_date) & (f["symbol"].isin(uni))].copy()
    if day.empty:
        print(f"[entry] no rows for universe on {target_date.date()}")
        return

    # Build risk metrics for target date
    day_adv = attach_adv_atr(f, target_date).rename(
        columns={"atr_pct_14": "atr_pct_14_from_panel"}
    )

    # Compute specialist scores and probs
    sentiment = None
    if args.news_sentiment or cfg.get("paths", {}).get("news_sentiment"):
        try:
            sentiment = load_sentiment_file(
                args.news_sentiment or cfg.get("paths", {}).get("news_sentiment")
            )
        except Exception:
            sentiment = None
    specs = compute_specialist_scores(
        day, news_sentiment=sentiment, params=cfg.get("specialists", {})
    )
    calibrators = _load_cal(
        calibrators_pkl=(
            args.calibrators_pkl
            or cfg.get("calibration", {}).get("calibrators_pkl", "")
        )
        or None,
        oof_path=(args.oof or cfg.get("paths", {}).get("oof", "")) or None,
        kind=cfg.get("calibration", {}).get("kind", "platt"),
    )
    for sc in [
        c for c in specs.columns if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = specs[sc].astype(float).values
        prob = (
            _apply_cal(calibrators.get(sc), raw)
            if (calibrators and sc in calibrators)
            else _naive(raw)
        )
        specs[f"{sc}_prob"] = prob

    # Meta probability
    with open(args.model_pkl, "rb") as fpk:
        meta = pickle.load(fpk)
    clf = meta.get("model")
    feature_names = meta.get("features") or [
        c for c in specs.columns if c.endswith("_prob")
    ]
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
    meta_prob = _apply_meta(
        args.meta_calibrator_pkl
        or cfg.get("calibration", {}).get("meta_calibrator_pkl"),
        meta_prob,
    )
    specs["meta_prob"] = meta_prob

    # Risk gates
    specs = specs.merge(day_adv, on="symbol", how="left")
    if "atr_pct_14_from_panel" in specs:
        specs["atr_pct_14"] = specs.get(
            "atr_pct_14", pd.Series(index=specs.index)
        ).fillna(specs["atr_pct_14_from_panel"])
        specs = specs.drop(columns=["atr_pct_14_from_panel"])
    # Risk thresholds: allow YAML overrides of CLI defaults
    try:
        _min_adv = cfg.get("risk", {}).get("min_adv_usd", args.min_adv_usd)
    except Exception:
        _min_adv = args.min_adv_usd
    try:
        _max_atr = cfg.get("risk", {}).get("max_atr_pct", args.max_atr_pct)
    except Exception:
        _max_atr = args.max_atr_pct
    liq = specs["adv20"].notna() & (specs["adv20"] >= float(_min_adv))
    atr_ok = (specs["atr_pct_14"] <= float(_max_atr)) | specs["atr_pct_14"].isna()
    specs["_ok"] = liq & atr_ok

    # Threshold + confirmations state
    state = _load_state(args.state_csv)
    counts: Dict[str, int] = {**state}
    specs["_above"] = specs["meta_prob"] >= float(args.entry_threshold)
    for sym, above in zip(specs["symbol"].values, specs["_above"].values):
        if bool(above):
            counts[sym] = counts.get(sym, 0) + 1
        else:
            counts[sym] = 0

    # Candidate set
    cands = specs[specs["_ok"].astype(bool)].copy()
    # Optional price gating
    if args.min_price is not None or args.max_price is not None:
        pcol = "adj_close" if args.entry_price == "close" else "adj_open"
        try:
            if args.min_price is not None:
                cands = cands[cands[pcol] >= float(args.min_price)]
            if args.max_price is not None:
                cands = cands[cands[pcol] <= float(args.max_price)]
        except Exception:
            pass
    if float(args.entry_threshold) > 0.0:
        cands = cands[cands["meta_prob"] >= float(args.entry_threshold)]
    # Require confirmations
    if int(args.confirmations) > 1:
        cands = cands[
            cands["symbol"].map(lambda s: counts.get(str(s).upper(), 0))
            >= int(args.confirmations)
        ]

    # Optional sector cap
    if args.sector_map_csv and int(args.sector_cap) > 0 and not cands.empty:
        try:
            cands = apply_sector_cap(
                cands, args.sector_map_csv, int(args.sector_cap), rank_col="meta_prob"
            )
        except Exception:
            pass

    # Optional ex-dividend blackout
    if int(args.exdiv_blackout_days) > 0 and args.dividends_csv and not cands.empty:
        try:
            dv = pd.read_csv(args.dividends_csv)
            if not dv.empty and "ex_dividend_date" in dv.columns:
                dv["symbol"] = dv["symbol"].astype(str).str.upper()
                dv["ex_dividend_date"] = pd.to_datetime(
                    dv["ex_dividend_date"], errors="coerce"
                )
                lo = pd.Timestamp(target_date) - pd.Timedelta(
                    days=int(args.exdiv_blackout_days)
                )
                hi = pd.Timestamp(target_date) + pd.Timedelta(
                    days=int(args.exdiv_blackout_days)
                )
                bad = set(
                    dv[(dv["ex_dividend_date"] >= lo) & (dv["ex_dividend_date"] <= hi)][
                        "symbol"
                    ].astype(str)
                )
                if bad:
                    cands = cands[~cands["symbol"].isin(bad)].copy()
        except Exception:
            pass

    # Rank and cap to top-K
    picks = cands.sort_values("meta_prob", ascending=False).head(int(args.top_k)).copy()

    if not picks.empty:
        state_df = picks.apply(lambda row: pd.Series(state_as_dict(row)), axis=1)
        for col in state_df.columns:
            picks[col] = state_df[col]

    # Reasoning and simple expectations
    def _reason(sym: str) -> str:
        return consensus_for_symbol(specs, sym)

    def _exp_and_hor(mp: float, atr: float):
        rsec = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
        base_p = rsec.get(
            "base_prob", args.risk_base_prob if args.risk_base_prob is not None else 0.5
        )
        k_scale = rsec.get("expected_k", None)
        cut1 = rsec.get("horizon_cut1", 0.55)
        cut2 = rsec.get("horizon_cut2", 0.60)
        cap_mult = rsec.get("expected_cap_mult", 2.0)
        e, h = expected_return_and_horizon(
            mp,
            atr,
            base_p,
            k_scale=k_scale,
            cap_mult=float(cap_mult),
            cut1=float(cut1),
            cut2=float(cut2),
        )
        return e, h

    picks["reason"] = picks["symbol"].map(_reason)
    _pairs = [
        _exp_and_hor(mp, atr)
        for mp, atr in zip(picks["meta_prob"].values, picks["atr_pct_14"].values)
    ]
    picks["expected_ret_pct"] = [p[0] for p in _pairs]
    picks["horizon"] = [p[1] for p in _pairs]

    # Prepare entry sizing
    entry_col = "adj_close" if args.entry_price == "close" else "adj_open"
    picks["ref_price"] = picks[entry_col].astype(float)
    # Live price override (optional)
    if str(args.price_source).lower() == "live":

        def _live_price(sym: str, fallback: float) -> float:
            pr = fallback
            try:
                # Prefer Polygon first
                from ..infra.http import HttpClient, HttpConfig

                api_key = os.environ.get("POLYGON_API_KEY", "")
                # Normalize plan to decide if polygon live endpoints are allowed
                plan_hint = normalize_plan(args.plan)
                allow_polygon_live = (plan_hint is None) or (
                    plan_hint not in ("starter",)
                )
                if api_key and allow_polygon_live:
                    url = f"https://api.polygon.io/v2/last/trade/{sym}"
                    client = HttpClient(
                        HttpConfig(requests_per_second=5.0, timeout=10.0)
                    )
                    try:
                        data = client.get_json(url, params={"apiKey": api_key}) or {}
                        p = (data.get("results", {}) or {}).get("p")
                        if p:
                            pr = float(p)
                    except Exception:
                        pass
                    # Fallback: Polygon previous close if live not authorized
                    if not np.isfinite(pr):
                        try:
                            prev_url = (
                                f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev"
                            )
                            data2 = (
                                client.get_json(
                                    prev_url,
                                    params={"adjusted": "true", "apiKey": api_key},
                                )
                                or {}
                            )
                            res = data2.get("results") or []
                            if res and res[0].get("c"):
                                pr = float(res[0]["c"])
                        except Exception:
                            pass
                # Yahoo fallback only if explicitly selected
                if not np.isfinite(pr) and str(args.live_provider).lower() == "yahoo":
                    import yfinance as yf  # type: ignore

                    tk = yf.Ticker(sym)
                    info = getattr(tk, "fast_info", None)
                    if info and getattr(info, "last_price", None) is not None:
                        pr = float(info.last_price)
                    else:
                        hist = tk.history(period="1d")
                        if not hist.empty:
                            pr = float(hist["Close"].iloc[-1])
            except Exception:
                pass
            return pr

        picks["ref_price"] = [
            _live_price(str(sym), float(px))
            for sym, px in zip(picks["symbol"].values, picks["ref_price"].values)
        ]
    stop_mult = float(args.stop_atr_mult)
    # Resolve caps from args or YAML (risk)
    cap_w = args.max_name_weight
    if cap_w is None:
        try:
            cap_w = (
                float(cfg.get("risk", {}).get("max_name_weight"))
                if cfg.get("risk", {})
                else None
            )
        except Exception:
            cap_w = None
    cap_notional_cfg = None
    try:
        cap_notional_cfg = (
            float(cfg.get("risk", {}).get("max_position_notional"))
            if cfg.get("risk", {})
            else None
        )
    except Exception:
        cap_notional_cfg = None
    cap_notional_arg = (
        args.max_position_notional if args.max_position_notional is not None else None
    )

    def _shares(row) -> int:
        pr = float(row.get("ref_price", np.nan))
        atrp = float(row.get("atr_pct_14", np.nan))
        if not np.isfinite(pr) or not np.isfinite(atrp) or atrp <= 0 or pr <= 0:
            return 0
        risk_frac = _risk_fraction(float(row.get("meta_prob", 0.5)), args)
        per_share = pr * stop_mult * atrp
        if per_share <= 0:
            return 0
        shares = int(np.floor((float(args.account_equity) * risk_frac) / per_share))
        # Notional caps
        cap_notional = 0.0
        if cap_w is not None and float(cap_w) > 0:
            cap_notional = max(cap_notional, float(cap_w) * float(args.account_equity))
        if cap_notional_arg is not None and float(cap_notional_arg) > 0:
            cap_notional = max(cap_notional, float(cap_notional_arg))
        if cap_notional_cfg is not None and float(cap_notional_cfg) > 0:
            cap_notional = max(cap_notional, float(cap_notional_cfg))
        if cap_notional > 0 and pr > 0:
            shares = min(shares, int(np.floor(cap_notional / pr)))
        return shares

    picks["shares"] = picks.apply(_shares, axis=1)
    picks = picks[picks["shares"] > 0]

    # Save state regardless of dry-run
    _save_state(args.state_csv, counts)

    # Filter out already-open positions
    already = set()
    if os.path.exists(args.positions_csv):
        try:
            pos = pd.read_csv(args.positions_csv)
            already = set(str(s).upper() for s in pos.get("symbol", pd.Series([])))
        except Exception:
            already = set()
    # One-per-session guardrail
    import json as _json

    session_file = os.path.join("data", "entries", "session_entries.json")
    os.makedirs(os.path.dirname(session_file), exist_ok=True)
    try:
        sess_key = pd.Timestamp.utcnow().date().isoformat()
    except Exception:
        from datetime import date as _date

        sess_key = _date.today().isoformat()
    session_state = {}
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as fh:
                session_state = _json.load(fh) or {}
        except Exception:
            session_state = {}
    seen_today = set(
        str(s).upper()
        for s in session_state.get(sess_key, [])
        if isinstance(session_state.get(sess_key, []), list)
    )
    new_entries = picks[~picks["symbol"].isin(already | seen_today)].copy()

    # Log
    spec_prob_cols = [
        c for c in picks.columns if c.startswith("spec_") and c.endswith("_prob")
    ]
    state_cols = [
        c for c in ("trend_state", "vol_state", "condition_label") if c in picks.columns
    ]
    diag_cols = [
        "symbol",
        "meta_prob",
        "adv20",
        "atr_pct_14",
        "ref_price",
        "shares",
        "expected_ret_pct",
        "horizon",
        "reason",
    ]
    diag_cols.extend(state_cols)
    diag_cols.extend(spec_prob_cols)
    if args.decision_log_csv and not new_entries.empty:
        os.makedirs(os.path.dirname(args.decision_log_csv), exist_ok=True)
        logdf = new_entries[diag_cols].copy()
        logdf.insert(0, "date", target_date.date())
        # Realized outcomes (requires fret_1d in features)
        if "fret_1d" in f.columns:
            # Build per-symbol forward return sequences
            f_sorted = f.sort_values(["symbol", "date"]).reset_index(drop=True)
            grp = {
                sym: df.reset_index(drop=True) for sym, df in f_sorted.groupby("symbol")
            }

            def _cum_ret(sym: str, dt: pd.Timestamp, n: int) -> float:
                g = grp.get(sym)
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

            look = int(max(1, args.realized_lookahead))
            # quick slices for 1/3/5 days
            for N in (1, 3, 5):
                logdf[f"ret_{N}d"] = [
                    _cum_ret(str(s), target_date, N) for s in logdf["symbol"].values
                ]
            logdf[f"cum_ret_{look}d"] = [
                _cum_ret(str(s), target_date, look) for s in logdf["symbol"].values
            ]

            # Time-to-target within lookahead
            def _time_to_target(
                sym: str, dt: pd.Timestamp, target: float, n: int
            ) -> float:
                g = grp.get(sym)
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

            logdf[f"t_hit_{look}d"] = [
                _time_to_target(
                    str(s), target_date, float(args.realized_target_pct), look
                )
                for s in logdf["symbol"].values
            ]
            logdf[f"hit_{look}d"] = logdf[f"t_hit_{look}d"].apply(
                lambda x: 1 if pd.notna(x) else 0
            )
        if os.path.exists(args.decision_log_csv):
            logdf.to_csv(args.decision_log_csv, mode="a", header=False, index=False)
        else:
            logdf.to_csv(args.decision_log_csv, index=False)
        print(f"[entry] appended log -> {args.decision_log_csv} rows={len(logdf)}")

    if args.dry_run:
        print(f"[entry] {target_date.date()} picks (new only):")
        if new_entries.empty:
            print("  (none)")
        else:
            print(new_entries[diag_cols].to_string(index=False))
            # Also print human-readable reasons
            for _, r in new_entries.iterrows():
                print(
                    f"  {r['symbol']}: Why: {r.get('reason','')} | Expected: +{float(r.get('expected_ret_pct', float('nan'))):.2f}% in {r.get('horizon','')}"
                )
        return

    # Enforce gross cap (optional)
    if not new_entries.empty:
        gross_cap = args.max_gross
        if gross_cap is None:
            try:
                gross_cap = (
                    float(cfg.get("risk", {}).get("max_gross"))
                    if cfg.get("risk", {})
                    else None
                )
            except Exception:
                gross_cap = None
        if gross_cap is not None and float(gross_cap) > 0:
            # Estimate current gross using ref_price for existing positions
            cur_notional = 0.0
            try:
                if os.path.exists(args.positions_csv):
                    cur_pos = pd.read_csv(args.positions_csv)
                    cur_pos["symbol"] = cur_pos["symbol"].astype(str).str.upper()
                    # Map prices for current positions from 'day' or fallback from picks map
                    price_map = (
                        pd.concat(
                            [
                                day[["symbol", entry_col]].rename(
                                    columns={entry_col: "px"}
                                ),
                                picks[["symbol", "ref_price"]].rename(
                                    columns={"ref_price": "px"}
                                ),
                            ]
                        )
                        .dropna()
                        .drop_duplicates("symbol")
                    )
                    cur_pos = cur_pos.merge(price_map, on="symbol", how="left")
                    cur_pos["px"] = cur_pos["px"].fillna(0.0)
                    cur_notional = float(
                        (
                            cur_pos["shares"].astype(float)
                            * cur_pos["px"].astype(float)
                        ).sum()
                    )
            except Exception:
                cur_notional = 0.0
            new_notional = float(
                (
                    new_entries["shares"].astype(float)
                    * new_entries["ref_price"].astype(float)
                ).sum()
            )
            cap_abs = float(gross_cap) * float(args.account_equity)
            avail = max(0.0, cap_abs - cur_notional)
            if new_notional > avail and new_notional > 0:
                scale = avail / new_notional
                # Scale shares proportionally and floor to integers
                new_entries["shares"] = (
                    new_entries["shares"].astype(float) * scale
                ).apply(lambda x: int(max(0, int(x))))
                new_entries = new_entries[new_entries["shares"] > 0]

    # Append to positions CSV
    if not new_entries.empty:
        os.makedirs(os.path.dirname(args.positions_csv), exist_ok=True)
        out = new_entries[
            ["symbol", "ref_price", "shares", "atr_pct_14", "meta_prob"]
        ].copy()
        out = out.rename(columns={"ref_price": "entry_price"})
        out.insert(1, "entry_date", target_date.date())
        # Initialize stop price from ATR and stop mult
        try:
            sm = float(args.stop_atr_mult)
            out["stop_price"] = out["entry_price"] * (
                1.0 - sm * out["atr_pct_14"].astype(float).clip(lower=0.0).fillna(0.0)
            )
        except Exception:
            out["stop_price"] = out["entry_price"] * 0.98
        out["peak_close"] = out["entry_price"]
        out["stop_date"] = target_date.date()
        out["atr_pct_entry"] = out.get("atr_pct_14", 0.0).astype(float)
        out["meta_prob_entry"] = out.get("meta_prob", 0.5).astype(float)
        out.drop(columns=["meta_prob"], inplace=True, errors="ignore")
        out = out.drop(columns=["atr_pct_14"]) if "atr_pct_14" in out.columns else out
        if os.path.exists(args.positions_csv):
            out.to_csv(args.positions_csv, mode="a", header=False, index=False)
        else:
            out.to_csv(args.positions_csv, index=False)
        print(f"[entry] appended positions -> {args.positions_csv} rows={len(out)}")
        # Update per-session guard
        try:
            cur = [str(s).upper() for s in out["symbol"].astype(str).tolist()]
            updated = sorted(list(set(session_state.get(sess_key, [])) | set(cur)))
            session_state[sess_key] = updated
            with open(session_file, "w", encoding="utf-8") as fh:
                fh.write(_json.dumps(session_state))
        except Exception:
            pass

        # Summarize all open positions after the update.
        try:
            pos_all = pd.read_csv(args.positions_csv)
            if not pos_all.empty:
                pos_all["symbol"] = pos_all["symbol"].astype(str).str.upper()
                # Best-effort mark price using today's panel or computed reference prices.
                price_frames = []
                try:
                    price_frames.append(
                        day[["symbol", entry_col]].rename(
                            columns={entry_col: "mark_price"}
                        )
                    )
                except Exception:
                    pass
                try:
                    price_frames.append(
                        picks[["symbol", "ref_price"]].rename(
                            columns={"ref_price": "mark_price"}
                        )
                    )
                except Exception:
                    pass
                mark_map = (
                    pd.concat(price_frames, axis=0, ignore_index=True)
                    if price_frames
                    else pd.DataFrame(columns=["symbol", "mark_price"])
                )
                mark_map = mark_map.dropna().drop_duplicates("symbol")
                pos_all = pos_all.merge(mark_map, on="symbol", how="left")
                pos_all["mark_price"] = pos_all["mark_price"].astype(float)
                pos_all["mark_price"] = pos_all["mark_price"].where(
                    pos_all["mark_price"].notna(), pos_all["entry_price"].astype(float)
                )
                shares = pos_all.get(
                    "shares", pd.Series(0, index=pos_all.index)
                ).astype(float)
                entry_px = pos_all["entry_price"].astype(float)
                mark_px = pos_all["mark_price"].astype(float)
                pos_all["pnl_usd"] = (mark_px - entry_px) * shares
                with np.errstate(divide="ignore", invalid="ignore"):
                    pos_all["pnl_pct"] = np.where(
                        entry_px != 0, (mark_px / entry_px - 1.0) * 100.0, np.nan
                    )
                # Build readable table.
                summary_cols = [
                    "symbol",
                    "entry_date",
                    "entry_price",
                    "mark_price",
                    "shares",
                    "stop_price",
                    "pnl_pct",
                    "pnl_usd",
                ]
                summary = pos_all[
                    [c for c in summary_cols if c in pos_all.columns]
                ].copy()
                fmt = {
                    "entry_price": lambda v: f"{float(v):.2f}",
                    "mark_price": lambda v: f"{float(v):.2f}",
                    "stop_price": lambda v: f"{float(v):.2f}",
                    "pnl_pct": lambda v: ("--" if pd.isna(v) else f"{float(v):+.2f}%"),
                    "pnl_usd": lambda v: f"{float(v):+.2f}",
                }
                print("[entry] open positions summary:")
                print(summary.to_string(index=False, formatters=fmt))
        except Exception as exc:
            print(f"[entry] warning: failed to render positions summary: {exc}")
        # Optional Discord notification to trade webhook
        hook = args.discord_webhook
        if not hook and isinstance(cfg.get("paper", {}), dict):
            try:
                hook = str(cfg.get("paper", {}).get("discord_webhook", ""))
            except Exception:
                hook = ""
        if hook and bool(args.send_discord):
            try:
                from ..infra.notify import send_discord

                lines = ["Trade Entry"]
                for _, r in out.iterrows():
                    sym = str(r.symbol).upper()
                    px = float(r.entry_price)
                    st = float(r.stop_price)
                    sh = int(r.shares)
                    lines.append(
                        f"${sym} • **Entry ${px:.2f}** • **Stop ${st:.2f}** • Shares {sh}"
                    )
                send_discord(hook, "@everyone\n" + "\n".join(lines))
                print("[entry] sent Discord trade entry notification")
            except Exception as e:
                print(f"[entry] discord send failed: {e}")
    else:
        print("[entry] no new entries.")


if __name__ == "__main__":
    main()
