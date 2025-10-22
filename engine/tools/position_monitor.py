"""Position monitor (daily prototype).

Monitors open positions and triggers exits based on:
- Stop-loss: price <= entry_price * (1 - stop_mult * ATR%) using today's ATR%.
- Probability exit: meta_prob < threshold (optional).

Inputs:
- positions CSV with columns: symbol,entry_date,entry_price[,shares]
- daily features parquet (with adj_close, atr_pct_14)
- trained meta model (and optional calibrators/meta-calibrator) to compute today's meta_prob

Outputs:
- Append exits to --exits-csv with exit_reason and realized PnL (using today's close)
- Overwrite positions CSV removing exited rows
- Optional Discord notification per exit

Note: This is a daily prototype; intraday monitoring requires an intraday data source.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional
from pathlib import Path
import warnings

# Suppress noisy numpy warnings emitted when intermediate windows are empty.
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..features.spec_news import load_sentiment as load_sentiment_file
from ..infra.yaml_config import load_yaml_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor positions and trigger daily exits")
    p.add_argument(
        "--config", type=str, default="", help="YAML with paths/specialists/risk"
    )
    p.add_argument("--features", required=True, help="Features parquet with daily rows")
    p.add_argument(
        "--positions-csv",
        required=True,
        help="CSV of open positions (symbol,entry_date,entry_price[,shares])",
    )
    p.add_argument("--exits-csv", type=str, default="data/paper/exits.csv")
    p.add_argument(
        "--model-pkl", required=True, help="Meta model pickle for meta_prob computation"
    )
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators (from run_cv)",
    )
    p.add_argument(
        "--oof",
        type=str,
        default="",
        help="OOF parquet to fit calibrators if calibrators-pkl not provided",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta-level calibrator for meta_prob",
    )
    p.add_argument(
        "--news-sentiment",
        type=str,
        default="",
        help="Optional sentiment file for spec_nlp",
    )
    # Exit policy
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=1.0,
        help="Stop distance in ATR× (uses today's ATR%)",
    )
    p.add_argument(
        "--prob-exit-thresh",
        type=float,
        default=0.45,
        help="Exit if meta_prob < threshold",
    )
    p.add_argument(
        "--prob-exit-consecutive",
        type=int,
        default=1,
        help="Require N consecutive days below threshold before exit",
    )
    p.add_argument(
        "--monitor-state-csv",
        type=str,
        default="data/state/monitor_state.csv",
        help="Persist counters for prob-exit hysteresis",
    )
    p.add_argument(
        "--take-profit-atr-mult",
        type=float,
        default=1.5,
        help="Take-profit distance in ATR× (0 disables)",
    )
    p.add_argument(
        "--trail-atr-mult",
        type=float,
        default=0.0,
        help="Trailing stop ATR× multiplier (0 disables)",
    )
    p.add_argument(
        "--intraday-dir",
        type=str,
        default="",
        help="Optional directory with intraday minute bars (per-symbol parquet)",
    )
    # Notifications
    p.add_argument(
        "--discord-webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    return p.parse_args(argv)


def _load_calibrators(path: str, oof_path: str, kind: str = "platt") -> dict:
    if path:
        import pickle

        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload.get("models", {})
    if oof_path:
        from ..models.calib_utils import fit_per_specialist_calibrators_from_oof

        oof = pd.read_parquet(oof_path)
        return fit_per_specialist_calibrators_from_oof(oof, kind)
    return {}


def _apply_spec_probs(specs: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    from ..models.calib_utils import (
        apply_calibrator as _apply,
        naive_prob_map as _naive,
    )

    out = specs.copy()
    for sc in [
        c for c in out.columns if c.startswith("spec_") and not c.endswith("_prob")
    ]:
        raw = out[sc].astype(float).values
        if calibrators and sc in calibrators:
            prob = _apply(calibrators[sc], raw)
        else:
            prob = _naive(raw)
        out[f"{sc}_prob"] = prob
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config) if args.config else {}
    f = pd.read_parquet(args.features)
    f["date"] = pd.to_datetime(f["date"])  # ensure dtype
    f["symbol"] = f["symbol"].astype(str).str.upper()
    today = f["date"].max()

    if not os.path.exists(args.positions_csv):
        print(f"[monitor] positions file not found: {args.positions_csv}")
        return
    pos = pd.read_csv(args.positions_csv)
    if pos.empty:
        print("[monitor] no open positions.")
        return
    for col in ("symbol", "entry_date", "entry_price"):
        if col not in pos.columns:
            raise RuntimeError(
                "positions CSV must include symbol,entry_date,entry_price"
            )
    pos["symbol"] = pos["symbol"].astype(str).str.upper()
    pos["entry_date"] = pd.to_datetime(pos["entry_date"]).dt.tz_localize(None)
    if "peak_close" not in pos.columns:
        pos["peak_close"] = pos["entry_price"].astype(float)
    else:
        pos["peak_close"] = pd.to_numeric(pos["peak_close"], errors="coerce").fillna(
            pos["entry_price"].astype(float)
        )
    if "stop_date" not in pos.columns:
        pos["stop_date"] = pos["entry_date"].dt.date
    else:
        pos["stop_date"] = pd.to_datetime(
            pos["stop_date"], errors="coerce"
        ).dt.date.fillna(pos["entry_date"].dt.date)

    # Prepare today's features for held symbols
    held = pos["symbol"].unique().tolist()
    day = f[(f["date"] == today) & (f["symbol"].isin(held))].copy()
    if day.empty:
        print(f"[monitor] no feature rows for today {today.date()} for held symbols.")
        return

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
    calibrators = _load_calibrators(
        args.calibrators_pkl or cfg.get("calibration", {}).get("calibrators_pkl", ""),
        args.oof or cfg.get("paths", {}).get("oof", ""),
        cfg.get("calibration", {}).get("kind", "platt"),
    )
    specs = _apply_spec_probs(specs, calibrators)

    # Meta prob
    import pickle

    with open(
        args.model_pkl or cfg.get("paths", {}).get("meta_model", ""), "rb"
    ) as fpk:
        meta = pickle.load(fpk)
    clf = meta.get("model")
    feature_names = meta.get("features")
    scaler = meta.get("scaler")
    if not feature_names:
        feature_names = [c for c in specs.columns if c.endswith("_prob")]
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
    if args.meta_calibrator_pkl or cfg.get("calibration", {}).get(
        "meta_calibrator_pkl"
    ):
        try:
            with open(
                args.meta_calibrator_pkl
                or cfg.get("calibration", {}).get("meta_calibrator_pkl"),
                "rb",
            ) as fpk:
                payload = pickle.load(fpk)
            mdl = payload.get("model", payload)
            if hasattr(mdl, "predict_proba"):
                meta_prob = mdl.predict_proba(meta_prob.reshape(-1, 1))[:, 1]
            elif hasattr(mdl, "transform"):
                meta_prob = mdl.transform(meta_prob)
        except Exception:
            pass
    specs["meta_prob"] = meta_prob

    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
    tp_mult_cfg = risk_cfg.get(
        "take_profit_atr_mult", risk_cfg.get("tp1_atr_mult", args.take_profit_atr_mult)
    )
    try:
        tp_mult = (
            float(tp_mult_cfg)
            if tp_mult_cfg is not None
            else float(args.take_profit_atr_mult)
        )
    except Exception:
        tp_mult = float(args.take_profit_atr_mult)
    if tp_mult < 0:
        tp_mult = 0.0
    trail_mult_cfg = risk_cfg.get(
        "trail_atr_mult", risk_cfg.get("trail_stop_mult", args.trail_atr_mult)
    )
    try:
        trail_mult = (
            float(trail_mult_cfg)
            if trail_mult_cfg is not None
            else float(args.trail_atr_mult)
        )
    except Exception:
        trail_mult = float(args.trail_atr_mult)
    if trail_mult < 0:
        trail_mult = 0.0

    intraday_dir_path = (
        Path(args.intraday_dir).expanduser() if args.intraday_dir else None
    )
    if intraday_dir_path and not intraday_dir_path.exists():
        intraday_dir_path = None
    day_start = pd.Timestamp(today).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    intraday_cache: dict[str, Optional[pd.DataFrame]] = {}

    def _get_intraday_slice(symbol: str) -> Optional[pd.DataFrame]:
        if intraday_dir_path is None:
            return None
        key = str(symbol).upper()
        if key not in intraday_cache:
            path = intraday_dir_path / f"{key}.parquet"
            if not path.exists():
                intraday_cache[key] = None
            else:
                try:
                    df_local = pd.read_parquet(path)
                    df_local["ts"] = pd.to_datetime(
                        df_local["ts"]
                    )  # ensure datetime dtype
                    intraday_cache[key] = df_local
                except Exception:
                    intraday_cache[key] = None
        df_local = intraday_cache[key]
        if df_local is None or df_local.empty:
            return None
        slice_df = df_local[(df_local["ts"] >= day_start) & (df_local["ts"] < day_end)]
        return slice_df if not slice_df.empty else None

    # Join entry info
    cur = specs.merge(pos, on="symbol", how="left")
    cur["meta_prob_entry"] = pd.to_numeric(cur.get("meta_prob_entry"), errors="coerce")
    cur["atr_pct_entry"] = pd.to_numeric(cur.get("atr_pct_entry"), errors="coerce")
    cur["atr_pct_14"] = (
        cur["atr_pct_14"].astype(float).fillna(cur["atr_pct_entry"]).fillna(0.0)
    )
    cur["meta_prob"] = cur["meta_prob"].fillna(cur["meta_prob_entry"]).fillna(0.5)
    cur["adj_close"] = cur["adj_close"].astype(float)
    cur["adj_high"] = cur.get("adj_high", cur["adj_close"]).astype(float)
    cur["entry_price"] = cur["entry_price"].astype(float)
    cur["peak_close"] = pd.to_numeric(cur.get("peak_close"), errors="coerce").fillna(
        cur["entry_price"]
    )
    cur["stop_date"] = pd.to_datetime(
        cur.get("stop_date", cur["entry_date"]), errors="coerce"
    ).dt.date
    cur["days_held"] = (
        today - pd.to_datetime(cur["entry_date"]).dt.tz_localize(None)
    ).dt.days.clip(lower=0)

    # Exit checks with hysteresis for probability exit
    stop_mult = float(cfg.get("risk", {}).get("stop_atr_mult", args.stop_atr_mult))
    stored_stop = pd.to_numeric(cur.get("stop_price"), errors="coerce")
    # Ensure a Series for downstream operations even if stop_price column is missing
    if not isinstance(stored_stop, pd.Series):
        stored_stop = pd.Series(np.nan, index=cur.index)
    atr_component = cur["atr_pct_14"].clip(lower=0.0).fillna(0.0)
    dynamic_stop = cur["entry_price"] * (1.0 - stop_mult * atr_component)
    # Trailing stop using peak close
    cur["peak_close"] = np.maximum(cur["peak_close"], cur["adj_close"])
    if trail_mult > 0:
        trail_stop = cur["peak_close"] * (1.0 - trail_mult * atr_component)
        dynamic_stop = np.maximum(dynamic_stop, trail_stop)
    # Robust check works for all-NaN Series
    if pd.isna(stored_stop).all():
        cur["stop_price_today"] = dynamic_stop
        cur["stop_date_today"] = cur["stop_date"]
    else:
        cur["stop_price_today"] = np.maximum(
            stored_stop.fillna(dynamic_stop), dynamic_stop
        )
        cur["stop_date_today"] = np.where(
            np.isclose(cur["stop_price_today"], stored_stop, rtol=1e-6, atol=1e-6),
            cur["stop_date"],
            today.date(),
        )
    cur["hit_stop"] = cur["adj_close"] <= cur["stop_price_today"]
    cur["hit_prob_raw"] = cur["meta_prob"] < float(args.prob_exit_thresh)

    # Load and update counters
    def _load_counts(path: str) -> dict[str, int]:
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path)
            return {
                str(r.symbol).upper(): int(r.get("consec", 0)) for _, r in df.iterrows()
            }
        except Exception:
            return {}

    def _save_counts(path: str, counts: dict[str, int]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(
            {"symbol": list(counts.keys()), "consec": list(counts.values())}
        )
        df.to_csv(path, index=False)

    counts = _load_counts(args.monitor_state_csv)
    new_counts: dict[str, int] = {}
    for _, r in cur.iterrows():
        sym = str(r["symbol"]).upper()
        if bool(r["hit_prob_raw"]):
            new_counts[sym] = counts.get(sym, 0) + 1
        else:
            new_counts[sym] = 0
    _save_counts(args.monitor_state_csv, new_counts)
    cur["hit_prob"] = cur["symbol"].map(
        lambda s: new_counts.get(str(s).upper(), 0)
    ) >= int(args.prob_exit_consecutive)

    cur["tp_price"] = np.nan
    cur["tp_fill_price"] = np.nan
    cur["hit_tp"] = False
    if tp_mult > 0:
        meta = cur["meta_prob"].astype(float).clip(0.0, 1.0)
        momentum_factor = 1.0 + np.clip(meta - 0.55, -0.25, 0.35) * 2.5
        time_factor = np.maximum(0.5, 1.0 - (cur["days_held"] / 90.0))
        tp_effective = tp_mult * momentum_factor * time_factor
        tp_effective = np.maximum(tp_mult * 0.4, tp_effective)
        atr_base = atr_component
        cur["tp_price"] = cur["entry_price"] * (1.0 + tp_effective * atr_base)
        for idx, row in cur.iterrows():
            target = float(row.get("tp_price", np.nan))
            if not np.isfinite(target) or target <= 0:
                continue
            sym = str(row["symbol"]).upper()
            intraday_slice = _get_intraday_slice(sym)
            hit_price = None
            if intraday_slice is not None and not intraday_slice.empty:
                hit_bars = intraday_slice[intraday_slice["high"] >= target]
                if not hit_bars.empty:
                    hit_price = float(hit_bars.iloc[0]["high"])
            else:
                day_high = row.get("adj_high", np.nan)
                if np.isfinite(day_high) and day_high >= target:
                    hit_price = float(day_high)
            if hit_price is not None:
                cur.at[idx, "hit_tp"] = True
                cur.at[idx, "tp_fill_price"] = hit_price

    exits = cur[(cur["hit_stop"]) | (cur["hit_prob"]) | (cur["hit_tp"])].copy()
    if exits.empty:
        print("[monitor] no exits today.")
        return
    exits["exit_reason"] = np.where(
        exits["hit_tp"], "tp", np.where(exits["hit_stop"], "stop", "prob")
    )
    exits["exit_date"] = today
    exits["exit_price"] = np.where(
        exits["exit_reason"] == "tp",
        exits["tp_fill_price"].astype(float).fillna(exits["adj_close"].astype(float)),
        exits["adj_close"].astype(float),
    )
    exits["pnl"] = (exits["exit_price"] - exits["entry_price"]) * exits.get(
        "shares", pd.Series(1, index=exits.index)
    ).astype(float)

    # Append to exits CSV
    os.makedirs(os.path.dirname(args.exits_csv), exist_ok=True)
    if os.path.exists(args.exits_csv):
        exits.to_csv(args.exits_csv, mode="a", header=False, index=False)
    else:
        exits.to_csv(args.exits_csv, index=False)
    print(f"[monitor] exits -> {args.exits_csv} rows={len(exits)}")

    # Remove exited from positions CSV
    exit_symbols = exits["symbol"].unique().tolist()
    remaining = pos[~pos["symbol"].isin(exit_symbols)].copy()
    if not remaining.empty:
        stay_updates = cur[~cur["symbol"].isin(exit_symbols)][
            ["symbol", "stop_price_today", "peak_close"]
        ].rename(
            columns={
                "stop_price_today": "stop_price_update",
                "peak_close": "peak_close_update",
            }
        )
        remaining = remaining.merge(stay_updates, on="symbol", how="left")
        if "stop_price_update" in remaining.columns:
            remaining["stop_price"] = remaining["stop_price_update"].combine_first(
                remaining.get("stop_price")
            )
            remaining.drop(columns=["stop_price_update"], inplace=True)
        if "peak_close_update" in remaining.columns:
            remaining["peak_close"] = remaining["peak_close_update"].combine_first(
                remaining.get("peak_close")
            )
            remaining.drop(columns=["peak_close_update"], inplace=True)
        remaining["peak_close"] = (
            remaining["peak_close"]
            .astype(float)
            .fillna(remaining["entry_price"].astype(float))
        )
        remaining["stop_price"] = pd.to_numeric(
            remaining.get("stop_price"), errors="coerce"
        )
    remaining.to_csv(args.positions_csv, index=False)
    print(f"[monitor] updated positions -> {args.positions_csv} open={len(remaining)}")

    # Optional Discord notification
    # Avoid sending notifications during tests unless explicitly desired
    if args.discord_webhook and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from ..infra.notify import send_discord

            lines: List[str] = [f"Exits — {today.date()}"]
            for _, r in exits.iterrows():
                lines.append(
                    f"- {r['symbol']}: {r['exit_reason']}  price=${float(r['exit_price']):.2f}  meta={float(r['meta_prob']):.3f}  ATR%={100*float(r['atr_pct_14']):.2f}"
                )
            send_discord(args.discord_webhook, "@everyone\n" + "\n".join(lines))
        except Exception:
            pass


if __name__ == "__main__":
    main()
