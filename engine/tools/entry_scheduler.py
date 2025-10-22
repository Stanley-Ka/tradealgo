"""Entry scheduler: triggers entry_loop at configured times during market hours.

Similar to real_time_alert, but calls entry_loop with pass-through args.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd

from ..infra.market_time import DEFAULT_CAL, is_open, today_session


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Schedule entry_loop during US market hours"
    )
    p.add_argument(
        "--calendar",
        type=str,
        default=DEFAULT_CAL,
        help="Market calendar: NASDAQ or NYSE",
    )
    p.add_argument("--poll", type=int, default=15, help="Seconds between checks")
    p.add_argument(
        "--times",
        type=str,
        default="09:35,15:55",
        help="Comma-separated HH:MM (local) trigger times",
    )
    p.add_argument(
        "--state",
        type=str,
        default="data/entries/entry_log.parquet",
        help="Log of trigger times to dedupe",
    )
    # Pass-through for entry_loop
    p.add_argument("--config", type=str, default="")
    p.add_argument("--features", type=str, default="")
    p.add_argument("--model-pkl", type=str, default="")
    p.add_argument("--universe-file", type=str, default="")
    p.add_argument("--oof", type=str, default="")
    p.add_argument("--calibrators-pkl", type=str, default="")
    p.add_argument("--meta-calibrator-pkl", type=str, default="")
    p.add_argument("--news-sentiment", type=str, default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--entry-threshold", type=float, default=0.0)
    p.add_argument("--confirmations", type=int, default=1)
    p.add_argument("--state-csv", type=str, default="data/state/entry_state.csv")
    p.add_argument("--sector-map-csv", type=str, default="")
    p.add_argument("--sector-cap", type=int, default=0)
    p.add_argument("--positions-csv", type=str, default="data/paper/positions.csv")
    p.add_argument("--entry-price", choices=["close", "open"], default="close")
    p.add_argument("--account-equity", type=float, default=100000.0)
    p.add_argument("--risk-mode", choices=["fixed", "auto"], default="auto")
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--risk-min-pct", type=float, default=0.002)
    p.add_argument("--risk-max-pct", type=float, default=0.006)
    p.add_argument("--risk-curve", choices=["linear", "quadratic"], default="quadratic")
    p.add_argument("--stop-atr-mult", type=float, default=1.0)
    p.add_argument("--min-adv-usd", type=float, default=1e7)
    p.add_argument("--max-atr-pct", type=float, default=0.05)
    p.add_argument("--decision-log-csv", type=str, default="data/paper/entry_log.csv")
    p.add_argument(
        "--recommendations-csv",
        type=str,
        default="data/alerts/recommendations.csv",
        help="Alerts log to drive dynamic universe",
    )
    p.add_argument(
        "--recommendation-lookback-mins",
        type=int,
        default=240,
        help="Only consider recommendations within this many minutes",
    )
    p.add_argument(
        "--discord-webhook",
        type=str,
        default=os.environ.get(
            "DISCORD_TRADES_WEBHOOK_URL", os.environ.get("DISCORD_WEBHOOK_URL", "")
        ),
        help="Optional Discord webhook override for entry notifications",
    )
    p.add_argument(
        "--send-discord",
        action="store_true",
        help="Forward Discord notifications to entry loop when set",
    )
    # Price gating (pass-through to entry_loop)
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
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Append plain-text audit logs to this file",
    )
    return p.parse_args(argv)


def _load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return pd.DataFrame(columns=["date", "time", "ran_at", "notes"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _save_log(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _recent_recommendations(path: str, lookback_mins: int) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "symbol" not in df.columns:
        return []
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if lookback_mins > 0 and "alert_ts" in df.columns:
        ts = pd.to_datetime(df["alert_ts"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(
            minutes=max(lookback_mins, 1)
        )
        mask = ts.notna() & (ts >= cutoff)
        df = df[mask]
    return sorted(df["symbol"].dropna().unique().tolist())


def _trigger_once(args: argparse.Namespace) -> None:
    from .entry_loop import main as entry_main

    rec_syms: List[str] = []
    temp_universe: Optional[tempfile.NamedTemporaryFile] = None
    if args.recommendations_csv:
        try:
            rec_syms = _recent_recommendations(
                args.recommendations_csv, int(max(0, args.recommendation_lookback_mins))
            )
        except Exception:
            rec_syms = []
    universe_override = None
    if rec_syms:
        temp_universe = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
        for sym in rec_syms:
            temp_universe.write(sym + "\n")
        temp_universe.flush()
        universe_override = temp_universe.name
        print(f"[entry-sched] using {len(rec_syms)} symbols from recommendations")

    eargs: List[str] = []
    # If provided via CLI, pass through; otherwise rely on YAML config
    if args.features:
        eargs += ["--features", args.features]
    if args.model_pkl:
        eargs += ["--model-pkl", args.model_pkl]
    universe_path = universe_override or args.universe_file
    if universe_path:
        eargs += ["--universe-file", universe_path]
    eargs += [
        "--top-k",
        str(args.top_k),
        "--entry-price",
        args.entry_price,
        "--account-equity",
        str(args.account_equity),
        "--risk-mode",
        args.risk_mode,
        "--risk-pct",
        str(args.risk_pct),
        "--risk-min-pct",
        str(args.risk_min_pct),
        "--risk-max-pct",
        str(args.risk_max_pct),
        "--risk-curve",
        args.risk_curve,
        "--stop-atr-mult",
        str(args.stop_atr_mult),
        "--min-adv-usd",
        str(args.min_adv_usd),
        "--max-atr-pct",
        str(args.max_atr_pct),
        "--confirmations",
        str(args.confirmations),
        "--state-csv",
        args.state_csv,
        "--positions-csv",
        args.positions_csv,
        "--decision-log-csv",
        args.decision_log_csv,
    ]
    if args.config:
        eargs = ["--config", args.config] + eargs
    if args.oof:
        eargs += ["--oof", args.oof]
    if args.calibrators_pkl:
        eargs += ["--calibrators-pkl", args.calibrators_pkl]
    if args.meta_calibrator_pkl:
        eargs += ["--meta-calibrator-pkl", args.meta_calibrator_pkl]
    if args.news_sentiment:
        eargs += ["--news-sentiment", args.news_sentiment]
    if args.entry_threshold and float(args.entry_threshold) > 0.0:
        eargs += ["--entry-threshold", str(args.entry_threshold)]
    if args.sector_map_csv:
        eargs += ["--sector-map-csv", args.sector_map_csv]
    if args.sector_cap and int(args.sector_cap) > 0:
        eargs += ["--sector-cap", str(args.sector_cap)]
    if args.min_price is not None:
        eargs += ["--min-price", str(args.min_price)]
    if args.max_price is not None:
        eargs += ["--max-price", str(args.max_price)]
    if args.discord_webhook:
        eargs += ["--discord-webhook", args.discord_webhook]
    if args.send_discord:
        eargs += ["--send-discord"]

    print(
        f"[entry-sched] triggering entry_loop top_k={args.top_k} threshold={args.entry_threshold}"
    )
    try:
        entry_main(eargs)
    finally:
        if temp_universe is not None:
            try:
                temp_universe.close()
            except Exception:
                pass
            try:
                os.unlink(temp_universe.name)
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    raw_times = (
        [t.strip() for t in args.times.split(",") if t is not None]
        if args.times
        else []
    )
    auto_mode = any((t or "").lower() == "auto" for t in raw_times) or not raw_times
    times = [t for t in raw_times if t and t.lower() != "auto"]
    log = _load_log(args.state)
    cal = args.calendar.upper()
    print(
        f"[entry-sched] calendar={cal} times={times if times else 'auto'} poll={args.poll}s"
    )
    always_fire = not times

    def _log(msg: str) -> None:
        if args.log_file:
            try:
                os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        pd.Timestamp.now()
                        .astimezone()
                        .strftime("%Y-%m-%d %H:%M:%S %Z ")
                        + msg
                        + "\n"
                    )
            except Exception:
                pass

    try:
        while True:
            now = datetime.now().astimezone()
            sess = today_session(cal)
            open_now = is_open(cal)
            print(
                f"[entry-sched] now={now.strftime('%Y-%m-%d %H:%M:%S %Z')} open={open_now}",
                end="\r",
            )
            if open_now and sess is not None:
                key_date = sess.open_ts.date()
                cur_hhmm = now.strftime("%H:%M")
                to_trigger: List[str] = []
                if times and cur_hhmm in times:
                    to_trigger.append(cur_hhmm)
                elif always_fire:
                    to_trigger.append(cur_hhmm)
                did_trigger = False
                for hhmm in to_trigger:
                    already = not log[
                        (log["date"] == key_date) & (log["time"] == hhmm)
                    ].empty
                    if already:
                        continue
                    try:
                        _trigger_once(args)
                        _log(f"entry triggered at {hhmm} top_k={args.top_k}")
                        row = pd.DataFrame(
                            {
                                "date": [key_date],
                                "time": [hhmm],
                                "ran_at": [now],
                                "notes": [
                                    f"entry_loop top_k={args.top_k} threshold={args.entry_threshold}"
                                ],
                            }
                        )
                        log = pd.concat([log, row], ignore_index=True)
                        _save_log(args.state, log)
                        did_trigger = True
                    except Exception as e:
                        emsg = f"entry trigger failed: {e}"
                        print("\n[entry-sched] " + emsg)
                        _log(emsg)
                if did_trigger:
                    time.sleep(60)
                    continue
            time.sleep(max(1, int(args.poll)))
    except KeyboardInterrupt:
        print("\n[entry-sched] stopped by user")


if __name__ == "__main__":
    main()
