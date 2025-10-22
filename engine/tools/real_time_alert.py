"""Real-time alert scheduler for US market hours.

Runs during market open and triggers the existing trade_alert at configured
clock times (e.g., 09:35, 15:55). Uses daily features (latest row), so alerts
are based on yesterday's EOD features unless you update features intraday.

Deduplicates alerts per date+time, and streams alerts to Discord via the
trade_alert tool (sizing supported via flags/YAML).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from ..infra.market_time import DEFAULT_CAL, is_open, today_session
from .entry_loop import main as entry_main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time alert scheduler during US market hours"
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
        help="Comma-separated HH:MM (local) times to trigger alerts while market is open",
    )
    p.add_argument(
        "--state",
        type=str,
        default="data/alerts/alert_log.parquet",
        help="Log to dedupe alerts",
    )
    # Interval/anchors (optional alternatives to --times)
    p.add_argument(
        "--every-minutes",
        type=int,
        default=0,
        help="If >0, trigger every N minutes during open session",
    )
    p.add_argument(
        "--at-open",
        action="store_true",
        help="Trigger once after market open (use --open-offset-mins)",
    )
    p.add_argument(
        "--open-offset-mins", type=int, default=5, help="Delay after open for --at-open"
    )
    p.add_argument(
        "--before-close-mins",
        type=int,
        default=0,
        help="If >0, trigger once N minutes before close",
    )
    # Pass-through to trade_alert
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--features", type=str, default="")
    p.add_argument("--model-pkl", type=str, default="")
    p.add_argument("--oof", type=str, default="")
    p.add_argument(
        "--calibrators-pkl",
        type=str,
        default="",
        help="Per-specialist calibrators pickle (forwarded to trade_alert)",
    )
    p.add_argument(
        "--meta-calibrator-pkl",
        type=str,
        default="",
        help="Optional meta calibrator pickle (forwarded to trade_alert/entry loop)",
    )
    p.add_argument("--universe-file", type=str, default="")
    p.add_argument("--provider", type=str, default="polygon")
    p.add_argument("--top-k", type=int, default=1)
    # Prefer dedicated alerts webhook if set
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
        "--alert-log-csv",
        type=str,
        default="",
        help="Optional CSV diagnostics passed to trade_alert",
    )
    p.add_argument(
        "--min-repeat-mins",
        type=int,
        default=60,
        help="Throttle duplicate alerts for identical symbol sets within N minutes",
    )
    p.add_argument(
        "--positions-csv",
        type=str,
        default="data/paper/positions.csv",
        help="Positions CSV to gate repeat paper trade alerts",
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
    # Sizing helpers
    p.add_argument("--account-equity", type=float, default=100000.0)
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--stop-atr-mult", type=float, default=1.0)
    p.add_argument("--entry-price", choices=["close", "open"], default="close")
    # Risk gates (optional overrides)
    p.add_argument("--min-adv-usd", type=float, default=None)
    p.add_argument("--max-atr-pct", type=float, default=None)
    p.add_argument(
        "--heartbeat",
        action="store_true",
        help="Send a small Discord heartbeat when no candidates",
    )
    # Optional: rebuild watchlist each trigger from current features/model
    p.add_argument(
        "--rebuild-watchlist",
        action="store_true",
        help="Rebuild a watchlist at each trigger using current features",
    )
    p.add_argument(
        "--watchlist-out", type=str, default="engine/data/universe/watchlist_rt.txt"
    )
    p.add_argument("--watch-topk", type=int, default=500)
    p.add_argument("--watch-min-price", type=float, default=None)
    p.add_argument("--watch-min-adv", type=float, default=None)
    p.add_argument(
        "--watch-bucket-adv",
        action="store_true",
        help="Diversify watchlist by ADV buckets",
    )
    p.add_argument("--watch-buckets", type=int, default=3)
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Append plain-text audit logs to this file",
    )
    # Optional pricing source pass-through for parity with trade_alert
    p.add_argument(
        "--price-source",
        choices=["feature", "live"],
        default=None,
        help="Use feature price or fetch live price (forwarded to trade_alert)",
    )
    p.add_argument(
        "--live-provider",
        choices=["yahoo", "polygon"],
        default=None,
        help="Provider for live price when --price-source=live",
    )
    p.add_argument(
        "--polygon-plan",
        choices=["auto", "starter", "pro", "enterprise"],
        default=None,
        help="Hint Polygon plan to skip unauthorized endpoints in trade_alert",
    )
    # Intraday mix (pass-through to trade_alert)
    p.add_argument(
        "--mix-intraday",
        type=float,
        default=None,
        help="Weight [0,1] for intraday blend; 0 disables",
    )
    p.add_argument(
        "--intraday-features",
        type=str,
        default=None,
        help="Latest intraday features snapshot parquet",
    )
    p.add_argument(
        "--mix-k",
        type=float,
        default=None,
        help="Logistic slope for converting intraday score to prob",
    )
    # Signal-driven mode (optional)
    p.add_argument(
        "--signal-threshold",
        type=float,
        default=None,
        help="If set, trigger alerts only when the top score >= threshold (disables clock schedule)",
    )
    p.add_argument(
        "--signal-use-median",
        action="store_true",
        help="Raise effective signal threshold to the median score when multiple symbols clear the base threshold",
    )
    p.add_argument(
        "--signal-min-delta",
        type=float,
        default=0.015,
        help="Minimum score increase vs last alert to re-trigger (unless cooldown expired)",
    )
    p.add_argument(
        "--signal-cooldown-mins",
        type=int,
        default=30,
        help="Cooldown window per symbol between alerts",
    )
    p.add_argument(
        "--signal-topk",
        type=int,
        default=1,
        help="Inspect the top N candidates for signal triggers",
    )
    p.add_argument(
        "--signal-state",
        type=str,
        default="data/alerts/signal_state.json",
        help="JSON file to persist last signal timestamps",
    )
    # Optional exploration knobs (forwarded)
    p.add_argument("--explore-prob", type=float, default=None)
    p.add_argument("--explore-topn", type=int, default=None)
    p.add_argument(
        "--explore-universe-file",
        type=str,
        default="",
        help="Optional broader universe file for exploratory scans",
    )
    p.add_argument(
        "--explore-top-k",
        type=int,
        default=1,
        help="Exploratory picks per trigger when explore universe is set",
    )
    p.add_argument(
        "--recommendations-csv",
        type=str,
        default="data/alerts/recommendations.csv",
        help="CSV log of alerted symbols for dedupe/entry scheduling",
    )
    # Cooldown to reduce repeats
    p.add_argument(
        "--cooldown-mins",
        type=int,
        default=120,
        help="Do not repeat alerted symbols within N minutes",
    )
    p.add_argument(
        "--cooldown-state",
        type=str,
        default="data/alerts/cooldown.json",
        help="Path to cooldown state JSON",
    )
    # Optional update message and demotion policy
    p.add_argument(
        "--send-updates",
        action="store_true",
        help="After trigger, send compact delta vs last picks",
    )
    p.add_argument(
        "--updates-state",
        type=str,
        default="data/alerts/last_picks.json",
        help="State file to diff picks across runs",
    )
    p.add_argument(
        "--demote-below-prob",
        type=float,
        default=None,
        help="If set, demote kept symbols whose prob falls below this",
    )
    p.add_argument(
        "--demote-cooldown-mins",
        type=int,
        default=60,
        help="Cooldown minutes for demoted symbols",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Trigger a single alert immediately (ignores market hours/time)",
    )
    p.add_argument(
        "--initial-watchlist-threshold",
        type=float,
        default=None,
        help="If set (>0), send startup watchlist alert for symbols meeting this probability",
    )
    p.add_argument(
        "--initial-watchlist-topk",
        type=int,
        default=25,
        help="Maximum symbols to include in the startup watchlist alert",
    )
    return p.parse_args(argv)


def _load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return pd.DataFrame(columns=["date", "time", "sent_at", "notes"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _save_log(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _trigger_once(
    args: argparse.Namespace, *, dry_run: bool = False
) -> Optional[pd.DataFrame]:
    # Call trade_alert as a module in-process
    from .trade_alert import main as alert_main

    # Load YAML (if provided) to derive defaults and fill missing paths
    cfg = {}
    if args.config:
        try:
            from ..infra.yaml_config import load_yaml_config

            cfg = load_yaml_config(args.config)
        except Exception:
            cfg = {}
    # Fill missing core paths from YAML if not provided via CLI
    if not args.features:
        try:
            args.features = str(cfg.get("paths", {}).get("features", ""))
        except Exception:
            args.features = ""
    if not args.model_pkl:
        try:
            args.model_pkl = str(cfg.get("paths", {}).get("meta_model", ""))
        except Exception:
            args.model_pkl = ""
    if not args.universe_file:
        try:
            args.universe_file = str(cfg.get("alert", {}).get("universe_file", ""))
        except Exception:
            args.universe_file = ""
    if args.signal_threshold is None:
        try:
            st = cfg.get("alert", {}).get("signal_threshold")
            if st is not None and st != "":
                args.signal_threshold = float(st)
        except Exception:
            pass
    # Optional on-the-fly watchlist rebuild
    universe_file = args.universe_file
    if bool(args.rebuild_watchlist):
        try:
            from .build_watchlist import main as wl_main

            wl_args: list[str] = [
                "--features",
                args.features,
                "--model-pkl",
                args.model_pkl,
                "--out",
                args.watchlist_out,
                "--top-k",
                str(int(args.watch_topk)),
            ]
            # Derive defaults from YAML if not explicitly set
            wl_min_price = args.watch_min_price
            if wl_min_price is None:
                try:
                    wl_min_price = float(cfg.get("alert", {}).get("min_price"))
                except Exception:
                    wl_min_price = None
            wl_min_adv = args.watch_min_adv
            if wl_min_adv is None:
                try:
                    wl_min_adv = float(cfg.get("risk", {}).get("min_adv_usd"))
                except Exception:
                    wl_min_adv = None
            if wl_min_price is not None:
                wl_args += ["--min-price", str(wl_min_price)]
            if wl_min_adv is not None:
                wl_args += ["--min-adv-usd", str(wl_min_adv)]
            if bool(args.watch_bucket_adv):
                wl_args += ["--bucket-adv", "--buckets", str(int(args.watch_buckets))]
            wl_main(wl_args)
            universe_file = args.watchlist_out
            print(f"[rt] rebuilt watchlist -> {universe_file}")
        except Exception as e:
            print(f"[rt] watchlist rebuild failed: {e}")

    # Build exclude list (cooldown)
    exclude_path = ""
    if int(args.cooldown_mins) > 0:
        try:
            import json as _json
            from datetime import datetime as _dt

            now = _dt.now().astimezone()
            # Load state and remove expired
            state = {}
            if os.path.exists(args.cooldown_state):
                state = (
                    _json.loads(open(args.cooldown_state, "r", encoding="utf-8").read())
                    or {}
                )
            keep = {}
            for sym, iso in state.items():
                try:
                    t = _dt.fromisoformat(iso)
                except Exception:
                    continue
                if (now - t).total_seconds() <= int(args.cooldown_mins) * 60:
                    keep[sym] = iso
            state = keep
            if state:
                exclude_path = os.path.join(os.path.dirname(args.state), "exclude.txt")
                with open(exclude_path, "w", encoding="utf-8") as f:
                    for s in state.keys():
                        f.write(str(s).upper() + "\n")
        except Exception:
            exclude_path = ""

    now_ts = datetime.now().astimezone()
    cal_code = args.calendar.upper() if hasattr(args, "calendar") else DEFAULT_CAL
    session = today_session(cal_code) if hasattr(args, "calendar") else None
    if session is not None and not isinstance(session, type(None)):
        try:
            session_open = session.open_ts
        except Exception:
            session_open = None
    else:
        session_open = None
    open_flag = is_open(cal_code) if hasattr(args, "calendar") else False
    alert_kind = "intraday"
    if not open_flag and session_open is not None:
        try:
            if now_ts < session_open:
                alert_kind = "pre-market"
        except Exception:
            pass

    base_args: List[str] = [
        "--provider",
        args.provider,
        "--entry-price",
        args.entry_price,
        "--account-equity",
        str(args.account_equity),
        "--risk-pct",
        str(args.risk_pct),
        "--stop-atr-mult",
        str(args.stop_atr_mult),
    ]
    if args.config:
        base_args = ["--config", args.config] + base_args
    if args.features:
        base_args += ["--features", args.features]
    if args.model_pkl:
        base_args += ["--model-pkl", args.model_pkl]
    if args.oof:
        base_args += ["--oof", args.oof]
    if args.calibrators_pkl:
        base_args += ["--calibrators-pkl", args.calibrators_pkl]
    if args.alert_log_csv:
        base_args += ["--alert-log-csv", args.alert_log_csv]
    if args.min_repeat_mins is not None:
        base_args += ["--min-repeat-mins", str(int(args.min_repeat_mins))]
    if args.price_source:
        base_args += ["--price-source", args.price_source]
    if args.live_provider:
        base_args += ["--live-provider", args.live_provider]
    if args.polygon_plan:
        base_args += ["--polygon-plan", args.polygon_plan]
    if args.mix_intraday is not None:
        base_args += ["--mix-intraday", str(args.mix_intraday)]
    if args.intraday_features:
        base_args += ["--intraday-features", args.intraday_features]
    if args.mix_k is not None:
        base_args += ["--mix-k", str(args.mix_k)]
    if args.min_price is not None:
        base_args += ["--min-price", str(args.min_price)]
    if args.max_price is not None:
        base_args += ["--max-price", str(args.max_price)]
    if args.min_adv_usd is not None:
        base_args += ["--min-adv-usd", str(args.min_adv_usd)]
    if args.max_atr_pct is not None:
        base_args += ["--max-atr-pct", str(args.max_atr_pct)]
    if args.explore_prob is not None:
        base_args += ["--explore-prob", str(args.explore_prob)]
    if args.explore_topn is not None:
        base_args += ["--explore-topn", str(args.explore_topn)]
    if args.heartbeat:
        base_args += ["--heartbeat-on-empty"]
    if exclude_path:
        base_args += ["--exclude-file", exclude_path]

    if (not dry_run) and args.recommendations_csv:
        base_args += [
            "--recommendations-csv",
            args.recommendations_csv,
            "--dedupe-per-day",
        ]

    def _invoke(
        alert_args: List[str], label: str, top_k: int
    ) -> Optional[pd.DataFrame]:
        call_args = alert_args + [
            "--top-k",
            str(top_k),
            "--alert-kind",
            alert_kind,
            "--alert-category",
            label,
        ]
        if args.discord_webhook and not dry_run:
            call_args += ["--discord-webhook", args.discord_webhook]
        if dry_run:
            call_args += ["--dry-run"]
        print(
            f"[rt] triggering alert ({label}): top_k={top_k} provider={args.provider} kind={alert_kind} dry_run={dry_run}"
        )
        return alert_main(call_args)

    watch_args = list(base_args)
    if universe_file:
        watch_args += ["--universe-file", universe_file]
    picks = _invoke(watch_args, "watchlist", args.top_k)

    if (not dry_run) and args.explore_universe_file:
        explore_path = args.explore_universe_file
        if os.path.exists(explore_path):
            explore_args = list(base_args)
            explore_args += ["--universe-file", explore_path]
            _invoke(explore_args, "explore", int(max(1, args.explore_top_k)))
        else:
            print(f"[rt] explore universe missing: {explore_path}")
    # Update cooldown state using recent alert diagnostics
    if (not dry_run) and int(args.cooldown_mins) > 0 and args.alert_log_csv:
        try:
            import json as _json
            from datetime import datetime as _dt

            now = _dt.now().astimezone()
            state = {}
            if os.path.exists(args.cooldown_state):
                state = (
                    _json.loads(open(args.cooldown_state, "r", encoding="utf-8").read())
                    or {}
                )
            if os.path.exists(args.alert_log_csv):
                df = pd.read_csv(args.alert_log_csv)
                # Take symbols from the most recent trigger entries (last 5 rows)
                recent_syms = (
                    df.tail(5)
                    .get("symbol", pd.Series([]))
                    .astype(str)
                    .str.upper()
                    .tolist()
                )
                for s in recent_syms:
                    state[s] = now.isoformat()
                os.makedirs(os.path.dirname(args.cooldown_state), exist_ok=True)
                with open(args.cooldown_state, "w", encoding="utf-8") as f:
                    f.write(_json.dumps(state))
        except Exception:
            pass
    return picks


def _load_signal_state(path: str) -> dict:
    if not path:
        return {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _save_signal_state(path: str, state: dict) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def _load_open_symbols(path: str) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty or "symbol" not in df.columns:
        return set()
    return set(df["symbol"].astype(str).str.upper())


def _deactivate_closed_positions(
    state: dict, open_symbols: Set[str], cooldown_mins: int
) -> bool:
    if not state:
        return False
    changed = False
    now = datetime.now().astimezone()
    cool = timedelta(minutes=max(0, int(cooldown_mins)))
    expiry = cool * 2 if cool.total_seconds() > 0 else timedelta(minutes=120)
    for sym in list(state.keys()):
        meta = state.get(sym) or {}
        if meta.get("active") and sym not in open_symbols:
            meta["active"] = False
            meta["pending"] = False
            state[sym] = meta
            changed = True
        if sym in open_symbols:
            continue
        ts = meta.get("timestamp")
        drop = False
        if ts:
            try:
                prev = datetime.fromisoformat(str(ts))
                drop = (now - prev) > expiry
            except Exception:
                drop = True
        else:
            drop = True
        if drop:
            state.pop(sym, None)
            changed = True
    return changed


def _session_date_str(calendar_code: str) -> str:
    """Return the current trading session date (YYYY-MM-DD) for dedupe keys.

    Uses the market calendar to anchor a single per-session identifier, so we
    can implement "alert once per night" semantics regardless of clock drift.
    """
    try:
        sess = today_session(calendar_code)
        if sess is not None:
            return sess.open_ts.date().isoformat()
    except Exception:
        pass
    # Fallback to local calendar date
    from datetime import datetime as _dt

    return _dt.now().astimezone().date().isoformat()


def _send_watchlist_alert(
    args: argparse.Namespace, picks: pd.DataFrame, threshold: float, rank_col: str
) -> None:
    if picks.empty:
        return
    hook = (
        args.discord_webhook
        or os.environ.get("DISCORD_ALERTS_WEBHOOK_URL")
        or os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    if not hook:
        return
    try:
        from ..infra.notify import send_discord
    except Exception:
        return
    now = datetime.now().astimezone()
    header = f"Daily Watchlist {now.strftime('%Y-%m-%d %H:%M %Z')} (p>={threshold:.3f})"
    lines = [header]
    for _, row in picks.iterrows():
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        prob = float(row.get(rank_col, float("nan")))
        parts = [f"${sym}", f"p={prob:.3f}"]
        adv = row.get("adv20", np.nan)
        if np.isfinite(adv):
            parts.append(f"ADV ${adv:,.0f}")
        px = row.get("ref_price", row.get("adj_close", row.get("close", np.nan)))
        if np.isfinite(px):
            parts.append(f"ref ${float(px):.2f}")
        lines.append(" • ".join(parts))
    try:
        send_discord(hook, "@everyone\n" + "\n".join(lines))
        print(
            f"[rt] sent startup watchlist alert ({len(picks)} symbols >= {threshold:.3f})"
        )
    except Exception as exc:
        print(f"[rt] watchlist alert failed: {exc}")


def _dispatch_initial_watchlist(args: argparse.Namespace) -> None:
    thr = args.initial_watchlist_threshold
    if thr is None or float(thr) <= 0:
        return
    try:
        picks = _trigger_once(args, dry_run=True)
    except Exception as exc:
        print(f"[rt] initial watchlist scan failed: {exc}")
        return
    if picks is None or picks.empty:
        print("[rt] initial watchlist scan: no candidates returned")
        return
    rank_col = "meta_prob_mix" if "meta_prob_mix" in picks.columns else "meta_prob"
    filt = picks[picks[rank_col] >= float(thr)].copy()
    if filt.empty:
        print(f"[rt] initial watchlist: no symbols >= {float(thr):.3f}")
        return
    topk = max(1, int(args.initial_watchlist_topk))
    filt = filt.sort_values(rank_col, ascending=False).head(topk)
    _send_watchlist_alert(args, filt, float(thr), rank_col)


def _send_signal_alert(
    args: argparse.Namespace,
    triggered: List[tuple[str, float, pd.Series]],
    effective_threshold: Optional[float] = None,
    base_threshold: Optional[float] = None,
) -> None:
    if not triggered:
        return
    # Repeat throttling for identical symbol sets within a short window (signal-mode only)
    try:
        repeat_mins = int(getattr(args, "min_repeat_mins", 0) or 0)
    except Exception:
        repeat_mins = 0
    if repeat_mins > 0 and getattr(args, "signal_state", ""):
        try:
            st_path = str(args.signal_state)
            st = {}
            if os.path.exists(st_path):
                with open(st_path, "r", encoding="utf-8") as fh:
                    st = json.load(fh) or {}
            now0 = datetime.now().astimezone()
            cur_set = sorted([str(sym).upper() for sym, _, _ in triggered])
            cur_key = ",".join(cur_set)
            meta = st.get("__last_signal__", {}) if isinstance(st, dict) else {}
            key_prev = str(meta.get("key", ""))
            ts_prev = meta.get("ts")
            if key_prev and ts_prev:
                try:
                    prev_dt = datetime.fromisoformat(str(ts_prev))
                    if (now0 - prev_dt) <= timedelta(
                        minutes=repeat_mins
                    ) and key_prev == cur_key:
                        print(
                            f"[rt] skip signal alert: identical set within {repeat_mins} mins"
                        )
                        return
                except Exception:
                    pass
            # Update and persist for next time
            meta = {"key": cur_key, "ts": now0.isoformat()}
            st["__last_signal__"] = meta
            os.makedirs(os.path.dirname(st_path), exist_ok=True)
            with open(st_path, "w", encoding="utf-8") as fh:
                json.dump(st, fh)
        except Exception:
            pass
    hook = (
        args.discord_webhook
        or os.environ.get("DISCORD_ALERTS_WEBHOOK_URL")
        or os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    if not hook:
        return
    try:
        from ..infra.notify import send_discord
    except Exception:
        return
    now = datetime.now().astimezone()
    eff = (
        float(effective_threshold)
        if effective_threshold is not None
        else float(args.signal_threshold or 0.0)
    )
    header = f"Signal Trigger {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (p>={eff:.3f})"
    if base_threshold is not None and eff > float(base_threshold) + 1e-6:
        header += f" • median from base {float(base_threshold):.3f}"
    lines = [header]
    rank_col = (
        "meta_prob_mix"
        if triggered and "meta_prob_mix" in triggered[0][2].index
        else "meta_prob"
    )
    for sym, score, row in triggered:
        parts = [f"${sym}", f"p={score:.3f}"]
        adv = row.get("adv20", np.nan)
        atr = row.get("atr_pct_14", np.nan)
        intr = row.get("intraday_prob", np.nan)
        if np.isfinite(adv):
            parts.append(f"ADV ${adv:,.0f}")
        if np.isfinite(atr):
            parts.append(f"ATR% {atr*100:.2f}")
        if np.isfinite(intr):
            parts.append(f"intraday {intr:.3f}")
        lines.append(" • ".join(parts))
    try:
        send_discord(hook, "@everyone\n" + "\n".join(lines))
    except Exception:
        pass
    # Append minimal recommendations log for signal events if configured
    try:
        rec_path = str(getattr(args, "recommendations_csv", "") or "")
        if rec_path:
            now_ts = datetime.now().astimezone()
            import pandas as _pd

            rows = []
            for sym, score, row in triggered:
                rows.append(
                    {
                        "alert_ts": now_ts.isoformat(),
                        "alert_date": now_ts.date().isoformat(),
                        "trade_date": now_ts.date().isoformat(),
                        "alert_kind": "intraday",
                        "alert_category": "signal",
                        "symbol": str(sym).upper(),
                        "meta_prob": float(row.get("meta_prob", score)),
                        "meta_prob_mix": float(row.get("meta_prob_mix", score)),
                        "adv20": float(row.get("adv20", float("nan"))),
                        "atr_pct_14": float(row.get("atr_pct_14", float("nan"))),
                    }
                )
            rec_df = _pd.DataFrame(rows)
            os.makedirs(os.path.dirname(rec_path), exist_ok=True)
            header = not os.path.exists(rec_path)
            rec_df.to_csv(rec_path, mode="a", index=False, header=header)
    except Exception:
        pass


def _execute_entry(
    args: argparse.Namespace, symbols: List[str], threshold: Optional[float] = None
) -> bool:
    if not symbols:
        return True
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    success = True
    try:
        for sym in symbols:
            tmp.write(str(sym).upper() + "\n")
        tmp.flush()
        entry_args: List[str] = [
            "--universe-file",
            tmp.name,
            "--top-k",
            str(len(symbols)),
            "--confirmations",
            "1",
        ]
        eff = threshold if threshold is not None else args.signal_threshold
        if eff is not None and float(eff) > 0:
            entry_args += ["--entry-threshold", f"{float(eff):.3f}"]
        if args.config:
            entry_args = ["--config", args.config] + entry_args
        if args.intraday_features:
            entry_args += ["--intraday-features", args.intraday_features]
        elif args.features:
            entry_args += ["--features", args.features]
        if args.model_pkl:
            entry_args += ["--model-pkl", args.model_pkl]
        if args.calibrators_pkl:
            entry_args += ["--calibrators-pkl", args.calibrators_pkl]
        elif args.oof:
            entry_args += ["--oof", args.oof]
        if getattr(args, "meta_calibrator_pkl", ""):
            entry_args += ["--meta-calibrator-pkl", args.meta_calibrator_pkl]
        if args.price_source:
            entry_args += ["--price-source", args.price_source]
        if args.live_provider:
            entry_args += ["--live-provider", args.live_provider]
        if args.min_adv_usd is not None:
            try:
                entry_args += ["--min-adv-usd", str(float(args.min_adv_usd))]
            except Exception:
                pass
        entry_args += ["--send-discord"]
        try:
            entry_main(entry_args)
        except SystemExit:
            success = True
        except Exception as exc:
            print(f"[rt] entry loop failed: {exc}")
            success = False
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass
    return success


def _poll_signals(
    args: argparse.Namespace, state: dict, open_symbols: Set[str]
) -> Tuple[dict, bool]:
    try:
        picks = _trigger_once(args, dry_run=True)
    except Exception as exc:
        print(f"[rt] signal poll failed: {exc}")
        return state, False
    if picks is None or picks.empty or args.signal_threshold is None:
        return state, False
    picks = picks.copy()
    rank_col = "meta_prob_mix" if "meta_prob_mix" in picks.columns else "meta_prob"
    sort_cols = [rank_col]
    if "adv20" in picks.columns:
        sort_cols.append("adv20")
    sort_cols.append("symbol")
    picks = picks.sort_values(sort_cols, ascending=[False, False, True]).reset_index(
        drop=True
    )
    now = datetime.now().astimezone()
    cooldown = timedelta(minutes=max(0, int(args.signal_cooldown_mins)))
    min_delta = float(max(0.0, args.signal_min_delta))
    triggered: List[tuple[str, float, pd.Series]] = []
    state_changed = False
    base_threshold = float(args.signal_threshold)
    eligible = picks[picks[rank_col] >= base_threshold]
    effective_threshold = base_threshold
    if bool(args.signal_use_median) and not eligible.empty and len(eligible) > 1:
        try:
            effective_threshold = float(
                max(base_threshold, np.median(eligible[rank_col].astype(float)))
            )
        except Exception:
            effective_threshold = base_threshold
    topn = max(1, int(args.signal_topk))
    # Per-session dedupe: alert each symbol at most once per trading session
    try:
        cal = args.calendar.upper() if hasattr(args, "calendar") else DEFAULT_CAL
        sess_key = _session_date_str(cal)
        alerted = set()
        meta_session = state.get("__session__", {}) if isinstance(state, dict) else {}
        if meta_session.get("date") == sess_key:
            alerted = set(
                str(s).upper()
                for s in meta_session.get("alerted", [])
                if isinstance(meta_session.get("alerted", []), list)
            )
        else:
            # reset session state at first trigger of the day
            state["__session__"] = {"date": sess_key, "alerted": []}
    except Exception:
        alerted = set()
        sess_key = None  # type: ignore
    for _, row in picks.head(topn).iterrows():
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        score = float(row.get(rank_col, 0.0))
        if not np.isfinite(score) or score < effective_threshold:
            continue
        # Skip symbols already alerted this session
        if alerted and sym in alerted:
            continue
        last = state.get(sym, {}) if isinstance(state, dict) else {}
        allow = True
        if last.get("active") or last.get("pending"):
            continue
        last_ts = last.get("timestamp")
        last_score = float(last.get("score", 0.0))
        if last_ts:
            try:
                prev = datetime.fromisoformat(str(last_ts))
                if cooldown.total_seconds() > 0 and (now - prev) < cooldown:
                    if score < last_score + min_delta:
                        allow = False
            except Exception:
                pass
        if sym in open_symbols:
            allow = False
        if not allow:
            continue
        state[sym] = {
            "timestamp": now.isoformat(),
            "score": score,
            "active": False,
            "pending": True,
            "threshold": effective_threshold,
        }
        # Record as alerted for this session (prevent further alert spam)
        try:
            meta_session = state.get("__session__", {})
            if meta_session.get("date") == sess_key:
                cur = list(meta_session.get("alerted", []))
                cur.append(sym)
                meta_session["alerted"] = sorted(list(set(cur)))
                state["__session__"] = meta_session
        except Exception:
            pass
        state_changed = True
        triggered.append((sym, score, row))
    if triggered:
        _send_signal_alert(args, triggered, effective_threshold, base_threshold)
        ok = _execute_entry(args, [sym for sym, _, _ in triggered], effective_threshold)
        if ok:
            for sym, _, _ in triggered:
                meta = state.get(sym, {}) or {}
                meta["active"] = True
                meta["pending"] = False
                state[sym] = meta
        else:
            for sym, _, _ in triggered:
                state.pop(sym, None)
        state_changed = True
    return state, (state_changed or bool(triggered))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Signal mode disables clock-based triggers
    signal_mode = args.signal_threshold is not None
    if signal_mode:
        args.every_minutes = 0
        args.at_open = False
        args.before_close_mins = 0
    times = (
        [] if signal_mode else [t.strip() for t in args.times.split(",") if t.strip()]
    )
    log = _load_log(args.state)
    cal = args.calendar.upper()
    print(
        f"[rt] calendar={cal} times={times} poll={args.poll}s interval={args.every_minutes} at_open={args.at_open} before_close={args.before_close_mins} signal_mode={signal_mode}"
    )
    if args.force:
        try:
            _trigger_once(args)
        except Exception as e:
            print(f"[rt] force trigger failed: {e}")
        return

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

    signal_state = _load_signal_state(args.signal_state) if signal_mode else {}
    _dispatch_initial_watchlist(args)
    try:
        while True:
            now = datetime.now().astimezone()
            sess = today_session(cal)
            open_now = is_open(cal)
            print(
                f"[rt] now={now.strftime('%Y-%m-%d %H:%M:%S %Z')} open={open_now}",
                end="\r",
            )
            if open_now and sess is not None:
                if signal_mode:
                    open_syms = _load_open_symbols(args.positions_csv)
                    if _deactivate_closed_positions(
                        signal_state, open_syms, int(args.signal_cooldown_mins)
                    ):
                        _save_signal_state(args.signal_state, signal_state)
                    signal_state, changed = _poll_signals(args, signal_state, open_syms)
                    if changed:
                        _save_signal_state(args.signal_state, signal_state)
                    time.sleep(max(1, int(args.poll)))
                    continue
                key_date = sess.open_ts.date()
                cur_hhmm = now.strftime("%H:%M")
                to_trigger: list[tuple[str, str]] = []  # (HH:MM, note)
                # explicit times list
                if times:
                    if cur_hhmm in times:
                        to_trigger.append((cur_hhmm, "fixed_time"))
                # interval mode
                if args.every_minutes and args.every_minutes > 0:
                    mins_since_open = int((now - sess.open_ts).total_seconds() // 60)
                    if (
                        mins_since_open >= 0
                        and mins_since_open % int(args.every_minutes) == 0
                    ):
                        to_trigger.append((cur_hhmm, f"interval_{args.every_minutes}"))
                # at open + offset
                if args.at_open:
                    target_time = (
                        sess.open_ts + pd.Timedelta(minutes=int(args.open_offset_mins))
                    ).strftime("%H:%M")
                    if cur_hhmm == target_time:
                        to_trigger.append(
                            (cur_hhmm, f"at_open+{args.open_offset_mins}")
                        )
                # before close mins
                if args.before_close_mins and args.before_close_mins > 0:
                    target_time = (
                        sess.close_ts
                        - pd.Timedelta(minutes=int(args.before_close_mins))
                    ).strftime("%H:%M")
                    if cur_hhmm == target_time:
                        to_trigger.append(
                            (cur_hhmm, f"before_close-{args.before_close_mins}")
                        )

                did_trigger = False
                for hhmm, note in to_trigger:
                    already = not log[
                        (log["date"] == key_date) & (log["time"] == hhmm)
                    ].empty
                    if already:
                        continue
                    try:
                        _trigger_once(args)
                        _log(f"triggered time={hhmm} note={note} top_k={args.top_k}")
                        row = pd.DataFrame(
                            {
                                "date": [key_date],
                                "time": [hhmm],
                                "sent_at": [now],
                                "notes": [
                                    f"{note} top_k={args.top_k} provider={args.provider}"
                                ],
                            }
                        )
                        log = pd.concat([log, row], ignore_index=True)
                        _save_log(args.state, log)
                        # Optional updates message based on diagnostics CSV
                        if (
                            args.send_updates
                            and args.alert_log_csv
                            and os.path.exists(args.alert_log_csv)
                        ):
                            try:
                                import json as _json

                                df = pd.read_csv(args.alert_log_csv)
                                cur = (
                                    df[
                                        (df.get("date").astype(str) == str(key_date))
                                        & (df.get("time") == hhmm)
                                    ]
                                    if ("date" in df.columns and "time" in df.columns)
                                    else df.tail(args.top_k)
                                )
                                cur_syms = [
                                    str(s).upper()
                                    for s in cur.get("symbol", pd.Series([]))
                                    .astype(str)
                                    .tolist()
                                ]
                                cur_meta = {
                                    str(r.symbol).upper(): float(
                                        getattr(
                                            r,
                                            "meta_prob_mix",
                                            getattr(r, "meta_prob", float("nan")),
                                        )
                                    )
                                    for _, r in cur.iterrows()
                                    if str(r.symbol).upper() in cur_syms
                                }
                                cur_px = {
                                    str(r.symbol).upper(): float(r.ref_price)
                                    for _, r in cur.iterrows()
                                    if "ref_price" in cur.columns
                                }
                                # Load previous
                                last = {}
                                if os.path.exists(args.updates_state):
                                    last = (
                                        _json.loads(
                                            open(
                                                args.updates_state,
                                                "r",
                                                encoding="utf-8",
                                            ).read()
                                        )
                                        or {}
                                    )
                                last_syms = set(last.keys())
                                new_syms = set(cur_syms)
                                added = sorted(list(new_syms - last_syms))
                                removed = sorted(list(last_syms - new_syms))
                                kept = sorted(list(new_syms & last_syms))
                                lines: list[str] = ["Alert Update"]
                                if added:
                                    lines.append("+ " + ", ".join(added))
                                if removed:
                                    lines.append("- " + ", ".join(removed))
                                # Deltas for kept
                                for s in kept[:5]:
                                    p0 = float(
                                        last.get(s, {}).get("ref_price", float("nan"))
                                    )
                                    p1 = float(cur_px.get(s, float("nan")))
                                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                                        d = (p1 / p0) - 1.0
                                        mp0 = float(
                                            last.get(s, {}).get(
                                                "meta_prob", float("nan")
                                            )
                                        )
                                        mp1 = float(cur_meta.get(s, float("nan")))
                                        dtxt = f"Δpx {100*d:+.2f}%" + (
                                            f" • Δp {mp1-mp0:+.3f}"
                                            if pd.notna(mp0) and pd.notna(mp1)
                                            else ""
                                        )
                                        lines.append(f"{s}: {dtxt}")
                                if len(lines) > 1:
                                    try:
                                        from ..infra.notify import send_discord

                                        hook = args.discord_webhook or os.environ.get(
                                            "DISCORD_ALERTS_WEBHOOK_URL", ""
                                        )
                                        if hook:
                                            send_discord(
                                                hook, "@everyone\n" + "\n".join(lines)
                                            )
                                    except Exception:
                                        pass
                                # Save new state
                                state = {
                                    s: {
                                        "meta_prob": float(
                                            cur_meta.get(s, float("nan"))
                                        ),
                                        "ref_price": float(cur_px.get(s, float("nan"))),
                                    }
                                    for s in cur_syms
                                }
                                os.makedirs(
                                    os.path.dirname(args.updates_state), exist_ok=True
                                )
                                with open(
                                    args.updates_state, "w", encoding="utf-8"
                                ) as f:
                                    f.write(_json.dumps(state))
                                # Demotion: add kept symbols below threshold to cooldown
                                if (
                                    args.demote_below_prob is not None
                                    and args.cooldown_state
                                    and args.demote_cooldown_mins
                                ):
                                    try:
                                        from datetime import datetime as _dt

                                        now2 = _dt.now().astimezone()
                                        cool = {}
                                        if os.path.exists(args.cooldown_state):
                                            cool = (
                                                _json.loads(
                                                    open(
                                                        args.cooldown_state,
                                                        "r",
                                                        encoding="utf-8",
                                                    ).read()
                                                )
                                                or {}
                                            )
                                        for s in kept:
                                            mp = float(cur_meta.get(s, 1.0))
                                            if mp < float(args.demote_below_prob):
                                                cool[s] = now2.isoformat()
                                        os.makedirs(
                                            os.path.dirname(args.cooldown_state),
                                            exist_ok=True,
                                        )
                                        with open(
                                            args.cooldown_state, "w", encoding="utf-8"
                                        ) as f:
                                            f.write(_json.dumps(cool))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        did_trigger = True
                    except Exception as e:
                        emsg = f"alert trigger failed: {e}"
                        print("\n[rt] " + emsg)
                        _log(emsg)
                if did_trigger:
                    time.sleep(60)
                    continue
            time.sleep(max(1, int(args.poll)))
    except KeyboardInterrupt:
        print("\n[rt] stopped by user")


if __name__ == "__main__":
    main()
