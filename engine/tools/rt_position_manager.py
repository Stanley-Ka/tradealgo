from __future__ import annotations

"""Real-time position manager (paper/hypo execution).

Monitors open positions during market hours, applies stops/targets, can trim partials,
and logs all actions. Live price cascade per poll:
Polygon last trade → Polygon snapshot → Yahoo fast_info → Yahoo quote → Polygon prev close.

Positions CSV columns (appended by entry_loop):
  symbol,entry_date,entry_price,shares,stop_price

Policy (configurable via flags):
- Hard stop at stop_price (sell remaining shares).
- Trailing: update stop to max(prev_stop, price - trail_mult * ATR% * price) if enabled.
- Take profits:
  - TP1 at entry + tp1_atr_mult * ATR% of entry: sell tp1_frac of shares; move stop to breakeven if enabled.
  - TP2 at entry + tp2_atr_mult * ATR%: sell remaining.
- Max hold days: close any remaining after N trading days.

Logs every action to --log-csv and writes updated positions back to CSV.
"""

import argparse
import os
import time
from typing import Optional, List

import numpy as np
import pandas as pd

from ..infra.market_time import DEFAULT_CAL, is_open
from ..infra.http import HttpClient, HttpConfig
from ..infra.env import load_env_files
from ..infra.yaml_config import load_yaml_config
from ..infra.log import get_logger

LOG = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time position manager (paper)")
    p.add_argument("--positions-csv", type=str, default="data/paper/positions.csv")
    p.add_argument("--log-csv", type=str, default="data/paper/trade_log.csv")
    p.add_argument("--calendar", type=str, default=DEFAULT_CAL)
    p.add_argument("--poll", type=int, default=15)
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config to read risk settings",
    )
    # Stops / trailing
    p.add_argument("--stop-atr-mult", type=float, default=1.0)
    p.add_argument("--trail-atr-mult", type=float, default=0.8)
    # Take profits / hold
    p.add_argument("--take-profit-atr-mult", type=float, default=1.5)
    p.add_argument("--min-hold-days", type=int, default=1)
    p.add_argument("--max-hold-days", type=int, default=120)
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.45,
        help="Fallback probability if no meta info",
    )
    # Live price
    p.add_argument("--price-source", choices=["live"], default="live")
    p.add_argument("--live-provider", choices=["yahoo", "polygon"], default="yahoo")
    return p.parse_args(argv)


def _now_local_str() -> str:
    from datetime import datetime as _dt

    return _dt.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _live_price(sym: str, provider: str = "yahoo") -> float:
    pr = float("nan")
    try:
        api_key = os.environ.get("POLYGON_API_KEY", "")
        client = HttpClient(HttpConfig(requests_per_second=5.0, timeout=10.0))
        # 1) Polygon last trade
        if api_key:
            try:
                url = f"https://api.polygon.io/v2/last/trade/{sym}"
                data = client.get_json(url, params={"apiKey": api_key}) or {}
                p = (data.get("results", {}) or {}).get("p")
                if p:
                    pr = float(p)
            except Exception:
                pass
        # 2) Polygon snapshot (single-ticker)
        if not np.isfinite(pr) and api_key:
            try:
                snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{sym}"
                s = client.get_json(snap_url, params={"apiKey": api_key}) or {}
                tkr = (s.get("ticker") or {}) if isinstance(s, dict) else {}
                lt = (tkr.get("lastTrade") or {}) if isinstance(tkr, dict) else {}
                lq = (tkr.get("lastQuote") or {}) if isinstance(tkr, dict) else {}
                day = (tkr.get("day") or {}) if isinstance(tkr, dict) else {}
                cand = None
                if isinstance(lt, dict) and (lt.get("p") or lt.get("P")):
                    cand = float(lt.get("p") or lt.get("P"))
                elif isinstance(lq, dict) and (lq.get("p") or lq.get("P")):
                    cand = float(lq.get("p") or lq.get("P"))
                elif isinstance(day, dict) and day.get("c"):
                    cand = float(day.get("c"))
                if cand is not None:
                    pr = float(cand)
            except Exception:
                pass
        # 3) Yahoo fast_info
        if not np.isfinite(pr):
            try:
                import yfinance as yf  # type: ignore

                tk = yf.Ticker(sym)
                info = getattr(tk, "fast_info", None)
                if info and getattr(info, "last_price", None) is not None:
                    pr = float(info.last_price)
                else:
                    hist = tk.history(period="1d")
                    if not hist.empty and float(hist["Close"].iloc[-1]) > 0:
                        pr = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        # 4) Yahoo public quote
        if not np.isfinite(pr):
            try:
                import requests as _rq

                q = _rq.get(
                    "https://query1.finance.yahoo.com/v7/finance/quote",
                    params={"symbols": sym},
                    timeout=6,
                )
                js = q.json() if q.ok else {}
                rmp = (
                    ((js or {}).get("quoteResponse", {}) or {}).get("result", [])
                    or [{}]
                )[0].get("regularMarketPrice")
                if rmp:
                    pr = float(rmp)
            except Exception:
                pass
        # 5) Polygon previous close
        if not np.isfinite(pr) and api_key:
            try:
                prev_url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev"
                data2 = (
                    client.get_json(
                        prev_url, params={"adjusted": "true", "apiKey": api_key}
                    )
                    or {}
                )
                res = data2.get("results") or []
                if res and res[0].get("c"):
                    pr = float(res[0]["c"])
            except Exception:
                pass
    except Exception:
        pass
    return pr


def _append_log(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def _days_since(d: pd.Timestamp) -> int:
    return int((pd.Timestamp.now().normalize() - pd.Timestamp(d).normalize()).days)


def main(argv: Optional[List[str]] = None) -> None:
    # Load env for direct runs so Polygon fallback works without scripts
    load_env_files()
    args = parse_args(argv)
    risk_cfg = {}
    if args.config:
        try:
            cfg = load_yaml_config(args.config)
            risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}
        except Exception as exc:
            LOG.warning("[pm] failed to load config %s: %s", args.config, exc)
    stop_mult = float(risk_cfg.get("stop_atr_mult", args.stop_atr_mult))
    trail_mult = float(risk_cfg.get("trail_atr_mult", args.trail_atr_mult))
    tp_mult = float(risk_cfg.get("take_profit_atr_mult", args.take_profit_atr_mult))
    min_hold = int(risk_cfg.get("min_hold_days", args.min_hold_days))
    max_hold = int(risk_cfg.get("max_hold_days", args.max_hold_days))
    prob_floor = float(args.prob_floor)
    cal = args.calendar.upper()
    print(f"[pm] starting RT position manager poll={args.poll}s cal={cal}")
    try:
        while True:
            open_now = is_open(cal)
            print(f"[pm] now={_now_local_str()} open={open_now}", end="\r")
            if open_now and os.path.exists(args.positions_csv):
                pos = pd.read_csv(args.positions_csv)
                if not pos.empty:
                    pos["symbol"] = pos["symbol"].astype(str).str.upper()
                    # Ensure required columns
                    for col in ("entry_price", "shares"):
                        if col not in pos.columns:
                            print(f"\n[pm] missing column {col} in positions")
                            break
                    updates: list[dict] = []
                    logs: list[dict] = []
                    new_states: list[dict] = []
                    for _, row in pos.iterrows():
                        sym = str(row.symbol).upper()
                        entry = float(row.entry_price)
                        shares = (
                            int(row.shares)
                            if not pd.isna(row.get("shares", np.nan))
                            else 0
                        )
                        if shares <= 0:
                            continue
                        stop = float(
                            row.get("stop_price", entry * (1.0 - stop_mult * 0.5))
                        )
                        atr_entry = row.get("atr_pct_entry")
                        if not np.isfinite(atr_entry) or atr_entry is None:
                            atr_entry = max(
                                1e-4, (entry - stop) / entry / max(stop_mult, 1e-4)
                            )
                        else:
                            atr_entry = float(atr_entry)
                        px = _live_price(sym, provider=args.live_provider)
                        if not np.isfinite(px) or px <= 0:
                            continue
                        entry_date = pd.to_datetime(
                            row.get("entry_date", pd.Timestamp.today())
                        )
                        days_held = _days_since(entry_date)
                        peak = float(row.get("peak_close", entry))
                        peak = max(peak, px)
                        meta_entry = row.get("meta_prob_entry", np.nan)
                        if not np.isfinite(meta_entry):
                            meta_entry = (
                                prob_floor if px >= entry else prob_floor - 0.05
                            )
                        meta_entry = float(np.clip(meta_entry, 0.0, 1.0))

                        base_stop = entry * (1.0 - stop_mult * atr_entry)
                        new_stop = max(stop, base_stop)
                        stop_date = pd.to_datetime(
                            row.get("stop_date", entry_date)
                        ).date()
                        if new_stop > stop + 1e-6:
                            stop_date = pd.Timestamp.today().date()
                        if trail_mult > 0:
                            trail_stop = peak * (1.0 - trail_mult * atr_entry)
                            if trail_stop > new_stop + 1e-6:
                                new_stop = trail_stop
                                stop_date = pd.Timestamp.today().date()

                        target_price = None
                        if tp_mult > 0:
                            momentum_factor = (
                                1.0 + np.clip(meta_entry - 0.55, -0.25, 0.35) * 2.5
                            )
                            time_factor = max(0.5, 1.0 - days_held / 90.0)
                            effective = max(
                                tp_mult * 0.4, tp_mult * momentum_factor * time_factor
                            )
                            target_price = entry * (1.0 + effective * atr_entry)

                        action = None
                        qty = 0
                        reason = ""
                        if px <= new_stop:
                            action = "sell_all"
                            qty = shares
                            reason = "SL"
                        elif (
                            target_price
                            and px >= target_price
                            and days_held >= min_hold
                        ):
                            action = "sell_all"
                            qty = shares
                            reason = "TP"
                        elif days_held >= max_hold:
                            action = "sell_all"
                            qty = shares
                            reason = "time"

                        state = row.to_dict()
                        state["entry_price"] = entry
                        state["shares"] = max(0, shares - qty) if action else shares
                        state["stop_price"] = new_stop
                        state["peak_close"] = peak
                        state["stop_date"] = stop_date
                        state["atr_pct_entry"] = atr_entry
                        state["meta_prob_entry"] = meta_entry

                        if action:
                            logs.append(
                                {
                                    "time": _now_local_str(),
                                    "symbol": sym,
                                    "action": action,
                                    "qty": qty,
                                    "price": px,
                                    "reason": reason,
                                }
                            )
                            print(
                                f"\n[pm] {sym} exit {reason} price=${px:.2f} qty={qty}"
                            )
                        if state["shares"] > 0:
                            new_states.append(state)

                    if logs:
                        _append_log(args.log_csv, logs)
                        print(f"\n[pm] actions logged -> {args.log_csv} n={len(logs)}")
                    if new_states:
                        new_df = pd.DataFrame(new_states)
                    else:
                        new_df = pd.DataFrame(columns=pos.columns)
                    if "entry_date" in new_df.columns:
                        new_df["entry_date"] = pd.to_datetime(
                            new_df["entry_date"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                    if "stop_date" in new_df.columns:
                        new_df["stop_date"] = pd.to_datetime(
                            new_df["stop_date"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                    new_df.to_csv(args.positions_csv, index=False)
            time.sleep(max(1, int(args.poll)))
    except KeyboardInterrupt:
        print("\n[pm] stopped by user")


if __name__ == "__main__":
    main()
