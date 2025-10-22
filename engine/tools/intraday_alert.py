from __future__ import annotations

"""Heuristic intraday alerts using snapshot features only (no daily meta).

Inputs: intraday snapshot parquet with columns date,symbol,adj_close, vwap_dev_20, breakout_high_20, vol_rel_20, atr_pct_14
Outputs: Discord message with top-K by intraday score; optional CSV log.

Usage:
  python -m engine.tools.intraday_alert --intraday-features data/datasets/features_intraday_latest.parquet --top-k 5 --discord-webhook ...
"""

import argparse
import os
from typing import Optional, List

import numpy as np
import pandas as pd

from ..infra.notify import send_discord


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday heuristic alerts from snapshot")
    p.add_argument("--intraday-features", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument(
        "--discord-webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    p.add_argument("--log-csv", type=str, default="")
    return p.parse_args(argv)


def score_row(r: pd.Series) -> float:
    # Simple bounded heuristic
    vwap = float(r.get("vwap_dev_20", 0.0))  # prefer above VWAP
    brk = float(r.get("breakout_high_20", 0.0))  # breakout flag
    volr = float(r.get("vol_rel_20", 1.0))  # relative volume
    atrp = float(r.get("atr_pct_14", 0.0))
    s = 0.0
    s += 2.0 * max(0.0, vwap)  # stronger if above VWAP
    s += 0.5 * brk
    s += 0.2 * max(0.0, min(volr - 1.0, 1.0))
    s += 0.5 * min(atrp, 0.05)  # modest weight to volatility up to 5%
    return float(s)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    df = pd.read_parquet(args.intraday_features)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    prc = df.get("adj_close", df.get("close"))
    if args.min_price is not None:
        df = df[prc >= float(args.min_price)]
    if args.max_price is not None:
        df = df[prc <= float(args.max_price)]
    if df.empty:
        print("[i-alert] no rows after filters")
        return
    df["intraday_score"] = df.apply(score_row, axis=1)
    picks = (
        df.sort_values("intraday_score", ascending=False).head(int(args.top_k)).copy()
    )
    lines = []
    for _, r in picks.iterrows():
        sym = str(r.symbol)
        px = float(r.get("adj_close", np.nan))
        score = float(r.intraday_score)
        vwap = float(r.get("vwap_dev_20", 0.0))
        volr = float(r.get("vol_rel_20", 1.0))
        atrp = float(r.get("atr_pct_14", 0.0))
        lines.append(
            f"{sym}: px=${px:.2f} score={score:.3f} vwap_dev={100*vwap:.2f}% vol_rel={volr:.2f} atr%={100*atrp:.2f}"
        )
    if not lines:
        print("[i-alert] no picks")
        return
    msg = "Intraday Alert\n" + "\n".join(lines)
    if args.discord_webhook:
        try:
            send_discord(args.discord_webhook, msg)
            print("[i-alert] sent to Discord")
        except Exception as e:
            print(f"[i-alert] Discord failed: {e}")
    if args.log_csv:
        try:
            os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
            picks.to_csv(args.log_csv, index=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
