from __future__ import annotations

"""Verify risk gates and sizing from config on a given date.

Examples:
  python -m engine.tools.verify_risk --config engine/presets/swing_aggressive.yaml \
    --features data/datasets/features_daily_1D.parquet --date 2024-09-30 --symbols AAPL,MSFT

Prints liquidity/volatility gate status and example sizing for each symbol.
"""

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from ..infra.yaml_config import load_yaml_config
from ..infra.feature_join import attach_adv_atr


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify risk gates and sizing for symbols")
    p.add_argument("--config", type=str, default="", help="YAML with paths/risk")
    p.add_argument(
        "--features",
        type=str,
        default="",
        help="Features parquet (date,symbol,adj_close,adj_open)",
    )
    p.add_argument(
        "--date", type=str, default="", help="Target date YYYY-MM-DD (default latest)"
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols to check",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config) if args.config else {}
    feat_path = args.features or (cfg.get("paths", {}) or {}).get("features", "")
    if not feat_path:
        raise RuntimeError("Provide --features or set paths.features in YAML")
    f = pd.read_parquet(feat_path)
    f["date"] = pd.to_datetime(f["date"])  # ensure dtype
    f["symbol"] = f["symbol"].astype(str).str.upper()
    target_date = pd.Timestamp(args.date) if args.date else f["date"].max()
    day = f[f["date"] == target_date].copy()
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        day = day[day["symbol"].isin(syms)]
    if day.empty:
        print(f"[risk] No rows on {target_date.date()} for given selection.")
        return
    # Attach ADV and ATR%
    day_adv = attach_adv_atr(f, target_date)
    day = day.merge(day_adv, on="symbol", how="left")
    # Resolve risk config
    r = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
    min_adv = float(r.get("min_adv_usd", 1e7))
    max_atr = float(r.get("max_atr_pct", 0.05))
    stop_mult = float(r.get("stop_atr_mult", 1.0))
    equity = float(r.get("account_equity", 100000.0))
    risk_mode = str(r.get("risk_mode", "fixed")).lower()
    risk_pct = float(r.get("risk_pct", 0.005))
    min_w = (
        float(r.get("min_risk_pct", 0.002))
        if r.get("min_risk_pct") is not None
        else 0.002
    )
    max_w = (
        float(r.get("max_risk_pct", 0.006))
        if r.get("max_risk_pct") is not None
        else 0.006
    )
    base_p = float(r.get("base_prob", 0.5))
    print(
        f"[risk] min_adv_usd={min_adv:,.0f} max_atr_pct={100*max_atr:.2f}% stop_mult={stop_mult} equity=${equity:,.0f}"
    )
    rows: List[str] = []
    for _, row in day.drop_duplicates("symbol").iterrows():
        sym = str(row.symbol).upper()
        adv = float(row.get("adv20", np.nan))
        atrp = float(row.get("atr_pct_14", np.nan))
        pr = float(row.get("adj_close", row.get("close", np.nan)))
        liq_ok = np.isfinite(adv) and adv >= min_adv
        atr_ok = (np.isfinite(atrp) and atrp <= max_atr) or not np.isfinite(atrp)
        risk_frac = risk_pct
        mp = float(row.get("meta_prob", 0.5))
        if risk_mode == "auto":
            denom = max(1e-6, 1.0 - base_p)
            conv_n = max(0.0, min(1.0, (mp - base_p) / denom))
            # default to quadratic as in entry_loop
            conv_n = conv_n * conv_n
            risk_frac = min_w + (max_w - min_w) * conv_n
        shares = 0
        stop_price = float("nan")
        if np.isfinite(pr) and pr > 0 and np.isfinite(atrp) and atrp > 0:
            stop_price = pr * (1.0 - stop_mult * atrp)
            per_share = max(1e-6, pr - stop_price)
            shares = int(np.floor((equity * risk_frac) / per_share))
        rows.append(
            f"{sym} • adv ${adv:,.0f} ({'ok' if liq_ok else 'low'}) • atr% {100*atrp:.2f}% ({'ok' if atr_ok else 'high'})"
            + (
                f" • entry ${pr:.2f} • stop ${stop_price:.2f} • risk {100*risk_frac:.2f}% • shares {shares}"
                if shares > 0
                else ""
            )
        )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
