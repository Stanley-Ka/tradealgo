"""Calibrate expected return and horizon from historical decisions.

Inputs:
- Features parquet with columns: date, symbol, fret_1d, atr_pct_14
- Decision log CSV with columns: date, symbol, meta_prob (optionally reason fields)

Outputs:
- Prints slope k for expected return mapping: E[ret_N] ~= k * (meta_prob - base_prob) * ATR%
- Prints median time-to-target for meta_prob buckets and suggested horizon cuts.

Usage:
  python -m engine.tools.calibrate_expectation \
    --features data/datasets/features_daily_1D.parquet \
    --decisions data/paper/entry_log.csv \
    --base-prob 0.5 --lookahead 5 --target-pct 0.01

You can then set in YAML under risk:
  risk:
    base_prob: 0.5
    expected_k: <k_suggested>
    expected_cap_mult: 2.0
    horizon_cut1: 0.55
    horizon_cut2: 0.60
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate expected return/horizon from decisions"
    )
    p.add_argument(
        "--features", required=True, help="Parquet with date,symbol,fret_1d,atr_pct_14"
    )
    p.add_argument(
        "--decisions",
        required=True,
        help="CSV with date,symbol,meta_prob (alert or entry log)",
    )
    p.add_argument(
        "--base-prob",
        type=float,
        default=0.5,
        help="Base probability for zero-conviction",
    )
    p.add_argument(
        "--lookahead", type=int, default=5, help="Max days to measure cumulative return"
    )
    p.add_argument(
        "--target-pct",
        type=float,
        default=0.01,
        help="Target return for horizon (e.g., 0.01=+1%)",
    )
    return p.parse_args(argv)


def _cum_forward_returns(
    f: pd.DataFrame, sym: str, start_idx: int, max_h: int
) -> tuple[float, Optional[int]]:
    r = f[f["symbol"] == sym].sort_values("date").reset_index(drop=True)
    # start_idx refers to the index of decision date in r; we want returns after it
    seq = r.loc[start_idx + 1 : start_idx + max_h, "fret_1d"].astype(float).values
    if seq.size == 0:
        return float("nan"), None
    cum = float(np.prod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0)
    # time-to-target: first day cumulative >= target
    cum_path = np.cumprod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0
    t_hit = int(np.argmax(cum_path >= 0)) if cum_path.size else None
    return cum, None


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    f = pd.read_parquet(args.features)
    f["date"] = pd.to_datetime(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()
    if "fret_1d" not in f.columns:
        raise RuntimeError("features must include fret_1d forward return")
    if "atr_pct_14" not in f.columns:
        raise RuntimeError("features must include atr_pct_14")
    d = pd.read_csv(args.decisions)
    # Accept either 'date' or 'date_decision'
    d_date_col = (
        "date"
        if "date" in d.columns
        else ("date_decision" if "date_decision" in d.columns else None)
    )
    if d_date_col is None:
        raise RuntimeError("decisions CSV must include 'date' or 'date_decision'")
    d["date"] = pd.to_datetime(d[d_date_col])
    d["symbol"] = d["symbol"].astype(str).str.upper()
    if "meta_prob" not in d.columns:
        raise RuntimeError("decisions CSV must include meta_prob")

    records = []
    # Build an index for features rows by symbol
    f = f.sort_values(["symbol", "date"])
    grp = {sym: df.reset_index(drop=True) for sym, df in f.groupby("symbol")}

    for _, row in d.iterrows():
        sym = str(row["symbol"]).upper()
        dt = pd.Timestamp(row["date"]).normalize()
        mp = float(row["meta_prob"]) if pd.notna(row["meta_prob"]) else np.nan
        if sym not in grp:
            continue
        gi = grp[sym]
        idx = gi.index[gi["date"].dt.normalize() == dt]
        if idx.empty:
            # Try previous date if not found (decisions may be intraday)
            prev_idx = gi.index[gi["date"].dt.normalize() < dt]
            if prev_idx.size == 0:
                continue
            start_idx = int(prev_idx.max())
        else:
            start_idx = int(idx.max())
        atrp = (
            float(gi.loc[start_idx, "atr_pct_14"])
            if "atr_pct_14" in gi.columns
            else np.nan
        )
        # Cumulative forward return up to lookahead
        seq = (
            gi.loc[start_idx + 1 : start_idx + int(args.lookahead), "fret_1d"]
            .astype(float)
            .values
        )
        if seq.size == 0:
            continue
        cum = float(np.prod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0)
        # time to target
        path = np.cumprod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0
        t_hit = None
        for i, v in enumerate(path, start=1):
            if v >= float(args.target_pct):
                t_hit = i
                break
        records.append(
            {
                "symbol": sym,
                "date": dt,
                "meta_prob": mp,
                "atr_pct_14": atrp,
                "cum_ret": cum,
                "t_hit": t_hit,
            }
        )

    if not records:
        print(
            "[cal] no matching decisions to features; check date alignment and symbols"
        )
        return
    R = pd.DataFrame.from_records(records)
    # Fit k via least squares through origin on positive edges
    x = (R["meta_prob"].astype(float) - float(args.base_prob)).clip(lower=0.0) * R[
        "atr_pct_14"
    ].astype(float)
    y = R["cum_ret"].astype(float)
    mask = x.notna() & y.notna() & (x > 0)
    if mask.sum() >= 5:
        k_hat = float((x[mask] * y[mask]).sum() / (x[mask] * x[mask]).sum())
        k_hat = float(max(0.1, min(k_hat, 10.0)))
    else:
        k_hat = 2.5
    # Bucket horizons by meta_prob
    cuts = [0.0, 0.55, 0.60, 1.0]
    cats = pd.cut(
        R["meta_prob"], bins=cuts, labels=["low", "med", "high"], include_lowest=True
    )
    med_t = R.groupby(cats)["t_hit"].median()

    print("[cal] Lookahead days:", int(args.lookahead))
    print(f"[cal] k_hat (suggested expected_k): {k_hat:.3f}")
    print(
        "[cal] Median days to +{:.1f}% by conviction bucket:".format(
            100 * float(args.target_pct)
        )
    )
    for label in ["low", "med", "high"]:
        v = med_t.get(label, np.nan)
        print(f"  {label:>4}: {v if pd.notna(v) else 'n/a'}")
    print("\nSuggested YAML overrides (paste under risk):")
    print("risk:")
    print(f"  base_prob: {float(args.base_prob):.3f}")
    print(f"  expected_k: {k_hat:.3f}")
    print("  expected_cap_mult: 2.0")
    print("  horizon_cut1: 0.55")
    print("  horizon_cut2: 0.60")


if __name__ == "__main__":
    main()
