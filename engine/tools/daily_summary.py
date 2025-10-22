from __future__ import annotations

import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
from ..infra.reason import consensus_for_symbol  # type: ignore


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize daily entries: expected vs realized returns"
    )
    p.add_argument(
        "--features",
        required=True,
        help="Features parquet with date,symbol,fret_1d,atr_pct_14",
    )
    p.add_argument(
        "--decisions",
        default="data/paper/entry_log.csv",
        help="Entry decisions CSV (from entry_scheduler/entry_loop)",
    )
    p.add_argument(
        "--date",
        default="",
        help="Target date YYYY-MM-DD (default latest in decisions)",
    )
    p.add_argument(
        "--lookahead",
        type=int,
        default=5,
        help="Days to compute realized cumulative return",
    )
    p.add_argument(
        "--target-pct",
        type=float,
        default=0.01,
        help="Target return for hit-rate (e.g., 0.01=+1%)",
    )
    p.add_argument(
        "--print-rows", type=int, default=10, help="Rows to print in detail block"
    )
    p.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV to write detailed per-symbol metrics",
    )
    p.add_argument(
        "--discord-webhook",
        default="",
        help="Optional Discord webhook to send a brief summary",
    )
    return p.parse_args(argv)


def _cum_forward(f: pd.DataFrame, sym: str, dt: pd.Timestamp, n: int) -> float:
    g = f[f["symbol"] == sym].sort_values("date").reset_index(drop=True)
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


def _time_to_target(
    f: pd.DataFrame, sym: str, dt: pd.Timestamp, target: float, n: int
) -> Optional[int]:
    g = f[f["symbol"] == sym].sort_values("date").reset_index(drop=True)
    idx = g.index[g["date"] == dt]
    if idx.size == 0:
        prev = g.index[g["date"] < dt]
        if prev.size == 0:
            return None
        start = int(prev.max())
    else:
        start = int(idx.max())
    seq = g.loc[start + 1 : start + int(n), "fret_1d"].astype(float).values
    if seq.size == 0:
        return None
    path = np.cumprod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0
    for i, v in enumerate(path, start=1):
        if v >= float(target):
            return int(i)
    return None


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    f = pd.read_parquet(args.features)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["symbol"] = f["symbol"].astype(str).str.upper()
    if "fret_1d" not in f.columns:
        raise RuntimeError("features must include fret_1d")
    d = pd.read_csv(args.decisions)
    d_date_col = (
        "date"
        if "date" in d.columns
        else ("date_decision" if "date_decision" in d.columns else None)
    )
    if d_date_col is None:
        raise RuntimeError("decisions CSV must include 'date' or 'date_decision'")
    d["date"] = pd.to_datetime(d[d_date_col]).dt.normalize()
    d["symbol"] = d["symbol"].astype(str).str.upper()
    all_dates = sorted(d["date"].unique())
    if not all_dates:
        print("[summary] no decisions found")
        return
    target_date = pd.Timestamp(args.date) if args.date else all_dates[-1]
    day = d[d["date"] == target_date].copy()
    if day.empty:
        print(f"[summary] no decisions on {target_date.date()}")
        return
    look = int(args.lookahead)
    tgt = float(args.target_pct)
    # Compute realized cumulative returns and hit metrics per symbol
    realized = [
        _cum_forward(f, str(sym), target_date, look)
        for sym in day["symbol"].astype(str).tolist()
    ]
    t_hits = [
        _time_to_target(f, str(sym), target_date, tgt, look)
        for sym in day["symbol"].astype(str).tolist()
    ]
    day["realized_ret"] = realized
    day["t_hit"] = t_hits
    day["hit"] = day["t_hit"].apply(lambda x: 1 if pd.notna(x) else 0)
    # Stats
    exp_col = "expected_ret_pct" if "expected_ret_pct" in day.columns else None
    n = len(day)
    avg_exp = float(day[exp_col].mean()) if exp_col else float("nan")
    med_exp = float(day[exp_col].median()) if exp_col else float("nan")
    avg_real = float(pd.to_numeric(day["realized_ret"], errors="coerce").mean())
    med_real = float(pd.to_numeric(day["realized_ret"], errors="coerce").median())
    hit_rate = float(day["hit"].mean()) if n else 0.0
    print(
        f"[summary] {target_date.date()} decisions={n} lookahead={look}d target={100*tgt:.1f}%"
    )
    if exp_col:
        print(f"  expected: mean={avg_exp:.2f}% median={med_exp:.2f}%")
    print(
        f"  realized: mean={100*avg_real:.2f}% median={100*med_real:.2f}%  hit={100*hit_rate:.1f}%"
    )
    # Print details
    cols = ["symbol", "meta_prob", "realized_ret", "t_hit", "hit"]
    if exp_col:
        cols.insert(2, exp_col)
    show = day[[c for c in cols if c in day.columns]].copy()
    # Scale to percent for readability
    if "realized_ret" in show.columns:
        show["realized_ret"] = (100.0 * show["realized_ret"].astype(float)).round(2)
    if exp_col and exp_col in show.columns:
        show[exp_col] = show[exp_col].astype(float).round(2)
    print("[summary] details (top by meta_prob):")
    try:
        print(
            show.sort_values("meta_prob", ascending=False)
            .head(int(args.print_rows))
            .to_string(index=False)
        )
    except Exception:
        print(show.head(int(args.print_rows)).to_string(index=False))
    if args.out_csv:
        day.to_csv(args.out_csv, index=False)
        print(f"[summary] wrote -> {args.out_csv}")
    if args.discord_webhook:
        try:
            import requests  # type: ignore

            # Winners/losers with reasons (top 3 each)
            def _top_k(block: pd.DataFrame, k: int = 3, asc: bool = False) -> List[str]:
                if block.empty or "realized_ret" not in block.columns:
                    return []
                b = block.copy()
                b = (
                    b[pd.notna(b["realized_ret"])]
                    .sort_values("realized_ret", ascending=asc)
                    .head(k)
                )
                out: List[str] = []
                for _, r in b.iterrows():
                    sym = str(r.get("symbol", "")).upper()
                    rr = float(r.get("realized_ret", 0.0))
                    # Try consensus from specialist probabilities if present
                    try:
                        reasons = consensus_for_symbol(block, sym)
                    except Exception:
                        reasons = ""
                    if not reasons:
                        mp = r.get("meta_prob")
                        reasons = f"meta {float(mp):.2f}" if pd.notna(mp) else ""
                    sign = "+" if rr >= 0 else ""
                    out.append(f"{sym} {sign}{100.0*rr:.2f}% — {reasons}")
                return out

            winners = _top_k(day, k=3, asc=False)
            losers = _top_k(day, k=3, asc=True)

            header = (
                f"Summary {target_date.date()} n={n} look={look}d target={100*tgt:.1f}%"
            )
            lines = [header]
            if exp_col:
                lines.append(f"expected: mean={avg_exp:.2f}% med={med_exp:.2f}%")
            lines.append(
                f"realized: mean={100*avg_real:.2f}% med={100*med_real:.2f}% hit={100*hit_rate:.1f}%"
            )
            if winners:
                lines.append("Winners:")
                lines.extend([f"- {w}" for w in winners])
            if losers:
                lines.append("Losers:")
                lines.extend([f"- {l}" for l in losers])
            msg = "\n".join(lines)
            # Truncate to Discord 2000 char limit conservatively
            if len(msg) > 1800:
                msg = msg[:1790] + "…"
            requests.post(args.discord_webhook, json={"content": msg}, timeout=10)
        except Exception:
            pass


if __name__ == "__main__":
    main()
