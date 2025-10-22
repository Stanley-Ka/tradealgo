from __future__ import annotations

"""Probe discrepancies between alert picks and decision log outcomes.

Inputs:
- Alert CSV (date,symbol,rank/prob if available)
- Decision log CSV (from paper_trader or backtest)

Outputs:
- CSV with columns: date_decision, symbol, in_alert, in_decision, alert_rank, decision_rank,
  reason (risk_gate_failed|rank_changed|price_shift|unknown), details
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe alert vs decision discrepancies")
    p.add_argument("--alert-csv", required=True)
    p.add_argument("--decision-log-csv", required=True)
    p.add_argument(
        "--out-csv", type=str, default="data/reports/alert_decision_discrepancies.csv"
    )
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="Filter to a specific decision date YYYY-MM-DD (optional)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    a = pd.read_csv(args.alert_csv)
    d = pd.read_csv(args.decision_log_csv)
    # Normalize dates
    date_col = "date_decision" if "date_decision" in d.columns else "date"
    d[date_col] = pd.to_datetime(d[date_col]).dt.date
    if "date" in a.columns:
        a["date"] = pd.to_datetime(a["date"]).dt.date
    else:
        a["date"] = d[date_col].max()
    if args.date:
        dt = pd.to_datetime(args.date).date()
        a = a[a["date"] == dt]
        d = d[d[date_col] == dt]

    # Rank within each source
    def _rank(df: pd.DataFrame, prob_col: str) -> pd.Series:
        if prob_col in df.columns:
            return df[prob_col].rank(ascending=False, method="min").astype(int)
        # fallback: preserve order
        return pd.Series(range(1, len(df) + 1), index=df.index)

    a_rank = _rank(
        a, next((c for c in ("prob", "meta_prob", "score") if c in a.columns), "")
    )
    d_rank = _rank(
        d,
        next(
            (c for c in ("meta_prob", "pre_score", "final_prob") if c in d.columns), ""
        ),
    )
    a = a.assign(alert_rank=a_rank)
    d = d.assign(decision_rank=d_rank)
    # Build union
    a = a.rename(columns={"date": "date_decision"})
    da = d[
        [
            date_col,
            "symbol",
            "decision_rank",
            "meta_prob",
            "_risk_ok",
            "liq_ok",
            "atr_ok",
            "earn_ok",
        ]
    ].rename(columns={date_col: "date_decision"})
    al = a[["date_decision", "symbol", "alert_rank"]]
    merged = pd.merge(al, da, on=["date_decision", "symbol"], how="outer")
    merged["in_alert"] = merged["alert_rank"].notna()
    merged["in_decision"] = merged["decision_rank"].notna()
    reasons: List[str] = []
    details: List[str] = []
    for _, row in merged.iterrows():
        if row["in_alert"] and not row["in_decision"]:
            # If missing in decision, try risk gates
            if (
                not bool(row.get("_risk_ok", True))
                or (row.get("liq_ok") == 0)
                or (row.get("atr_ok") == 0)
                or (row.get("earn_ok") == 0)
            ):
                reasons.append("risk_gate_failed")
                details.append("one or more risk gates false")
            else:
                reasons.append("rank_changed")
                details.append("dropped below top-K or ties broke differently")
        elif not row["in_alert"] and row["in_decision"]:
            reasons.append("rank_changed")
            details.append("added in decision ranking")
        else:
            # both present: check ranking difference
            diff = (
                (int(row["decision_rank"]) - int(row["alert_rank"]))
                if (
                    pd.notna(row.get("decision_rank"))
                    and pd.notna(row.get("alert_rank"))
                )
                else np.nan
            )
            reasons.append("none" if (pd.isna(diff) or diff == 0) else "rank_changed")
            details.append("rank_diff=" + ("nan" if pd.isna(diff) else str(int(diff))))
    merged["reason"], merged["details"] = reasons, details
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[probe] wrote discrepancies -> {args.out_csv}")


if __name__ == "__main__":
    main()
