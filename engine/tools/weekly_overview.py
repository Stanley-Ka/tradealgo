from __future__ import annotations

"""Weekly overview summarizing trades, specialist performance, and best combos.

Inputs:
- Decision log CSV produced by paper_trader with --log-specialists enabled. Expected columns:
  date_decision, symbol, selected, w_tgt, fret_1d_next, and specialist *_prob columns.

Outputs:
- Prints a concise weekly summary to stdout
- Optionally writes a Markdown report and updates a YAML of specialist weights based on win rates

Trade definition:
- A trade is a contiguous run of days where a symbol was selected (selected==True) on consecutive
  decision dates. The trade return is the compounded product of fret_1d_next over the run.
- Trade entry specialist signals are taken from the first day of the run.

Dynamic weights:
- Weights are proportional to smoothed win rates per specialist over the evaluated set.
  weight_k ~ (wins_k + alpha) / (attempts_k + 2*alpha), then normalized to sum to 1.
"""

import argparse
import itertools
import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weekly overview: trades, specialist win rates, combos"
    )
    p.add_argument("--decision-log", type=str, default="data/paper/decision_log.csv")
    p.add_argument("--out-md", type=str, default="data/reports/weekly_overview.md")
    p.add_argument(
        "--weights-out", type=str, default="data/paper/specialist_weights.yaml"
    )
    p.add_argument(
        "--weeks",
        type=int,
        default=1,
        help="Number of weeks ending at latest date to include",
    )
    p.add_argument(
        "--min-combo-samples",
        type=int,
        default=5,
        help="Minimum samples required to report a combo",
    )
    # Optional drift/calibration reports
    p.add_argument(
        "--cv-drift-csv", type=str, default="D:\\EngineData\\reports\\cv_drift.csv"
    )
    p.add_argument(
        "--calib-monitor-csv", type=str, default="data/reports/calibration_monitor.csv"
    )
    p.add_argument("--consensus-threshold", type=float, default=0.5)
    return p.parse_args(argv)


def _detect_spec_prob_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("spec_") and c.endswith("_prob")]


def _group_trade_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily selections into trade-level episodes by symbol.

    Returns a DataFrame with columns: symbol, start_date, end_date, n_days,
    ret, entry_row_idx (index of the first row in df), and indices of the rows in the run.
    """
    d = df.copy().reset_index(drop=False).rename(columns={"index": "_row"})
    d["date_decision"] = pd.to_datetime(d["date_decision"]).dt.normalize()
    d = d.sort_values(["symbol", "date_decision"]).reset_index(drop=True)
    episodes: List[Dict] = []
    for sym, g in d.groupby("symbol"):
        g = g.reset_index(drop=True)
        if g.empty:
            continue
        # Build episode breaks where date gap > 1 business day
        gap = g["date_decision"].diff().dt.days.fillna(0).astype(int)
        # Consider any gap >= 2 days as break (weekend will be 3; trading calendar is not available here)
        is_break = gap.ge(2)
        seg_id = is_break.cumsum()
        for _, seg in g.groupby(seg_id):
            rows = seg.index.tolist()
            if not len(rows):
                continue
            start_idx = int(seg.iloc[0]["_row"])  # original index of first row
            start_date = pd.Timestamp(seg.iloc[0]["date_decision"]).date()
            end_date = pd.Timestamp(seg.iloc[-1]["date_decision"]).date()
            # trade return = compounded product of fret_1d_next across segment
            rr = seg.get("fret_1d_next")
            if rr is None or rr.isna().all():
                trade_ret = float("nan")
            else:
                vals = rr.astype(float).values
                trade_ret = float(np.prod(1.0 + np.nan_to_num(vals, nan=0.0)) - 1.0)
            episodes.append(
                {
                    "symbol": sym,
                    "start_date": start_date,
                    "end_date": end_date,
                    "n_days": int(len(seg)),
                    "ret": trade_ret,
                    "entry_row_idx": start_idx,
                    "row_indices": rows,
                }
            )
    return pd.DataFrame(episodes)


def _date_limits(df: pd.DataFrame, weeks: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.to_datetime(df["date_decision"]).dt.normalize()
    end = d.max()
    start = end - pd.Timedelta(days=7 * int(max(1, weeks)))
    return start, end


def _win_rates(
    trades: pd.DataFrame, df: pd.DataFrame, spec_cols: List[str], thr: float = 0.5
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in spec_cols:
        wins = 0
        tot = 0
        for _, tr in trades.iterrows():
            entry = int(tr["entry_row_idx"]) if pd.notna(tr["entry_row_idx"]) else None
            if entry is None:
                continue
            try:
                p = float(df.loc[entry, col])
            except Exception:
                p = float("nan")
            if not np.isfinite(p):
                continue
            pos = p > thr
            if pos:
                tot += 1
                if float(tr["ret"]) > 0:
                    wins += 1
        rate = (wins / tot) if tot > 0 else float("nan")
        stats[col] = {"wins": float(wins), "tot": float(tot), "win_rate": float(rate)}
    return stats


def _combo_stats(
    trades: pd.DataFrame,
    df: pd.DataFrame,
    spec_cols: List[str],
    comb_k: int = 3,
    thr: float = 0.5,
) -> List[Dict]:
    out: List[Dict] = []
    for combo in itertools.combinations(spec_cols, comb_k):
        wins = 0
        tot = 0
        for _, tr in trades.iterrows():
            entry = int(tr["entry_row_idx"]) if pd.notna(tr["entry_row_idx"]) else None
            if entry is None:
                continue
            try:
                pv = [float(df.loc[entry, c]) for c in combo]
            except Exception:
                pv = []
            if len(pv) != comb_k or any(not np.isfinite(x) for x in pv):
                continue
            all_pos = all(x > thr for x in pv)
            if all_pos:
                tot += 1
                if float(tr["ret"]) > 0:
                    wins += 1
        rate = (wins / tot) if tot > 0 else float("nan")
        out.append({"combo": combo, "wins": wins, "tot": tot, "win_rate": rate})
    out.sort(
        key=lambda r: (0 if np.isnan(r["win_rate"]) else r["win_rate"], r["tot"]),
        reverse=True,
    )
    return out


def _weights_from_winrates(
    stats: Dict[str, Dict[str, float]], alpha: float = 2.0
) -> Dict[str, float]:
    # Laplace smoothing around 0.5 prior
    sc: Dict[str, float] = {}
    for k, v in stats.items():
        wins = float(v.get("wins", 0.0))
        tot = float(v.get("tot", 0.0))
        p = (wins + alpha) / (tot + 2.0 * alpha) if tot > 0 else 0.5
        sc[k] = p
    # Normalize to sum to 1
    s = sum(sc.values()) or 1.0
    return {k: float(v / s) for k, v in sc.items()}


def _write_yaml(path: str, m: Dict[str, float]) -> None:
    import yaml  # type: ignore

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(m, f, sort_keys=True)


def _write_md(
    path: str, summary: Dict, top_combos: List[Dict], extras: List[str] | None = None
) -> None:
    lines: List[str] = []
    lines.append(f"# Weekly Overview ({summary['start']} → {summary['end']})\n")
    lines.append(
        f"- Trades closed: {summary['trades']}  Win rate: {summary['win_rate']:.1%}  Avg ret: {summary['avg_ret']:+.2%}\n"
    )
    lines.append(f"- Specialists tracked: {len(summary['spec_cols'])}\n")
    lines.append("\n## Top Specialist Combos (3-way)\n")
    if not top_combos:
        lines.append("(no combos with enough samples)\n")
    else:
        for r in top_combos:
            combo = ", ".join(r["combo"])
            lines.append(f"- {combo}: win_rate={r['win_rate']:.1%}  n={r['tot']}\n")
    # Optional extras (drift/calibration snippets)
    if extras:
        lines.append("\n## Calibration & Drift Notes\n")
        for bl in extras:
            lines.append(bl + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.decision_log):
        raise FileNotFoundError(args.decision_log)
    df = pd.read_csv(args.decision_log)
    if df.empty:
        print("[weekly] decision log is empty; nothing to summarize")
        return
    # Filter to selected rows if column exists
    if "selected" in df.columns:
        try:
            mask = df["selected"].astype(bool)
            df = df[mask]
        except Exception:
            pass
    # Ensure date
    if "date_decision" not in df.columns:
        raise RuntimeError(
            "decision log is missing date_decision column (enable --decision-log-csv in paper_trader)"
        )
    df["date_decision"] = pd.to_datetime(df["date_decision"]).dt.normalize()
    start, end = _date_limits(df, int(args.weeks))
    # Restrict rows to period of interest (by decision date)
    dfp = df[(df["date_decision"] >= start) & (df["date_decision"] <= end)].copy()
    if dfp.empty:
        print("[weekly] no decisions in the requested window")
        return
    # Build episodes from the full log, then keep those that end within window
    epis_all = _group_trade_episodes(df)
    epis = epis_all[pd.to_datetime(epis_all["end_date"]) <= end].copy()
    epis = epis[pd.to_datetime(epis["end_date"]) >= start]
    if epis.empty:
        print("[weekly] no closed trades in window")
        return
    # Specialist prob columns
    spec_cols = _detect_spec_prob_cols(df)
    # Summary metrics
    wins = int((epis["ret"] > 0).sum())
    ntr = int(len(epis))
    avg_ret = float(epis["ret"].astype(float).mean()) if ntr else float("nan")
    win_rate = (wins / ntr) if ntr else float("nan")
    print(
        f"[weekly] window {start.date()} → {end.date()}  closed={ntr}  win_rate={win_rate:.1%}  avg_ret={avg_ret:+.2%}"
    )
    # Specialist win rates and derived weights
    stats = _win_rates(epis, df, spec_cols, thr=float(args.consensus_threshold))
    weights = _weights_from_winrates(stats, alpha=2.0)
    # Best 3-way combos
    raw_combos = _combo_stats(
        epis, df, spec_cols, comb_k=3, thr=float(args.consensus_threshold)
    )
    top_combos = [r for r in raw_combos if r["tot"] >= int(args.min_combo_samples)]
    top_combos = top_combos[:10]
    # Persist outputs
    try:
        _write_yaml(args.weights_out, weights)
        print(f"[weekly] updated specialist weights -> {args.weights_out}")
    except Exception as e:
        print(f"[weekly] warning: failed to write weights: {e}")
    try:
        # Extras: include short drift and calibration summaries if available
        extras: List[str] = []
        try:
            if args.cv_drift_csv and os.path.exists(args.cv_drift_csv):
                drift = pd.read_csv(args.cv_drift_csv).head(5)
                extras.append(
                    "CV Drift (PSI) — first rows:\n" + drift.to_string(index=False)
                )
        except Exception:
            pass
        try:
            if args.calib_monitor_csv and os.path.exists(args.calib_monitor_csv):
                mon = pd.read_csv(args.calib_monitor_csv)
                # show worst 5 by brier and those disabled if column exists
                worst = mon.sort_values("brier", ascending=False).head(5)
                extras.append(
                    "Calibrator Monitor — worst by Brier:\n"
                    + worst.to_string(index=False)
                )
        except Exception:
            pass
        _write_md(
            args.out_md,
            {
                "start": start.date(),
                "end": end.date(),
                "trades": ntr,
                "win_rate": win_rate,
                "avg_ret": avg_ret,
                "spec_cols": spec_cols,
            },
            top_combos,
            extras,
        )
        print(f"[weekly] wrote report -> {args.out_md}")
    except Exception as e:
        print(f"[weekly] warning: failed to write report: {e}")


if __name__ == "__main__":
    main()
