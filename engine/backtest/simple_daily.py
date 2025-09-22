"""Simple daily backtester for probability-based signals.

Assumptions:
- Use next-day simple return `fret_1d` from the features dataset.
- Select top-K names by probability each day, equal-weighted.
- Apply costs via turnover (bps per 1.0 of weight change).

Example:
  python -m engine.backtest.simple_daily \
    --features data/datasets/features_daily_1D.parquet \
    --pred data/datasets/meta_predictions.parquet --prob-col meta_prob \
    --top-k 20 --cost-bps 5
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import pandas as pd
from typing import List, Optional

from ..data.store import storage_root


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple daily top-K backtest using probabilities")
    p.add_argument("--features", required=True, help="Features parquet with fret_1d and date/symbol")
    p.add_argument("--pred", required=True, help="Predictions parquet with date/symbol and prob column")
    p.add_argument("--prob-col", type=str, default="meta_prob", help="Probability column name in pred file")
    p.add_argument("--top-k", type=int, default=20, help="Number of names to hold each day")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Cost per unit turnover (basis points)")
    p.add_argument("--out", type=str, default="", help="Output parquet for daily results")
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="daily", help="Rebalance frequency")
    p.add_argument("--rebal-weekday", choices=["MON","TUE","WED","THU","FRI"], default="MON", help="Weekly rebal weekday")
    p.add_argument("--turnover-cap", type=float, default=None, help="Cap on daily turnover (e.g., 0.5 for 50%)")
    p.add_argument("--report-csv", type=str, default="", help="Optional CSV export of daily results")
    p.add_argument("--report-html", type=str, default="", help="Optional HTML report (summary + last 20 days)")
    p.add_argument("--mlflow", action="store_true", help="Log run to MLflow if available")
    p.add_argument("--mlflow-experiment", type=str, default="research-backtest", help="MLflow experiment name")
    return p.parse_args(argv)


def _is_rebal_day(date: pd.Timestamp, mode: str, weekday: str, prev_date: Optional[pd.Timestamp]) -> bool:
    if mode == "daily":
        return True
    wd_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    if mode == "weekly":
        return date.weekday() == wd_map[weekday]
    # monthly: rebalance on the first trading day of each new month
    if prev_date is None:
        return True
    return date.month != prev_date.month or date.year != prev_date.year


def daily_backtest(
    features_path: str,
    pred_path: str,
    prob_col: str,
    top_k: int,
    cost_bps: float,
    rebalance: str = "daily",
    rebal_weekday: str = "MON",
    turnover_cap: Optional[float] = None,
) -> pd.DataFrame:
    f = pd.read_parquet(features_path)
    p = pd.read_parquet(pred_path)
    f["date"] = pd.to_datetime(f["date"]) 
    p["date"] = pd.to_datetime(p["date"]) 
    df = p.merge(f[["date", "symbol", "fret_1d"]], on=["date", "symbol"], how="left")
    df = df.dropna(subset=[prob_col, "fret_1d"]).copy()

    # Sort by date and probability per day
    df = df.sort_values(["date", prob_col], ascending=[True, False]).reset_index(drop=True)

    results = []
    prev_weights = pd.Series(dtype=float)
    cur_weights = pd.Series(dtype=float)
    prev_date: Optional[pd.Timestamp] = None
    for date, grp in df.groupby("date", sort=True):
        ts = pd.Timestamp(date)
        if _is_rebal_day(ts, rebalance, rebal_weekday, prev_date):
            picks = grp.head(top_k).copy()
            if len(picks) == 0:
                prev_date = ts
                continue
            w = pd.Series(1.0 / len(picks), index=picks["symbol"].values)
            # Compute turnover cost on rebal days only
            aligned_prev = prev_weights.reindex(w.index).fillna(0.0)
            aligned_curr = w
            delta = aligned_curr - aligned_prev
            turnover = float(delta.abs().sum())
            if turnover_cap is not None and turnover > turnover_cap:
                # Scale changes to respect the cap: w_new = prev + alpha * delta
                alpha = max(0.0, min(1.0, turnover_cap / max(1e-12, turnover)))
                aligned_curr = aligned_prev + alpha * delta
                # Renormalize to sum to 1 if needed
                s = float(aligned_curr.sum())
                if s > 0:
                    aligned_curr = aligned_curr / s
                turnover = float((aligned_curr - aligned_prev).abs().sum())
                w = aligned_curr
            prev_weights = w.copy()
            cur_weights = w
        else:
            # No rebalance: keep weights, zero turnover
            turnover = 0.0
        # Daily portfolio return always computed from current weights
        # Use returns from all symbols we currently hold; if missing, treat as 0
        fret_map = grp.set_index("symbol")["fret_1d"]
        aligned_w = cur_weights.reindex(fret_map.index).fillna(0.0)
        port_ret = float((aligned_w * fret_map).sum())
        cost = (cost_bps / 1e4) * turnover
        net_ret = port_ret - cost
        results.append({
            "date": date,
            "gross_ret": port_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "names": int((cur_weights > 0).sum()),
        })
        prev_date = ts

    res = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    res["equity"] = (1.0 + res["net_ret"]).cumprod()
    return res


def summary_stats(res: pd.DataFrame) -> dict:
    if res.empty:
        return {}
    daily = res["net_ret"].values
    ann = 252
    cagr = res["equity"].iloc[-1] ** (ann / max(1, len(res))) - 1.0
    vol = np.std(daily) * np.sqrt(ann)
    sharpe = (np.mean(daily) * ann) / vol if vol > 0 else float("nan")
    mdd = float((res["equity"].cummax() - res["equity"]) .max()) / float(res["equity"].cummax().max())
    turn = res["turnover"].mean()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd, "AvgTurnover": turn}


def write_reports(res: pd.DataFrame, stats: dict, csv_path: Optional[str], html_path: Optional[str]) -> None:
    if csv_path:
        try:
            res.to_csv(csv_path, index=False)
            print(f"[bt] CSV report -> {csv_path}")
        except Exception as e:
            print(f"[bt] CSV export failed: {e}")
    if html_path:
        try:
            tail = res.tail(20).copy()
            # Build a simple SVG equity curve
            svg = _build_equity_svg(res[["date", "equity"]])
            html = []
            html.append("<html><head><meta charset='utf-8'><title>Backtest Report</title>")
            html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px} table{border-collapse:collapse} th,td{border:1px solid #eee;padding:6px 10px} .k{font-weight:bold}</style>")
            html.append("</head><body>")
            html.append("<h2>Backtest Summary</h2>")
            html.append("<table>")
            for k in ["CAGR", "Sharpe", "MaxDD", "AvgTurnover"]:
                v = stats.get(k, float('nan'))
                html.append(f"<tr><td class='k'>{k}</td><td>{v:.6f}</td></tr>")
            html.append("</table>")
            html.append("<h3>Equity Curve</h3>")
            html.append(svg)
            html.append("<h3>Last 20 days</h3>")
            html.append(tail.to_html(index=False))
            html.append("</body></html>")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
            print(f"[bt] HTML report -> {html_path}")
        except Exception as e:
            print(f"[bt] HTML export failed: {e}")


def _build_equity_svg(df: pd.DataFrame, width: int = 800, height: int = 300, margin: int = 20) -> str:
    if df.empty:
        return "<svg width='800' height='300'></svg>"
    xs = list(range(len(df)))
    ys = df["equity"].astype(float).values
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if ymax - ymin < 1e-12:
        ymax = ymin + 1e-6
    def scale_x(i: int) -> float:
        return margin + (width - 2*margin) * (i / max(1, len(xs)-1))
    def scale_y(y: float) -> float:
        # invert y for SVG (0 at top)
        return height - margin - (height - 2*margin) * ((y - ymin) / (ymax - ymin))
    points = " ".join(f"{scale_x(i):.1f},{scale_y(y):.1f}" for i, y in enumerate(ys))
    svg = f"""
<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' stroke='#ddd'/>
  <polyline fill='none' stroke='#1f77b4' stroke-width='2' points='{points}'/>
</svg>
""".strip()
    return svg


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    res = daily_backtest(
        args.features,
        args.pred,
        args.prob_col,
        args.top_k,
        args.cost_bps,
        rebalance=args.rebalance,
        rebal_weekday=args.rebal_weekday,
        turnover_cap=args.turnover_cap,
    )
    stats = summary_stats(res)
    print(f"Stats: {stats}")
    out_path = args.out.strip()
    if not out_path:
        root = storage_root()
        os.makedirs(os.path.join(root, "backtests"), exist_ok=True)
        out_path = os.path.join(root, "backtests", "daily_topk_results.parquet")
    res.to_parquet(out_path, index=False)
    print(f"[bt] daily results -> {out_path}")
    # Optional reports
    csv_path = args.report_csv.strip() or None
    html_path = args.report_html.strip() or None
    write_reports(res, stats, csv_path, html_path)

    # Optional MLflow logging
    if args.mlflow:
        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name="simple_daily"):
                mlflow.log_params(
                    {
                        "rebalance": args.rebalance,
                        "rebal_weekday": args.rebal_weekday,
                        "top_k": args.top_k,
                        "cost_bps": args.cost_bps,
                        "turnover_cap": args.turnover_cap if args.turnover_cap is not None else "none",
                    }
                )
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                # log artifacts
                mlflow.log_artifact(out_path)
                if csv_path and os.path.exists(csv_path):
                    mlflow.log_artifact(csv_path)
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path)
                print("[bt] MLflow logging complete.")
        except Exception as e:
            print(f"[bt] MLflow logging skipped/failed: {e}")


if __name__ == "__main__":
    main()
