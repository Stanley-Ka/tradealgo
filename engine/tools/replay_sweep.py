from __future__ import annotations

"""Run many day replays in parallel and aggregate results.

Examples:
  python -m engine.tools.replay_sweep \
    --start 2022-01-01 --end 2023-12-31 --step 5 --sample 200 \
    --config engine/presets/swing_aggressive.yaml --workers 4 \
    --top-k 5 --entry-threshold 0.48 --exit-threshold 0.45 --exit-consecutive 2 \
    --out-root data/backtests/replays

Notes:
  - This wraps engine.tools.replay_day.main repeatedly with different dates.
  - It is IO/network heavy (Yahoo/Polygon/news). Use reasonable --workers.
  - After finishing, it optionally aggregates all summaries under --out-root.
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .replay_day import main as replay_day_main


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch replay many dates in parallel")
    p.add_argument(
        "--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--end", type=str, required=True, help="End date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--step", type=int, default=5, help="Stride in days between sampled dates"
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, randomly sample this many dates from the range",
    )
    p.add_argument(
        "--style",
        type=str,
        default="",
        help="Style alias to resolve preset (overrides --config)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="engine/presets/swing_aggressive.yaml",
        help="Preset YAML path",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="History days to include before target date",
    )
    p.add_argument(
        "--hold-days",
        type=int,
        default=-1,
        help="Max hold days after entry (-1 for auto)",
    )
    p.add_argument(
        "--max-symbols", type=int, default=120, help="Universe cap per replay"
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override top-K (fallback to config when 0)",
    )
    p.add_argument("--provider", choices=["yahoo"], default="yahoo")
    p.add_argument("--skip-news", action="store_true")
    p.add_argument(
        "--news-provider", choices=["polygon", "finnhub", "none"], default="polygon"
    )
    p.add_argument("--news-window", type=int, default=3)
    p.add_argument("--news-max-symbols", type=int, default=40)
    p.add_argument(
        "--entry-threshold",
        type=float,
        default=np.nan,
        help="Meta prob entry threshold (NaN to use config)",
    )
    p.add_argument("--exit-threshold", type=float, default=0.45)
    p.add_argument("--exit-consecutive", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (0 = use min(4, cpu_count()))",
    )
    # ATR control overrides (passed through to replay_day)
    p.add_argument("--stop-atr-mult", type=float, default=None)
    p.add_argument("--take-profit-atr-mult", type=float, default=None)
    p.add_argument("--trail-atr-mult", type=float, default=None)
    p.add_argument(
        "--out-root",
        type=str,
        default="data/backtests/replays",
        help="Root output directory",
    )
    p.add_argument(
        "--aggregate", action="store_true", help="Aggregate summaries after completion"
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def _date_list(start: str, end: str, step: int, sample: int) -> List[str]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    dates = []
    cur = s
    one = pd.Timedelta(days=step if step > 0 else 1)
    while cur <= e:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = cur + one
    if sample and sample > 0 and sample < len(dates):
        rng = np.random.default_rng(42)
        picks = rng.choice(len(dates), size=sample, replace=False)
        dates = [dates[i] for i in sorted(picks)]
    return dates


def _args_for_date(date_str: str, args: argparse.Namespace) -> List[str]:
    cmd: List[str] = [
        "--date",
        date_str,
        "--lookback-days",
        str(args.lookback_days),
        "--hold-days",
        str(args.hold_days),
        "--max-symbols",
        str(args.max_symbols),
        "--provider",
        str(args.provider),
        "--news-provider",
        str(args.news_provider),
        "--news-window",
        str(args.news_window),
        "--news-max-symbols",
        str(args.news_max_symbols),
        "--exit-threshold",
        f"{float(args.exit_threshold):.6f}",
        "--exit-consecutive",
        str(args.exit_consecutive),
        "--output-dir",
        str(args.out_root),
    ]
    if args.style:
        cmd += ["--style", args.style]
    elif args.config:
        cmd += ["--config", args.config]
    if args.top_k and int(args.top_k) > 0:
        cmd += ["--top-k", str(int(args.top_k))]
    if args.skip_news:
        cmd += ["--skip-news"]
    if not np.isnan(args.entry_threshold):
        cmd += ["--entry-threshold", f"{float(args.entry_threshold):.6f}"]
    if args.verbose:
        cmd += ["--verbose"]
    # ATR overrides
    if args.stop_atr_mult is not None:
        cmd += ["--stop-atr-mult", f"{float(args.stop_atr_mult):.6f}"]
    if args.take_profit_atr_mult is not None:
        cmd += ["--take-profit-atr-mult", f"{float(args.take_profit_atr_mult):.6f}"]
    if args.trail_atr_mult is not None:
        cmd += ["--trail-atr-mult", f"{float(args.trail_atr_mult):.6f}"]
    return cmd


def _run_one(
    date_str_and_args: Tuple[str, argparse.Namespace]
) -> Tuple[str, bool, Optional[str]]:
    date_str, args = date_str_and_args
    run_args = _args_for_date(date_str, args)
    try:
        replay_day_main(run_args)
        return date_str, True, None
    except Exception as exc:  # pragma: no cover
        return date_str, False, str(exc)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    dates = _date_list(args.start, args.end, args.step, args.sample)
    if not dates:
        print("[sweep] no dates to run")
        return
    workers = args.workers if args.workers and args.workers > 0 else min(4, cpu_count())
    os.makedirs(args.out_root, exist_ok=True)
    print(
        f"[sweep] running {len(dates)} replays with workers={workers} -> {args.out_root}"
    )
    payload = [(d, args) for d in dates]
    if workers <= 1:
        results = [_run_one(x) for x in payload]
    else:  # pragma: no cover
        with Pool(processes=workers) as pool:
            results = pool.map(_run_one, payload)
    ok = sum(1 for _, s, _ in results if s)
    fail = [(d, err) for d, s, err in results if not s]
    print(f"[sweep] completed ok={ok} failed={len(fail)}")
    if fail:
        for d, e in fail[:10]:
            print(f"[sweep] fail {d}: {e}")
    if args.aggregate:
        try:
            from .aggregate_replays import main as agg_main

            out_csv = os.path.join(args.out_root, "replays_aggregate.csv")
            agg_main(["--root", args.out_root, "--out-csv", out_csv, "--print-summary"])
        except Exception as exc:  # pragma: no cover
            print(f"[sweep] aggregation failed: {exc}")


if __name__ == "__main__":
    main()
