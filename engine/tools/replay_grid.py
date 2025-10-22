from __future__ import annotations

"""Parameter grid over replay_sweep.

Runs multiple replay sweeps over a grid of parameters (top_k, entry thresholds,
stop/take-profit ATR multiples, exit thresholds), stores each run under a
distinct subdirectory, and aggregates all results into a single CSV with the
grid parameters attached.
"""

import argparse
import os
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .replay_sweep import main as sweep_main


def _parse_list_floats(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def _parse_list_ints(text: str) -> List[int]:
    vals: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def _slug(**kw) -> str:
    def _fmt(v):
        if isinstance(v, float):
            return ("%0.3f" % v).replace(".", "p").rstrip("0").rstrip("p")
        return str(v)

    parts = [f"{k}{_fmt(v)}" for k, v in kw.items()]
    return "_".join(parts)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run replay sweeps over parameter grid and aggregate"
    )
    # Date/sample scheduling (same as replay_sweep)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    # Base config
    p.add_argument("--style", type=str, default="")
    p.add_argument("--config", type=str, default="engine/presets/swing_aggressive.yaml")
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--hold-days", type=int, default=-1)
    p.add_argument("--max-symbols", type=int, default=120)
    p.add_argument("--provider", choices=["yahoo"], default="yahoo")
    p.add_argument("--skip-news", action="store_true")
    p.add_argument(
        "--news-provider", choices=["polygon", "finnhub", "none"], default="polygon"
    )
    p.add_argument("--news-window", type=int, default=3)
    p.add_argument("--news-max-symbols", type=int, default=40)
    # Grid params
    p.add_argument("--topk-list", type=str, default="5,10")
    p.add_argument("--entry-thresholds", type=str, default="0.45,0.48,0.50")
    p.add_argument("--exit-thresholds", type=str, default="0.45")
    p.add_argument("--stop-atr-list", type=str, default="1.0,1.25")
    p.add_argument("--tp-atr-list", type=str, default="0.0,1.5")
    p.add_argument("--exit-consecutive", type=int, default=2)
    # Output
    p.add_argument("--out-root", type=str, default="data/backtests/replay_grid")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="If set, skip sweeps and only aggregate existing subdirs",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def _sweep_one(
    sub_root: str,
    base_args: argparse.Namespace,
    topk: int,
    entry_t: float,
    exit_t: float,
    stop_m: float,
    tp_m: float,
) -> None:
    args = [
        "--start",
        base_args.start,
        "--end",
        base_args.end,
        "--step",
        str(base_args.step),
        "--sample",
        str(base_args.sample),
        "--workers",
        str(base_args.workers),
        "--lookback-days",
        str(base_args.lookback_days),
        "--hold-days",
        str(base_args.hold_days),
        "--max-symbols",
        str(base_args.max_symbols),
        "--provider",
        base_args.provider,
        "--news-provider",
        base_args.news_provider,
        "--news-window",
        str(base_args.news_window),
        "--news-max-symbols",
        str(base_args.news_max_symbols),
        "--top-k",
        str(topk),
        "--entry-threshold",
        f"{entry_t:.6f}",
        "--exit-threshold",
        f"{exit_t:.6f}",
        "--exit-consecutive",
        str(base_args.exit_consecutive),
        "--out-root",
        sub_root,
        "--aggregate",
    ]
    if base_args.style:
        args += ["--style", base_args.style]
    elif base_args.config:
        args += ["--config", base_args.config]
    if base_args.skip_news:
        args += ["--skip-news"]
    if base_args.verbose:
        args += ["--verbose"]
    # ATR overrides
    args += ["--stop-atr-mult", f"{stop_m:.6f}"]
    args += ["--take-profit-atr-mult", f"{tp_m:.6f}"]
    sweep_main(args)


def _aggregate_grid(out_root: str, combos_meta: Optional[List[dict]] = None) -> str:
    root = Path(out_root)
    rows = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        agg_path = sub / "replays_aggregate.csv"
        if not agg_path.exists():
            continue
        try:
            df = pd.read_csv(agg_path)
        except Exception:
            continue
        # Infer meta from folder name if not provided
        meta = {}
        parts = sub.name.split("_")
        for part in parts:
            if part.startswith("topk"):
                meta["top_k"] = int(part[4:])
            elif part.startswith("entry"):
                meta["entry_threshold"] = float(part[5:].replace("p", "."))
            elif part.startswith("exit"):
                meta["exit_threshold"] = float(part[4:].replace("p", "."))
            elif part.startswith("stop"):
                meta["stop_atr_mult"] = float(part[4:].replace("p", "."))
            elif part.startswith("tp"):
                meta["take_profit_atr_mult"] = float(part[2:].replace("p", "."))
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        print(f"[grid] no aggregated files found under {out_root}")
        return ""
    out_df = pd.concat(rows, axis=0, ignore_index=True)
    out_path = os.path.join(out_root, "grid_aggregate.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[grid] wrote {len(out_df)} rows -> {out_path}")
    return out_path


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)
    topk_list = _parse_list_ints(args.topk_list)
    entry_list = _parse_list_floats(args.entry_thresholds)
    exit_list = _parse_list_floats(args.exit_thresholds)
    stop_list = _parse_list_floats(args.stop_atr_list)
    tp_list = _parse_list_floats(args.tp_atr_list)

    combos = list(product(topk_list, entry_list, exit_list, stop_list, tp_list))
    if not args.aggregate_only:
        print(f"[grid] running {len(combos)} combos under {args.out_root}")
        for idx, (tk, et, xt, sm, tm) in enumerate(combos, start=1):
            slug = _slug(topk=tk, entry=et, exit=xt, stop=sm, tp=tm)
            sub_root = os.path.join(args.out_root, slug)
            os.makedirs(sub_root, exist_ok=True)
            print(f"[grid] ({idx}/{len(combos)}) -> {slug}")
            _sweep_one(sub_root, args, tk, et, xt, sm, tm)
    _aggregate_grid(args.out_root)


if __name__ == "__main__":
    main()
