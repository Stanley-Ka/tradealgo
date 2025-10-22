"""Walk-forward orchestration: train specialists/meta on a period, test on next.

This script wires together existing tools:
- run_cv: produce specialist OOF + calibrators on the train window
- train_meta: fit meta model on OOF
- predict_daily: generate top-K predictions for each day in the next window
- simple_daily: evaluate backtest on that next window

Usage (example):
  python -m engine.tools.walkforward \
    --features data/datasets/features_daily_1D.parquet \
    --start 2018-01-01 --end 2023-12-31 --freq quarterly \
    --label label_up_1d --top-k 20 --cost-bps 5 --out-dir data/backtests/wf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward train->test over periods")
    p.add_argument("--features", required=True, help="Features parquet path")
    p.add_argument(
        "--start", type=str, default="", help="Start date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--end", type=str, default="", help="End date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--freq",
        choices=["monthly", "quarterly"],
        default="monthly",
        help="Walk-forward period granularity for train/test",
    )
    p.add_argument(
        "--label",
        type=str,
        default="label_up_1d",
        help="Binary label column in features",
    )
    p.add_argument("--spec-calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument(
        "--kfolds", type=int, default=6, help="Time K-folds for specialist CV"
    )
    p.add_argument("--purge-days", type=int, default=5)
    p.add_argument("--embargo-days", type=int, default=5)
    p.add_argument("--meta-model", choices=["lr", "hgb"], default="hgb")
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument(
        "--meta-calibration", choices=["none", "platt", "isotonic"], default="isotonic"
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--cost-model", choices=["flat", "spread"], default="flat")
    p.add_argument("--spread-k", type=float, default=1e8)
    p.add_argument("--spread-cap-bps", type=float, default=25.0)
    p.add_argument("--spread-min-bps", type=float, default=2.0)
    p.add_argument("--out-dir", type=str, default="data/backtests/walkforward")
    return p.parse_args(argv)


@dataclass
class Period:
    label: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _periods(dates: List[pd.Timestamp], freq: str) -> List[Period]:
    # Build period edges by month/quarter from available trading dates
    df = pd.DataFrame({"date": pd.to_datetime(dates)}).dropna().sort_values("date")
    if df.empty:
        return []
    grp_key = (
        df["date"].dt.to_period("Q")
        if freq == "quarterly"
        else df["date"].dt.to_period("M")
    )
    periods = df.groupby(grp_key)
    ordered = []
    for key, g in periods:
        ordered.append((str(key), g["date"].min(), g["date"].max()))
    out: List[Period] = []
    for i in range(len(ordered) - 1):
        cur, nxt = ordered[i], ordered[i + 1]
        label = f"{cur[0]}_to_{nxt[0]}"
        out.append(
            Period(
                label=label,
                train_start=cur[1],
                train_end=cur[2],
                test_start=nxt[1],
                test_end=nxt[2],
            )
        )
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    f = pd.read_parquet(args.features, columns=["date", "symbol", args.label])
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    if args.start:
        f = f[f["date"] >= pd.Timestamp(args.start)]
    if args.end:
        f = f[f["date"] <= pd.Timestamp(args.end)]
    dates = sorted(f["date"].unique().tolist())
    periods = _periods(dates, freq=args.freq)
    if not periods:
        raise RuntimeError("No periods found; check features/date filters")
    root = args.out_dir
    _ensure_dir(root)

    # Import tools lazily to avoid heavy import cost until needed
    from engine.models import run_cv as _cv
    from engine.models import train_meta as _meta
    from engine.tools import predict_daily as _pred
    from engine.backtest import simple_daily as _bt

    all_results: List[pd.DataFrame] = []
    for i, per in enumerate(periods, start=1):
        per_dir = os.path.join(root, f"{per.label}")
        _ensure_dir(per_dir)
        print(
            f"[wf] {i}/{len(periods)} Train {per.train_start.date()}..{per.train_end.date()} | Test {per.test_start.date()}..{per.test_end.date()}"
        )

        # 1) Specialists CV on train window
        oof_path = os.path.join(per_dir, "oof.parquet")
        cal_path = os.path.join(per_dir, "spec_calibrators.pkl")
        _cv.main(
            [
                "--features",
                args.features,
                "--label",
                args.label,
                "--calibration",
                args.spec_calibration,
                "--cv-scheme",
                "time_kfold",
                "--kfolds",
                str(args.kfolds),
                "--purge-days",
                str(args.purge_days),
                "--embargo-days",
                str(args.embargo_days),
                "--start",
                per.train_start.strftime("%Y-%m-%d"),
                "--end",
                per.train_end.strftime("%Y-%m-%d"),
                "--out",
                oof_path,
                "--calibrators-out",
                cal_path,
            ]
        )

        # 2) Train meta on OOF
        model_path = os.path.join(per_dir, "meta_model.pkl")
        meta_cal_path = os.path.join(per_dir, "meta_calibrator.pkl")
        meta_args = [
            "--oof",
            oof_path,
            "--model",
            args.meta_model,
            "--model-out",
            model_path,
            "--train-folds",
            "all-but-last:1",
            "--test-folds",
            "last:1",
        ]
        if args.meta_model == "hgb":
            meta_args += ["--hgb-max-iter", str(args.hgb_max_iter)]
        if args.meta_calibration != "none":
            meta_args += [
                "--meta-calibration",
                args.meta_calibration,
                "--meta-calibrator-out",
                meta_cal_path,
            ]
        _meta.main(meta_args)

        # 3) Generate predictions for each date in the test window (top-K only)
        preds_paths: List[str] = []
        test_dates = [d for d in dates if per.test_start <= d <= per.test_end]
        for d in test_dates:
            out_csv = os.path.join(per_dir, f"pred_{d.date()}.csv")
            _pred.main(
                [
                    "--features",
                    args.features,
                    "--model-pkl",
                    model_path,
                    "--top-k",
                    str(args.top_k),
                    "--out-csv",
                    out_csv,
                    "--date",
                    str(d.date()),
                    "--calibrators-pkl",
                    cal_path,
                    *(
                        ["--meta-calibrator-pkl", meta_cal_path]
                        if args.meta_calibration != "none"
                        else []
                    ),
                ]
            )
            preds_paths.append(out_csv)
        # Concatenate predictions across the test window
        if not preds_paths:
            print("[wf] no predictions produced for test window; skipping")
            continue
        preds_df = pd.concat(
            [pd.read_csv(p) for p in preds_paths], axis=0, ignore_index=True
        )
        preds_all_path = os.path.join(per_dir, "predictions.csv")
        preds_df.to_csv(preds_all_path, index=False)

        # 4) Backtest on the test window
        bt_out = os.path.join(per_dir, "daily_results.parquet")
        bt_args = [
            "--features",
            args.features,
            "--pred",
            preds_all_path,
            "--prob-col",
            "meta_prob",
            "--top-k",
            str(args.top_k),
            "--cost-bps",
            str(args.cost_bps),
            "--out",
            bt_out,
        ]
        if args.cost_model == "spread":
            bt_args += [
                "--cost-model",
                "spread",
                "--spread-k",
                str(args.spread_k),
                "--spread-cap-bps",
                str(args.spread_cap_bps),
                "--spread-min-bps",
                str(args.spread_min_bps),
            ]
        _bt.main(bt_args)
        # Read the backtest results and trim to test window
        res = pd.read_parquet(bt_out)
        res["date"] = pd.to_datetime(res["date"]).dt.normalize()
        res = res[
            (res["date"] >= per.test_start) & (res["date"] <= per.test_end)
        ].copy()
        res.insert(0, "period", per.label)
        all_results.append(res)

    if not all_results:
        print("[wf] no results produced")
        return
    agg = pd.concat(all_results, axis=0, ignore_index=True)
    agg_path = os.path.join(root, "walkforward_results.parquet")
    agg.to_parquet(agg_path, index=False)
    print(f"[wf] aggregated results -> {agg_path} rows={len(agg)}")


if __name__ == "__main__":
    main()
