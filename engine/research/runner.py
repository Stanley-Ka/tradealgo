"""End-to-end research pipeline runner.

Chains dataset build (Alpha Vantage) -> feature build -> CV+calibration for
specialists -> meta-learner training -> simple daily backtest.

You can skip steps and override inputs/outputs via flags.

Example (NASDAQ-100 universe):
  python -m engine.research.runner \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --start 2015-01-01 --calibration platt --top-k 20 --cost-bps 5
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from ..data.store import storage_root

# Import module mains so we can pass argv lists
from ..data import build_dataset as mod_build_dataset
from ..features import build_features as mod_build_features
from ..models import run_cv as mod_run_cv
from ..models import train_meta as mod_train_meta
from ..backtest import simple_daily as mod_backtest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end research pipeline runner")
    p.add_argument("--universe-file", required=True, help="Path to a text file with one SYMBOL per line")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD (default=all available)")
    p.add_argument("--api-key", type=str, default=os.environ.get("ALPHAVANTAGE_API_KEY", ""), help="Alpha Vantage API key")
    p.add_argument("--max-per-minute", type=int, default=5, help="Alpha Vantage calls per minute")

    p.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--news-sentiment", type=str, default="", help="Optional path to news sentiment CSV/Parquet")
    p.add_argument("--train-folds", type=str, default="all-but-last:1")
    p.add_argument("--test-folds", type=str, default="last:1")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="daily")
    p.add_argument("--rebal-weekday", choices=["MON","TUE","WED","THU","FRI"], default="MON")
    p.add_argument("--turnover-cap", type=float, default=None)
    p.add_argument("--report-csv", type=str, default="")
    p.add_argument("--report-html", type=str, default="")

    # Override outputs
    p.add_argument("--features-out", type=str, default="")
    p.add_argument("--oof-out", type=str, default="")
    p.add_argument("--meta-out", type=str, default="")
    p.add_argument("--model-out", type=str, default="")
    p.add_argument("--bt-out", type=str, default="")

    # Skip flags
    p.add_argument("--skip-build", action="store_true", help="Skip dataset build step")
    p.add_argument("--skip-features", action="store_true", help="Skip feature build step")
    p.add_argument("--skip-cv", action="store_true", help="Skip CV + calibration step")
    p.add_argument("--skip-meta", action="store_true", help="Skip meta training step")
    p.add_argument("--skip-backtest", action="store_true", help="Skip backtest step")
    # MLflow
    p.add_argument("--mlflow", action="store_true", help="Log the end-to-end run to MLflow if available")
    p.add_argument("--mlflow-experiment", type=str, default="research-e2e")
    p.add_argument("--run-name", type=str, default="pipeline")
    return p.parse_args(argv)


def default_paths() -> dict:
    root = storage_root()
    os.makedirs(os.path.join(root, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    os.makedirs(os.path.join(root, "backtests"), exist_ok=True)
    return {
        "features": os.path.join(root, "datasets", "features_daily_1D.parquet"),
        "oof": os.path.join(root, "datasets", "oof_specialists.parquet"),
        "meta": os.path.join(root, "datasets", "meta_predictions.parquet"),
        "model": os.path.join(root, "models", "meta_lr.pkl"),
        "bt": os.path.join(root, "backtests", "daily_topk_results.parquet"),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    paths = default_paths()
    features_out = args.features_out or paths["features"]
    oof_out = args.oof_out or paths["oof"]
    meta_out = args.meta_out or paths["meta"]
    model_out = args.model_out or paths["model"]
    bt_out = args.bt_out or paths["bt"]

    # Optional MLflow run wrapper
    mlflow_active = False
    if args.mlflow:
        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run(run_name=args.run_name)
            mlflow_active = True
            # Log high-level params
            mlflow.log_params(
                {
                    "universe_file": args.universe_file,
                    "start": args.start,
                    "end": args.end or "",
                    "calibration": args.calibration,
                    "train_folds": args.train_folds,
                    "test_folds": args.test_folds,
                    "top_k": args.top_k,
                    "cost_bps": args.cost_bps,
                    "rebalance": args.rebalance,
                    "rebal_weekday": args.rebal_weekday,
                    "turnover_cap": args.turnover_cap if args.turnover_cap is not None else "none",
                }
            )
        except Exception as e:
            print(f"[runner] MLflow init failed: {e}")

    try:
        # 1) Build dataset from Alpha Vantage (per-symbol Parquets)
        if not args.skip_build:
            print("[runner] Step 1/5: Building dataset (Alpha Vantage)…")
            mod_build_dataset.main([
                "--universe-file", args.universe_file,
                "--start", args.start,
                *( ["--end", args.end] if args.end else [] ),
                *( ["--api-key", args.api_key] if args.api_key else [] ),
                "--max-per-minute", str(args.max_per_minute),
            ])
        else:
            print("[runner] Step 1/5: Skipped dataset build.")

        # 2) Build features Parquet
        if not args.skip_features:
            print("[runner] Step 2/5: Building features…")
            mod_build_features.main([
                "--universe-file", args.universe_file,
                "--start", args.start,
                *( ["--end", args.end] if args.end else [] ),
                "--out", features_out,
            ])
            if mlflow_active and os.path.exists(features_out):
                import mlflow  # type: ignore

                mlflow.log_artifact(features_out)
        else:
            print("[runner] Step 2/5: Skipped features.")

        # 3) Run CV + calibration to produce OOF probs
        if not args.skip_cv:
            print("[runner] Step 3/5: CV + calibration…")
            mod_run_cv.main([
                "--features", features_out,
                "--label", "label_up_1d",
                "--calibration", args.calibration,
                *( ["--news-sentiment", args.news_sentiment] if args.news_sentiment else [] ),
                *( ["--start", args.start] if args.start else [] ),
                *( ["--end", args.end] if args.end else [] ),
                "--out", oof_out,
            ])
            if mlflow_active and os.path.exists(oof_out):
                import mlflow  # type: ignore

                mlflow.log_artifact(oof_out)
        else:
            print("[runner] Step 3/5: Skipped CV.")

        # 4) Train meta-learner and produce test predictions
        if not args.skip_meta:
            print("[runner] Step 4/5: Training meta-learner…")
            mod_train_meta.main([
                "--oof", oof_out,
                "--train-folds", args.train_folds,
                "--test-folds", args.test_folds,
                "--out", meta_out,
                "--model-out", model_out,
            ])
            if mlflow_active:
                import mlflow  # type: ignore

                if os.path.exists(meta_out):
                    mlflow.log_artifact(meta_out)
                if os.path.exists(model_out):
                    mlflow.log_artifact(model_out)
        else:
            print("[runner] Step 4/5: Skipped meta.")

        # 5) Backtest daily top-K
        if not args.skip_backtest:
            print("[runner] Step 5/5: Backtesting…")
            # Compose backtest args including reports
            bt_args = [
                "--features", features_out,
                "--pred", meta_out,
                "--prob-col", "meta_prob",
                "--top-k", str(args.top_k),
                "--cost-bps", str(args.cost_bps),
                "--rebalance", args.rebalance,
                "--rebal-weekday", args.rebal_weekday,
                "--out", bt_out,
            ]
            if args.turnover_cap is not None:
                bt_args += ["--turnover-cap", str(args.turnover_cap)]
            if args.report_csv:
                bt_args += ["--report-csv", args.report_csv]
            if args.report_html:
                bt_args += ["--report-html", args.report_html]
            # Do not forward MLflow; runner logs artifacts instead (to avoid nested runs)
            mod_backtest.main(bt_args)
            if mlflow_active:
                import mlflow  # type: ignore

                if os.path.exists(bt_out):
                    mlflow.log_artifact(bt_out)
                if args.report_csv and os.path.exists(args.report_csv):
                    mlflow.log_artifact(args.report_csv)
                if args.report_html and os.path.exists(args.report_html):
                    mlflow.log_artifact(args.report_html)
        else:
            print("[runner] Step 5/5: Skipped backtest.")
    finally:
        if mlflow_active:
            try:
                import mlflow  # type: ignore

                mlflow.end_run()
                print("[runner] MLflow run completed.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
