"""
Live/paper trading entry point stub.
Computes daily signals and submits orders via a broker adapter.
"""

from __future__ import annotations

from .infra.config import Settings
import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live/paper trading entry stub")
    p.add_argument("--features", type=str, default="", help="Features parquet path for latest signals")
    p.add_argument("--model-pkl", type=str, default="", help="Trained meta model pickle")
    p.add_argument("--oof", type=str, default="", help="OOF parquet (for calibrators)")
    p.add_argument("--news-sentiment", type=str, default="", help="Optional sentiment CSV/Parquet")
    p.add_argument("--top-k", type=int, default=20, help="Number of names to print as signals")
    return p.parse_args()


def main() -> None:
    settings = Settings.load()
    print("[LIVE] Loaded settings:", settings.project_name, settings.mode)
    args = parse_args()
    if args.features and args.model_pkl:
        # Thin wrapper around predict_daily to make live testing tactile
        from .tools.predict_daily import main as predict_main

        pd_args = [
            "--features", args.features,
            "--model-pkl", args.model_pkl,
            "--top-k", str(args.top_k),
        ]
        if args.oof:
            pd_args += ["--oof", args.oof]
        if args.news_sentiment:
            pd_args += ["--news-sentiment", args.news_sentiment]
        predict_main(pd_args)
    else:
        print("[LIVE] Provide --features and --model-pkl to print today’s picks.")
        print("[LIVE] Live streaming available:")
        print("       python -m engine.data.stream_finnhub --symbols AAPL,MSFT --ohlc-interval 1m")
        print("       Set token via FINNHUB_API_KEY or --token TOKEN")
        print("       See ENGINE_CAPABILITIES.txt for full options.")


if __name__ == "__main__":
    main()
