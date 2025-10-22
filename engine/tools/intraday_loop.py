from __future__ import annotations

"""Simple intraday loop wrapper to refresh snapshot every N minutes.

Usage:
  python -m engine.tools.intraday_loop --config engine/config.intraday.example.yaml --every 5
"""

import argparse
import time
from typing import Optional, List

from .intraday_pipeline import main as pipeline_main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run intraday pipeline in a loop")
    p.add_argument("--config", required=True)
    p.add_argument("--every", type=int, default=5, help="Minutes between runs (min 1)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    interval_sec = max(60, int(args.every) * 60)
    print(
        f"[i-loop] starting intraday loop every {args.every} min (sec={interval_sec})"
    )
    try:
        while True:
            try:
                pipeline_main(["--config", args.config])
            except Exception as e:
                print(f"[i-loop] intraday pipeline failed: {e}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("[i-loop] stopped")


if __name__ == "__main__":
    main()
