"""Run a command only during US market open hours (local time aware).

Example: start live stream at NASDAQ open:
  python -m engine.tools.run_during_market -- \
    python -m engine.data.stream_finnhub --symbols AAPL,MSFT --ohlc-interval 1m

Notes:
- Uses NASDAQ calendar by default; change with --calendar NYSE
- On close, the child process is terminated (SIGTERM on POSIX; best-effort on Windows)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

from ..infra.market_time import DEFAULT_CAL, is_open, next_session, seconds_until


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a command only during US market hours")
    p.add_argument(
        "--calendar",
        type=str,
        default=DEFAULT_CAL,
        help="Market calendar: NASDAQ or NYSE",
    )
    p.add_argument(
        "--poll", type=int, default=15, help="Seconds between open/close checks"
    )
    p.add_argument(
        "--",
        dest="cmdsep",
        action="store_true",
        help="Separator before the command to run",
    )
    p.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to run during market hours"
    )
    return p.parse_args(argv)


def _terminate(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cal = args.calendar.upper()
    cmd = args.cmd
    if not cmd:
        print(
            "Provide a command after --, e.g., -- python -m engine.data.stream_finnhub --symbols AAPL"
        )
        sys.exit(2)

    child: Optional[subprocess.Popen] = None
    try:
        while True:
            if is_open(cal):
                if child is None or child.poll() is not None:
                    print(f"[runner] Market open. Starting: {' '.join(cmd)}")
                    child = subprocess.Popen(cmd)
            else:
                if child is not None and child.poll() is None:
                    print("[runner] Market closed. Stopping child…")
                    _terminate(child)
                    child.wait(timeout=10)
                    child = None
                # sleep until next open if far away
                nxt = next_session(cal)
                if nxt:
                    secs = seconds_until(nxt.open_ts)
                    if secs > args.poll:
                        sleep_for = min(secs, 300)
                        print(
                            f"[runner] Sleeping {sleep_for}s until next check (next open in {secs}s)"
                        )
                        time.sleep(sleep_for)
                        continue
            time.sleep(max(1, args.poll))
    except KeyboardInterrupt:
        print("[runner] Interrupted. Cleaning up…")
    finally:
        if child is not None and child.poll() is None:
            _terminate(child)
            try:
                child.wait(timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    main()
