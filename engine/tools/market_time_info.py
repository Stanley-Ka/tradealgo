"""Print US market session info in your local timezone.

Examples:
  python -m engine.tools.market_time_info              # default NASDAQ
  python -m engine.tools.market_time_info --calendar NYSE
"""

from __future__ import annotations

import argparse
from typing import Optional

from ..infra.market_time import (
    DEFAULT_CAL,
    is_open,
    next_session,
    today_session,
    seconds_until,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show US market session times in local timezone"
    )
    p.add_argument(
        "--calendar",
        type=str,
        default=DEFAULT_CAL,
        help="Market calendar: NASDAQ or NYSE",
    )
    return p.parse_args()


def main(argv: Optional[list[str]] = None) -> None:  # argv unused for -m entry
    args = parse_args()
    cal = args.calendar.upper()
    print(f"[market] calendar={cal}")
    open_now = is_open(cal)
    print(f"[market] open_now={open_now}")
    today = today_session(cal)
    if today:
        print(
            f"[market] today open  : {today.open_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        print(
            f"[market] today close : {today.close_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
    nxt = next_session(cal)
    if nxt:
        print(
            f"[market] next  open  : {nxt.open_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}  (in {seconds_until(nxt.open_ts)}s)"
        )
        print(
            f"[market] next  close : {nxt.close_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}  (in {seconds_until(nxt.close_ts)}s)"
        )


if __name__ == "__main__":
    main()
