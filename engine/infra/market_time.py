"""US market time utilities with local-time output.

Uses pandas_market_calendars to compute trading sessions. Default calendar is
NASDAQ; you can change via CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import lru_cache

import pandas_market_calendars as mcal  # type: ignore


DEFAULT_CAL = "NASDAQ"


@dataclass
class Session:
    open_ts: datetime
    close_ts: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=8)
def get_calendar(code: str = DEFAULT_CAL):
    """Cached calendar instance for faster repeated queries."""
    return mcal.get_calendar(code)


def today_session(
    calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None
) -> Optional[Session]:
    """Return the trading session that contains 'now' (UTC-based), if any.

    Times in the returned Session are converted to the local timezone for display.
    """
    now_utc = now if now is not None else _now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    cal = get_calendar(calendar_code)
    # Span +/- 1 day to handle timezone boundaries
    sched = cal.schedule(
        start_date=(now_utc - timedelta(days=1)).date(),
        end_date=(now_utc + timedelta(days=1)).date(),
    )
    for _, row in sched.iterrows():
        op_utc = row["market_open"].to_pydatetime()
        cl_utc = row["market_close"].to_pydatetime()
        if op_utc <= now_utc <= cl_utc:
            return Session(open_ts=op_utc.astimezone(), close_ts=cl_utc.astimezone())
    return None


def next_session(
    calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None
) -> Optional[Session]:
    now_utc = now if now is not None else _now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    cal = get_calendar(calendar_code)
    sched = cal.schedule(
        start_date=now_utc.date(), end_date=(now_utc + timedelta(days=10)).date()
    )
    for _, row in sched.iterrows():
        op_utc = row["market_open"].to_pydatetime()
        cl_utc = row["market_close"].to_pydatetime()
        if cl_utc > now_utc:
            return Session(open_ts=op_utc.astimezone(), close_ts=cl_utc.astimezone())
    return None


def is_open(calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None) -> bool:
    now_utc = now if now is not None else _now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    cal = get_calendar(calendar_code)
    # Span +/- 1 day to ensure 'now' falls into a scheduled window regardless of local timezone
    sched = cal.schedule(
        start_date=(now_utc - timedelta(days=1)).date(),
        end_date=(now_utc + timedelta(days=1)).date(),
    )
    if sched.empty:
        return False
    try:
        return bool(cal.open_at_time(sched, now_utc))
    except Exception:
        return False


def seconds_until(dt: datetime, now: Optional[datetime] = None) -> int:
    now = now or _now()
    return max(0, int((dt - now).total_seconds()))
