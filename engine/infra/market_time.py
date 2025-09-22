"""US market time utilities with local-time output.

Uses pandas_market_calendars to compute trading sessions. Default calendar is
NASDAQ; you can change via CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import pandas_market_calendars as mcal  # type: ignore


DEFAULT_CAL = "NASDAQ"


@dataclass
class Session:
    open_ts: datetime
    close_ts: datetime


def _now() -> datetime:
    return datetime.now().astimezone()


def get_calendar(code: str = DEFAULT_CAL):
    return mcal.get_calendar(code)


def today_session(calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None) -> Optional[Session]:
    now = now or _now()
    cal = get_calendar(calendar_code)
    sched = cal.schedule(start_date=now.date(), end_date=now.date())
    if sched.empty:
        return None
    row = sched.iloc[0]
    return Session(open_ts=row["market_open"].to_pydatetime().astimezone(),
                   close_ts=row["market_close"].to_pydatetime().astimezone())


def next_session(calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None) -> Optional[Session]:
    now = now or _now()
    cal = get_calendar(calendar_code)
    sched = cal.schedule(start_date=now.date(), end_date=(now + timedelta(days=10)).date())
    for _, row in sched.iterrows():
        op = row["market_open"].to_pydatetime().astimezone()
        cl = row["market_close"].to_pydatetime().astimezone()
        if cl > now:
            return Session(op, cl)
    return None


def is_open(calendar_code: str = DEFAULT_CAL, now: Optional[datetime] = None) -> bool:
    now = now or _now()
    cal = get_calendar(calendar_code)
    return bool(cal.open_at_time(schedule=cal.schedule(start_date=now.date(), end_date=now.date()),
                                 ts=now))


def seconds_until(dt: datetime, now: Optional[datetime] = None) -> int:
    now = now or _now()
    return max(0, int((dt - now).total_seconds()))

