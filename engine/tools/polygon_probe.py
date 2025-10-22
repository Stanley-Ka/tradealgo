from __future__ import annotations

"""Probe Polygon API capabilities with the current API key.

Checks common endpoints and prints whether each is accessible, with a brief note
on how the engine uses them.

Usage:
  python -m engine.tools.polygon_probe
"""

import os
from typing import Optional

from ..data.polygon_client import PolygonClient
from ..infra.env import load_env_files


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _try(fn, *args, **kwargs) -> bool:
    try:
        js = fn(*args, **kwargs) or {}
        # consider 401/403 style payloads
        if isinstance(js, dict) and any(
            k in js for k in ("status", "error", "message")
        ):
            st = str(js.get("status", "")).lower()
            msg = str(js.get("message" or js.get("error", "")))
            if "auth" in msg.lower() or st in ("unauthorized", "error"):
                return False
        return True
    except Exception:
        return False


def main(argv: Optional[list[str]] = None) -> None:  # noqa: ARG001
    # Auto-load scripts/api.env or .env so users don't need to export in shell
    load_env_files()
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        _warn("POLYGON_API_KEY not set; only public Yahoo fallbacks will work.")
    client = PolygonClient()
    # Live pricing
    s = "AAPL"
    if _try(client.last_trade, s):
        _ok("v2 last trade: OK (real-time pricing available)")
    else:
        _warn("v2 last trade: unauthorized or blocked (use Yahoo/prev close fallback)")
    if _try(client.snapshot_ticker, s):
        _ok("v2 snapshot: OK (latest trade/quote/day)")
    else:
        _warn("v2 snapshot: unauthorized (Starter often blocks this)")
    if _try(client.previous_close, s):
        _ok("v2 previous close: OK (fallback sizing price)")
    else:
        _warn("v2 previous close: failed (unexpected)")
    # Reference/news
    if _try(client.reference_tickers, market="stocks", active=True, limit=1):
        _ok("v3 reference tickers: OK (universe building)")
    else:
        _warn("v3 reference tickers: failed")
    if _try(client.ticker_details, s):
        _ok("v3 ticker details: OK (sector/industry/market-cap)")
    else:
        _warn("v3 ticker details: failed")
    if _try(client.dividends, ticker=s, limit=1):
        _ok("v3 dividends: OK (ex-div blackout)")
    else:
        _warn("v3 dividends: failed")
    if _try(client.news, ticker=s, limit=1):
        _ok("v2 reference news: OK (news sentiment specialist)")
    else:
        _warn("v2 reference news: failed")


if __name__ == "__main__":
    main()
