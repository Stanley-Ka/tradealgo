from __future__ import annotations

"""Build a dynamic US stocks universe from Polygon and save to a text file.

Requires POLYGON_API_KEY in env. Filters:
- market=stocks, active=true, currency=USD
- type filter (common stock only by default)
- exchange filter (optional: e.g., NYSE,NASDAQ,ARCA)

Usage:
  python -m engine.tools.build_universe_polygon --out engine/data/universe/us_all.txt \
    --types CS --exchanges NASDAQ,NYSE,ARCA
"""

import argparse
import os
from typing import List, Optional

from engine.infra.http import HttpClient, HttpConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build US stocks universe from Polygon")
    p.add_argument(
        "--out", required=True, help="Output text file path (one SYMBOL per line)"
    )
    p.add_argument(
        "--types",
        type=str,
        default="CS",
        help="Comma-separated Polygon types to include (e.g., CS,ADR)",
    )
    p.add_argument(
        "--exchanges",
        type=str,
        default="",
        help="Comma-separated exchange codes (e.g., NASDAQ,NYSE,ARCA)",
    )
    p.add_argument("--include-etf", action="store_true", help="Include ETFs (type=ETF)")
    return p.parse_args(argv)


def fetch_polygon_tickers(
    types: List[str], exchanges: Optional[List[str]]
) -> List[str]:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=20.0))
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "apiKey": key,
    }
    # v3 supports server-side type filtering; do that if a single type requested
    if types and len(types) == 1:
        params["type"] = types[0]
    out: List[str] = []
    next_url: Optional[str] = None
    pages = 0
    total_seen = 0
    while True:
        data = (
            client.get_json(next_url or url, params=None if next_url else params) or {}
        )
        # Debug: if API returns error/status without results
        if not data.get("results") and data.get("status") not in (None, "OK"):
            print(
                f"[universe] polygon status={data.get('status')} message={data.get('message') or data.get('error')}"
            )
        results = data.get("results", [])
        for row in results:
            t = str(row.get("type", "")).upper()
            sym = str(row.get("ticker", "")).upper()
            if not sym:
                continue
            if types:
                # Allow if matches any requested type
                if t not in types and not ("ETF" in types and t == "ETF"):
                    continue
            # Client-side exchange filter (optional). Polygon uses MIC codes like XNAS, XNYS, ARCX
            if exchanges:
                mic = str(row.get("primary_exchange", "")).upper()
                # Accept either MIC or common names; map a few commons
                name_map = {"NASDAQ": "XNAS", "NYSE": "XNYS", "ARCA": "ARCX"}
                allowed = set(e.upper() for e in exchanges)
                if mic and (
                    mic in allowed or any(name_map.get(e) == mic for e in allowed)
                ):
                    out.append(sym)
                elif not mic:
                    # if missing, skip
                    continue
                else:
                    continue
            else:
                out.append(sym)
            total_seen += 1
        next_url = data.get("next_url")
        if not next_url:
            break
        # next_url already contains apiKey in Polygon responses; ensure
        if "apiKey" not in next_url:
            next_url = next_url + ("&" if "?" in next_url else "?") + f"apiKey={key}"
        pages += 1
    syms = sorted(list(dict.fromkeys(out)))
    print(f"[universe] fetched pages={pages} raw_count={total_seen} unique={len(syms)}")
    return syms


def fetch_polygon_tickers_v2(
    types: List[str], exchanges: Optional[List[str]]
) -> List[str]:
    """Fallback using v2 reference tickers if v3 returns no data on the account plan."""
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=20.0))
    url = "https://api.polygon.io/v2/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
        "apiKey": key,
    }
    out: List[str] = []
    pages = 0
    while True:
        data = client.get_json(url, params=params) or {}
        results = data.get("results") or data.get("tickers") or []
        for row in results:
            sym = str(row.get("ticker", "")).upper()
            if not sym:
                continue
            t = str(row.get("type", "")).upper()
            if types and t and t not in types:
                continue
            if exchanges:
                mic = (
                    str(row.get("primary_exchange", "")).upper()
                    or str(row.get("exchange", "")).upper()
                )
                name_map = {"NASDAQ": "XNAS", "NYSE": "XNYS", "ARCA": "ARCX"}
                allowed = set(e.upper() for e in exchanges)
                if mic and (
                    mic in allowed or any(name_map.get(e) == mic for e in allowed)
                ):
                    out.append(sym)
                elif not mic:
                    continue
                else:
                    continue
            else:
                out.append(sym)
        pages += 1
        # v2 uses 'next_url' (full URL) or 'cursor' param
        next_url = data.get("next_url")
        cursor = data.get("cursor")
        if next_url:
            url = (
                next_url
                if ("apiKey=" in next_url)
                else (next_url + ("&" if "?" in next_url else "?") + f"apiKey={key}")
            )
            params = None
            continue
        if cursor:
            params["cursor"] = cursor
            continue
        break
    syms = sorted(list(dict.fromkeys(out)))
    print(f"[universe:v2] pages={pages} unique={len(syms)}")
    return syms


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    types = [
        t.strip().upper()
        for t in (args.types.split(",") if args.types else [])
        if t.strip()
    ]
    if args.include_etf and "ETF" not in types:
        types.append("ETF")
    exchanges = [
        e.strip().upper()
        for e in (args.exchanges.split(",") if args.exchanges else [])
        if e.strip()
    ]
    syms = fetch_polygon_tickers(types, exchanges or None)
    if not syms and exchanges:
        print(
            "[universe] v3 returned 0 with exchange filter; retrying without exchange filter…"
        )
        syms = fetch_polygon_tickers(types, None)
    if not syms:
        print("[universe] v3 returned 0; attempting v2 fallback…")
        try:
            syms = fetch_polygon_tickers_v2(types, exchanges or None)
        except Exception as e:
            print(f"[universe] v2 fallback failed: {e}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(s + "\n")
    print(f"[universe] wrote {len(syms)} symbols -> {args.out}")


if __name__ == "__main__":
    main()
