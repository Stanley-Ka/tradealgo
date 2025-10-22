from __future__ import annotations

"""Build/refresh an earnings dates map for a universe.

Fetches earnings events per symbol between a date range and writes a de-duplicated
CSV/Parquet with columns: date, symbol, source, and optional fields like time.

Providers:
- finnhub: `/calendar/earnings` (requires FINNHUB_API_KEY)
- yfinance (fallback per-symbol): best-effort next earnings date from `Ticker.calendar`

Usage:
  python -m engine.tools.build_earnings_map \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --start 2024-01-01 --end 2024-12-31 \
    --provider finnhub --out data/events/earnings.parquet
"""

import argparse
import os
from typing import Optional, List, Dict

import pandas as pd

from ..infra.env import load_env_files
from ..infra.http import HttpClient, HttpConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build earnings dates map for a universe")
    p.add_argument("--universe-file", required=True)
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--provider", choices=["finnhub", "yfinance"], default="finnhub")
    p.add_argument(
        "--token",
        type=str,
        default="",
        help="API token (optional; pickup from env if empty)",
    )
    p.add_argument("--out", required=True, help="Output CSV/Parquet path")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests (rate limit)",
    )
    return p.parse_args(argv)


def _read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def _fetch_earnings_finnhub(
    symbols: List[str], start: str, end: str, token: str
) -> pd.DataFrame:
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=15.0))
    rows: List[Dict] = []
    url = "https://finnhub.io/api/v1/calendar/earnings"
    for s in symbols:
        try:
            js = (
                client.get_json(
                    url, params={"symbol": s, "from": start, "to": end, "token": token}
                )
                or {}
            )
            for it in js.get("earningsCalendar", []) or []:
                d = (
                    pd.to_datetime(it.get("date")).normalize()
                    if it.get("date")
                    else None
                )
                if d is None:
                    continue
                rows.append(
                    {
                        "date": d,
                        "symbol": s,
                        "source": "finnhub",
                        "time": str(it.get("hour", "")),
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _fetch_earnings_yf(symbols: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["date", "symbol", "source", "time"])
    for s in symbols:
        try:
            tk = yf.Ticker(s)
            cal = getattr(tk, "calendar", None)
            d = None
            t = ""
            if cal is not None and not getattr(cal, "empty", True):
                # yfinance returns a dataframe with an index containing 'Earnings Date'
                # and a column with values; pull the first if available
                try:
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"].values[0]
                        d = pd.to_datetime(val).normalize()
                except Exception:
                    pass
            if d is not None:
                rows.append({"date": d, "symbol": s, "source": "yfinance", "time": t})
        except Exception:
            continue
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> None:
    load_env_files()
    args = parse_args(argv)
    syms = _read_universe(args.universe_file)
    if args.provider == "finnhub":
        key = args.token or os.environ.get("FINNHUB_API_KEY", "")
        if not key:
            raise RuntimeError("FINNHUB_API_KEY must be set for provider=finnhub")
        df = _fetch_earnings_finnhub(syms, args.start, args.end, key)
    else:
        df = _fetch_earnings_yf(syms)
        # Not filtered by date range since yfinance typically returns next event; keep as-is
    if df.empty:
        print("[earnings] no rows fetched; nothing to write")
        return
    # Merge with existing output and de-duplicate
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["source"] = df["source"].astype(str)
    if os.path.exists(args.out):
        try:
            ext = (
                pd.read_parquet(args.out)
                if args.out.lower().endswith(".parquet")
                else pd.read_csv(args.out)
            )
            ext["date"] = pd.to_datetime(ext["date"]).dt.normalize()
            ext["symbol"] = ext["symbol"].astype(str).str.upper()
            merged = pd.concat([ext, df], ignore_index=True)
        except Exception:
            merged = df.copy()
    else:
        merged = df.copy()
    merged = (
        merged.drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".parquet"):
        merged.to_parquet(args.out, index=False)
    else:
        merged.to_csv(args.out, index=False)
    print(f"[earnings] wrote -> {args.out} rows={len(merged)} (added {len(df)})")


if __name__ == "__main__":
    main()
