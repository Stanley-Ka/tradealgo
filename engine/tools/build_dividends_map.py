from __future__ import annotations

"""Build a dividends events table for a universe from Polygon.

Outputs CSV with: symbol,ex_dividend_date,declared_date,payment_date,cash_amount

Usage:
  python -m engine.tools.build_dividends_map --universe-file engine/data/universe/us_all.txt \
      --start 2020-01-01 --end 2030-01-01 --out engine/data/dividends.csv
"""

import argparse
import os
from typing import List, Optional

import pandas as pd

from ..data.polygon_client import PolygonClient


def _read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dividends map via Polygon")
    p.add_argument("--universe-file", required=True)
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default="2035-12-31")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    syms = _read_universe(args.universe_file)
    if not os.environ.get("POLYGON_API_KEY"):
        raise RuntimeError("POLYGON_API_KEY required in environment")
    client = PolygonClient()
    rows = []
    for i, sym in enumerate(syms, 1):
        try:
            cur: Optional[str] = None
            while True:
                js = (
                    client.dividends(
                        ticker=sym,
                        ex_div_gte=args.start,
                        ex_div_lte=args.end,
                        limit=1000,
                        cursor_url=cur,
                    )
                    or {}
                )
                res = js.get("results") or []
                for r in res:
                    rows.append(
                        {
                            "symbol": sym,
                            "ex_dividend_date": r.get("ex_dividend_date")
                            or r.get("exDate")
                            or "",
                            "declared_date": r.get("declaration_date")
                            or r.get("declaredDate")
                            or "",
                            "payment_date": r.get("pay_date")
                            or r.get("paymentDate")
                            or "",
                            "cash_amount": r.get("cash_amount")
                            or r.get("cashAmount")
                            or 0.0,
                        }
                    )
                nxt = js.get("next_url")
                if nxt:
                    cur = nxt
                    continue
                break
        except Exception:
            pass
        if i % 200 == 0:
            print(f"[dividends] {i}/{len(syms)}")
    df = pd.DataFrame(rows)
    # Normalize dates
    for c in ("ex_dividend_date", "declared_date", "payment_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[dividends] rows={len(df)} symbols={len(syms)} -> {args.out}")


if __name__ == "__main__":
    main()
