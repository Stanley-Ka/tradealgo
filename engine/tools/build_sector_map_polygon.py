from __future__ import annotations

"""Build a sector/industry/market-cap map from Polygon reference data.

Outputs a CSV with columns: symbol,sector,industry,market_cap,primary_exchange

Usage:
  python -m engine.tools.build_sector_map_polygon --universe-file engine/data/universe/us_all.txt \
      --out engine/data/sector_map.csv
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
    p = argparse.ArgumentParser(description="Build sector map via Polygon reference")
    p.add_argument("--universe-file", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    syms = _read_universe(args.universe_file)
    # Require API key via env; PolygonClient will attach header
    _ = os.environ.get("POLYGON_API_KEY", "")
    client = PolygonClient()
    rows = []
    for i, sym in enumerate(syms, 1):
        try:
            data = client.ticker_details(sym) or {}
            res = data.get("results") or {}
            sector = (
                res.get("sic_description")
                or res.get("industry")
                or res.get("sector")
                or ""
            ).strip()
            industry = (res.get("industry") or "").strip()
            mcap = res.get("market_cap") or res.get("marketCap") or None
            exch = res.get("primary_exchange") or res.get("primaryExchange") or ""
            rows.append(
                {
                    "symbol": sym,
                    "sector": str(sector)[:64],
                    "industry": str(industry)[:64],
                    "market_cap": float(mcap) if mcap is not None else None,
                    "primary_exchange": str(exch),
                }
            )
        except Exception:
            rows.append(
                {
                    "symbol": sym,
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "primary_exchange": "",
                }
            )
        if i % 200 == 0:
            print(f"[sector-map] {i}/{len(syms)}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[sector-map] rows={len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
