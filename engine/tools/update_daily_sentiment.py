"""Fetch and score daily news sentiment per symbol and maintain a rolling Parquet.

This utility appends FinBERT-scored sentiment rows aggregated by (date, symbol)
into a single Parquet or CSV file, de-duplicated by keys.

Examples:
  # Update last 3 days for NASDAQ-100 and append to a Parquet
  python -m engine.tools.update_daily_sentiment \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --days 3 --out data/datasets/sentiment_finbert.parquet

  # Explicit date range
  python -m engine.tools.update_daily_sentiment \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --start 2024-09-01 --end 2024-09-10 \
    --out data/datasets/sentiment_finbert.parquet
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from ..news.finbert import FinBERTSentiment, FinBERTConfig
from ..news.providers import fetch_news
from ..infra.log import get_logger

log = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update rolling daily sentiment via FinBERT"
    )
    p.add_argument(
        "--universe-file",
        type=str,
        required=True,
        help="Universe file with one SYMBOL per line",
    )
    p.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD")
    p.add_argument(
        "--end", type=str, default="", help="End date YYYY-MM-DD (inclusive)"
    )
    p.add_argument(
        "--days",
        type=int,
        default=3,
        help="If start/end not set, use last N days up to today",
    )
    p.add_argument("--provider", type=str, default="polygon")
    p.add_argument("--token", type=str, default=os.environ.get("POLYGON_API_KEY", ""))
    p.add_argument("--model", type=str, default="yiyanghkust/finbert-tone")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between symbol fetches (rate limiting)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output Parquet/CSV path (appends, de-duplicates)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Do not write output, only print summary"
    )
    p.add_argument(
        "--skip-on-rate-limit",
        action="store_true",
        help="If a 429 Too Many Requests is encountered, skip the rest of this update step",
    )
    return p.parse_args(argv)


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def _daterange(args: argparse.Namespace) -> tuple[str, str]:
    if args.start and args.end:
        return args.start, args.end
    # default: last N days up to today (inclusive)
    today = date.today()
    start = today - timedelta(days=max(0, int(args.days)))
    return start.isoformat(), today.isoformat()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    start, end = _daterange(args)
    uni = read_universe(args.universe_file)
    rows: List[dict] = []
    for sym in uni:
        try:
            items = fetch_news(
                sym, start, end, provider=args.provider, token=args.token
            )
        except Exception as e:
            msg = str(e)
            log.warning("fetch failed %s: %s", sym, e)
            if args.skip_on_rate_limit and ("429" in msg or "Too Many Requests" in msg):
                log.warning(
                    "rate limit encountered; skipping remainder of sentiment update step."
                )
                # Early exit without writing
                return
            continue
        for it in items:
            rows.append(
                {
                    "date": it.date.date(),
                    "symbol": it.symbol,
                    "headline": it.headline,
                    "summary": it.summary,
                }
            )
        if args.sleep and args.sleep > 0:
            import time as _time

            _time.sleep(float(args.sleep))
    if not rows:
        log.info("no news rows fetched; nothing to do.")
        return
    # Score with FinBERT in batches
    texts = []
    for r in rows:
        t = str(r.get("headline", ""))
        s = str(r.get("summary", ""))
        texts.append((t + ("\n" + s if s else "")).strip())
    model = FinBERTSentiment(FinBERTConfig(model_name=args.model))
    scores = np.zeros((len(texts),), dtype=float)
    step = max(1, int(args.batch))
    for i in range(0, len(texts), step):
        sc, _ = model.score(texts[i : i + step])
        scores[i : i + step] = sc
    df_sc = pd.DataFrame(rows)
    df_sc["sentiment_item"] = scores
    # Aggregate to daily per symbol
    grp = df_sc.groupby(["date", "symbol"])  # date-only above
    out_new = (
        grp["sentiment_item"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_item": "sentiment"})
    )
    out_new["sentiment"] = out_new["sentiment"].astype(float).clip(-1.0, 1.0)

    if args.dry_run:
        log.info("would write %d rows to %s", len(out_new), args.out)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Merge with existing file if present and de-duplicate by (date,symbol)
    if os.path.exists(args.out):
        try:
            extant = (
                pd.read_parquet(args.out)
                if args.out.lower().endswith(".parquet")
                else pd.read_csv(args.out)
            )
            # Ensure dtypes
            extant["date"] = pd.to_datetime(extant["date"]).dt.date
            extant["symbol"] = extant["symbol"].astype(str).str.upper()
            extant["sentiment"] = extant["sentiment"].astype(float)
            merged = pd.concat([extant, out_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
            merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
        except Exception as e:
            log.warning("failed to load existing output, overwriting: %s", e)
            merged = out_new
    else:
        merged = out_new

    if args.out.lower().endswith(".parquet"):
        merged.to_parquet(args.out, index=False)
    else:
        merged.to_csv(args.out, index=False)
    log.info("wrote -> %s rows=%d (added %d)", args.out, len(merged), len(out_new))


if __name__ == "__main__":
    main()
