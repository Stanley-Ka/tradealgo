"""Build daily symbol sentiment using FinBERT from news headlines/summaries.

Inputs:
  - Either provide an input file with columns [date,symbol,headline,summary?]
    via --input-news, or fetch via Finnhub with --universe-file/--start/--end.

Outputs:
  - A CSV/Parquet with columns [date,symbol,sentiment] in [-1,1], aggregated per day.

Examples:
  # From existing news CSV
  python -m engine.tools.build_sentiment_finbert \
    --input-news data/news/headlines.csv --out data/datasets/sentiment_finbert.parquet

  # Fetch via Finnhub and score
  python -m engine.tools.build_sentiment_finbert \
    --universe-file engine/data/universe/nasdaq100.example.txt \
    --start 2024-01-01 --end 2024-02-01 --out data/datasets/sentiment_finbert.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from ..news.finbert import FinBERTSentiment, FinBERTConfig
from ..news.providers import fetch_news
from ..infra.log import get_logger

log = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily symbol sentiment via FinBERT")
    p.add_argument(
        "--input-news",
        type=str,
        default="",
        help="CSV/Parquet with date,symbol,headline,summary? (optional)",
    )
    p.add_argument(
        "--universe-file",
        type=str,
        default="",
        help="Universe file to fetch news for (one SYMBOL per line)",
    )
    p.add_argument(
        "--start",
        type=str,
        default="",
        help="Start date YYYY-MM-DD (required for fetching)",
    )
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="End date YYYY-MM-DD (required for fetching)",
    )
    p.add_argument("--provider", type=str, default="finnhub")
    p.add_argument("--token", type=str, default=os.environ.get("FINNHUB_API_KEY", ""))
    p.add_argument("--model", type=str, default="yiyanghkust/finbert-tone")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between symbol fetches (rate limiting)",
    )
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--skip-on-rate-limit",
        action="store_true",
        help="If a 429 Too Many Requests is encountered, skip this build (exit early)",
    )
    return p.parse_args(argv)


def read_universe(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip().upper()
            for ln in f.readlines()
            if ln.strip() and not ln.startswith("#")
        ]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    rows: List[dict] = []
    if args.input_news:
        news_path = args.input_news
        df = (
            pd.read_parquet(news_path)
            if news_path.lower().endswith(".parquet")
            else pd.read_csv(news_path)
        )
        if "date" not in df.columns or "symbol" not in df.columns:
            raise RuntimeError(
                "input-news must have columns: date,symbol,headline[,summary]"
            )
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = df["symbol"].astype(str).str.upper()
        rows = df[
            [
                "date",
                "symbol",
                "headline",
                *(["summary"] if "summary" in df.columns else []),
            ]
        ].to_dict("records")
    else:
        if not (args.universe_file and args.start and args.end):
            raise RuntimeError(
                "Provide --input-news or (--universe-file, --start, --end) to fetch news"
            )
        uni = read_universe(args.universe_file)
        for sym in uni:
            try:
                items = fetch_news(
                    sym, args.start, args.end, provider=args.provider, token=args.token
                )
            except Exception as e:
                msg = str(e)
                log.warning("failed fetching %s: %s", sym, e)
                if args.skip_on_rate_limit and (
                    "429" in msg or "Too Many Requests" in msg
                ):
                    log.warning(
                        "rate limit encountered; skipping sentiment build step."
                    )
                    rows = []
                    break
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
        raise RuntimeError("No news rows to score")
    texts = []
    for r in rows:
        t = str(r.get("headline", ""))
        s = str(r.get("summary", ""))
        texts.append((t + ("\n" + s if s else "")).strip())

    model = FinBERTSentiment(FinBERTConfig(model_name=args.model))
    # Batch scoring for memory
    scores = np.zeros((len(texts),), dtype=float)
    step = max(1, int(args.batch))
    for i in range(0, len(texts), step):
        sc, _ = model.score(texts[i : i + step])
        scores[i : i + step] = sc

    # Build DataFrame and aggregate per (date,symbol)
    df_sc = pd.DataFrame(rows)
    df_sc["sentiment_item"] = scores
    grp = df_sc.groupby(["date", "symbol"])  # date is already date-only above
    out = (
        grp["sentiment_item"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_item": "sentiment"})
    )
    # Clip to [-1,1]
    out["sentiment"] = out["sentiment"].astype(float).clip(-1.0, 1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    log.info("wrote sentiment -> %s rows=%d", args.out, len(out))


if __name__ == "__main__":
    main()
