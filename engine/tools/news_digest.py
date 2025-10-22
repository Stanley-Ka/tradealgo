from __future__ import annotations

"""Fetch recent news for a set of symbols and produce a digest.

Supports Polygon or Finnhub providers. Uses lightweight heuristic sentiment by
default; can optionally just list headlines without scoring. Outputs to stdout,
optionally to CSV, and can send a digest to Discord.
"""

import argparse
import os
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

from ..infra.yaml_config import load_yaml_config
from ..news.providers import fetch_news_multi
from ..news.sentiment import score_news


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="News digest from Polygon/Finnhub for a set of symbols"
    )
    p.add_argument("--config", type=str, default="", help="YAML with defaults")
    p.add_argument(
        "--universe-file",
        type=str,
        default="",
        help="Universe file (one SYMBOL per line)",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols (overrides universe)",
    )
    p.add_argument("--provider", type=str, default="polygon")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--start", type=str, default="", help="YYYY-MM-DD (overrides days)")
    p.add_argument("--end", type=str, default="", help="YYYY-MM-DD (overrides days)")
    p.add_argument(
        "--limit", type=int, default=10, help="Max items per symbol (best-effort)"
    )
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument(
        "--discord-webhook", type=str, default=os.environ.get("DISCORD_WEBHOOK_URL", "")
    )
    p.add_argument("--skip-on-rate-limit", action="store_true")
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
    cfg = load_yaml_config(args.config) if args.config else {}
    syms: List[str]
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe_file:
        syms = read_universe(args.universe_file)
    else:
        raise RuntimeError("Provide --symbols or --universe-file")
    if not syms:
        print("[news] empty symbol list")
        return
    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=int(args.days))
        start, end = start_dt.isoformat(), end_dt.isoformat()
    token = (
        os.environ.get("POLYGON_API_KEY", "")
        if args.provider == "polygon"
        else os.environ.get("FINNHUB_API_KEY", "")
    )

    items = []
    try:
        items = fetch_news_multi(
            syms, start=start, end=end, provider=args.provider, token=token
        )
    except Exception as e:
        msg = str(e)
        if args.skip_on_rate_limit and (
            "429" in msg or "Too Many Requests" in msg or "403" in msg
        ):
            print(f"[news] rate limited: {msg}; skipping")
            return
        raise
    if not items:
        print("[news] no items fetched")
        return

    # Score via heuristic sentiment
    avg, cnt, details = score_news(items)
    print(f"[news] items={len(items)} avg_sentiment={avg:+.3f}")
    # Build DataFrame
    rows = []
    for it, s in details:
        rows.append(
            {
                "date": it.date.date(),
                "symbol": it.symbol,
                "source": it.source,
                "headline": it.headline,
                "sentiment": s,
                "url": it.url,
            }
        )
    df = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    print(df.head(20).to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[news] saved -> {args.out_csv}")

    if args.discord_webhook:
        try:
            from ..infra.notify import send_discord

            lines = [
                "News Digest",
                f"Symbols: {', '.join(syms[:20])}",
                f"Avg sentiment {avg:+.2f} from {len(items)} items",
                "Top headlines:",
            ]
            for it, s in details[:5]:
                lines.append(
                    f"- {it.symbol} {it.date.date()} {s:+.2f}: {it.headline[:120]}"
                )
            send_discord(args.discord_webhook, "@everyone\n" + "\n".join(lines))
            print("[news] Discord digest sent")
        except Exception as e:
            print(f"[news] Discord failed: {e}")


if __name__ == "__main__":
    main()
