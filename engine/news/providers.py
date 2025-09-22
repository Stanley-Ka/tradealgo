from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import pandas as pd


@dataclass
class NewsItem:
    date: pd.Timestamp
    symbol: str
    source: str
    headline: str
    summary: str
    url: str


def _finnhub_company_news(symbol: str, start: str, end: str, token: Optional[str] = None) -> List[NewsItem]:
    key = token or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set; cannot fetch news.")
    url = "https://finnhub.io/api/v1/company-news"
    resp = requests.get(url, params={"symbol": symbol, "from": start, "to": end, "token": key}, timeout=30)
    resp.raise_for_status()
    data = resp.json() or []
    out: List[NewsItem] = []
    for row in data:
        # Finnhub fields: datetime (timestamp), headline, summary, url, source
        ts = pd.to_datetime(row.get("datetime", 0), unit="s") if "datetime" in row else pd.Timestamp.utcnow()
        out.append(
            NewsItem(
                date=ts.tz_localize(None),
                symbol=symbol.upper(),
                source=str(row.get("source", "finnhub")),
                headline=str(row.get("headline", "")),
                summary=str(row.get("summary", "")),
                url=str(row.get("url", "")),
            )
        )
    return out


def fetch_news(
    symbol: str,
    start: str,
    end: str,
    provider: str = "finnhub",
    token: Optional[str] = None,
) -> List[NewsItem]:
    """Fetch recent company news for a symbol.

    Currently supports provider "finnhub" via REST.
    """
    provider = provider.lower()
    if provider == "finnhub":
        return _finnhub_company_news(symbol, start, end, token=token)
    raise NotImplementedError(f"Unknown provider: {provider}")

