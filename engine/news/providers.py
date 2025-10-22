from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..infra.http import HttpClient, HttpConfig
import pandas as pd


@dataclass
class NewsItem:
    date: pd.Timestamp
    symbol: str
    source: str
    headline: str
    summary: str
    url: str


def _finnhub_company_news(
    symbol: str, start: str, end: str, token: Optional[str] = None
) -> List[NewsItem]:
    key = token or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set; cannot fetch news.")
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": start, "to": end, "token": key}
    # Use resilient HTTP client with modest rate limit
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=20.0))
    data = client.get_json(url, params=params) or []
    out: List[NewsItem] = []
    for row in data:
        # Finnhub fields: datetime (timestamp), headline, summary, url, source
        ts = (
            pd.to_datetime(row.get("datetime", 0), unit="s")
            if "datetime" in row
            else pd.Timestamp.utcnow()
        )
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


def _polygon_company_news(
    symbol: str, start: str, end: str, token: Optional[str] = None
) -> List[NewsItem]:
    key = token or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set; cannot fetch news.")
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": symbol.upper(),
        "published_utc.gte": start,
        "published_utc.lte": end,
        "order": "asc",
        "limit": 1000,
        "apiKey": key,
    }
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=20.0))
    data = client.get_json(url, params=params) or {}
    results = data.get("results", [])
    out: List[NewsItem] = []
    for row in results:
        ts = (
            pd.to_datetime(row.get("published_utc", None))
            if row.get("published_utc")
            else pd.Timestamp.utcnow()
        )
        out.append(
            NewsItem(
                date=ts.tz_localize(None),
                symbol=symbol.upper(),
                source=str(row.get("source", "polygon")),
                headline=str(row.get("title", "")),
                summary=str(row.get("description", "")),
                url=str(row.get("article_url", "")),
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
    if provider == "polygon":
        return _polygon_company_news(symbol, start, end, token=token)
    raise NotImplementedError(f"Unknown provider: {provider}")


def fetch_news_multi(
    symbols: List[str],
    start: str,
    end: str,
    provider: str = "polygon",
    token: Optional[str] = None,
) -> List[NewsItem]:
    """Fetch news for multiple symbols and concatenate the results.

    Uses the per-symbol fetchers under the hood to remain rate-limit friendly.
    """
    out: List[NewsItem] = []
    for s in symbols:
        try:
            out.extend(fetch_news(s, start, end, provider=provider, token=token))
        except Exception:
            continue
    return sorted(out, key=lambda x: x.date)


# --------------- Advanced (query/limit/order/sort) ---------------


def _polygon_news_generic(
    symbol: Optional[str],
    start: str,
    end: str,
    token: Optional[str],
    limit: Optional[int] = None,
    order: str = "asc",
    sort: str = "published_utc",
    search: str = "",
) -> List[NewsItem]:
    key = token or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set; cannot fetch news.")
    url = "https://api.polygon.io/v2/reference/news"
    params: Dict[str, Any] = {
        "order": order,
        "sort": sort,
        "published_utc.gte": start,
        "published_utc.lte": end,
        "apiKey": key,
    }
    if symbol:
        params["ticker"] = symbol.upper()
    if limit:
        params["limit"] = int(limit)
    if search:
        params["search"] = str(search)
    client = HttpClient(HttpConfig(requests_per_second=3.0, timeout=20.0))
    data = client.get_json(url, params=params) or {}
    results = data.get("results", [])
    out: List[NewsItem] = []
    for row in results:
        ts = (
            pd.to_datetime(row.get("published_utc", None))
            if row.get("published_utc")
            else pd.Timestamp.utcnow()
        )
        out.append(
            NewsItem(
                date=ts.tz_localize(None),
                symbol=(
                    row.get("tickers", [symbol])[0]
                    if row.get("tickers")
                    else (symbol or "")
                ),
                source=str(row.get("source", "polygon")),
                headline=str(row.get("title", "")),
                summary=str(row.get("description", "")),
                url=str(row.get("article_url", "")),
            )
        )
    return out


def fetch_news_advanced(
    symbols: Optional[List[str]],
    start: str,
    end: str,
    provider: str = "polygon",
    limit: Optional[int] = None,
    order: str = "asc",
    sort: str = "published_utc",
    search: str = "",
    token: Optional[str] = None,
) -> List[NewsItem]:
    """Advanced news fetch with limit/order/sort/search and optional symbols.

    For Polygon, when symbols is None or empty, fetches global news by query/date range.
    """
    provider = provider.lower()
    if provider != "polygon":
        # For now delegate to legacy functions per symbol without advanced params
        syms = symbols or []
        out: List[NewsItem] = []
        for s in syms:
            out.extend(fetch_news(s, start, end, provider=provider, token=token))
        return sorted(out, key=lambda x: x.date)
    if not symbols:
        return _polygon_news_generic(
            None, start, end, token, limit=limit, order=order, sort=sort, search=search
        )
    out: List[NewsItem] = []
    for s in symbols:
        out.extend(
            _polygon_news_generic(
                s, start, end, token, limit=limit, order=order, sort=sort, search=search
            )
        )
    return sorted(out, key=lambda x: x.date)
