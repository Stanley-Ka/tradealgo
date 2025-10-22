Polygon API Usage (Starter Plan Friendly)

This engine uses a resilient price and news cascade that works with Polygon’s Stocks Starter plan (15‑minute delay) and improves gracefully when you have real‑time access.

Pricing endpoints (order of attempts)
- v2 last trade: `GET https://api.polygon.io/v2/last/trade/{ticker}`
  - Fields: `results.p` (price)
  - Notes: real‑time if entitled; may rate‑limit or return auth errors on some plans.
- v2 single‑ticker snapshot: `GET https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}`
  - Fields: `ticker.lastTrade.p`, `ticker.lastQuote.p`, `ticker.day.c`
  - Notes: provides a consolidated view (latest trade/quote + today’s OHLC). Delayed on Starter; effectively “near‑real‑time” within your plan’s constraints.
- v2 previous close (adjusted): `GET https://api.polygon.io/v2/aggs/ticker/{ticker}/prev`
  - Fields: `results[0].c` (close)
  - Notes: last resort for a valid reference price when live routes fail.

Yahoo fallbacks (no key required)
- yfinance fast_info: last_price via Python `yfinance.Ticker(sym).fast_info.last_price`
- Public quote API: `https://query1.finance.yahoo.com/v7/finance/quote?symbols={sym}` → `quoteResponse.result[0].regularMarketPrice`

News endpoint (Polygon)
- v2 reference news: `GET https://api.polygon.io/v2/reference/news`
  - Common params: `ticker`, `published_utc.gte`, `published_utc.lte`, `order`, `limit`
  - Fields used: `title`, `description`, `article_url`, `published_utc`, `source`, `tickers`

Plans and what we use
- Starter (common): No entitlement to last trade/last quote. Use: Aggregates Previous Day (prev) and reference/news. The engine will skip last trade and snapshot when `alert.polygon_plan: starter` or `--polygon-plan starter` is set, and rely on Yahoo for intraday, falling back to Polygon `prev` if Yahoo is unavailable.
- Pro/Enterprise: Attempt last trade first, then snapshot, then Yahoo, then Polygon `prev`.

Reference coverage and helpers
- Universe: `engine.tools.build_universe_polygon` pulls active US stocks via v3 reference.
- Sector/industry/market cap: `engine.tools.build_sector_map_polygon` queries v3 ticker details per symbol.
- Dividends: `engine.tools.build_dividends_map` queries v3 dividends (by ticker and date range); use with `entry_loop --dividends-csv ... --exdiv-blackout-days N` to avoid entries near ex-div.

How pricing is picked at runtime
- Default cascade: Polygon last trade → Polygon snapshot → Yahoo (fast_info → quote) → Polygon previous close.
- With `polygon_plan: starter`: Skip last trade and snapshot; use Yahoo fast_info → Yahoo quote → Polygon previous close.
- The `live_provider` is treated as a preference; the engine still tries the best available route allowed by your plan.

Operational tips
- Set `POLYGON_API_KEY` in environment. Use `scripts/api.env` and load with the helper scripts.
- Expect rate limits on free tiers; Yahoo fallback helps avoid hard failures and “prev close” sizing.
- To force live pricing even with stale features, use `alert.price_source: live` in your YAML.

Implementation notes
- All REST calls go through `engine.data.polygon_client.PolygonClient` to apply retry/backoff and basic rate limiting.
- Daily bars use Aggregates v2 with `adjusted=true` to produce `adj_*` columns for the feature pipeline.
