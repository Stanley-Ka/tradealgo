Phased Roadmap (Concise)
========================

Phase 0 — Daily Research (current)
- Daily features, specialist calibration, logistic meta, top‑K backtest. Tools: alerts, paper trader, Streamlit UI.

Phase 1 — Intraday Data Bridge
- Collect 1m/5m bars (Polygon) to rolling Parquets, restart‑safe.

Phase 2 — Intraday Features
- Port core specialists to bar windows; build “latest snapshot”.

Phase 3 — Intraday Scoring Loop
- Load calibrators/meta once; on each bar compute probs → rank with hysteresis.

Phase 4 — Entry Loop
- Threshold/top‑K policy with risk gates and sizing; persist positions and notify Discord.

Phase 5 — Position Manager
- Trailing stops, probability exits with confirmation, trims, max hold; log exits and PnL.

Phase 6 — Risk/Portfolio Layer
- Sector/name caps, turnover budget, cash accounting; improved sizing.

Phase 7 — Intraday Backtesting
- Bar‑level simulation validating entry/exit and constraints.

Immediate Next Items
- Implement `engine/tools/entry_loop.py` (daily proxy, already scaffolded) to produce entries on a schedule.
- Add Polygon bar collector stub and dry‑run path.
- Optional diversification: sector cap for alerts/paper trader.
