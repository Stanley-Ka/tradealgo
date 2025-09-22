# Trading Engine Scaffold

This repository initializes a research and execution pipeline for a US equities swing-trading bot built around four specialist models whose calibrated probabilities are combined by a meta-learner into a final trading decision.

Key ideas mirrored here:
- Four specialists (V0): patterns (candlesticks), technical indicators, light sequence proxy, and news/NLP (optional sentiment input).
- Calibrate probabilities (e.g., Platt or isotonic) before combining.
- Meta-learner (stacking) over calibrated probabilities + context features.
- Portfolio and execution layers incorporate costs, constraints, and risk checks.

## Layout

```
engine/
  data/            # parquet, cached metadata (storage, ignored by git)
  features/        # five specialists + calibration utilities
  models/          # meta-learner, model registry
  portfolio/       # sizing, constraints, risk checks
  exec/            # broker adapters (IBKR, Tradier) + base interface
  infra/           # config, logging, scheduling stubs
  main_backtest.py # entry for research/backtests
  main_live.py     # entry for live/paper execution
  settings.toml    # project settings (non-secret)
```

## Quick start

1) Create a virtual environment and install requirements:
```
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Configure settings in `engine/settings.toml`; copy secrets to a local `.env` (do not commit).

3) Run backtest stub:
```
python -m engine.main_backtest
```

4) Run live stub (paper):
```
python -m engine.main_live
```

Capabilities overview: see `ENGINE_CAPABILITIES.txt` for all settings and commands.
Market hours (local timezone):
- Show NASDAQ/NYSE session times: `python -m engine.tools.market_time_info --calendar NASDAQ`
- Run only during market hours: `python -m engine.tools.run_during_market -- python -m engine.data.stream_finnhub --symbols AAPL,MSFT`

Legal/tax notes for AU→US trading: see `LEGAL_NOTES_AU_US.md` (not legal advice).

### Live price stream (no visualization)

To stream live US stock trades in your terminal using Finnhub WebSocket:

```
# Set your Finnhub API key (https://finnhub.io)
export FINNHUB_API_KEY=YOUR_KEY_HERE   # Windows PowerShell: $env:FINNHUB_API_KEY="YOUR_KEY_HERE"

# Stream trades for AAPL and MSFT
python -m engine.data.stream_finnhub --symbols AAPL,MSFT
```

Notes:
- Free Finnhub plans have rate/usage limits and may differ in coverage/latency.
- Output prints a line per trade with timestamp, symbol, price, and size.

To also print OHLCV candles aggregated from trades (e.g., 1-minute bars):

```
python -m engine.data.stream_finnhub --symbols AAPL,MSFT --ohlc-interval 1m --status-interval 10
```

- `--ohlc-interval` supports `1s,5s,1m,5m,1h` etc. Candles print when the interval closes.
- Use `--debug` to see pings and raw messages if nothing prints (off-hours).

## Next steps

- Implement data adapters (e.g., Polygon/Tiingo) in `engine/data/` and persist to Parquet.
- Alpha Vantage daily-adjusted fetcher and dataset builder are available:
  - Build dataset: `python -m engine.data.build_dataset --universe-file engine/data/universe/nasdaq100.example.txt --start 2010-01-01`
  - Set `ALPHAVANTAGE_API_KEY` (or use `--api-key`) and respect rate limits.
- Flesh out the five specialists in `engine/features/` and add calibration.
- Build baseline features for a universe/date range:
  - `python -m engine.features.build_features --universe-file engine/data/universe/nasdaq100.example.txt --start 2015-01-01 --out datasets/features_nasdaq100_1D.parquet`
- Implement the meta-learner in `engine/models/meta_learner.py` and wire into backtests.
- Run time-based CV + calibration on specialists (optionally with news sentiment file):
  - `python -m engine.models.run_cv --features data/datasets/features_nasdaq100_1D.parquet --label label_up_1d --calibration platt --out data/datasets/oof_specialists.parquet [--news-sentiment path/to/sentiment.parquet]`
- Train the meta-learner and backtest daily top-K:
  - Train meta: `python -m engine.models.train_meta --oof data/datasets/oof_specialists.parquet --train-folds all-but-last:1 --test-folds last:1 --out data/datasets/meta_predictions.parquet`
  - Backtest: `python -m engine.backtest.simple_daily --features data/datasets/features_daily_1D.parquet --pred data/datasets/meta_predictions.parquet --prob-col meta_prob --top-k 20 --cost-bps 5 --rebalance weekly --rebal-weekday MON --turnover-cap 0.5 --report-csv data/backtests/daily_topk_results.csv --report-html data/backtests/daily_topk_report.html --mlflow --mlflow-experiment research-backtest`

End-to-end research runner (chains all steps with sensible defaults):
- `python -m engine.research.runner --universe-file engine/data/universe/nasdaq100.example.txt --start 2015-01-01 --calibration platt --top-k 20 --cost-bps 5 --rebalance weekly --rebal-weekday MON --turnover-cap 0.5 --report-csv data/backtests/daily_topk_results.csv --report-html data/backtests/daily_topk_report.html --mlflow --mlflow-experiment research-e2e --run-name pipeline`

CV options:
- Use time_kfold with purge/embargo and MLflow logging: `python -m engine.models.run_cv --features data/datasets/features_nasdaq100_1D.parquet --cv-scheme time_kfold --kfolds 5 --purge-days 5 --embargo-days 5 --out data/datasets/oof_specialists.parquet --mlflow --mlflow-experiment research-cv`
- Add cost model + constraints to `engine/portfolio/` and connect a backtest engine (Lean/backtrader).
- Implement broker adapters in `engine/exec/` (IBKR via ib_insync, Tradier via REST) and test paper trading.
### Predict Daily Picks (Meta)

After training a meta model, you can produce a ranked list for the latest date:

```
python -m engine.tools.predict_daily \
  --features data/datasets/features_daily_1D.parquet \
  --model-pkl data/models/meta_lr.pkl \
  --oof data/datasets/oof_specialists.parquet \
  --news-sentiment data/datasets/dummy_sentiment.parquet \
  --top-k 20 --out-csv data/signals/picks.csv
```

Notes:
- If `--oof` is provided, per-specialist calibrators are fit from OOF raw->prob and applied to today’s scores.
- Without `--oof`, a naive mapping maps raw [-1,1] to prob [0,1].

### Dummy Sentiment Builder

Create a placeholder sentiment file (for format/testing only):

```
python -m engine.data.build_dummy_sentiment \
  --features data/datasets/features_daily_1D.parquet \
  --out data/datasets/dummy_sentiment.parquet --noise 0.1
```

Outputs columns `date,symbol,sentiment` in [-1,1].

### Trade Alert (Top Setup + News)

Find the most promising setup in a universe (e.g., NASDAQ‑100) and fetch recent news with a simple sentiment score:

```
python -m engine.tools.trade_alert \
  --features data/datasets/features_daily_1D.parquet \
  --model-pkl data/models/meta_lr.pkl \
  --oof data/datasets/oof_specialists.parquet \
  --universe-file engine/data/universe/nasdaq100.example.txt \
  --provider finnhub --from-days 3 --top-k 1
```

Notes:
- Requires `FINNHUB_API_KEY` for news fetching.
- Optional calibrators file (avoids needing OOF at prediction time): run CV with `--calibrators-out data/models/spec_calibrators.pkl` and then pass `--calibrators-pkl` to predictors/alert.
- Risk gates: use `--min-adv-usd` (20D average dollar volume), `--max-atr-pct`, and optionally provide an `--earnings-file` with `date,symbol` to apply a blackout via `--earnings-blackout`.
- Notifications: set `--slack-webhook` or `--discord-webhook` (or env `SLACK_WEBHOOK_URL` / `DISCORD_WEBHOOK_URL`).

### YAML Configuration (sensitivity and defaults)

You can centralize critical parameters in a YAML file. An example is provided at `engine/config.example.yaml`:

```
python -m engine.tools.trade_alert --config engine/config.example.yaml
python -m engine.tools.predict_daily --config engine/config.example.yaml
```

CLI flags still override YAML values. The YAML includes specialist weights/windows, risk gates, calibrator paths, universe file, provider, and Discord webhook.
