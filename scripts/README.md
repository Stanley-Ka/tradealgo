Scripts overview

Core scripts (manual use)
- daily-run.ps1: Build watchlist → paper trade step (logs specialists) → online updates; optional weekly overview on weekends.
  - If `engine/data/universe/premarket_refined.txt` exists (from pre-market.ps1), it is used as the trading universe instead of the broad daily watchlist.
- pre-market.ps1: Just before open — fetch recent news sentiment for your current watchlist and refine it to a top‑5 shortlist for the evening trade.
  - Options: `-IntradayFeatures` + `-MixIntraday` to blend intraday snapshot into scoring, `-IntradayAlpha` for extra alpha from intraday gap/volume/VWAP/breakout, `-SectorMapCsv` + `-SectorBoost` to favor strong sectors (no diversification), `-EarningsFile` + `-EarningsBlackout` to gate around earnings, `-BuildEarnings` to refresh earnings map.
- daily-build.ps1: Build a daily watchlist (top 50 by meta_prob) with liquidity/price filters → engine/data/universe/daily_watchlist.txt.
- paper-trade.ps1: Step the paper trader once using the watchlist/universe; logs specialists and decisions; supports dynamic specialist weights.
- daily-train.ps1: Online updates from the decision log (per-specialist calibrators + meta refit).
- weekly-overview.ps1: Generate weekly summary (closed trades, win rate), report best 3‑specialist combos, and update specialist_weights.yaml.
- weekly-build.ps1: Build a larger weekly watchlist (top 200) for research → engine/data/universe/weekly_watchlist.txt.
- backtest.ps1: Run simple daily top‑K backtest with an existing predictions parquet.

Real-time (optional)
- start-overnight.ps1: Launch real-time alerts and entry scheduler; optional RT position manager; writes PID files.
- stop-rt-alert.ps1 / stop-entry.ps1: Stop running workers via PID files.
- status.ps1: Show status of real-time alerts and entry scheduler.
- start-rt-alert.ps1 / start-entry.ps1: Launch one worker at a time.

Logging outputs (by default paths)
- Alerts diagnostics: `data/alerts/alert_diag.csv` (from trade_alert; includes symbol, meta_prob, gates).
- Alert triggers: `data/alerts/alert_log.parquet` (real_time_alert dedupe state).
- Entries decided: `data/paper/entry_log.csv` (from entry_scheduler → entry_loop).
- Positions: `data/paper/positions.csv`; RT manager actions: `data/paper/trade_log.csv`.
- Paper ledger/equity: `data/paper/ledger.parquet`, `data/paper/weights.parquet`.

Nightly reflection (recommended)
- Update expectation mapping from decisions and realized outcomes:
  ./scripts/update-expectation.ps1 -DryRun   # review suggestion
  ./scripts/update-expectation.ps1           # apply to YAML (risk.expected_k)

Model training cadence
- Keep meta fixed day-to-day; retrain weekly with isotonic + HGB:
  python -m engine.models.run_cv --features data/datasets/features_daily_1D.parquet --label label_up_1d --cv-scheme time_kfold --kfolds 5 --purge-days 3 --embargo-days 3 --calibration isotonic --out data/datasets/oof_specialists.parquet --calibrators-out data/models/spec_calibrators.pkl
  python -m engine.models.train_meta --oof data/datasets/oof_specialists.parquet --train-folds all-but-last:1 --test-folds last:1 --model hgb --hgb-learning-rate 0.05 --hgb-max-iter 400 --meta-calibration isotonic --replace-prob-with-calibrated --out data/datasets/meta_predictions.parquet --model-out data/models/meta_hgb.pkl

Advanced/optional
- daily.ps1: YAML-driven daily pipeline (sentiment → predict → alert).
- run-trade-alert.ps1 / run-entry.ps1: One-off invocations for alert or entry selection.
- start-intraday.ps1: Intraday snapshot loop scaffold.
- intraday-entry-demo.ps1 / intraday-one-symbol.ps1: Demo scaffolds for intraday bar building and sample runs.
- build-yahoo.ps1 / e2e-nasdaq5.ps1: Dataset build/backtest demo (useful for research/testing).

Tips
- All scripts load environment variables from scripts/api.env if present (see example keys below).
- Direct Python tools also auto-load scripts/api.env now (no manual env needed).
- For unattended runs, you can still use Windows Task Scheduler to trigger any of these scripts on a schedule if needed.
- Prefer start-overnight.ps1 for unattended runs during US hours.
 - For larger universes (e.g., US all), run update-universe.ps1 to refresh the list, then rebuild-polygon.ps1 to rebuild data/features.

Example: Train + backtest on D:\
- ./scripts/weekly-train.ps1 -Features "C:\\EngineData\\datasets\\features_daily_1D.parquet" -Preset engine/presets/research.yaml -OOF "C:\\EngineData\\datasets\\oof_specialists.parquet" -Calibrators "C:\\EngineData\\models\\spec_calibrators.pkl" -MetaPred "C:\\EngineData\\datasets\\meta_predictions.parquet" -Model "C:\\EngineData\\models\\meta_hgb.pkl" -BacktestReport "C:\\EngineData\\backtests\\daily_topk_report.html" -BuildWatchlist -WatchlistOut engine/data/universe/watchlist.txt -WLTopK 500 -RunAlert -DryRun

Routine vs specific
- Routine: `pre-market.ps1` before open, `daily-run.ps1` after close (daily).
- Specific: `weekly-overview.ps1`, `build-earnings-map.ps1`, `backtest.ps1`, `train-and-backtest.ps1` as needed.

Quick recipes
- Daily (manual): `./scripts/daily-run.ps1 -Config engine/config.research.yaml`
- Pre‑market shortlist (top 5): `./scripts/pre-market.ps1 -Config engine/config.research.yaml`
  - Add intraday mix: `./scripts/pre-market.ps1 -Config engine/config.research.yaml -IntradayFeatures data/datasets/features_intraday_latest.parquet -MixIntraday 0.3 -IntradayAlpha 0.15`
  - Build/refresh earnings map during run: `./scripts/pre-market.ps1 -Config engine/config.research.yaml -BuildEarnings -EarningsFile data/events/earnings.parquet`
- Run specific scripts at times without Task Scheduler:
  ```
  ./scripts/run-at-times.ps1 -Events @(
    '08:45 | ./scripts/pre-market.ps1 -Config engine/config.research.yaml -IntradayFeatures data/datasets/features_intraday_latest.parquet -MixIntraday 0.3 -IntradayAlpha 0.15 -EarningsFile data/events/earnings.parquet -EarningsBlackout 2',
    '16:10 | ./scripts/daily-run.ps1 -Config engine/config.research.yaml'
  )
  ```
  - Add `-RepeatDaily` to loop every day.
  - Use day filters: `MON-FRI 08:45 | …` or `MON,WED,FRI 09:00 | …`.
  - YAML schedule:
    ```
    repeatDaily: true
    events:
      - time: "08:45"
        days: [MON, TUE, WED, THU, FRI]
        cmd: ./scripts/pre-market.ps1 -Config engine/config.research.yaml -IntradayFeatures data/datasets/features_intraday_latest.parquet -MixIntraday 0.3 -IntradayAlpha 0.15 -EarningsFile data/events/earnings.parquet -EarningsBlackout 2
      - time: "16:10"
        days: [MON, TUE, WED, THU, FRI]
        cmd: ./scripts/daily-run.ps1 -Config engine/config.research.yaml
    ```
    Run with: `./scripts/run-at-times.ps1 -ScheduleFile schedule.yaml`
- Friday/Sat weekly report: `./scripts/weekly-overview.ps1`

Notes on legacy scripts
- train-and-backtest.ps1 remains the core orchestrator (parameterized). The `train-d.ps1` and `train-d-conservative.ps1` wrappers are now superseded by `weekly-train.ps1` but kept for compatibility.


Environment (api.env)
POLYGON_API_KEY=your_polygon_key
# Alerts (recommendations) channel webhook
DISCORD_ALERTS_WEBHOOK_URL=https://discord.com/api/webhooks/...
# Paper trading executions channel webhook
DISCORD_TRADES_WEBHOOK_URL=https://discord.com/api/webhooks/...
# Legacy fallback used if the above are not set
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
FINNHUB_API_KEY=your_finnhub_key   # if using finnhub for news
