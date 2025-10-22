# Trading Engine Scaffold

Research‑first US equities engine focused on swing trading: daily features, specialist ensemble with calibration, logistic/meta learners, simple backtests, Discord trade alerts, and a paper trader. A Streamlit UI is included.

Note: The engine is now swing‑only. Intraday style presets have been removed to simplify training and operations for swing strategies. Use the swing presets in `engine/presets/` (aggressive/conservative).

See `ENGINE_CAPABILITIES.txt` for command references and `ROADMAP.md` for planned phases. For Polygon endpoint details and pricing fallbacks, read `docs/polygon_api.md`.

## Install

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Secrets: never commit API keys. Use environment variables or `scripts/api.env` (used by PowerShell scripts).

## Quick Start

Build data and features (Yahoo, free):
```
python -m engine.data.build_dataset --provider yahoo --universe-file engine/data/universe/nasdaq100.example.txt --start 2015-01-01
python -m engine.features.build_features --universe-file engine/data/universe/nasdaq100.example.txt --start 2015-01-01 --out data/datasets/features_daily_1D.parquet
```

Cross‑validate specialists and train meta:
```
python -m engine.models.run_cv --features data/datasets/features_daily_1D.parquet --label label_up_1d --calibration platt --out data/datasets/oof_specialists.parquet --calibrators-out data/models/spec_calibrators.pkl
python -m engine.models.train_meta --oof data/datasets/oof_specialists.parquet --train-folds all-but-last:1 --test-folds last:1 --out data/datasets/meta_predictions.parquet --model-out data/models/meta_lr.pkl
```

Backtest top‑K strategy:
```
python -m engine.backtest.simple_daily --features data/datasets/features_daily_1D.parquet --pred data/datasets/meta_predictions.parquet --prob-col meta_prob --top-k 20 --cost-bps 5 --report-html data/backtests/daily_topk_report.html
```

Trade alert (Discord or dry‑run):
```
python -m engine.tools.trade_alert --features data/datasets/features_daily_1D.parquet --model-pkl data/models/meta_lr.pkl --universe-file engine/data/universe/nasdaq100.example.txt --provider none --top-k 3 --dry-run
```

Live price source notes:
- When `--price-source live` is used, the engine now tries: Polygon last trade → Polygon snapshot → Yahoo fast_info → Yahoo quote → Polygon previous close. This works even if `--live-provider polygon` is set.
- Quick workaround if Polygon last trade is blocked by plan/rate‑limit: add `--price-source live --live-provider yahoo`.
 - Starter plan hint: set `alert.polygon_plan: starter` (or `provider.polygon_plan: starter`) in any preset to gracefully skip unauthorized Polygon live endpoints.

Polygon reference utilities:
- Build US stocks universe: `python -m engine.tools.build_universe_polygon --out engine/data/universe/us_all.txt --types CS --exchanges NASDAQ,NYSE,ARCA`
- Build sector map: `python -m engine.tools.build_sector_map_polygon --universe-file engine/data/universe/us_all.txt --out engine/data/sector_map.csv`
- Build dividends map and enable ex‑div blackout in entries: `python -m engine.tools.build_dividends_map --universe-file engine/data/universe/us_all.txt --start 2020-01-01 --out engine/data/dividends.csv`, then run entry loop with `--dividends-csv engine/data/dividends.csv --exdiv-blackout-days 1`

Real‑time alerts during market hours (Windows PowerShell):
```
./scripts/start-rt-alert.ps1 -Every 5
./scripts/stop-rt-alert.ps1
```
Or via Python directly (not limited to fixed times):
```
python -m engine.tools.real_time_alert --config engine/presets/ver1_polygon.yaml \
  --every-minutes 1 --price-source live --mix-intraday 0.3 --intraday-features data/datasets/features_intraday_latest.parquet \
  --alert-log-csv data/alerts/diag.csv --send-updates --demote-below-prob 0.50
```
This recalculates meta probabilities every minute (optionally blending intraday snapshot), sends compact Discord alerts with bold entry/stop, and follows up with delta updates (price/prob changes). Use `--cooldown-mins` to avoid repeats.

Test Discord webhooks and Polygon capabilities:
```
python -m engine.tools.validate_config --config engine/presets/ver1_polygon.yaml --send-discord-test
python -m engine.tools.polygon_probe
```

Paper trader (daily ledger update):
```
./scripts/update-paper-ledger.ps1 -Config engine/config.research.yaml
```

Weekly overview and dynamic specialist weighting:
```
# Step paper trader daily with specialist logging and dynamic weights
./scripts/paper-trade.ps1 -Config engine/config.research.yaml

# End of week: summarize closed trades, update weights, and write report
./scripts/weekly-overview.ps1 -DecisionLog data/paper/decision_log.csv -OutMD data/reports/weekly_overview.md -WeightsOut data/paper/specialist_weights.yaml -Weeks 1
```

Daily routine (manual, compact):
```
./scripts/daily-run.ps1 -Config engine/config.research.yaml
```

- Build a swing training dataset (multi‑horizon labels):
```
python -m engine.tools.build_swing_dataset \
  --features data/datasets/features_daily_1D.parquet \
  --model-pkl data/models/meta_hgb.pkl \
  --universe-file engine/data/universe/swing_aggressive.watchlist.txt \
  --timeframes 3,7,14 --entry-price close --top-k 20 \
  --start 2017-01-01 --out data/datasets/swing_training_dataset.parquet --resume
```
This writes per-entry rows with meta/specialist probabilities, ADV/ATR context, and ret_/label_up_3d/7d/14d for supervised training.

## UI and Utilities

- UI: `streamlit run ui/app.py`
- Market hours info: `python -m engine.tools.market_time_info --calendar NASDAQ`
- Run only during market hours: `python -m engine.tools.run_during_market -- python -m engine.tools.trade_alert --dry-run`
- Secret scanner: `python -m engine.tools.scan_secrets --all`

Legal/tax notes (AU→US): `LEGAL_NOTES_AU_US.md` (not legal advice).
