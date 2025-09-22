# Trading Engine Scaffold

This repository initializes a research and execution pipeline for a US equities swing-trading bot built around five specialist models whose calibrated probabilities are combined by a meta-learner into a final trading decision.

Key ideas mirrored here:
- Five specialists: patterns, technical indicators, sequence model, news/NLP, and an alt-data model.
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

## Next steps

- Implement data adapters (e.g., Polygon/Tiingo) in `engine/data/` and persist to Parquet.
- Flesh out the five specialists in `engine/features/` and add calibration.
- Implement the meta-learner in `engine/models/meta_learner.py` and wire into backtests.
- Add cost model + constraints to `engine/portfolio/` and connect a backtest engine (Lean/backtrader).
- Implement broker adapters in `engine/exec/` (IBKR via ib_insync, Tradier via REST) and test paper trading.

