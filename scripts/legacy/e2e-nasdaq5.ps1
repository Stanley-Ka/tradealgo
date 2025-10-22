param(
  [string]$Universe = "engine/data/universe/nasdaq5.example.txt",
  [string]$Start = "2018-01-01",
  [string]$End = "2020-01-01",
  [string]$FeaturesOut = "data/datasets/features_daily_1D.parquet",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibs = "data/models/spec_calibrators.pkl",
  [string]$MetaPred = "data/datasets/meta_predictions.parquet",
  [string]$MetaModel = "data/models/meta_lr.pkl",
  [string]$ReportHtml = "data/backtests/daily_topk_report.html"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\..\load-env.ps1" -EnvFile (Join-Path (Join-Path $scriptDir "..") "api.env") | Out-Null

Write-Host "[e2e] Building dataset (Yahoo) for $Universe ..."
python -m engine.data.build_dataset --provider yahoo --universe-file $Universe --start $Start --end $End

Write-Host "[e2e] Building features ..."
python -m engine.features.build_features --universe-file $Universe --provider yahoo --start $Start --out $FeaturesOut

Write-Host "[e2e] Running CV + calibration (time_kfold) ..."
python -m engine.models.run_cv --features $FeaturesOut --label label_up_1d --calibration platt --cv-scheme time_kfold --kfolds 3 --purge-days 3 --embargo-days 3 --out $OOF --calibrators-out $Calibs

Write-Host "[e2e] Training meta-learner ..."
python -m engine.models.train_meta --oof $OOF --train-folds all-but-last:1 --test-folds last:1 --out $MetaPred --model-out $MetaModel

Write-Host "[e2e] Backtesting ..."
python -m engine.backtest.simple_daily --features $FeaturesOut --pred $MetaPred --prob-col meta_prob --top-k 5 --cost-bps 5 --report-html $ReportHtml

Write-Host "[e2e] Done. Report: $ReportHtml"
