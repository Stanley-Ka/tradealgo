param(
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Config = "engine/presets/swing_aggressive.yaml",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [string]$MetaPred = "data/datasets/meta_predictions.parquet",
  [string]$Model = "data/models/meta_hgb.pkl",
  [string]$MetaCalibrator = "",
  [switch]$WritePaths,
  [string]$ConfigOut = "",
  [string]$BacktestReport = "data/backtests/daily_topk_report.html",
  [string]$Universe = "engine/data/universe/us_all.txt",
  [int]$TopK = 20,
  [int]$CostBps = 5,
  [string]$Rebalance = "weekly",
  [string]$Weekday = "MON",
  [string]$Start = "",
  [string]$End = "",
  [switch]$BuildWatchlist = $true,
  [string]$WatchlistOut = "engine/data/universe/watchlist.txt",
  [int]$WLTopK = 500,
  [double]$WLMinPrice = 1.0,
  [double]$WLMinADV = 5000000,
  [switch]$RunAlert = $true,
  [switch]$DryRun = $true,
  [string]$PriceSource = "live",
  [string]$LiveProvider = "polygon",
  [int]$FromDays = 3
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

Write-Host "[train] Cross-validating specialists (isotonic, purged time-kfold + drift report)..."
$drift = "D:\\EngineData\\reports\\cv_drift.csv"
$cvArgs = @("-m","engine.models.run_cv","--features",$Features,"--label","label_up_1d","--cv-scheme","time_kfold","--kfolds","5","--purge-days","3","--embargo-days","3","--calibration","isotonic","--out",$OOF,"--calibrators-out",$Calibrators,"--spec-config",$Config,"--drift-report",$drift)
if ($Start -and $Start.Trim() -ne "") { $cvArgs += @("--start", $Start) }
if ($End -and $End.Trim() -ne "") { $cvArgs += @("--end", $End) }
& $py @cvArgs
if ($LASTEXITCODE -ne 0) { Write-Error "run_cv failed"; exit 1 }

Write-Host "[train] Training meta-learner (HGB) with isotonic meta calibration + odds/interactions..."
& $py -m engine.models.train_meta --oof $OOF --train-folds all-but-last:1 --test-folds last:1 --select-top-specs 6 --min-auc 0.505 --model hgb --hgb-learning-rate 0.05 --hgb-max-iter 400 --meta-calibration isotonic --replace-prob-with-calibrated --add-odds --add-interactions --out $MetaPred --model-out $Model $(if ($MetaCalibrator -and $MetaCalibrator.Trim() -ne "") {"--meta-calibrator-out", $MetaCalibrator})
if ($LASTEXITCODE -ne 0) { Write-Error "train_meta failed"; exit 1 }

Write-Host "[bt] Backtesting top-$TopK ($Rebalance)..."
& $py -m engine.backtest.simple_daily --features $Features --pred $MetaPred --prob-col meta_prob --top-k $TopK --cost-bps $CostBps --rebalance $Rebalance --rebal-weekday $Weekday --report-html $BacktestReport
if ($LASTEXITCODE -ne 0) { Write-Error "backtest failed"; exit 1 }

if ($BuildWatchlist) {
  Write-Host "[watchlist] Building Top-$WLTopK -> $WatchlistOut ..."
  & $py -m engine.tools.build_watchlist --features $Features --model-pkl $Model --oof $OOF --out $WatchlistOut --top-k $WLTopK --min-price $WLMinPrice --min-adv-usd $WLMinADV
}

if ($RunAlert) {
  Write-Host "[alert] Running trade alert (DryRun=$($DryRun.IsPresent)) ..."
  $args = @(
    "-m", "engine.tools.trade_alert",
    "--features", $Features,
    "--model-pkl", $Model,
    "--calibrators-pkl", $Calibrators,
    "--universe-file", $Universe,
    "--provider", "polygon",
    "--top-k", "5",
    "--price-source", $PriceSource,
    "--live-provider", $LiveProvider,
    "--from-days", $FromDays
  )
  if ($DryRun) { $args += "--dry-run" }
  if ($MetaCalibrator -and $MetaCalibrator.Trim() -ne "") { $args += @("--meta-calibrator-pkl", $MetaCalibrator) }
  & $py @args
}

Write-Host "[done] Train -> Meta -> Backtest -> (Watchlist) -> (Alert) complete."

if ($WritePaths -and $ConfigOut -and $ConfigOut.Trim() -ne "") {
  Write-Host "[paths] Writing resolved paths to $ConfigOut ..."
  $upd = @(
    "-m", "engine.tools.update_yaml_paths",
    "--in", $Config,
    "--out", $ConfigOut,
    "--features", $Features,
    "--oof", $OOF,
    "--calibrators", $Calibrators,
    "--meta", $MetaPred,
    "--meta-model", $Model,
    "--universe", $Universe
  )
  if ($MetaCalibrator -and $MetaCalibrator.Trim() -ne "") { $upd += @("--meta-calibrator", $MetaCalibrator) }
  & $py @upd
}
