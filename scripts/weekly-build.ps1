param(
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [int]$TopK = 200,
  [double]$MinADV = 15000000,
  [double]$MinPrice = 1,
  [double]$MaxPrice = 10000,
  [string]$Out = "engine/data/universe/weekly_watchlist.txt"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.build_watchlist",
  "--features", $Features,
  "--model-pkl", $Model,
  "--out", $Out,
  "--top-k", $TopK,
  "--min-adv-usd", $MinADV,
  "--min-price", $MinPrice,
  "--max-price", $MaxPrice
)
if (Test-Path $OOF) { $argsList += @("--oof", $OOF) }
if (Test-Path $Calibrators) { $argsList += @("--calibrators-pkl", $Calibrators) }

python @argsList
