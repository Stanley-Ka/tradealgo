param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [int]$TopK = 50,
  [double]$MinADV = 10000000,
  [double]$MinPrice = 1,
  [double]$MaxPrice = 10000,
  [string]$Out = "engine/data/universe/daily_watchlist.txt"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.build_watchlist",
  "--config", $Config,
  "--out", $Out,
  "--top-k", $TopK,
  "--min-adv-usd", $MinADV,
  "--min-price", $MinPrice,
  "--max-price", $MaxPrice
)
if (Test-Path $Features) { $argsList += @("--features", $Features) }
if (Test-Path $Model) { $argsList += @("--model-pkl", $Model) }
if (Test-Path $OOF) { $argsList += @("--oof", $OOF) }
if (Test-Path $Calibrators) { $argsList += @("--calibrators-pkl", $Calibrators) }

python @argsList
