param(
  [string]$Features = "D:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Model = "D:\\EngineData\\models\\meta_hgb.pkl",
  [string]$Universe = "engine/data/universe/swing_aggressive.watchlist.txt",
  [string]$Out = "D:\\EngineData\\datasets\\swing_training_dataset.parquet",
  [string]$Timeframes = "3,7,14",
  [ValidateSet('close','open')][string]$EntryPrice = 'close',
  [int]$TopK = 20,
  [string]$Start = "2017-01-01",
  [string]$End = "",
  [double]$MinADV = 1e7,
  [double]$MaxATR = 0.05,
  [double]$TP = 0.02,
  [double]$SL = 0.03,
  [switch]$RequireAll,
  [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$python = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $python = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

$args = @(
  "-m", "engine.tools.build_swing_dataset",
  "--features", $Features,
  "--model-pkl", $Model,
  "--universe-file", $Universe,
  "--out", $Out,
  "--timeframes", $Timeframes,
  "--entry-price", $EntryPrice,
  "--top-k", $TopK,
  "--start", $Start,
  "--min-adv-usd", $MinADV,
  "--max-atr-pct", $MaxATR,
  "--tp-pct", $TP,
  "--sl-pct", $SL,
  "--add-hit-before-stop"
)
if ($End -and $End.Trim()) { $args += @("--end", $End.Trim()) }
if ($RequireAll) { $args += "--require-all-horizons" }
if ($Resume) { $args += "--resume" }

& $python $args
