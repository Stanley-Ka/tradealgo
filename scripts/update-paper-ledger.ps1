param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Pred = "data/datasets/meta_predictions.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [int]$TopK = 20,
  [string]$StateDir = "data/paper",
  [double]$CostBps = 5,
  [double]$TurnoverCap = 0.3
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Prefer predictions if present; else fall back to model+oof/calibrators
$argsList = @(
  "-m", "engine.tools.paper_trader",
  "--config", $Config,
  "--features", $Features,
  "--universe-file", $Universe,
  "--top-k", $TopK,
  "--state-dir", $StateDir,
  "--cost-bps", $CostBps
)
if (Test-Path $Pred) {
  $argsList += @("--pred", $Pred)
} else {
  $argsList += @("--model-pkl", $Model)
  if (Test-Path $OOF) { $argsList += @("--oof", $OOF) }
  if (Test-Path $Calibrators) { $argsList += @("--calibrators-pkl", $Calibrators) }
}
if ($TurnoverCap -gt 0) { $argsList += @("--turnover-cap", $TurnoverCap) }

python @argsList
