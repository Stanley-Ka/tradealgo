param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Pred = "",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [string]$Universe = "engine/data/universe/daily_watchlist.txt",
  [int]$TopK = 20,
  [double]$CostBps = 5,
  [double]$TurnoverCap = 0.3,
  [string]$DecisionLog = "data/paper/decision_log.csv",
  [string]$WeightsYaml = "data/paper/specialist_weights.yaml",
  [switch]$UseSpecialistWeights=$true,
  [switch]$FallbackPrevDate=$true,
  [string]$MetaCalibrator = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.paper_trader",
  "--config", $Config,
  "--universe-file", $Universe,
  "--top-k", $TopK,
  "--state-dir", "data/paper",
  "--cost-bps", $CostBps,
  "--decision-log-csv", $DecisionLog,
  "--log-specialists"
)
if (Test-Path $Features) { $argsList += @("--features", $Features) }
if ($Pred -and (Test-Path $Pred) -and (-not $UseSpecialistWeights)) {
  $argsList += @("--pred", $Pred)
} else {
  $argsList += @("--model-pkl", $Model)
  if (Test-Path $OOF) { $argsList += @("--oof", $OOF) }
  if (Test-Path $Calibrators) { $argsList += @("--calibrators-pkl", $Calibrators) }
}
if ($TurnoverCap -gt 0) { $argsList += @("--turnover-cap", $TurnoverCap) }
if ($UseSpecialistWeights) {
  $argsList += @("--use-specialist-weights")
  if ($WeightsYaml) { $argsList += @("--specialist-weights-yaml", $WeightsYaml) }
}
if ($FallbackPrevDate) { $argsList += @("--fallback-prev-date") }
if ($MetaCalibrator -and (Test-Path $MetaCalibrator)) { $argsList += @("--meta-calibrator-pkl", $MetaCalibrator) }

python @argsList
