param(
  [string]$DecisionLog = "data/paper/decision_log.csv",
  [string]$OutMD = "data/reports/weekly_overview.md",
  [string]$WeightsOut = "data/paper/specialist_weights.yaml",
  [int]$Weeks = 1,
  [int]$MinComboSamples = 5
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.weekly_overview",
  "--decision-log", $DecisionLog,
  "--out-md", $OutMD,
  "--weights-out", $WeightsOut,
  "--weeks", $Weeks,
  "--min-combo-samples", $MinComboSamples
)

python @argsList
