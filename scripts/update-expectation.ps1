param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Decisions = "data/paper/entry_log.csv",
  [double]$BaseProb = 0.5,
  [int]$Lookahead = 5,
  [int]$MinObs = 30,
  [switch]$DryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.update_expectation_from_logs",
  "--config", $Config,
  "--features", $Features,
  "--decisions", $Decisions,
  "--base-prob", $BaseProb,
  "--lookahead", $Lookahead,
  "--min-observations", $MinObs
)
if ($DryRun) { $argsList += "--dry-run" }

python @argsList
