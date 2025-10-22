param(
  [string]$Config = "engine/config.research.yaml",
  [int]$TopK = 3,
  [switch]$DryRun,
  [string]$Log = "data/alerts/alerts_log.csv",
  [switch]$PrintSpecs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Load API keys from scripts/api.env if present; fallback to .env
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.trade_alert",
  "--config", $Config,
  "--provider", "polygon",
  "--top-k", $TopK,
  "--alert-log-csv", $Log
)
if ($DryRun) { $argsList += "--dry-run" }
if ($PrintSpecs) { $argsList += "--print-specialist-probs" }

python @argsList
