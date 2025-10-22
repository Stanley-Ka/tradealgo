param(
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Decisions = "",
  [int]$Lookahead = 5,
  [double]$TargetPct = 0.01,
  [int]$Rows = 10
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Prefer alert webhook; fallback to generic webhook
$hook = $env:DISCORD_ALERTS_WEBHOOK_URL
if (-not $hook -or $hook.Trim() -eq "") { $hook = $env:DISCORD_WEBHOOK_URL }

$argsList = @(
  "-m", "engine.tools.daily_summary",
  "--features", $Features,
  "--lookahead", $Lookahead,
  "--target-pct", $TargetPct,
  "--print-rows", $Rows
)
if ($Decisions -and $Decisions.Trim() -ne "") {
  $argsList += @("--decisions", $Decisions)
}
if ($hook -and $hook.Trim() -ne "") { $argsList += @("--discord-webhook", $hook) }

python @argsList
