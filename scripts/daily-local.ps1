# Requires: PowerShell 7+ (pwsh) or Windows PowerShell
# One-command daily routine using a local preset.

param(
  [string]$Config = "engine/presets/swing_aggressive.local.yaml"
)

$ErrorActionPreference = 'Stop'

# Resolve script root (repo scripts folder)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Load API keys if present (scripts/api.env)
try {
  if (Test-Path (Join-Path $root 'load-env.ps1')) {
    & (Join-Path $root 'load-env.ps1')
  }
} catch {
  Write-Warning "Failed to load scripts/api.env via load-env.ps1: $($_.Exception.Message)"
}

# Optional readiness check
try {
  $checkReady = Join-Path $root 'check-ready.ps1'
  if (Test-Path $checkReady) {
    & $checkReady -Config $Config
  }
} catch {
  Write-Warning "check-ready.ps1 reported issues (continuing): $($_.Exception.Message)"
}

# Run the daily workflow (watchlist → paper step → online updates)
$dailyRun = Join-Path $root 'daily-run.ps1'
if (-not (Test-Path $dailyRun)) {
  throw "daily-run.ps1 not found in $root"
}

& $dailyRun -Config $Config

Write-Host "Daily routine completed using config: $Config" -ForegroundColor Green
