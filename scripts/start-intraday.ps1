param(
  [string]$Config = "engine/config.intraday.example.yaml",
  [int]$Every = 5
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Run intraday pipeline every N minutes (simple loop)
Write-Host "[intraday] Starting loop every $Every minutes using $Config ..."
try {
  while ($true) {
    python -m engine.tools.intraday_pipeline --config $Config
    Start-Sleep -Seconds ([Math]::Max(60, $Every*60))
  }
} catch {
  Write-Host "[intraday] stopped: $($_.Exception.Message)"
}
