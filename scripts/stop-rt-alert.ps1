param(
  [string]$State = "data/alerts/alert_log.parquet"
)

$pidPath = Join-Path (Split-Path $State -Parent) "rt.pid"
if (-not (Test-Path $pidPath)) {
  Write-Host "[rt] No PID file found at $pidPath."
  exit 0
}
$procId = Get-Content $pidPath | Select-Object -First 1
if (-not $procId) {
  Write-Host "[rt] PID file empty."
  exit 0
}
try {
  Stop-Process -Id $procId -Force -ErrorAction Stop
  Write-Host "[rt] Stopped real-time alerts process (PID ${procId})."
} catch {
  Write-Host "[rt] Could not stop PID ${procId}: $($_.Exception.Message)"
}
