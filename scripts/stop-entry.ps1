param(
  [string]$State = "data/entries/entry_log.parquet"
)

$pidPath = Join-Path (Split-Path $State -Parent) "entry.pid"
if (-not (Test-Path $pidPath)) {
  Write-Host "[entry] No PID file found at $pidPath."
  exit 0
}
$procId = Get-Content $pidPath | Select-Object -First 1
if (-not $procId) {
  Write-Host "[entry] PID file empty."
  exit 0
}
try {
  Stop-Process -Id $procId -Force -ErrorAction Stop
  Write-Host "[entry] Stopped entry scheduler process (PID ${procId})."
} catch {
  Write-Host "[entry] Could not stop PID ${procId}: $($_.Exception.Message)"
}
