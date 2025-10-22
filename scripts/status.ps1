function Get-ProcessStatus($pidPath) {
  if (-not (Test-Path $pidPath)) { return "missing" }
  $procId = Get-Content $pidPath | Select-Object -First 1
  if (-not $procId) { return "empty" }
  try {
    $p = Get-Process -Id $procId -ErrorAction Stop
    return "running (PID $procId)"
  } catch {
    return "not running (stale PID $procId)"
  }
}

$rtPid = "data/alerts/rt.pid"
$enPid = "data/entries/entry.pid"
Write-Host "Real-time alerts: $(Get-ProcessStatus $rtPid)"
Write-Host "Entry scheduler : $(Get-ProcessStatus $enPid)"
