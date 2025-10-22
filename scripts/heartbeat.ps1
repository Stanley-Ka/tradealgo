param(
  [string]$LogDir = "data/logs",
  [string]$DiscordWebhook = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

function Get-ProcStatus($pidPath) {
  if (-not (Test-Path $pidPath)) { return @{ status = 'missing'; pid = 0 } }
  $pid = Get-Content $pidPath | Select-Object -First 1
  if (-not $pid) { return @{ status = 'empty'; pid = 0 } }
  try {
    $p = Get-Process -Id $pid -ErrorAction Stop
    return @{ status = 'running'; pid = $pid }
  } catch {
    return @{ status = 'stale'; pid = $pid }
  }
}

$rtPid = "data/alerts/rt.pid"
$enPid = "data/entries/entry.pid"
$rt = Get-ProcStatus $rtPid
$en = Get-ProcStatus $enPid

Write-Host "[hb] RT alerts: $($rt.status) (PID $($rt.pid))"
Write-Host "[hb] Entry scheduler: $($en.status) (PID $($en.pid))"

# Compose a short message
$msg = "RT alerts: $($rt.status) (PID $($rt.pid))`nEntry scheduler: $($en.status) (PID $($en.pid))"

# Determine webhook
$hook = $DiscordWebhook
if (-not $hook -or $hook.Trim() -eq "") { $hook = $env:DISCORD_ALERTS_WEBHOOK_URL }
if (-not $hook -or $hook.Trim() -eq "") { $hook = $env:DISCORD_WEBHOOK_URL }

if ($hook -and $hook.Trim() -ne "") {
  try {
    $payload = @{ content = "Heartbeat: `n$msg" } | ConvertTo-Json
    Invoke-RestMethod -Uri $hook -Method Post -ContentType 'application/json' -Body $payload | Out-Null
    Write-Host "[hb] Posted heartbeat to Discord"
  } catch {
    Write-Host "[hb] Failed to post heartbeat: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}
