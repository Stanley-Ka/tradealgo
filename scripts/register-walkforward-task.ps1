param(
  [string]$TaskName = 'EngineWalkforwardReports',
  [ValidateSet('Daily','Weekly')][string]$Schedule = 'Weekly',
  [string]$At = '02:30',
  [ValidateSet('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')][string]$DaysOfWeek = 'Sunday',
  [string]$Features = 'C:\\EngineData\\datasets\\features_daily_1D.parquet',
  [string]$Start = '2017-01-01',
  [string]$End = '',
  [ValidateSet('monthly','quarterly')][string]$Freq = 'quarterly',
  [int]$TopK = 20,
  [double]$CostBps = 5.0,
  [ValidateSet('flat','spread')][string]$CostModel = 'spread',
  [double]$SpreadK = 1e8,
  [double]$SpreadCapBps = 25.0,
  [double]$SpreadMinBps = 2.0,
  [string]$OutDir = 'C:\\EngineData\\backtests\\walkforward',
  [string]$ReportsDir = 'C:\\EngineData\\reports',
  [string]$OOFPath = 'C:\\EngineData\\datasets\\oof_specialists.parquet',
  [switch]$ForceUpdate
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $here '..')
$psExe = (Get-Command powershell.exe).Source
$script = (Resolve-Path (Join-Path $repoRoot 'scripts/walkforward-reports.ps1')).Path

# Build PowerShell argument string for the action
$args = @('-NoProfile','-ExecutionPolicy','Bypass','-File', '"' + $script + '"',
          '-Features', '"' + $Features + '"',
          '-Start', '"' + $Start + '"',
          '-Freq', $Freq,
          '-TopK', $TopK,
          '-CostBps', $CostBps,
          '-CostModel', $CostModel,
          '-SpreadK', $SpreadK,
          '-SpreadCapBps', $SpreadCapBps,
          '-SpreadMinBps', $SpreadMinBps,
          '-OutDir', '"' + $OutDir + '"',
          '-ReportsDir', '"' + $ReportsDir + '"',
          '-OOFPath', '"' + $OOFPath + '"')
if ($End -and $End.Trim()) { $args += @('-End', '"' + $End.Trim() + '"') }
$argString = ($args -join ' ')

try {
  $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
} catch { $existing = $null }
if ($existing -and $ForceUpdate) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
}

$action = New-ScheduledTaskAction -Execute $psExe -Argument $argString
if ($Schedule -eq 'Daily') {
  $trigger = New-ScheduledTaskTrigger -Daily -At $At
} else {
  $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $At
}
# Use S4U so task can run whether user is logged on; requires Windows 8+/Server 2012+
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries)

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Write-Host "[task] Registered '$TaskName' to run $Schedule at $At." -ForegroundColor Green
Write-Host "        Action: powershell.exe $argString" -ForegroundColor DarkGray
