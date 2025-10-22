param(
  [string]$TaskName = 'EngineSwingDataset',
  [ValidateSet('Weekly','Daily')][string]$Schedule = 'Weekly',
  [string]$At = '03:00',
  [ValidateSet('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')][string]$DaysOfWeek = 'Sunday',
  [string]$Features = 'D:\\EngineData\\datasets\\features_daily_1D.parquet',
  [string]$Model = 'D:\\EngineData\\models\\meta_hgb.pkl',
  [string]$Universe = 'engine/data/universe/swing_aggressive.watchlist.txt',
  [string]$Out = 'D:\\EngineData\\datasets\\swing_training_dataset.parquet',
  [string]$Timeframes = '3,7,14',
  [ValidateSet('close','open')][string]$EntryPrice = 'close',
  [int]$TopK = 20,
  [string]$Start = '2017-01-01',
  [string]$End = '',
  [double]$MinADV = 1e7,
  [double]$MaxATR = 0.05,
  [double]$TP = 0.02,
  [double]$SL = 0.03,
  [switch]$RequireAll,
  [switch]$Resume,
  [switch]$ForceUpdate
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $here '..')
$psExe = (Get-Command powershell.exe).Source
$script = (Resolve-Path (Join-Path $repoRoot 'scripts/build-swing-dataset.ps1')).Path

$args = @('-NoProfile','-ExecutionPolicy','Bypass','-File', '"' + $script + '"',
          '-Features', '"' + $Features + '"',
          '-Model', '"' + $Model + '"',
          '-Universe', '"' + $Universe + '"',
          '-Out', '"' + $Out + '"',
          '-Timeframes', '"' + $Timeframes + '"',
          '-EntryPrice', $EntryPrice,
          '-TopK', $TopK,
          '-Start', '"' + $Start + '"',
          '-MinADV', $MinADV,
          '-MaxATR', $MaxATR,
          '-TP', $TP,
          '-SL', $SL)
if ($End -and $End.Trim()) { $args += @('-End', '"' + $End.Trim() + '"') }
if ($RequireAll) { $args += '-RequireAll' }
if ($Resume) { $args += '-Resume' }
$argString = ($args -join ' ')

try { $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { $existing = $null }
if ($existing -and $ForceUpdate) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
}

$action = New-ScheduledTaskAction -Execute $psExe -Argument $argString
if ($Schedule -eq 'Daily') { $trigger = New-ScheduledTaskTrigger -Daily -At $At }
else { $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $At }
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries)
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Write-Host "[task] Registered '$TaskName' to run $Schedule at $At." -ForegroundColor Green
Write-Host "        Action: powershell.exe $argString" -ForegroundColor DarkGray
