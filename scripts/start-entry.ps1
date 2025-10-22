param(
  [string]$Config = "",
  [string]$Features = "",
  [string]$Model = "",
  [string]$Universe = "",
  [string]$Times = "09:35,15:55",
  [int]$Poll = 15,
  [int]$TopK = 5,
  [double]$Threshold = 0.0,
  [int]$Confirm = 1,
  [string]$SectorMap = "",
  [int]$SectorCap = 0,
  [string]$State = "data/entries/entry_log.parquet"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.entry_scheduler",
  "--times", $Times,
  "--poll", $Poll,
  "--top-k", $TopK,
  "--confirmations", $Confirm,
  "--state", $State,
  "--send-discord",
  "--recommendations-csv", "data/alerts/recommendations.csv"
)
if ($Config -ne "") { $argsList = @("--config", $Config) + $argsList }
if ($PSBoundParameters.ContainsKey('Features') -and $Features -ne "") { $argsList += @("--features", $Features) }
if ($PSBoundParameters.ContainsKey('Model') -and $Model -ne "") { $argsList += @("--model-pkl", $Model) }
if ($PSBoundParameters.ContainsKey('Universe') -and $Universe -ne "") { $argsList += @("--universe-file", $Universe) }
if ($Threshold -gt 0) { $argsList += @("--entry-threshold", $Threshold) }
if ($SectorMap -ne "") { $argsList += @("--sector-map-csv", $SectorMap) }
if ($SectorCap -gt 0) { $argsList += @("--sector-cap", $SectorCap) }

$argsStr = ($argsList -join ' ')
$proc = Start-Process -FilePath "python" -ArgumentList $argsStr -PassThru -WindowStyle Hidden
$pidPath = Join-Path (Split-Path $State -Parent) "entry.pid"
New-Item -ItemType Directory -Path (Split-Path $pidPath -Parent) -Force | Out-Null
Set-Content -Path $pidPath -Value $proc.Id
Write-Host "[entry] Started entry scheduler (PID $($proc.Id)). PID file: $pidPath"
