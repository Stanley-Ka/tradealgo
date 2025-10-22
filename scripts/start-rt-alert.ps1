param(
  [string]$Config = "",
  [string]$Features = "",
  [string]$Model = "",
  [string]$Universe = "",
  [int]$Every = 5,
  [int]$Poll = 15,
  [string]$State = "data/alerts/alert_log.parquet",
  [string]$Times = "",
  [string]$AlertLog = "data/alerts/alert_diag.csv",
  [int]$CooldownMins = 120
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.real_time_alert",
  "--provider", "polygon",
  "--top-k", 3,
  "--poll", $Poll,
  "--state", $State
)
if ($Config -ne "") { $argsList = @("--config", $Config) + $argsList }
if ($PSBoundParameters.ContainsKey('Features') -and $Features -ne "") { $argsList += @("--features", $Features) }
if ($PSBoundParameters.ContainsKey('Model') -and $Model -ne "") { $argsList += @("--model-pkl", $Model) }
if ($PSBoundParameters.ContainsKey('Universe') -and $Universe -ne "") { $argsList += @("--universe-file", $Universe) }
if ($Every -gt 0) { $argsList += @("--every-minutes", $Every) }
if ($Times -ne "") { $argsList += @("--times", $Times) }
if ($AlertLog -ne "") { $argsList += @("--alert-log-csv", $AlertLog) }
if ($CooldownMins -gt 0) { $argsList += @("--cooldown-mins", $CooldownMins) }

# If DISCORD_WEBHOOK_URL is set in env, engine will pick it up by default
$argsStr = ($argsList -join ' ')
$proc = Start-Process -FilePath "python" -ArgumentList $argsStr -PassThru -WindowStyle Hidden
$pidPath = Join-Path (Split-Path $State -Parent) "rt.pid"
New-Item -ItemType Directory -Path (Split-Path $pidPath -Parent) -Force | Out-Null
Set-Content -Path $pidPath -Value $proc.Id
Write-Host "[rt] Started real-time alerts (PID $($proc.Id)). PID file: $pidPath"
