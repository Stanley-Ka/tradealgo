param(
  [string]$Config = "engine/config.example.daily.yaml",
  [string]$Date = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @("-m", "engine.tools.daily_pipeline", "--config", $Config)
if ($Date -ne "") { $argsList += @("--date", $Date) }

python @argsList
