param(
  [string]$Config = "engine/presets/autopilot.yaml",
  [switch]$StartManager = $true,
  [switch]$StartIntraday = $true,
  [double]$MixWeight = 0.3,
  [int]$AlertEvery = 5,
  [int]$AlertPoll = 15,
  [int]$MinRepeatMins = 90
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$args = @(
  "-Config", $Config,
  "-AlertEvery", $AlertEvery,
  "-AlertPoll", $AlertPoll,
  "-EntryTimes", "auto",
  "-StartManager":$StartManager,
  "-StartIntraday":$StartIntraday,
  "-MixIntraday",
  "-MixWeight", $MixWeight,
  "-MinRepeatMins", $MinRepeatMins
)
& (Join-Path $scriptDir "start-overnight.ps1") @args
