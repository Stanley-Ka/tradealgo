param(
  [string]$Config = "engine/presets/swing_aggressive.yaml",
  [string]$Features = "",
  [string]$Model = "",
  [string]$Universe = "",
  [int]$AlertEvery = 5,
  [string]$EntryTimes = "09:35,15:55",
  [switch]$StartManager = $true,
  [switch]$MixIntraday = $false,
  [double]$MixWeight = 0.5,
  [string]$IntradayFeatures = "",
  [switch]$StartIntraday = $false,
  [string]$IntradayConfig = "engine/config.intraday.example.yaml",
  [int]$IntradayEvery = 5,
  [int]$AlertPoll = 15,
  [double]$SignalThreshold = -1,
  [double]$SignalMinDelta = 0.015,
  [int]$SignalCooldown = 30,
  [int]$SignalTopK = 1,
  [string]$SignalState = "data/alerts/signal_state.json",
  [bool]$SignalUseMedian = $true,
  [double]$InitialWatchlistThreshold = 0.55,
  [int]$InitialWatchlistTopK = 25,
  [double]$MinAdvUsd = -1,
  [string]$PositionsCsv = "data/paper/positions.csv",
  [switch]$AutoTrain = $false,
  [string[]]$AutoTrainLogs = @(),
  [switch]$AutoTrainSkipEntryLog = $false,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PassThru = @()
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Always stop previous workers before starting fresh
try {
  & "$scriptDir\stop-rt-alert.ps1" -State "data/alerts/alert_log.parquet" | Out-Null
} catch { }
try {
  & "$scriptDir\stop-entry.ps1" -State "data/entries/entry_log.parquet" | Out-Null
} catch { }
# Best-effort: stop previous position manager if pid file exists
try {
  $pmPidPath = "data/paper/pm.pid"
  if (Test-Path $pmPidPath) {
    $pmPid = Get-Content $pmPidPath | Select-Object -First 1
    if ($pmPid) {
      try { Stop-Process -Id ([int]$pmPid) -Force -ErrorAction Stop } catch { }
    }
  }
} catch { }

# Resolve config (prefer .local)
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$cfgCandidate = if (Test-Path $Config) { $Config } else { Join-Path $repoRoot $Config }
if (-not (Test-Path $cfgCandidate)) {
  throw "Config not found: $Config"
}
$localCandidate = [System.IO.Path]::ChangeExtension($cfgCandidate, '.local.yaml')
if (Test-Path $localCandidate) {
  $cfgCandidate = $localCandidate
}

# Defaults for optional paths
if ([string]::IsNullOrWhiteSpace($Features)) {
  $Features = "data/datasets/features_daily_1D.parquet"
}
if ([string]::IsNullOrWhiteSpace($Model)) {
  $Model = "data/models/meta_lr.pkl"
}
if ([string]::IsNullOrWhiteSpace($Universe)) {
  $Universe = "engine/data/universe/nasdaq100.example.txt"
}
if ([string]::IsNullOrWhiteSpace($IntradayFeatures)) {
  $IntradayFeatures = "data/datasets/features_intraday_latest.parquet"
}
if ([string]::IsNullOrWhiteSpace($EntryTimes) -or $EntryTimes.ToLower() -eq "auto") {
  $EntryTimes = "auto"
}

$parms = @{
  Config     = $cfgCandidate
  AlertEvery = $AlertEvery
  EntryTimes = $EntryTimes
  Features   = $Features
  Model      = $Model
  Universe   = $Universe
  SignalUseMedian = $SignalUseMedian
  InitialWatchlistThreshold = $InitialWatchlistThreshold
  InitialWatchlistTopK = $InitialWatchlistTopK
  PositionsCsv = $PositionsCsv
  AutoTrain = $AutoTrain
  AutoTrainLogs = $AutoTrainLogs
  AutoTrainSkipEntryLog = $AutoTrainSkipEntryLog
}
if ($StartManager) { $parms.StartManager = $true }
if ($MixIntraday) {
  $parms.MixIntraday = $true
  $parms.MixWeight = $MixWeight
  if ($IntradayFeatures -and $IntradayFeatures.Trim() -ne "") {
    $parms.IntradayFeatures = $IntradayFeatures
  }
}
if ($StartIntraday) { $parms.StartIntraday = $true }
if ($PSBoundParameters.ContainsKey('IntradayConfig')) { $parms.IntradayConfig = $IntradayConfig }
if ($PSBoundParameters.ContainsKey('IntradayEvery')) { $parms.IntradayEvery = $IntradayEvery }
if ($PSBoundParameters.ContainsKey('AlertPoll')) { $parms.AlertPoll = $AlertPoll }
if ($PSBoundParameters.ContainsKey('SignalThreshold')) { $parms.SignalThreshold = $SignalThreshold }
if ($PSBoundParameters.ContainsKey('SignalMinDelta')) { $parms.SignalMinDelta = $SignalMinDelta }
if ($PSBoundParameters.ContainsKey('SignalCooldown')) { $parms.SignalCooldown = $SignalCooldown }
if ($PSBoundParameters.ContainsKey('SignalTopK')) { $parms.SignalTopK = $SignalTopK }
if ($PSBoundParameters.ContainsKey('SignalState')) { $parms.SignalState = $SignalState }
if ($PSBoundParameters.ContainsKey('InitialWatchlistThreshold')) { $parms.InitialWatchlistThreshold = $InitialWatchlistThreshold }
if ($PSBoundParameters.ContainsKey('InitialWatchlistTopK')) { $parms.InitialWatchlistTopK = $InitialWatchlistTopK }
if ($PSBoundParameters.ContainsKey('EnableExplore')) { $parms.EnableExplore = $EnableExplore }
if ($PSBoundParameters.ContainsKey('MinAdvUsd')) { $parms.MinAdvUsd = $MinAdvUsd }

$startScript = Join-Path $scriptDir "start-overnight.ps1"
if ($PassThru -and $PassThru.Count -gt 0) {
  & $startScript @parms @PassThru
} else {
  & $startScript @parms
}
