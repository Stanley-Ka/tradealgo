param(
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [string]$Config = "engine/config.research.yaml",
  # Optional intraday snapshot integration
  [switch]$StartIntraday,
  [string]$IntradayConfig = "engine/config.intraday.example.yaml",
  [string]$IntradayFeatures = "data/datasets/features_intraday_latest.parquet",
  [int]$IntradayEvery = 5,
  # Optional: build a daily watchlist before starting
  [switch]$BuildWatchlist,
  [string]$WatchlistOut = "engine/data/universe/watchlist.txt",
  [int]$WatchTopK = 500,
  [double]$WatchMinPrice = 1.0,
  [double]$WatchMinADV = 5000000,
  [int]$AlertEvery = 5,
  [int]$AlertPoll = 15,
  [string]$AlertTimes = "",
  [string]$AlertState = "data/alerts/alert_log.parquet",
  [string]$AlertLog = "data/alerts/alert_diag.csv",
  [double]$SignalThreshold = -1,
  [double]$SignalMinDelta = 0.015,
  [int]$SignalCooldown = 30,
  [int]$SignalTopK = 1,
  [string]$SignalState = "data/alerts/signal_state.json",
  [double]$MinPrice = -1,
  [double]$MaxPrice = -1,
  [double]$MinAdvUsd = -1,
  [switch]$Heartbeat,
  [string]$LogDir = "data/logs",
  [int]$EntryTopK = 5,
  [string]$EntryTimes = "09:35,15:55",
  [int]$EntryPoll = 15,
  [string]$EntryState = "data/entries/entry_log.parquet"
  , [switch]$StartManager
  , [string]$ManagerPositions = "data/paper/positions.csv"
  , [string]$ManagerLog = "data/paper/trade_log.csv"
  , [int]$ManagerPoll = 15
  , [switch]$RebuildWatchlist
  , [switch]$MixIntraday
  , [double]$MixWeight = 0.0
  , [switch]$WatchBucketAdv
  , [int]$WatchBuckets = 3
  , [bool]$SignalUseMedian = $false
  , [double]$InitialWatchlistThreshold = 0.0
  , [int]$InitialWatchlistTopK = 25
  , [string]$PositionsCsv = "data/paper/positions.csv"
  , [switch]$AutoTrain
  , [string[]]$AutoTrainLogs = @()
  , [switch]$AutoTrainSkipEntryLog
  , [switch]$Parity
  , [switch]$DisableEntryScheduler
  , [switch]$EnableExplore
  , [int]$MinRepeatMins = 60
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Respect YAML defaults: only set local defaults if explicitly provided
if (-not $PSBoundParameters.ContainsKey('IntradayFeatures') -or [string]::IsNullOrWhiteSpace($IntradayFeatures) -or $IntradayFeatures.StartsWith('-')) {
  $IntradayFeatures = "data/datasets/features_intraday_latest.parquet"
}
if (-not $PSBoundParameters.ContainsKey('EntryTimes') -or $null -eq $EntryTimes -or $EntryTimes -eq "" -or $EntryTimes.ToLower() -eq "auto") {
  $EntryTimes = "auto"
}
# Prefer current virtualenv Python if available
$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

# If requested, build a daily watchlist from features/model
if ($BuildWatchlist) {
  try {
    $wlArgs = @(
      "-m", "engine.tools.build_watchlist",
      "--features", $Features,
      "--model-pkl", $Model,
      "--out", $WatchlistOut,
      "--top-k", $WatchTopK,
      "--min-price", $WatchMinPrice,
      "--min-adv-usd", $WatchMinADV
    )
    python @wlArgs
    if (Test-Path $WatchlistOut) {
      $Universe = $WatchlistOut
      Write-Host "[overnight] Using watchlist as universe -> $Universe"
    }
  } catch {
    Write-Host "[overnight] Watchlist build failed: $($_.Exception.Message)"
  }
}

# If requested, start intraday snapshot loop (keep daily features for core models)
$featuresPath = $Features
$intradayFeaturesPath = ""
if ($StartIntraday) {
  try {
    # Launch a Python loop module to avoid nested PowerShell issues
    $intrArgs = @(
      "-m", "engine.tools.intraday_loop",
      "--config", $IntradayConfig,
      "--every", $IntradayEvery
    )
    Start-Process -FilePath $py -ArgumentList ($intrArgs -join ' ') -PassThru -WindowStyle Hidden | Out-Null
    if ($IntradayFeatures -and $IntradayFeatures.Trim() -ne "") {
      $intradayFeaturesPath = $IntradayFeatures
    }
    Write-Host "[overnight] Started intraday snapshot loop; intraday features -> $intradayFeaturesPath"
  } catch {
    Write-Host "[overnight] Failed to start intraday loop: $($_.Exception.Message)"
    Write-Host "[overnight] Fallback: will continue without intraday snapshots."
  }
}

if ($AutoTrain) {
  try {
    $trainArgs = @(
      "-m", "engine.tools.auto_train_from_log",
      "--entry-log", "data/paper/entry_log.csv",
      "--min-rows", 200,
      "--calibrator-out", "data/models/meta_calibrator.auto.pkl",
      "--weights-out", "data/models/specialist_condition_weights.json",
      "--lookahead-col", "ret_3d",
      "--dedupe"
    )
    if ($AutoTrainSkipEntryLog) {
      $trainArgs += "--skip-entry-log"
    }
    if ($AutoTrainLogs -and $AutoTrainLogs.Count -gt 0) {
      foreach ($logPath in $AutoTrainLogs) {
        if ($logPath -and $logPath -ne "") {
          $trainArgs += @("--log", $logPath)
        }
      }
    }
    & $py @trainArgs
  } catch {
    Write-Host "[overnight] Auto-train step failed: $($_.Exception.Message)"
  }
}

# Launch real-time alerts directly via python
$rtArgsList = @(
  "-m", "engine.tools.real_time_alert",
  "--config", $Config,
  "--provider", $(if ($Parity) { "none" } else { "polygon" }),
  "--top-k", 3,
  "--poll", $AlertPoll,
  "--state", $AlertState,
  "--log-file", (Join-Path $LogDir "rt-alert.log"),
  "--recommendations-csv", "data/alerts/recommendations.csv"
)
# Pass core paths only if explicitly provided; otherwise real_time_alert loads from YAML
if ($PSBoundParameters.ContainsKey('Features') -and $featuresPath -and $featuresPath.Trim() -ne "") {
  $rtArgsList += @("--features", $featuresPath)
}
if ($PSBoundParameters.ContainsKey('Model') -and $Model -and $Model.Trim() -ne "") {
  $rtArgsList += @("--model-pkl", $Model)
}
if ($PSBoundParameters.ContainsKey('Universe') -and $Universe -and $Universe.Trim() -ne "") {
  $rtArgsList += @("--universe-file", $Universe)
}
if ($EnableExplore) {
  $rtArgsList += @(
    "--explore-universe-file", "engine/data/universe/us_all.txt",
    "--explore-top-k", 1
  )
}
if ($PositionsCsv -and $PositionsCsv -ne "") {
  $rtArgsList += @("--positions-csv", $PositionsCsv)
}
if ($SignalThreshold -lt 0) {
  if ($AlertEvery -gt 0) { $rtArgsList += @("--every-minutes", $AlertEvery) }
  if ($EntryTimes -and $EntryTimes -ne "auto") { $rtArgsList += @("--times", $EntryTimes) }
} else {
  $rtArgsList += @(
    "--signal-threshold", $SignalThreshold,
    "--signal-min-delta", $SignalMinDelta,
    "--signal-cooldown-mins", $SignalCooldown,
    "--signal-topk", $SignalTopK,
    "--signal-state", $SignalState
  )
  if ($SignalUseMedian) { $rtArgsList += "--signal-use-median" }
}
if ($InitialWatchlistThreshold -gt 0) {
  $rtArgsList += @(
    "--initial-watchlist-threshold", ([string]::Format("{0:F3}", $InitialWatchlistThreshold)),
    "--initial-watchlist-topk", $InitialWatchlistTopK
  )
}
if ($AlertLog -ne "") { $rtArgsList += @("--alert-log-csv", $AlertLog) }
if ($MinRepeatMins -ge 0) { $rtArgsList += @("--min-repeat-mins", $MinRepeatMins) }
if ($MinPrice -ge 0) { $rtArgsList += @("--min-price", $MinPrice) }
if ($MaxPrice -ge 0) { $rtArgsList += @("--max-price", $MaxPrice) }
if ($MinAdvUsd -ge 0) { $rtArgsList += @("--min-adv-usd", $MinAdvUsd) }
if ($Heartbeat) { $rtArgsList += "--heartbeat" }
if ($RebuildWatchlist) {
  $rtArgsList += @(
    "--rebuild-watchlist",
    "--watchlist-out", $WatchlistOut,
    "--watch-topk", $WatchTopK,
    "--watch-min-price", $WatchMinPrice,
    "--watch-min-adv", $WatchMinADV
  )
  if ($WatchBucketAdv) { $rtArgsList += @("--watch-bucket-adv", "--watch-buckets", $WatchBuckets) }
}
if ($MixIntraday -and $intradayFeaturesPath -ne "") {
  $rtArgsList += @(
    "--mix-intraday", $MixWeight,
    "--intraday-features", $intradayFeaturesPath
  )
}
# Parity mode: enforce deterministic settings and suppress sends
if ($Parity) {
  $rtArgsList += @(
    "--cooldown-mins", 0,
    "--price-source", "feature",
    "--explore-prob", 0,
    "--discord-webhook", ""
  )
}
else {
  # Default to live pricing for sizing/messages when not in Parity mode
  $rtArgsList += @("--price-source", "live", "--live-provider", "polygon")
}
$rtArgStr = ($rtArgsList -join ' ')
$rtOut = Join-Path $LogDir "rt-alert.out"
$rtErr = Join-Path $LogDir "rt-alert.err"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
function Get-AlivePid {
  param([string]$PidFile)
  if (-not (Test-Path $PidFile)) { return $null }
  try {
    $pidVal = Get-Content -Path $PidFile -ErrorAction Stop | Select-Object -First 1
    if (-not $pidVal) { return $null }
    $p = Get-Process -Id ([int]$pidVal) -ErrorAction SilentlyContinue
    if ($null -ne $p) { return [int]$pidVal }
  } catch { }
  return $null
}

  $rtPidPath = Join-Path (Split-Path $AlertState -Parent) "rt.pid"
  New-Item -ItemType Directory -Path (Split-Path $rtPidPath -Parent) -Force | Out-Null
$existingRtPid = Get-AlivePid -PidFile $rtPidPath
if ($null -ne $existingRtPid) {
  Write-Host "[overnight] Detected running real-time alerts (PID $existingRtPid); skipping new launch."
} else {
  $rtProc = Start-Process -FilePath $py -ArgumentList $rtArgStr -PassThru -RedirectStandardOutput $rtOut -RedirectStandardError $rtErr -WindowStyle Hidden
  Set-Content -Path $rtPidPath -Value $rtProc.Id
}

# Decide whether to launch entry scheduler.
# In signal-driven mode (SignalThreshold >= 0) with auto entry times, real_time_alert will execute entries.
$signalMode = ($SignalThreshold -ge 0)
$autoEntry = ($EntryTimes -and $EntryTimes.ToLower() -eq "auto")
$shouldStartEntry = (-not $DisableEntryScheduler) -and ((-not $signalMode) -or (-not $autoEntry))
if ($shouldStartEntry) {
  # Launch entry scheduler directly via python
  $enArgsList = @(
    "-m", "engine.tools.entry_scheduler",
    "--config", $Config,
    "--times", $EntryTimes,
    "--poll", $EntryPoll,
    "--top-k", $EntryTopK,
    "--state", $EntryState,
    "--log-file", (Join-Path $LogDir "entry-sched.log"),
    "--universe-file", $Universe
  )
  if ($PSBoundParameters.ContainsKey('Features') -and $featuresPath -and $featuresPath.Trim() -ne "") {
    $enArgsList += @("--features", $featuresPath)
  }
  if ($PSBoundParameters.ContainsKey('Model') -and $Model -and $Model.Trim() -ne "") {
    $enArgsList += @("--model-pkl", $Model)
  }
  if ($MinPrice -ge 0) { $enArgsList += @("--min-price", $MinPrice) }
  if ($MaxPrice -ge 0) { $enArgsList += @("--max-price", $MaxPrice) }
  if ($MinAdvUsd -ge 0) { $enArgsList += @("--min-adv-usd", $MinAdvUsd) }
  if (-not $Parity) {
    $enArgsList += "--send-discord"
  }
  $enArgsList += @("--recommendations-csv", "data/alerts/recommendations.csv")
  $enArgStr = ($enArgsList -join ' ')
  $enOut = Join-Path $LogDir "entry-sched.out"
  $enErr = Join-Path $LogDir "entry-sched.err"
  $enPidPath = Join-Path (Split-Path $EntryState -Parent) "entry.pid"
  New-Item -ItemType Directory -Path (Split-Path $enPidPath -Parent) -Force | Out-Null
  $existingEntryPid = Get-AlivePid -PidFile $enPidPath
  if ($null -ne $existingEntryPid) {
    Write-Host "[overnight] Detected running entry scheduler (PID $existingEntryPid); skipping new launch."
  } else {
    $enProc = Start-Process -FilePath $py -ArgumentList $enArgStr -PassThru -RedirectStandardOutput $enOut -RedirectStandardError $enErr -WindowStyle Hidden
    Set-Content -Path $enPidPath -Value $enProc.Id
  }
} else {
  Write-Host "[overnight] Entry scheduler suppressed (signal-mode=$signalMode, auto-entry=$autoEntry)."
}

# Optionally start RT position manager
if ($StartManager) {
  $pmArgs = @(
    "-m", "engine.tools.rt_position_manager",
    "--positions-csv", $ManagerPositions,
    "--log-csv", $ManagerLog,
    "--poll", $ManagerPoll,
    "--config", $Config
  )
  $pmOut = Join-Path $LogDir "pm.out"
  $pmErr = Join-Path $LogDir "pm.err"
  $pmPidPath = Join-Path (Split-Path $ManagerPositions -Parent) "pm.pid"
  New-Item -ItemType Directory -Path (Split-Path $pmPidPath -Parent) -Force | Out-Null
  $existingPmPid = Get-AlivePid -PidFile $pmPidPath
  if ($null -ne $existingPmPid) {
    Write-Host "[overnight] Detected running position manager (PID $existingPmPid); skipping new launch."
  } else {
    $pmProc = Start-Process -FilePath $py -ArgumentList ($pmArgs -join ' ') -PassThru -RedirectStandardOutput $pmOut -RedirectStandardError $pmErr -WindowStyle Hidden
    Set-Content -Path $pmPidPath -Value $pmProc.Id
    Write-Host "[overnight] Started RT position manager (poll=$ManagerPoll s)"
  }
}

Write-Host "[overnight] Real-time alerts: PID file $rtPidPath."
if ($shouldStartEntry) {
  Write-Host "[overnight] Entry scheduler: PID file $(Join-Path (Split-Path $EntryState -Parent) 'entry.pid')."
} else {
  Write-Host "[overnight] Entry scheduler not started per settings."
}
Write-Host "[overnight] Use scripts/status.ps1 to check status; stop with scripts/stop-rt-alert.ps1 and scripts/stop-entry.ps1."
