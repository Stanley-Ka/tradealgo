param(
  [Parameter(Mandatory = $true)][string]$Start,
  [Parameter(Mandatory = $true)][string]$End,
  [int]$Step = 5,
  [int]$Sample = 0,
  [string]$Config = "engine/presets/swing_aggressive.yaml",
  [string]$Style = "",
  [int]$LookbackDays = 60,
  [int]$HoldDays = -1,
  [int]$MaxSymbols = 120,
  [string]$Provider = "yahoo",
  [switch]$SkipNews,
  [string]$NewsProvider = "polygon",
  [int]$NewsWindow = 3,
  [int]$NewsMaxSymbols = 40,
  [string]$TopKList = "5,10",
  [string]$EntryThresholds = "0.45,0.48,0.50",
  [string]$ExitThresholds = "0.45",
  [string]$StopAtrList = "1.0,1.25",
  [string]$TpAtrList = "0.0,1.5",
  [int]$ExitConsecutive = 2,
  [int]$Workers = 0,
  [string]$OutRoot = "data/backtests/replay_grid",
  [switch]$AggregateOnly
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$python = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $python = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

$argsList = @(
  "-m", "engine.tools.replay_grid",
  "--start", $Start,
  "--end", $End,
  "--step", $Step,
  "--sample", $Sample,
  "--workers", $Workers,
  "--lookback-days", $LookbackDays,
  "--hold-days", $HoldDays,
  "--max-symbols", $MaxSymbols,
  "--provider", $Provider,
  "--news-provider", $NewsProvider,
  "--news-window", $NewsWindow,
  "--news-max-symbols", $NewsMaxSymbols,
  "--topk-list", $TopKList,
  "--entry-thresholds", $EntryThresholds,
  "--exit-thresholds", $ExitThresholds,
  "--stop-atr-list", $StopAtrList,
  "--tp-atr-list", $TpAtrList,
  "--exit-consecutive", $ExitConsecutive,
  "--out-root", $OutRoot
)

if ($Style -and $Style.Trim()) {
  $argsList += @("--style", $Style.Trim())
} elseif ($Config -and $Config.Trim()) {
  $argsList += @("--config", $Config.Trim())
}
if ($SkipNews) { $argsList += "--skip-news" }
if ($AggregateOnly) { $argsList += "--aggregate-only" }
if ($PSBoundParameters.ContainsKey('Verbose') -and $VerbosePreference -ne 'SilentlyContinue') { $argsList += "--verbose" }

& $python $argsList
