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
  [int]$TopK = 0,
  [string]$Provider = "yahoo",
  [switch]$SkipNews,
  [string]$NewsProvider = "polygon",
  [int]$NewsWindow = 3,
  [int]$NewsMaxSymbols = 40,
  [double]$EntryThreshold = [double]::NaN,
  [double]$ExitThreshold = 0.45,
  [int]$ExitConsecutive = 2,
  [int]$Workers = 0,
  [string]$OutRoot = "data/backtests/replays",
  [switch]$Aggregate
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$python = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $python = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

$argsList = @(
  "-m", "engine.tools.replay_sweep",
  "--start", $Start,
  "--end", $End,
  "--step", $Step,
  "--sample", $Sample,
  "--lookback-days", $LookbackDays,
  "--hold-days", $HoldDays,
  "--max-symbols", $MaxSymbols,
  "--provider", $Provider,
  "--news-provider", $NewsProvider,
  "--news-window", $NewsWindow,
  "--news-max-symbols", $NewsMaxSymbols,
  "--exit-threshold", $ExitThreshold,
  "--exit-consecutive", $ExitConsecutive,
  "--workers", $Workers,
  "--out-root", $OutRoot
)

if ($Style -and $Style.Trim()) {
  $argsList += @("--style", $Style.Trim())
} elseif ($Config -and $Config.Trim()) {
  $argsList += @("--config", $Config.Trim())
}
if ($TopK -gt 0) { $argsList += @("--top-k", $TopK) }
if ($SkipNews) { $argsList += "--skip-news" }
if (-not [double]::IsNaN($EntryThreshold)) { $argsList += @("--entry-threshold", $EntryThreshold) }
if ($Aggregate) { $argsList += "--aggregate" }
if ($PSBoundParameters.ContainsKey('Verbose') -and $VerbosePreference -ne 'SilentlyContinue') {
  $argsList += "--verbose"
}

& $python $argsList
