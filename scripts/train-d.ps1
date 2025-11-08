param(
  [string]$Preset = "engine/presets/research.yaml",
  [string]$OutPreset = "engine/presets/swing_aggressive.local.yaml"
)

$features = "C:\\EngineData\\datasets\\features_daily_1D.parquet"
$oof = "C:\\EngineData\\datasets\\oof_specialists.parquet"
$cal = "C:\\EngineData\\models\\spec_calibrators.pkl"
$meta = "C:\\EngineData\\datasets\\meta_predictions.parquet"
$model = "C:\\EngineData\\models\\meta_hgb.pkl"
$metaCal = "C:\\EngineData\\models\\meta_calibrator.pkl"
$bt = "C:\\EngineData\\backtests\\daily_topk_report.html"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

function Resolve-PresetPath([string]$Path) {
  $candidate = if (Test-Path $Path) { $Path } else { Join-Path $repoRoot $Path }
  if (-not (Test-Path $candidate)) {
    throw "Preset not found: $Path"
  }
  $localCandidate = [System.IO.Path]::ChangeExtension($candidate, '.local.yaml')
  if (Test-Path $localCandidate) {
    return $localCandidate
  }
  return $candidate
}

$presetPath = Resolve-PresetPath $Preset
$outPath = if ([System.IO.Path]::IsPathRooted($OutPreset)) { $OutPreset } else { Join-Path $repoRoot $OutPreset }

$params = @{
  Features      = $features
  Config        = $presetPath
  OOF           = $oof
  Calibrators   = $cal
  MetaPred      = $meta
  Model         = $model
  MetaCalibrator= $metaCal
  BacktestReport= $bt
  Universe      = "engine/data/universe/us_all.txt"
  BuildWatchlist= $true
  WatchlistOut  = "engine/data/universe/watchlist.txt"
  WLTopK        = 500
  RunAlert      = $true
  DryRun        = $true
  WritePaths    = $true
  ConfigOut     = $outPath
}

& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "train-and-backtest.ps1") @params
