param(
  # Core inputs
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  # Outputs/paths
  [string]$WatchlistOut = "engine/data/universe/daily_watchlist.txt",
  [string]$DecisionLog = "data/paper/decision_log.csv",
  [string]$WeightsYaml = "data/paper/specialist_weights.yaml",
  [string]$WeeklyMD = "data/reports/weekly_overview.md",
  # Parameters
  [int]$TopKWatch = 50,
  [int]$TopKPaper = 20,
  [double]$MinADV = 10000000,
  [double]$MinPrice = 1,
  [double]$MaxPrice = 10000,
  [double]$CostBps = 5,
  [double]$TurnoverCap = 0.3,
  [int]$CalibratorMinRows = 200,
  [int]$MetaMinRows = 300,
  [switch]$RunWeeklyOverview
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

Write-Host "[daily-run] 1/3 Build watchlist..."
& (Join-Path $scriptDir "daily-build.ps1") -Features $Features -Model $Model -OOF $OOF -Calibrators $Calibrators -TopK $TopKWatch -MinADV $MinADV -MinPrice $MinPrice -MaxPrice $MaxPrice -Out $WatchlistOut

Write-Host "[daily-run] 2/3 Paper trade step (with specialist logging + dynamic weights)..."
# Prefer premarket refined shortlist if present
$UniverseForRun = $WatchlistOut
$Refined = "engine/data/universe/premarket_refined.txt"
if (Test-Path $Refined) { $UniverseForRun = $Refined }
& (Join-Path $scriptDir "paper-trade.ps1") -Config $Config -Features $Features -Model $Model -OOF $OOF -Calibrators $Calibrators -Universe $UniverseForRun -TopK $TopKPaper -CostBps $CostBps -TurnoverCap $TurnoverCap -DecisionLog $DecisionLog -WeightsYaml $WeightsYaml -UseSpecialistWeights

Write-Host "[daily-run] 3/3 Online updates (calibrators + meta) from decision log..."
& (Join-Path $scriptDir "daily-train.ps1") -DecisionLog $DecisionLog -Features $Features -Config $Config -CalibratorsOut $Calibrators -LabelThreshold 0 -CalibratorMinRows $CalibratorMinRows -MetaMinRows $MetaMinRows

# Optional daily summary with Discord (uses alert webhook env if set)
if (Test-Path $DecisionLog) {
  Write-Host "[daily-run] 3.5/4 Daily summary (Discord if configured)..."
  & (Join-Path $scriptDir "daily-summary.ps1") -Features $Features -Decisions $DecisionLog -Lookahead 5 -TargetPct 0.01 -Rows 10
} else {
  Write-Host "[daily-run] 3.5/4 Daily summary skipped (decision log missing)" -ForegroundColor Yellow
}

if ($RunWeeklyOverview -or ("Saturday","Sunday" -contains (Get-Date).DayOfWeek.ToString())) {
  Write-Host "[daily-run] Weekly overview & specialist weight update..."
  & (Join-Path $scriptDir "weekly-overview.ps1") -DecisionLog $DecisionLog -OutMD $WeeklyMD -WeightsOut $WeightsYaml -Weeks 1 -MinComboSamples 5
}

Write-Host "[daily-run] Done."
