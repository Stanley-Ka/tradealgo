param(
  [string]$Features = "D:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Start = "2017-01-01",
  [string]$End = "",
  [ValidateSet('monthly','quarterly')][string]$Freq = 'quarterly',
  [string]$Label = "label_up_1d",
  [int]$TopK = 20,
  [double]$CostBps = 5.0,
  [ValidateSet('flat','spread')][string]$CostModel = 'spread',
  [double]$SpreadK = 1e8,
  [double]$SpreadCapBps = 25.0,
  [double]$SpreadMinBps = 2.0,
  [string]$OutDir = "D:\\EngineData\\backtests\\walkforward",
  [string]$ReportsDir = "D:\\EngineData\\reports",
  [string]$OOFPath = "D:\\EngineData\\datasets\\oof_specialists.parquet"
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$python = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $python = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

Write-Host "[wf] Running walk-forward..." -ForegroundColor Cyan
$wfArgs = @(
  "-m", "engine.tools.walkforward",
  "--features", $Features,
  "--start", $Start,
  "--freq", $Freq,
  "--label", $Label,
  "--top-k", $TopK,
  "--cost-bps", $CostBps,
  "--cost-model", $CostModel,
  "--spread-k", $SpreadK,
  "--spread-cap-bps", $SpreadCapBps,
  "--spread-min-bps", $SpreadMinBps,
  "--out-dir", $OutDir
)
if ($End -and $End.Trim()) { $wfArgs += @("--end", $End.Trim()) }
& $python $wfArgs

$agg = Join-Path $OutDir "walkforward_results.parquet"
if (-not (Test-Path $agg)) {
  throw "Walk-forward aggregate not found: $agg"
}

Write-Host "[wf] Computing bootstrap CIs..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
$ciOut = Join-Path $ReportsDir "bootstrap_ci.csv"
& $python -m engine.tools.bootstrap_ci --results $agg --block-size 10 --samples 1000 --out-csv $ciOut

if (Test-Path $OOFPath) {
  Write-Host "[wf] Computing reliability (OOF)..." -ForegroundColor Cyan
  $relOut = Join-Path $ReportsDir "reliability_oof.csv"
  & $python -m engine.tools.reliability --oof $OOFPath --label-col y_true --bins 10 --all-specialists --out-csv $relOut
} else {
  Write-Host "[wf] OOF not found at $OOFPath — skipping reliability." -ForegroundColor Yellow
}

Write-Host "[wf] Done. Results:" -ForegroundColor Green
Write-Host "  Walk-forward: $agg"
Write-Host "  Bootstrap CI: $ciOut"
if (Test-Path (Join-Path $ReportsDir "reliability_oof.csv")) { Write-Host "  Reliability: $(Join-Path $ReportsDir 'reliability_oof.csv')" }
