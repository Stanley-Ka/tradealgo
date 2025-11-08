param(
  # Walk-forward params
  [string]$Features = "C:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Start = "2017-01-01",
  [string]$End = "",
  [ValidateSet('monthly','quarterly')][string]$Freq = 'quarterly',
  # Dataset params
  [string]$Model = "C:\\EngineData\\models\\meta_hgb.pkl",
  [string]$Universe = "engine/data/universe/swing_aggressive.watchlist.txt",
  [string]$OutDataset = "C:\\EngineData\\datasets\\swing_training_dataset.parquet",
  [string]$Timeframes = "3,7,14",
  [ValidateSet('close','open')][string]$EntryPrice = 'close',
  [int]$TopK = 20,
  [double]$MinADV = 1e7,
  [double]$MaxATR = 0.05,
  [double]$TP = 0.02,
  [double]$SL = 0.03,
  # Outputs
  [string]$OutDir = "C:\\EngineData\\backtests\\walkforward",
  [string]$ReportsDir = "C:\\EngineData\\reports"
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$python = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $python = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
$wfAgg = Join-Path $OutDir "walkforward_results.parquet"
$summaryJson = Join-Path $ReportsDir "weekly_summary.json"

# Pre-dataset row count
$prevCount = 0
if (Test-Path $OutDataset) {
  try {
    $tmp = & $python -m engine.tools.summarize_outputs --dataset $OutDataset --timeframes $Timeframes
    $prevCount = ([object]::ReferenceEquals($tmp,$null)) ? 0 : ( ($tmp | ConvertFrom-Json).dataset.rows )
  } catch { $prevCount = 0 }
}

Write-Host "[weekly] Running walk-forward + reports..." -ForegroundColor Cyan
& "$scriptDir/walkforward-reports.ps1" -Features $Features -Start $Start -End $End -Freq $Freq -OutDir $OutDir -ReportsDir $ReportsDir | Out-Null

Write-Host "[weekly] Building swing dataset..." -ForegroundColor Cyan
& "$scriptDir/build-swing-dataset.ps1" -Features $Features -Model $Model -Universe $Universe -Out $OutDataset -Timeframes $Timeframes -EntryPrice $EntryPrice -TopK $TopK -Start $Start -End $End -MinADV $MinADV -MaxATR $MaxATR -TP $TP -SL $SL -RequireAll -Resume | Out-Null

# Summaries
$dsSum = & $python -m engine.tools.summarize_outputs --dataset $OutDataset --timeframes $Timeframes --prev-count $prevCount
$wfSum = & $python -m engine.tools.summarize_outputs --walkforward $wfAgg
$dsObj = $null; $wfObj = $null
try { $dsObj = $dsSum | ConvertFrom-Json } catch {}
try { $wfObj = $wfSum | ConvertFrom-Json } catch {}

$payload = [ordered]@{
  timestamp   = (Get-Date).ToString('s')
  dataset     = $dsObj.dataset
  walkforward = $wfObj.walkforward
}
$payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $summaryJson -Encoding UTF8
Write-Host "[weekly] Summary -> $summaryJson" -ForegroundColor Green
