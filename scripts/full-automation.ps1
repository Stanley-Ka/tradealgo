param(
  [ValidateSet("yahoo","alphavantage","polygon")]
  [string]$DataProvider = "yahoo",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [string]$Config = "engine/config.research.yaml",
  [string]$Start = "2015-01-01",
  [string]$End = "",
  [string]$Features = "C:\\EngineData\datasets\features_daily_polygon.parquet",
  [string]$OOF = "C:\\EngineData\\datasets\\oof_specialists.parquet",
  [string]$Calibrators = "C:\\EngineData\\models\\spec_calibrators.pkl",
  [string]$MetaPred = "C:\\EngineData\\datasets\\meta_predictions.parquet",
  [string]$Model = "C:\\EngineData\\models\\meta_hgb.pkl",
  [double]$MaxPerMinute = 5,
  [string]$SentimentOut = "data/datasets/sentiment_finbert.parquet",
  [string]$IntradayFeatures = "",
  [string]$RecommendationsCsv = "data/alerts/recommendations.csv",
  [string]$AlertKind = "pre-market",
  [string]$AlertCategory = "watchlist",
  [int]$RefineTopK = 5,
  [int]$CalibratorMinRows = 75,
  [int]$MetaMinRows = 90,
  [switch]$SkipData,
  [switch]$SkipFeatures,
  [switch]$RunTraining,
  [switch]$SkipPreMarket,
  [switch]$SkipDailyRun,
  [switch]$SkipAlert,
  [switch]$AlertDryRun,
  [switch]$DisableAlertDedupe
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptDir

function Resolve-RepoPath([string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return (Join-Path $repoRoot $Path)
}

& (Join-Path $scriptDir "load-env.ps1") -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

Push-Location $repoRoot
try {
  $pythonExe = "python"
  if ($env:VIRTUAL_ENV) {
    $candidatePy = Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"
    if (Test-Path $candidatePy) {
      $pythonExe = $candidatePy
    }
  }

  $universePath = Resolve-RepoPath $Universe
  $configPath = Resolve-RepoPath $Config
  $featuresPath = Resolve-RepoPath $Features
  $oofPath = Resolve-RepoPath $OOF
  $calibratorsPath = Resolve-RepoPath $Calibrators
  $metaPredPath = Resolve-RepoPath $MetaPred
  $modelPath = Resolve-RepoPath $Model
  $sentimentPath = Resolve-RepoPath $SentimentOut
  $intradayPath = Resolve-RepoPath $IntradayFeatures
  $recommendationsPath = Resolve-RepoPath $RecommendationsCsv

  foreach ($required in @(@{Name="Universe";Path=$universePath}, @{Name="Config";Path=$configPath})) {
    if (-not (Test-Path $required.Path)) {
      throw "Missing $($required.Name) file: $($required.Path)"
    }
  }

  foreach ($path in @($featuresPath,$oofPath,$calibratorsPath,$metaPredPath,$modelPath,$sentimentPath,$recommendationsPath)) {
    if ([string]::IsNullOrWhiteSpace($path)) { continue }
    $dir = Split-Path -Parent $path
    if ($dir -and -not (Test-Path $dir)) {
      New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
  }

  $hasEnd = -not [string]::IsNullOrWhiteSpace($End)
  $featuresLastDate = $null
  if (Test-Path $featuresPath) {
    try {
      $code = @"
import pandas as pd
path = r'''$featuresPath'''
try:
    d = pd.read_parquet(path, columns=['date'])
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    print(d['date'].max().isoformat())
except Exception as e:
    print(f'ERR:{e}')
"@
      $probe = & $pythonExe -c $code
      if ($probe -and -not $probe.StartsWith('ERR')) {
        $featuresLastDate = Get-Date $probe
      } else {
        Write-Warning "[auto] Unable to read features parquet date: $probe"
      }
    } catch {
      Write-Warning "[auto] Features date probe failed: $($_.Exception.Message)"
    }
  }

  if ($featuresLastDate) {
    $staleDays = ((Get-Date).Date - $featuresLastDate.Date).TotalDays
    if ($staleDays -gt 2 -and $SkipData -and $SkipFeatures) {
      Write-Warning "[auto] Features are $([math]::Round($staleDays,0)) days old ($($featuresLastDate.ToString('yyyy-MM-dd'))). Consider rerunning without -SkipData/-SkipFeatures."
    }
  }

  $tasks = @()

  if (-not $SkipData) {
    $tasks += @{
      Name = "Build $DataProvider daily dataset"
      Action = {
        $args = @("-m","engine.data.build_dataset","--universe-file",$universePath,"--provider",$DataProvider,"--start",$Start)
        if ($hasEnd) { $args += @("--end",$End) }
        $args += @("--max-per-minute", ([string]$MaxPerMinute))
        & $pythonExe @args
      }
    }
  }

  if (-not $SkipFeatures) {
    $tasks += @{
      Name = "Build daily features parquet"
      Action = {
        $args = @("-m","engine.features.build_features","--universe-file",$universePath,"--provider",$DataProvider,"--start",$Start,"--out",$featuresPath)
        if ($hasEnd) { $args += @("--end",$End) }
        & $pythonExe @args
      }
    }
  }

  if ($RunTraining) {
    $tasks += @{
      Name = "Cross-validate specialists"
      Action = {
        $args = @(
          "-m","engine.models.run_cv",
          "--features",$featuresPath,
          "--label","label_up_1d",
          "--calibration","platt",
          "--out",$oofPath,
          "--calibrators-out",$calibratorsPath
        )
        & $pythonExe @args
      }
    }
    $tasks += @{
      Name = "Train meta model"
      Action = {
        $args = @(
          "-m","engine.models.train_meta",
          "--oof",$oofPath,
          "--train-folds","all-but-last:1",
          "--test-folds","last:1",
          "--out",$metaPredPath,
          "--model-out",$modelPath
        )
        & $pythonExe @args
      }
    }
  }

  if (-not $SkipPreMarket) {
    $tasks += @{
      Name = "Pre-market shortlist"
      Action = {
        $pmParams = @{
          Config        = $configPath
          Features      = $featuresPath
          Model         = $modelPath
          OOF           = $oofPath
          Calibrators   = $calibratorsPath
          TopK          = $RefineTopK
          SentimentOut  = $sentimentPath
        }
        if (-not [string]::IsNullOrWhiteSpace($intradayPath)) {
          $pmParams["IntradayFeatures"] = $intradayPath
        }
        & (Join-Path $scriptDir "pre-market.ps1") @pmParams
      }
    }
  }

  if (-not $SkipDailyRun) {
    $tasks += @{
      Name = "Daily watchlist + paper trader"
      Action = {
        $drParams = @{
          Config       = $configPath
          Features     = $featuresPath
          Model        = $modelPath
          OOF          = $oofPath
          Calibrators  = $calibratorsPath
          CalibratorMinRows = $CalibratorMinRows
          MetaMinRows = $MetaMinRows
        }
        & (Join-Path $scriptDir "daily-run.ps1") @drParams
      }
    }
  }

  if (-not $SkipAlert) {
    $tasks += @{
      Name = "Trade alert"
      Action = {
        $args = @(
          "-m","engine.tools.trade_alert",
          "--config",$configPath,
          "--alert-kind",$AlertKind,
          "--alert-category",$AlertCategory,
          "--recommendations-csv",$recommendationsPath
        )
        if (-not $DisableAlertDedupe) { $args += "--dedupe-per-day" }
        if ($AlertDryRun) { $args += "--dry-run" }
        & $pythonExe @args
      }
    }
  }

  if ($tasks.Count -eq 0) {
    Write-Host "[auto] Nothing to do (all stages skipped)."
    return
  }

  $step = 0
  $total = $tasks.Count
  foreach ($task in $tasks) {
    $step += 1
    Write-Host ("[auto] Step {0}/{1}: {2}..." -f $step, $total, $task.Name)
    & $task.Action
  }

  Write-Host "[auto] Pipeline complete."
} finally {
  Pop-Location
}
