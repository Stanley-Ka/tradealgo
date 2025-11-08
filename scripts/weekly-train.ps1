param(
  [string]$Features = "C:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Preset = "engine/presets/research.yaml",
  [string]$OOF = "C:\\EngineData\\datasets\\oof_specialists.parquet",
  [string]$Calibrators = "C:\\EngineData\\models\\spec_calibrators.pkl",
  [string]$MetaPred = "C:\\EngineData\\datasets\\meta_predictions.parquet",
  [string]$Model = "C:\\EngineData\\models\\meta_hgb.pkl",
  [string]$BacktestReport = "C:\\EngineData\\backtests\\daily_topk_report.html",
  [string]$Universe = "engine/data/universe/us_all.txt",
  [string]$Start = "",
  [string]$End = "",
  [switch]$BuildWatchlist = $true,
  [string]$WatchlistOut = "engine/data/universe/watchlist.txt",
  [int]$WLTopK = 500,
  [switch]$RunAlert = $true,
  [switch]$DryRun = $true,
  [switch]$WritePaths = $true,
  [string]$ConfigOut = "engine/presets/swing_aggressive.local.yaml"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

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
$configOutPath = if ([System.IO.Path]::IsPathRooted($ConfigOut)) { $ConfigOut } else { Join-Path $repoRoot $ConfigOut }

# Echo preset and features status (last date) for quick sanity
try {
  $py = "python"
  if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
    $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
  }
  if (Test-Path $Features) {
    $code = @"
import pandas as pd
import sys
path = r'''$Features'''
try:
    d = pd.read_parquet(path, columns=['date'])
    d['date'] = pd.to_datetime(d['date'])
    print(d['date'].max().date())
except Exception as e:
    print(f'ERR: {e}')
"@
    $last = & $py -c $code
    if ($last -and -not $last.StartsWith('ERR:')) {
      Write-Host "[train] preset: $presetPath"
      Write-Host "[train] features path: $Features"
      Write-Host "[train] features last date: $last"
    } else {
      Write-Host "[train] could not read features date: $last" -ForegroundColor Yellow
    }
  } else {
    Write-Host "[train] features path not found: $Features" -ForegroundColor Yellow
  }
} catch {
  Write-Host "[train] features date probe failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

$params = @{
  Features       = $Features
  Config         = $presetPath
  OOF            = $OOF
  Calibrators    = $Calibrators
  MetaPred       = $MetaPred
  Model          = $Model
  BacktestReport = $BacktestReport
  Universe       = $Universe
  Start         = $Start
  End           = $End
  BuildWatchlist = $BuildWatchlist
  WatchlistOut   = $WatchlistOut
  WLTopK         = $WLTopK
  RunAlert       = $RunAlert
  DryRun         = $DryRun
  WritePaths     = $WritePaths
  ConfigOut      = $configOutPath
}

& (Join-Path $scriptDir "train-and-backtest.ps1") @params

# Optional: post drift summary to Discord alerts channel if available
try {
  $drift = "C:\\EngineData\\reports\\cv_drift.csv"
  if (Test-Path $drift) {
    $hook = $env:DISCORD_ALERTS_WEBHOOK_URL
    if (-not $hook -or $hook.Trim() -eq "") { $hook = $env:DISCORD_WEBHOOK_URL }
    if ($hook) {
      $lines = Get-Content $drift | Select-Object -First 6
      $msg = "CV Drift (first rows):`n" + ($lines -join "`n")
      $payload = @{ content = $msg } | ConvertTo-Json
      Invoke-RestMethod -Uri $hook -Method Post -ContentType 'application/json' -Body $payload | Out-Null
      Write-Host "[weekly-train] posted drift summary to Discord"
    }
  }
} catch {
  Write-Host "[weekly-train] drift Discord post failed: $($_.Exception.Message)" -ForegroundColor Yellow
}
