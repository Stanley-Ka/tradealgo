param(
  [string]$DecisionLog = "data/paper/decision_log.csv",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Config = "engine/config.research.yaml",
  [string]$OOFOut = "data/datasets/oof_specialists.parquet",
  [string]$CalibratorsOut = "data/models/spec_calibrators.pkl",
  [string]$MetaOut = "data/models/meta_online.pkl",
[double]$LabelThreshold = 0.0,
[int]$CalibratorMinRows = 200,
[int]$MetaMinRows = 300
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# 1) Update per-specialist calibrators from decision log if possible
$rowCount = 0
if (Test-Path $DecisionLog) {
  try {
    $rowCount = [Math]::Max(0, ([System.IO.File]::ReadAllLines($DecisionLog).Length - 1))
  } catch {
    $rowCount = 0
  }
  if ($rowCount -ge $CalibratorMinRows) {
    try {
      python -m engine.models.online_update --decision-log $DecisionLog --out-calibrator $CalibratorsOut --label-threshold $LabelThreshold --min-rows $CalibratorMinRows
    } catch {
      Write-Host "[daily-train] calibrator update failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
  } else {
    Write-Host "[daily-train] calibrator update skipped ($rowCount < $CalibratorMinRows rows)" -ForegroundColor Yellow
  }
}

# 2) Refit online meta-learner from decision log
if (Test-Path $DecisionLog) {
  if ($rowCount -ge $MetaMinRows) {
    try {
      python -m engine.models.online_meta_refit --decision-log $DecisionLog --out-model $MetaOut --label-threshold $LabelThreshold --min-rows $MetaMinRows --algo logreg --C 1.0 --class-weight-balanced
    } catch {
      Write-Host "[daily-train] meta refit skipped/failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
  } else {
    Write-Host "[daily-train] meta refit skipped ($rowCount < $MetaMinRows rows)" -ForegroundColor Yellow
  }
} else {
  Write-Host "[daily-train] no decision log found; skipping online updates" -ForegroundColor Yellow
}

# 3) Monitor calibration drift and auto-fallback drifty calibrators
if (Test-Path $DecisionLog) {
  try {
    $calOutFiltered = "$CalibratorsOut.filtered.pkl"
    python -m engine.tools.monitor_calibrators --decision-log $DecisionLog --features $Features --spec-config $Config --calibrators-in $CalibratorsOut --calibrators-out $calOutFiltered --report-csv "data/reports/calibration_monitor.csv" --window-days 30 --auc-floor 0.505 --brier-ceil 0.26 --overwrite
  } catch {
    Write-Host "[daily-train] calibration monitor failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 4) Fit rolling meta calibrator (per regime) from decision log
if (Test-Path $DecisionLog) {
  try {
    $metaCal = "data/models/meta_calibrator.pkl"
    python -m engine.tools.fit_meta_calibrator --decision-log $DecisionLog --features $Features --out $metaCal --kind isotonic --window-days 60 --per-regime regime_vol
  } catch {
    Write-Host "[daily-train] meta calibrator fit failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}
