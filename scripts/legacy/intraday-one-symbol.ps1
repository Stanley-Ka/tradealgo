param(
  [string]$Symbol = "MSFT",
  [string]$Interval = "1m",
  [int]$Minutes = 60,
  [string]$BarsRoot = "data/equities/polygon",
  [string]$SnapshotOut = "data/datasets/features_intraday_latest.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [int]$TopK = 5
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\..\load-env.ps1" -EnvFile (Join-Path (Join-Path $scriptDir "..") "api.env") | Out-Null

Write-Host "[intraday-1sym] Fetching recent bars (may be limited on free plan)..."
python -m engine.data.polygon_fetch_intraday --symbol $Symbol --interval $Interval --minutes $Minutes --bars-root $BarsRoot --skip-on-rate-limit

# Ensure bars directory exists; scaffold if not
$barsDir = Join-Path $BarsRoot ("intraday_" + $Interval)
if (-not (Test-Path $barsDir)) {
  Write-Host "[intraday-1sym] Bars folder missing; scaffolding (dry-run) ..."
  python -m engine.data.polygon_stream_bars --symbols $Symbol --interval $Interval --out-root $BarsRoot --dry-run
}

Write-Host "[intraday-1sym] Building latest intraday snapshot ..."
try {
  python -m engine.tools.build_intraday_latest --bars-root $BarsRoot --interval $Interval --lookback-bars 200 --out $SnapshotOut
} catch {
  Write-Host "[intraday-1sym] Snapshot build failed (likely no bars); skipping prediction."
  Write-Host "[intraday-1sym] Done."
  exit 0
}

if (-not (Test-Path $SnapshotOut)) {
  Write-Host "[intraday-1sym] Snapshot file not found; skipping prediction."
  Write-Host "[intraday-1sym] Done."
  exit 0
}

Write-Host "[intraday-1sym] Predicting latest picks ..."
python -m engine.tools.predict_intraday --intraday-features $SnapshotOut --model-pkl $Model --top-k $TopK

Write-Host "[intraday-1sym] Done. Snapshot: $SnapshotOut"
