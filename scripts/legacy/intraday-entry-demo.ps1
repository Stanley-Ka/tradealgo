param(
  [string]$Symbols = "MSFT,AAPL",
  [string]$Interval = "1m",
  [string]$BarsRoot = "data/equities/polygon",
  [int]$LookbackBars = 200,
  [string]$SnapshotOut = "data/datasets/features_intraday_latest.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [int]$TopK = 5,
  [double]$Threshold = 0.6,
  [int]$Confirm = 2,
  [string]$SectorMap = "",
  [int]$SectorCap = 1,
  [string]$Positions = "data/paper/positions.csv",
  [switch]$DryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\..\load-env.ps1" -EnvFile (Join-Path (Join-Path $scriptDir "..") "api.env") | Out-Null

Write-Host "[intraday-demo] Scaffolding bars (dry-run)..."
python -m engine.data.polygon_stream_bars --symbols $Symbols --interval $Interval --out-root $BarsRoot --dry-run

Write-Host "[intraday-demo] Building latest intraday snapshot ..."
python -m engine.tools.build_intraday_latest --bars-root $BarsRoot --interval $Interval --lookback-bars $LookbackBars --out $SnapshotOut

Write-Host "[intraday-demo] Selecting entries from snapshot ..."
$argsList = @(
  "-m", "engine.tools.entry_loop",
  "--intraday-features", $SnapshotOut,
  "--model-pkl", $Model,
  "--universe-file", $Universe,
  "--top-k", $TopK,
  "--entry-threshold", $Threshold,
  "--confirmations", $Confirm,
  "--positions-csv", $Positions
)
if ($SectorMap -ne "") { $argsList += @("--sector-map-csv", $SectorMap) }
if ($SectorCap -gt 0) { $argsList += @("--sector-cap", $SectorCap) }
if ($DryRun) { $argsList += "--dry-run" }

python @argsList

Write-Host "[intraday-demo] Done. Snapshot: $SnapshotOut"
