param(
  [string]$Watchlist = "engine/data/universe/watchlist.txt",
  [string]$BarsRoot = "C:\\EngineData\\equities\\polygon",
  [string]$Interval = "1m",
  [int]$LookbackBars = 200,
  [string]$SnapshotOut = "C:\\EngineData\\datasets\\features_intraday_latest.parquet"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

if (-not (Test-Path $Watchlist)) { Write-Error "Watchlist not found: $Watchlist"; exit 1 }
$syms = (Get-Content $Watchlist | Where-Object { $_ -and -not $_.StartsWith('#') }) -join ','
if (-not $syms) { Write-Error "Watchlist empty"; exit 1 }

$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) { $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe") }

Write-Host "[intraday] Fetching Polygon bars for watchlist ($Interval) ..."
& $py -m engine.data.polygon_stream_bars --symbols $syms --interval $Interval --out-root $BarsRoot

Write-Host "[intraday] Building latest intraday snapshot -> $SnapshotOut ..."
& $py -m engine.tools.build_intraday_latest --bars-root $BarsRoot --interval $Interval --lookback-bars $LookbackBars --out $SnapshotOut

Write-Host "[intraday] Done."
