param(
  [string]$Features = "C:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Model = "C:\\EngineData\\models\\meta_hgb.pkl",
  [string]$Config = "engine/presets/research.yaml",
  [string]$Universe = "engine/data/universe/us_all.txt",
  [string]$Calibrators = "C:\\EngineData\\models\\spec_calibrators.pkl",
  [int]$TopK = 5,
  [double]$MinPrice = 5,
  [double]$MinADV = 10000000,
  [double]$MaxATR = 0.05,
  [string]$PriceSource = "feature",
  [string]$Provider = "none"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$cfgCandidate = if (Test-Path $Config) { $Config } else { Join-Path $repoRoot $Config }
if (-not (Test-Path $cfgCandidate)) {
  throw "Config not found: $Config"
}
$localCandidate = [System.IO.Path]::ChangeExtension($cfgCandidate, '.local.yaml')
if (Test-Path $localCandidate) {
  $cfgCandidate = $localCandidate
}

$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

# Paths for diagnostics
$rtDiag = "data/alerts/parity_rt.csv"
$drDiag = "data/alerts/parity_dr.csv"
New-Item -ItemType Directory -Path (Split-Path $rtDiag -Parent) -Force | Out-Null

# 1) Dry-run trade_alert
Write-Host "[parity] Running dry-run trade_alert..."
$drArgs = @(
  "-m", "engine.tools.trade_alert",
  "--config", $cfgCandidate,
  "--features", $Features,
  "--model-pkl", $Model,
  "--calibrators-pkl", $Calibrators,
  "--universe-file", $Universe,
  "--provider", $Provider,
  "--top-k", $TopK,
  "--min-price", $MinPrice,
  "--max-atr-pct", $MaxATR,
  "--min-adv-usd", $MinADV,
  "--price-source", $PriceSource,
  "--alert-log-csv", $drDiag,
  "--dry-run"
)
& $py @drArgs
if ($LASTEXITCODE -ne 0) { Write-Error "dry-run trade_alert failed"; exit 1 }

# 2) Real-time trigger (single shot at current minute)
Write-Host "[parity] Running real_time_alert (single trigger)..."
$now = Get-Date -Format "HH:mm"
$rtArgs = @(
  "-m", "engine.tools.real_time_alert",
  "--config", $cfgCandidate,
  "--features", $Features,
  "--model-pkl", $Model,
  "--universe-file", $Universe,
  "--calibrators-pkl", $Calibrators,
  "--provider", $Provider,
  "--top-k", $TopK,
  "--cooldown-mins", 0,
  "--alert-log-csv", $rtDiag,
  "--price-source", $PriceSource,
  "--force"
)
& $py @rtArgs

if (-not (Test-Path $rtDiag)) { Write-Error "real_time_alert did not produce diagnostics ($rtDiag). Ensure current minute matches --times and market calendar isn't blocking."; exit 1 }

# 3) Compare symbols
Write-Host "[parity] Comparing picks..."
$dr = Import-Csv -Path $drDiag
$rt = Import-Csv -Path $rtDiag
$drSyms = ($dr | Select-Object -ExpandProperty symbol | ForEach-Object { $_.ToUpper() })
$rtSyms = ($rt | Select-Object -ExpandProperty symbol | ForEach-Object { $_.ToUpper() })
$onlyDr = $drSyms | Where-Object { $_ -and ($_ -notin $rtSyms) }
$onlyRt = $rtSyms | Where-Object { $_ -and ($_ -notin $drSyms) }
Write-Host "Dry-run picks: $($drSyms -join ', ')"
Write-Host "RT picks   : $($rtSyms -join ', ')"
if ($onlyDr -or $onlyRt) {
  Write-Host "[parity] Differences detected. Only in dry-run: $($onlyDr -join ', ') | Only in RT: $($onlyRt -join ', ')"
  exit 2
} else {
  Write-Host "[parity] Picks match under controlled settings."
}
