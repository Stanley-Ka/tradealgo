param(
  [string]$Universe = "engine/data/universe/us_all.txt",
  [string]$Start = "2015-01-01",
  [string]$End = "",
  [string]$FeaturesOut = "data/datasets/features_daily_1D.parquet",
  [int]$MaxPerMinute = 0,
  [switch]$Overwrite
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Prefer current virtualenv Python if available
$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

Write-Host "[polygon] Building daily dataset via Polygon for $Universe ..."
$dsArgs = @(
  "-m", "engine.data.build_dataset",
  "--provider", "polygon",
  "--universe-file", $Universe,
  "--start", $Start,
  "--max-per-minute", $MaxPerMinute
)
if ($End -ne "") { $dsArgs += @("--end", $End) }
if ($Overwrite) { $dsArgs += "--overwrite" }
& $py @dsArgs

Write-Host "[polygon] Building features from Polygon parquets ..."
$featArgs = @(
  "-m", "engine.features.build_features",
  "--provider", "polygon",
  "--universe-file", $Universe,
  "--start", $Start,
  "--out", $FeaturesOut
)
if ($End -ne "") { $featArgs += @("--end", $End) }
& $py @featArgs

Write-Host "[polygon] Done. Features -> $FeaturesOut"
