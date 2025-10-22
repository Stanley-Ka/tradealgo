param(
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [string]$Start = "2015-01-01",
  [string]$End = "2020-01-01",
  [string]$FeaturesOut = "data/datasets/features_daily_1D.parquet"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\..\load-env.ps1" -EnvFile (Join-Path (Join-Path $scriptDir "..") "api.env") | Out-Null

python -m engine.data.build_dataset --provider yahoo --universe-file $Universe --start $Start --end $End
python -m engine.features.build_features --universe-file $Universe --provider yahoo --start $Start --out $FeaturesOut
