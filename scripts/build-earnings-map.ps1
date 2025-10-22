param(
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [string]$Start = (Get-Date).AddDays(-30).ToString('yyyy-MM-dd'),
  [string]$End = (Get-Date).AddDays(30).ToString('yyyy-MM-dd'),
  [ValidateSet("finnhub","yfinance")][string]$Provider = "finnhub",
  [string]$Out = "data/events/earnings.parquet"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.build_earnings_map",
  "--universe-file", $Universe,
  "--start", $Start,
  "--end", $End,
  "--provider", $Provider,
  "--out", $Out
)

python @argsList
