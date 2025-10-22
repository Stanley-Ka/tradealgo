param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$WatchlistIn = "engine/data/universe/daily_watchlist.txt",
  [string]$SentimentOut = "data/datasets/sentiment_finbert.parquet",
  [string]$IntradayFeatures = "",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$OOF = "data/datasets/oof_specialists.parquet",
  [string]$Calibrators = "data/models/spec_calibrators.pkl",
  [int]$SentimentDays = 3,
  [int]$TopK = 5,
  [double]$BlendSentiment = 0.25,
  [double]$MixIntraday = 0.3,
  [double]$IntradayAlpha = 0.15,
  [double]$MinADV = 10000000,
  [double]$MinPrice = 1,
  [double]$MaxPrice = 10000,
  [string]$SectorMapCsv = "engine/data/sector_map.csv",
  [double]$SectorBoost = 0.10,
  [ValidateSet("meta","sentiment","pre")][string]$SectorScore = "meta",
  [int]$BreadthLookback = 252,
  [double]$BreadthWeight = 0.10,
  [string]$EarningsFile = "",
  [int]$EarningsBlackout = 2,
  [switch]$BuildEarnings,
  [int]$EarningsStartDaysBack = 0,
  [int]$EarningsEndDaysAhead = 21,
  [string]$Out = "engine/data/universe/premarket_refined.txt",
  [string]$OutCSV = "data/signals/premarket_refined.csv"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

# Ensure a watchlist exists; if missing, build it from latest features/model
if (-not (Test-Path $WatchlistIn)) {
  Write-Host "[pre-market] Watchlist not found; building via daily-build.ps1 ..."
  try {
    & (Join-Path $scriptDir "daily-build.ps1") -Config $Config -Features $Features -Model $Model -OOF $OOF -Calibrators $Calibrators -TopK 50 -Out $WatchlistIn
  } catch {
    Write-Host "[pre-market] watchlist build failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 0) Optional: build/refresh earnings map around current date
if ($BuildEarnings -or ($EarningsFile -and -not (Test-Path $EarningsFile))) {
  try {
    $start = (Get-Date).AddDays(-1 * $EarningsStartDaysBack).ToString('yyyy-MM-dd')
    $end = (Get-Date).AddDays($EarningsEndDaysAhead).ToString('yyyy-MM-dd')
    $argsE = @(
      "-m", "engine.tools.build_earnings_map",
      "--universe-file", $WatchlistIn,
      "--start", $start,
      "--end", $end,
      "--provider", "finnhub",
      "--out", $(if ($EarningsFile) { $EarningsFile } else { "data/events/earnings.parquet" })
    )
    python @argsE
  } catch {
    Write-Host "[pre-market] earnings map step failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 1) Update daily news sentiment for the watchlist universe
Write-Host "[pre-market] Updating news sentiment for watchlist..."
try {
  $uni = $WatchlistIn
  if (-not (Test-Path $uni)) {
    throw "Watchlist not found: $uni"
  }
  $provider = "polygon"
  $token = $env:POLYGON_API_KEY
  $argsA = @(
    "-m", "engine.tools.update_daily_sentiment",
    "--universe-file", $uni,
    "--days", $SentimentDays,
    "--provider", $provider,
    "--out", $SentimentOut,
    "--skip-on-rate-limit"
  )
  if ($token) { $argsA += @("--token", $token) }
  python @argsA
} catch {
  Write-Host "[pre-market] sentiment update failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 2) Refine watchlist to top-K using sentiment + meta
Write-Host "[pre-market] Refining watchlist (top $TopK)..."
$hook = $env:DISCORD_ALERTS_WEBHOOK_URL
if (-not $hook -or $hook.Trim() -eq "") { $hook = $env:DISCORD_WEBHOOK_URL }
$argsB = @(
  "-m", "engine.tools.refine_watchlist",
  "--config", $Config,
  "--watchlist-in", $WatchlistIn,
  "--top-k", $TopK,
  "--blend-sentiment", $BlendSentiment,
  "--news-sentiment", $SentimentOut,
  "--mix-intraday", $MixIntraday,
  "--intraday-alpha", $IntradayAlpha,
  "--min-adv-usd", $MinADV,
  "--min-price", $MinPrice,
  "--max-price", $MaxPrice,
  "--sector-map-csv", $SectorMapCsv,
  "--sector-boost", $SectorBoost,
  "--sector-score", $SectorScore,
  "--breadth-lookback-days", $BreadthLookback,
  "--breadth-weight", $BreadthWeight,
  "--two-tower",
  "--tower-regime", "regime_vol",
  "--tower-weight-low", "0.20",
  "--tower-weight-high", "0.50",
  "--out", $Out,
  "--out-csv", $OutCSV
)
if (Test-Path $Features) { $argsB += @("--features", $Features) }
if (Test-Path $Model) { $argsB += @("--model-pkl", $Model) }
if ($EarningsFile) { $argsB += @("--earnings-file", $EarningsFile, "--earnings-blackout", $EarningsBlackout) }
if ($IntradayFeatures -and $IntradayFeatures.Trim() -ne "") { $argsB += @("--intraday-features", $IntradayFeatures) }
if (Test-Path $OOF) { $argsB += @("--oof", $OOF) }
if (Test-Path $Calibrators) { $argsB += @("--calibrators-pkl", $Calibrators) }
if ($hook -and $hook.Trim() -ne "") { $argsB += @("--discord-webhook", $hook) }

python @argsB

Write-Host "[pre-market] Done. Shortlist -> $Out"
