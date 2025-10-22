param(
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Decisions = "data/paper/entry_log.csv",
  [double]$BaseProb = 0.5,
  [int]$Lookahead = 5,
  [int]$MinObs = 30,
  [switch]$Summary,
  [string]$SummaryFeatures = "",
  [string]$SummaryDiscord = "",
  [int]$SummaryLookahead = 5,
  [double]$SummaryTarget = 0.01
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

Write-Host "[nightly] Updating paper trader ledger..."
& "$scriptDir\update-paper-ledger.ps1" -Config $Config

Write-Host "[nightly] Updating expectation mapping (risk.expected_k)..."
& "$scriptDir\update-expectation.ps1" -Config $Config -Features $Features -Decisions $Decisions -BaseProb $BaseProb -Lookahead $Lookahead -MinObs $MinObs

Write-Host "[nightly] Done."

if ($Summary) {
  $sf = if ($SummaryFeatures -and $SummaryFeatures.Trim() -ne "") { $SummaryFeatures } else { $Features }
  Write-Host "[nightly] Posting daily summary..."
  $sargs = @('-Features', $sf, '-Decisions', $Decisions, '-Lookahead', $SummaryLookahead, '-TargetPct', $SummaryTarget)
  if ($SummaryDiscord -and $SummaryDiscord.Trim() -ne '') { $sargs += @('-Discord', $SummaryDiscord) }
  & "$scriptDir\summary.ps1" @sargs
}
