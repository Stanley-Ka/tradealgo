param(
  [string]$Features = "D:\\EngineData\\datasets\\features_daily_1D.parquet",
  [string]$Decisions = "data/paper/entry_log.csv",
  [string]$Date = "",
  [int]$Lookahead = 5,
  [double]$TargetPct = 0.01,
  [int]$Rows = 10,
  [string]$OutCsv = "",
  [string]$Discord = ""
)

$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}

$args = @(
  '-m','engine.tools.daily_summary',
  '--features', $Features,
  '--decisions', $Decisions,
  '--lookahead', $Lookahead,
  '--target-pct', $TargetPct,
  '--print-rows', $Rows
)
if ($Date -ne '') { $args += @('--date', $Date) }
if ($OutCsv -ne '') { $args += @('--out-csv', $OutCsv) }
if ($Discord -ne '') { $args += @('--discord-webhook', $Discord) }
& $py @args
