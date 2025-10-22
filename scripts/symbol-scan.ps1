param(
  [string]$Symbols,
  [string]$Config = "engine/config.research.yaml",
  [string]$Features = "",
  [string]$Model = "",
  [string]$Calibrators = "",
  [string]$OOF = "",
  [string]$MetaCalibrator = "",
  [string]$NewsSentiment = "",
  [string]$EarningsFile = "",
  [string]$Date = "",
  [int]$HistoryDays = 15
)

if (-not $Symbols -or $Symbols.Trim() -eq "") {
  throw "Provide one or more tickers via -Symbols (comma-separated)."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$py = "python"
if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
  $py = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
}

$argsList = @(
  "-m", "engine.tools.symbol_scanner",
  "--symbols", $Symbols,
  "--config", $Config,
  "--history-days", $HistoryDays
)
if ($Features -and $Features.Trim() -ne "") { $argsList += @("--features", $Features) }
if ($Model -and $Model.Trim() -ne "") { $argsList += @("--model-pkl", $Model) }
if ($Calibrators -and $Calibrators.Trim() -ne "") { $argsList += @("--calibrators-pkl", $Calibrators) }
if ($OOF -and $OOF.Trim() -ne "") { $argsList += @("--oof", $OOF) }
if ($MetaCalibrator -and $MetaCalibrator.Trim() -ne "") { $argsList += @("--meta-calibrator-pkl", $MetaCalibrator) }
if ($NewsSentiment -and $NewsSentiment.Trim() -ne "") { $argsList += @("--news-sentiment", $NewsSentiment) }
if ($EarningsFile -and $EarningsFile.Trim() -ne "") { $argsList += @("--earnings-file", $EarningsFile) }
if ($Date -and $Date.Trim() -ne "") { $argsList += @("--date", $Date) }

& $py @argsList
