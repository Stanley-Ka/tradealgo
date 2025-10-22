param(
  [string]$Features = "data/datasets/features_daily_1D.parquet",
  [string]$Model = "data/models/meta_lr.pkl",
  [string]$Universe = "engine/data/universe/nasdaq100.example.txt",
  [int]$TopK = 5,
  [double]$Threshold = 0.0,
  [int]$Confirm = 1,
  [string]$SectorMap = "",
  [int]$SectorCap = 0,
  [string]$Positions = "data/paper/positions.csv",
  [string]$Log = "data/paper/entry_log.csv",
  [switch]$DryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.entry_loop",
  "--features", $Features,
  "--model-pkl", $Model,
  "--universe-file", $Universe,
  "--top-k", $TopK,
  "--confirmations", $Confirm,
  "--positions-csv", $Positions,
  "--decision-log-csv", $Log
)
if ($Threshold -gt 0) { $argsList += @("--entry-threshold", $Threshold) }
if ($SectorMap -ne "") { $argsList += @("--sector-map-csv", $SectorMap) }
if ($SectorCap -gt 0) { $argsList += @("--sector-cap", $SectorCap) }
if ($DryRun) { $argsList += "--dry-run" }

python @argsList
