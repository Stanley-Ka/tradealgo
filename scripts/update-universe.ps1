param(
  [string]$Out = "engine/data/universe/us_all.txt",
  [string]$Types = "CS",
  # Use MIC codes by default for reliability: XNAS (NASDAQ), XNYS (NYSE), ARCX (NYSE ARCA)
  [string]$Exchanges = "XNAS,XNYS,ARCX"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$scriptDir\load-env.ps1" -EnvFile (Join-Path $scriptDir "api.env") | Out-Null

$argsList = @(
  "-m", "engine.tools.build_universe_polygon",
  "--out", $Out,
  "--types", $Types,
  "--exchanges", $Exchanges
)

python @argsList
