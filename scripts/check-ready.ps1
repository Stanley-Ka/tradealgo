param(
  [string]$Config = "engine/presets/swing_aggressive.yaml",
  [switch]$SendDiscordTest
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

$args = @('-m','engine.tools.validate_config','--config', $cfgCandidate)
if ($SendDiscordTest) { $args += '--send-discord-test' }
& $py @args
