param(
  [ValidateSet('aggressive','conservative')]
  [string]$Preset = 'aggressive',
  [int]$AlertEvery = 5,
  [string]$EntryTimes = '09:35,15:55',
  [switch]$StartManager
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

$baseCfg = switch ($Preset) {
  'aggressive'   { 'engine/presets/swing_aggressive.yaml' }
  'conservative' { 'engine/presets/swing_conservative.yaml' }
}

$cfgPath = Join-Path $repoRoot $baseCfg
$localPath = [System.IO.Path]::ChangeExtension($cfgPath, '.local.yaml')
$cfg = if (Test-Path $localPath) { $localPath } else { $cfgPath }

$params = @{
  Config     = $cfg
  AlertEvery = $AlertEvery
  EntryTimes = $EntryTimes
}
if ($StartManager) { $params["StartManager"] = $true }

# Intraday presets removed for swing-only focus; no intraday mixing here

& (Join-Path $scriptDir 'start-overnight.ps1') @params
