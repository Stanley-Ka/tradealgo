param(
  [string]$SettingsPath = "engine/settings.toml",
  [string]$StorageRoot = "C:\\EngineData"
)

if (-not (Test-Path $SettingsPath)) {
  Write-Error "Settings file not found: $SettingsPath"
  exit 1
}

$content = Get-Content -Raw -Path $SettingsPath -Encoding UTF8
$pattern = '(?m)^storage_root\s*=\s*".*"'
$replacement = "storage_root = \"$StorageRoot\""
if ($content -match $pattern) {
  $new = [System.Text.RegularExpressions.Regex]::Replace($content, $pattern, $replacement)
} else {
  # Insert under [data] section if key missing
  $new = $content -replace '(?m)^\[data\]\s*$', "[data]`r`n$replacement"
}
Set-Content -Path $SettingsPath -Value $new -Encoding UTF8
Write-Host "[settings] Updated storage_root -> $StorageRoot in $SettingsPath"
