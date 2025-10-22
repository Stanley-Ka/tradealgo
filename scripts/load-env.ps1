param(
  [string]$EnvFile = ".env"
)

# Resolve default if file not found: prefer scripts\api.env
if (-not (Test-Path $EnvFile)) {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $candidate = Join-Path $scriptDir "api.env"
  if (Test-Path $candidate) {
    $EnvFile = $candidate
  } else {
    $candidate2 = Join-Path (Join-Path $PSScriptRoot "..") "scripts\api.env"
    if (Test-Path $candidate2) {
      $EnvFile = $candidate2
    }
  }
}

if (Test-Path $EnvFile) {
  Get-Content $EnvFile | ForEach-Object {
    if (-not [string]::IsNullOrWhiteSpace($_) -and -not $_.Trim().StartsWith('#')) {
      $kv = $_ -split '=', 2
      if ($kv.Length -eq 2) {
        $name = $kv[0].Trim()
        $value = $kv[1].Trim()
        # Set for current process and PS Env: provider
        [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
        Set-Item -Path Env:$name -Value $value | Out-Null
      }
    }
  }
  Write-Host "[env] loaded: $EnvFile"
} else {
  Write-Host "[env] file not found: $EnvFile"
}
