param(
  # Events in the form:
  #   "HH:mm | <command>"
  #   "MON-FRI HH:mm | <command>"
  #   "MON,WED,FRI HH:mm | <command>"
  [string[]]$Events = @(),
  # Optional YAML/JSON schedule file with structure:
  #   repeatDaily: true
  #   events:
  #     - time: "08:45"
  #       days: [MON,TUE,WED,THU,FRI]
  #       cmd: "./scripts/pre-market.ps1 -Config engine/config.research.yaml"
  #       parallel: false
  [string]$ScheduleFile = "",
  # Run each command without waiting (can also be set per-event in YAML)
  [switch]$Parallel,
  # Repeat the schedule every day
  [switch]$RepeatDaily
)

function Parse-Days($token) {
  $map = @{ MON=1; TUE=2; WED=3; THU=4; FRI=5; SAT=6; SUN=0 }
  $tok = ($token.Trim().ToUpper())
  if ($tok -eq 'WEEKDAYS') { return @(1,2,3,4,5) }
  if ($tok -eq 'WEEKENDS') { return @(0,6) }
  if ($tok -eq '*' -or $tok -eq 'ALL') { return @() } # empty => any day
  if ($tok -match '^[A-Z]{3}-[A-Z]{3}$') {
    $a,$b = $tok.Split('-')
    $ia = $map[$a]; $ib = $map[$b]
    if ($null -eq $ia -or $null -eq $ib) { throw "Invalid day range '$token'" }
    $out = @()
    $i = $ia
    while ($true) {
      $out += $i
      if ($i -eq $ib) { break }
      $i = ($i + 1) % 7
    }
    return $out
  }
  # Comma-separated list
  $days = @()
  foreach ($d in $tok.Split(',')) {
    $dd = $d.Trim()
    if (-not $dd) { continue }
    if (-not $map.ContainsKey($dd)) { throw "Invalid day '$dd'" }
    $days += $map[$dd]
  }
  return $days
}

function Parse-EventLine($line) {
  $parts = $line -split '\|', 2
  if ($parts.Count -lt 2) { throw "Invalid event format. Use: HH:mm | <command> or MON-FRI HH:mm | <command>" }
  $left = $parts[0].Trim()
  $cmd = $parts[1].Trim()
  $tokens = $left -split '\s+'
  $timeStr = $tokens[-1]
  $days = @()
  if ($tokens.Count -gt 1) {
    $dayTok = ($tokens[0..($tokens.Count-2)] -join ' ')
    if ($dayTok -match '[A-Za-z]') { $days = Parse-Days $dayTok }
  }
  try { $t = [DateTime]::ParseExact($timeStr, 'HH:mm', $null) }
  catch { throw "Invalid time '$timeStr'. Use 24h HH:mm (e.g., 08:45)" }
  return @{ Time=$t; Command=$cmd; Days=$days; Parallel=$false }
}

function Next-Occurrence($today, $t, $days) {
  # days: empty => any day; else array of ints Sun=0..Sat=6
  for ($k=0; $k -lt 8; $k++) {
    $cand = (Get-Date -Hour $t.Hour -Minute $t.Minute -Second 0).AddDays($k)
    if ($days.Count -eq 0 -or $days -contains [int]$cand.DayOfWeek) {
      if ($cand -ge $today) { return $cand }
    }
  }
  return (Get-Date -Hour $t.Hour -Minute $t.Minute -Second 0).AddDays(1)
}

function Load-ScheduleFile($path) {
  if (-not (Test-Path $path)) { throw "Schedule file not found: $path" }
  $ext = [System.IO.Path]::GetExtension($path).ToLower()
  if ($ext -eq '.json') {
    return (Get-Content $path -Raw | ConvertFrom-Json)
  }
  # YAML: use python with PyYAML to convert to JSON
  $py = 'python'
  $code = "import json, yaml, sys; p=sys.argv[1]; data=yaml.safe_load(open(p,'r',encoding='utf-8')); print(json.dumps(data))"
  try {
    $json = & $py -c $code $path
    return ($json | ConvertFrom-Json)
  } catch {
    throw "Failed to parse YAML schedule file. Ensure Python + PyYAML available. $_"
  }
}

$_events = @()
$_repeat = $RepeatDaily
if ($ScheduleFile) {
  $cfg = Load-ScheduleFile $ScheduleFile
  if ($cfg.PSObject.Properties.Name -contains 'repeatDaily') { $_repeat = [bool]$cfg.repeatDaily }
  if ($cfg.PSObject.Properties.Name -contains 'events') {
    foreach ($ev in $cfg.events) {
      $t = [DateTime]::ParseExact([string]$ev.time, 'HH:mm', $null)
      $days = @()
      if ($ev.days) {
        if ($ev.days -is [string]) { $days = Parse-Days $ev.days }
        else { $days = @(); foreach ($d in $ev.days) { $days += (Parse-Days ([string]$d)) } }
      }
      $_events += @{ Time=$t; Command=[string]$ev.cmd; Days=$days; Parallel=[bool]($ev.parallel) }
    }
  }
}
foreach ($e in $Events) { $_events += (Parse-EventLine $e) }
if (-not $_events -or $_events.Count -eq 0) { Write-Host "[schedule] No events provided."; exit 2 }
Write-Host "[schedule] Loaded $($_events.Count) event(s)."

while ($true) {
  $now = Get-Date
  $parsed = $_events
  # Build next occurrences and sort ascending
  $occ = @()
  foreach ($p in $parsed) {
    $when = Next-Occurrence $now $p.Time $p.Days
    $occ += @{ When=$when; Command=$p.Command; Parallel=($Parallel -or $p.Parallel) }
  }
  $occ = $occ | Sort-Object When
  foreach ($ev in $occ) {
    $now = Get-Date
    $delta = [int]([TimeSpan]($ev.When - $now)).TotalSeconds
    if ($delta -le 0) {
      # already passed for today and RepeatDaily is false: skip
      if (-not $_repeat) { continue }
      # schedule next day if repeating
      $ev.When = $ev.When.AddDays(1)
      $delta = [int]([TimeSpan]($ev.When - (Get-Date))).TotalSeconds
    }
    Write-Host "[schedule] Waiting $delta s until $($ev.When.ToString('yyyy-MM-dd HH:mm')) to run:`n  $($ev.Command)"
    Start-Sleep -Seconds ([Math]::Max($delta, 1))
    try {
      $args = @('-NoProfile','-ExecutionPolicy','Bypass','-Command', $ev.Command)
      if ($ev.Parallel) {
        Start-Process -FilePath "powershell" -ArgumentList $args | Out-Null
        Write-Host "[schedule] Launched (parallel) at $(Get-Date -Format 'HH:mm:ss')"
      } else {
        Start-Process -FilePath "powershell" -ArgumentList $args -Wait | Out-Null
        Write-Host "[schedule] Completed at $(Get-Date -Format 'HH:mm:ss')"
      }
    } catch {
      Write-Host "[schedule] Command failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
  if (-not $_repeat) { break }
  # Repeat next day: sleep to shortly after midnight local
  $tomorrow = (Get-Date).Date.AddDays(1).AddMinutes(1)
  $sleep = [int]([TimeSpan]($tomorrow - (Get-Date))).TotalSeconds
  Write-Host "[schedule] Day complete. Sleeping $sleep s until next cycle…"
  Start-Sleep -Seconds ([Math]::Max($sleep, 1))
}
