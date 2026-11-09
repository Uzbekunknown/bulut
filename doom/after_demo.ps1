# after_demo.ps1 - after a demo recording in DOOM (Insert: walk with WASD and turn for ~7 s):
#   1. fit_move.py: the keys -> camera move model; written into the settings (sim_move=1 and the
#      speed, tau, lag) only when the fit holds (R^2 >= 0.5), else the settings stay as they were;
#   2. eval_move.py: what the move compensation adds at 50 and 100 ms, walking and standing;
#   3. demo_doom.py: the 2x2 colour videos at 100 and 50 ms.
# The layer reads the settings twice a second, so a running game gets the move model at once.
#
#   powershell -ExecutionPolicy Bypass -File doom\after_demo.ps1
#   ... [-Demo <demo-* folder>] (default: the newest in sim_rec_dir) [-Settings <file>] [-NoWrite] [-NoVideo]
param(
	[string]$Demo = "",
	[string]$Settings = "",
	[switch]$NoWrite,
	[switch]$NoVideo
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# Python 3 with numpy: the one PYTHON names, else python on the PATH
$py = $env:PYTHON
if (-not $py) { $cmd = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1; if ($cmd) { $py = $cmd.Source } }
$settings = $Settings
if (-not $settings) { $settings = Join-Path $here "layer\build\cloud-sim-vk.txt" }
$results = Join-Path $here "results"

function Fail([string]$msg) {
	Write-Host "after_demo: $msg" -ForegroundColor Red
	exit 1
}

if (-not $py -or -not (Test-Path $py)) { Fail "no Python: put python on the PATH or set PYTHON to python.exe" }
if (-not (Test-Path $settings)) { Fail "no settings at $settings" }
if (-not $Demo) {
	$recdir = Join-Path $env:USERPROFILE "datasets\doom-rec"
	foreach ($line in Get-Content $settings) {
		if ($line -match '^\s*sim_rec_dir\s*=\s*(.+?)\s*$') { $recdir = $Matches[1] }
	}
	$last = Get-ChildItem $recdir -Directory -Filter "demo-*" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
	if (-not $last) { Fail "no demo-* recording in ${recdir}: press Insert in the game first (with the layer on)" }
	$Demo = $last.FullName
}
if (-not (Test-Path $Demo)) { Fail "no recording at $Demo" }
# info.txt is the last file the layer writes: wait for it (the disk writer may still be busy)
$t0 = Get-Date
while (-not (Test-Path (Join-Path $Demo "info.txt"))) {
	if (((Get-Date) - $t0).TotalSeconds -gt 60) { Fail "$Demo has no info.txt after 60 s: the recording did not finish" }
	Start-Sleep -Milliseconds 500
}
$name = Split-Path -Leaf $Demo
Write-Host "after_demo: $Demo"
New-Item -ItemType Directory -Force $results | Out-Null

# 1. the move model
$backup = "$settings.before-$name"
if (-not $NoWrite) { Copy-Item $settings $backup -Force }
$fitOut = Join-Path $results "fit_move_$name.txt"
if ($NoWrite) { & $py -I (Join-Path $here "fit_move.py") $Demo --settings $settings | Tee-Object -FilePath $fitOut }
else { & $py -I (Join-Path $here "fit_move.py") $Demo --settings $settings --write | Tee-Object -FilePath $fitOut }
if ($LASTEXITCODE -ne 0) {
	if (-not $NoWrite) { Move-Item $backup $settings -Force }
	Fail "fit_move.py failed (settings as they were)"
}
$r2 = $null
foreach ($line in Get-Content $fitOut) {
	if ($line -match 'R\^2 (-?[0-9.]+)') { $r2 = [double]$Matches[1] }
}
if (-not $NoWrite) {
	if ($null -eq $r2 -or $r2 -lt 0.5) {
		Move-Item $backup $settings -Force
		Write-Host "after_demo: the move fit does not hold (R^2 $r2): settings as they were, sim_move stays off" -ForegroundColor Yellow
		Write-Host "  (walk longer and straighter in the next demo recording: W / S / A / D held for a second or more)"
	} else {
		Write-Host "after_demo: the move model is in the settings (R^2 $r2), sim_move=1; the old ones in $backup" -ForegroundColor Green
	}
}

# 2. what the move adds (with the speed just fitted, even when it was not written)
$evalOut = Join-Path $results "move_eval_$name.txt"
$speed = $null
foreach ($line in Get-Content $fitOut) {
	if ($line -match '^\s*sim_move_speed=([0-9.eE+-]+)') { $speed = $Matches[1] }
}
if ($speed) { & $py -I (Join-Path $here "eval_move.py") $Demo --settings $settings --speed $speed | Tee-Object -FilePath $evalOut }
else { & $py -I (Join-Path $here "eval_move.py") $Demo --settings $settings | Tee-Object -FilePath $evalOut }

# 3. the videos
if (-not $NoVideo) {
	foreach ($ms in 100, 50) {
		$mp4 = Join-Path $here "demo_doom_${ms}ms_colour.mp4"
		& $py -I (Join-Path $here "demo_doom.py") $Demo --settings $settings --latency $ms --out $mp4
		if ($LASTEXITCODE -eq 0) { Write-Host "after_demo: $mp4 ($([math]::Round((Get-Item $mp4).Length / 1MB, 1)) MB)" }
	}
}
Write-Host "after_demo: done; the fit in $fitOut, the check in $evalOut"
