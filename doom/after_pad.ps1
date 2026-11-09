# after_pad.ps1 - after a gamepad calibration recording in DOOM (End, while turning with the right
# stick alone for ~10 s: slow and fast, left, right, up, down, circles; aim assist off):
#   fit_pad.py: the stick -> camera model and the cross-validated miss at 10-150 ms, with and without
#   the stick warp; written into the settings (sim_pad=1 and the tables) only when the fit holds
#   (R^2 >= 0.5), else the settings stay as they were. The layer reads the settings twice a second,
#   so a running game gets the stick warp at once (F6 to warp, F7 for the delay).
#
#   powershell -ExecutionPolicy Bypass -File doom\after_pad.ps1
#   ... [-Rec <rec-* folder>] (default: the newest in sim_rec_dir) [-Settings <file>] [-NoWrite]
param(
	[string]$Rec = "",
	[string]$Settings = "",
	[switch]$NoWrite
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
	Write-Host "after_pad: $msg" -ForegroundColor Red
	exit 1
}

if (-not $py -or -not (Test-Path $py)) { Fail "no Python: put python on the PATH or set PYTHON to python.exe" }
if (-not (Test-Path $settings)) { Fail "no settings at $settings" }
if (-not $Rec) {
	$recdir = Join-Path $env:USERPROFILE "datasets\doom-rec"
	foreach ($line in Get-Content $settings) {
		if ($line -match '^\s*sim_rec_dir\s*=\s*(.+?)\s*$') { $recdir = $Matches[1] }
	}
	$last = Get-ChildItem $recdir -Directory -Filter "rec-*" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
	if (-not $last) { Fail "no rec-* recording in ${recdir}: press End in the game first (with the layer on)" }
	$Rec = $last.FullName
}
if (-not (Test-Path $Rec)) { Fail "no recording at $Rec" }
$t0 = Get-Date
while (-not (Test-Path (Join-Path $Rec "info.txt"))) {
	if (((Get-Date) - $t0).TotalSeconds -gt 60) { Fail "$Rec has no info.txt after 60 s: the recording did not finish" }
	Start-Sleep -Milliseconds 500
}
$name = Split-Path -Leaf $Rec
Write-Host "after_pad: $Rec"
New-Item -ItemType Directory -Force $results | Out-Null

$backup = "$settings.before-$name"
if (-not $NoWrite) { Copy-Item $settings $backup -Force }
$fitOut = Join-Path $results "fit_pad_$name.txt"
if ($NoWrite) { & $py -I (Join-Path $here "fit_pad.py") $Rec --settings $settings | Tee-Object -FilePath $fitOut }
else { & $py -I (Join-Path $here "fit_pad.py") $Rec --settings $settings --write | Tee-Object -FilePath $fitOut }
if ($LASTEXITCODE -ne 0) {
	if (-not $NoWrite) { Move-Item $backup $settings -Force }
	Fail "fit_pad.py failed (settings as they were)"
}
$r2 = $null
foreach ($line in Get-Content $fitOut) {
	if ($line -match '^R\^2 (-?[0-9.]+)') { $r2 = [double]$Matches[1] }
}
if (-not $NoWrite) {
	if ($null -eq $r2 -or $r2 -lt 0.5) {
		Move-Item $backup $settings -Force
		Write-Host "after_pad: the stick fit does not hold (R^2 $r2): settings as they were, sim_pad stays off" -ForegroundColor Yellow
		Write-Host "  (turn with the right stick alone in the next recording, aim assist off, standing still)"
	} else {
		Write-Host "after_pad: the stick model is in the settings (R^2 $r2), sim_pad=1; the old ones in $backup" -ForegroundColor Green
	}
}
Write-Host "after_pad: done; the fit and the miss at 10-150 ms in $fitOut"
