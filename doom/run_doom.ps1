<#
run_doom.ps1 - DOOM (2016) with the cloud simulator layer.

  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1              start the game with the layer
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Record      a recording, as End does
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Snapshot    a snapshot, as Home does
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Mode 2 -Latency 100   as F6 / F7 do
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Demo        a demo recording, as Insert does
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Dump        the screen the layer shows, to a file
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -Stop        close the game
  powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1 -GameArgs '+g_fov 90'   more game options
    (motion blur is turned off by +r_motionblur 0 unless -KeepBlur)

The game starts from its own DOOMx64vk.exe: in the folder -Game names, else the one found through
Steam's library folders. DISABLE_NR_PROXY=1 makes a DLSS-NR vulkan-1.dll beside the game, if there
is one, pass every call on to System32's loader. Our layer is found through VK_ADD_LAYER_PATH.
Steam must be running (the game asks it for the licence); SteamAppId keeps the game from restarting
itself through Steam, which would drop this environment. Steam's fossilize layer is kept out
(DISABLE_VK_LAYER_VALVE_steam_fossilize_1), as DLSS-NR found for DOOM on an Intel Arc 140V.
The Steam overlay's Vulkan layer is kept out too (DISABLE_VK_LAYER_VALVE_steam_overlay_1) unless
-SteamOverlay: on 2026-11-08, with a gamepad, the simulator stopped getting the game's frames 75 s in
(F6, End did nothing) while the game went on; the overlay, between the game and our layer, most likely
began presenting them itself. Steam Input for the gamepad does not need that layer.
The layer's log is doom\layer\build\cloud-sim-vk.log; its settings cloud-sim-vk.txt beside it.
#>
param([string]$Game = '', [switch]$Record, [switch]$Snapshot, [switch]$Demo, [switch]$Dump, [int]$Mode = -1, [int]$Latency = -1, [switch]$Stop,
	[int]$Wait = 120, [string]$GameArgs = '', [switch]$KeepBlur, [switch]$SteamOverlay)
# DOOM's menu offers motion blur Low / Medium / High only; the cvar r_motionblur (in DOOMx64vk.exe's
# strings, 2026-11-08) turns it off. Blur smears the frames the calibration registers and the old
# frame the warp turns, so it is off unless -KeepBlur.
if (-not $KeepBlur) { $GameArgs = ('+r_motionblur 0 ' + $GameArgs).Trim() }
$ErrorActionPreference = 'Stop'
# DOOM (2016) in one of Steam's library folders (steamapps\libraryfolders.vdf)
function Find-Doom {
	$steamPath = (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath
	if (-not $steamPath) { return $null }
	$roots = @($steamPath)
	$vdf = Join-Path $steamPath 'steamapps\libraryfolders.vdf'
	if (Test-Path $vdf) {
		foreach ($m in @(Select-String -Path $vdf -Pattern '"path"\s+"([^"]+)"')) { $roots += $m.Matches[0].Groups[1].Value.Replace('\\', '\') }
	}
	foreach ($r in $roots) {
		$d = Join-Path $r 'steamapps\common\DOOM'
		if (Test-Path (Join-Path $d 'DOOMx64vk.exe')) { return $d }
	}
	return $null
}
$game = if ($Game) { $Game } else { Find-Doom }
$exe = if ($game) { Join-Path $game 'DOOMx64vk.exe' } else { '' }
$layer = Join-Path $PSScriptRoot 'layer\build'
$settings = Join-Path $layer 'cloud-sim-vk.txt'
$log = Join-Path $layer 'cloud-sim-vk.log'

function Set-Key([string]$name, [string]$value) {
	$lines = @(if (Test-Path $settings) { Get-Content $settings })
	$found = $false
	$out = @(foreach ($l in $lines) { if ($l -match "^\s*$name\s*=") { $found = $true; "$name=$value" } else { $l } })
	if (-not $found) { $out += "$name=$value" }
	Set-Content -Path $settings -Value $out -Encoding ascii
}

function Fail([string]$why) {
	Write-Host $why -ForegroundColor Red
	exit 1
}

function Step-Key([string]$name) {
	$v = 0
	if (Test-Path $settings) {
		$m = Select-String -Path $settings -Pattern "^\s*$name\s*=\s*(\d+)"
		if ($m) { $v = [int]$m.Matches[0].Groups[1].Value }
	}
	Set-Key $name ([string]($v + 1))
}

$running = @(Get-Process -Name DOOMx64vk -ErrorAction SilentlyContinue)
if ($Stop) {
	if ($running.Count) { $running | Stop-Process; 'DOOM closed' } else { 'DOOM is not running' }
	return
}
if ($Record -or $Snapshot -or $Demo -or $Dump -or $Mode -ge 0 -or $Latency -ge 0) {
	if (-not $running.Count) { Fail 'DOOM is not running' }
	if ($Mode -ge 0) { Set-Key 'sim_mode' ([string]$Mode) }
	if ($Latency -ge 0) { Set-Key 'sim_latency' ([string]$Latency) }
	if ($Record) { Step-Key 'sim_record_now' }
	if ($Snapshot) { Step-Key 'sim_snapshot_now' }
	if ($Demo) { Step-Key 'sim_demo_now' }
	if ($Dump) { Step-Key 'sim_dump_now' }
	'asked; the layer reads its settings twice a second (see cloud-sim-vk.log)'
	return
}
if ($running.Count) { Fail 'DOOM is already running' }
if (-not (Test-Path (Join-Path $layer 'cloud_sim_vk.dll'))) { Fail "no layer in ${layer}: run doom\layer\build.bat" }
if (-not $exe -or -not (Test-Path $exe)) { Fail 'DOOM (2016) is not found: pass -Game <the folder with DOOMx64vk.exe>' }
$steam = @(Get-Process -Name steam -ErrorAction SilentlyContinue)
if (-not $steam.Count) { Fail 'Steam is not running: start Steam first (the game asks it for the licence)' }
# Started from a sandboxed shell, the game cannot reach Steam: SteamAPI_Init fails and DOOM ends
# within 0.3 s with 0xE06D7363, with or without the layer. Run this script from your own window.

$env:DISABLE_NR_PROXY = '1'
$env:VK_ADD_LAYER_PATH = $layer
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_UZ_cloud_sim'
$env:SteamAppId = '379720'
$env:SteamGameId = '379720'
$env:DISABLE_VK_LAYER_VALVE_steam_fossilize_1 = '1'
if ($SteamOverlay) { Remove-Item Env:DISABLE_VK_LAYER_VALVE_steam_overlay_1 -ErrorAction SilentlyContinue }
else { $env:DISABLE_VK_LAYER_VALVE_steam_overlay_1 = '1' }
Remove-Item Env:VK_LAYER_PATH -ErrorAction SilentlyContinue
Remove-Item Env:CLOUD_SIM_VK_SETTINGS -ErrorAction SilentlyContinue
Remove-Item Env:CLOUD_SIM_VK_LOG -ErrorAction SilentlyContinue

$started = Get-Date
$p = if ($GameArgs) { Start-Process -FilePath $exe -WorkingDirectory $game -ArgumentList $GameArgs -PassThru }
	else { Start-Process -FilePath $exe -WorkingDirectory $game -PassThru }
"DOOM started (pid $($p.Id)); waiting up to $Wait s for the layer"
$deadline = $started.AddSeconds($Wait)
$on = $false
while ((Get-Date) -lt $deadline) {
	Start-Sleep -Milliseconds 500
	if ((Test-Path $log) -and (Get-Item $log).LastWriteTime -gt $started) {
		if (Select-String -Path $log -Pattern 'sim: on for' -Quiet) { $on = $true; break }
	}
	if ($p.HasExited) {
		$again = @(Get-Process -Name DOOMx64vk -ErrorAction SilentlyContinue)
		if ($again.Count) {
			"The game restarted itself (now pid $($again[0].Id)), most likely through Steam: then it runs without this environment."
			'Close it (-Stop); a steam_appid.txt with 379720 in it, beside the exe, may keep it from doing that.'
		} else {
			"DOOMx64vk.exe ended (exit code $($p.ExitCode))."
		}
		break
	}
}
if ((Test-Path $log) -and (Get-Item $log).LastWriteTime -gt $started) {
	'--- cloud-sim-vk.log'
	Get-Content $log | Select-Object -First 40
} else {
	'The layer wrote no log: it is not in the game.'
}
if ($on) {
	'The layer is on. In the game: F6 mode (local -> cloud -> warp), F7 delay, F8 blind, End record, Home snapshot, Insert demo recording.'
	if (-not $SteamOverlay) { '(the Steam overlay is off for this run: Shift+Tab does nothing; -SteamOverlay keeps it)' }
}
