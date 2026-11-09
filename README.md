# Bulut

Cloud gaming sends every frame from a server, so the picture follows the mouse a whole round trip late. Bulut.gg turns each late frame on the player's computer to where the camera already is, using the mouse input the client has right now.

This repository is the prototype in a real game: a Vulkan layer that makes DOOM (2016) behave as if it were streamed from a server, and then compensates the delay.

**Result:** at 100 ms of simulated cloud delay, the p90 camera miss at the centre of a 1080p screen drops from 65–113 px to 2.5–5.4 px (two in-game recordings, cross-validated).

## What is here

| Path | What it does |
|---|---|
| `doom/layer/cloud_sim_vk.c` | The layer, `VK_LAYER_UZ_cloud_sim`. Modes: `local` (the game as it is), `cloud` (every frame held back by the delay), `warp` (the old frame turned to where the camera is now, by the mouse). It finds the game's depth buffer to keep the weapon in place. The gamepad's right stick and the movement keys are in too, checked on synthetic frames so far. |
| `doom/layer/sim.comp` | The layer's compute shaders: the warp, hole filling, frame history, the move splat, the overlay. |
| `doom/layer/input_logger.c`, `mouse_shm.h` | Raw mouse, keys and XInput gamepad, timestamped and shared with the layer. |
| `doom/layer/test_vk_sim.c` | A small Vulkan app that checks the layer's modes against a known camera, optionally under the Khronos validation layer. |
| `doom/layer/build/cloud-sim-vk.txt` | The layer's settings, calibrated for DOOM (2016); the layer reads them again twice a second. |
| `doom/fit_*.py` | Fit the mouse, gamepad and movement models from a recording the layer made. |
| `doom/eval_*.py` | The screen's miss with and without compensation at any delay, cross-validated. |
| `doom/demo_doom.py` | A side-by-side video: cloud today, compensated, no delay. |
| `doom/*.ps1` | Start DOOM with the layer; fit and check after a recording. |
| `doom/snap_doom.py`, `watch_log.py`, `blind_report.py`, `layer/bench_vkcube.py` | A snapshot's depth, the layer's log, the blind test's answers, the GPU cost. |

## Build and test

Windows, Visual Studio 2022 with the C++ tools, and the Vulkan SDK:

```
doom\layer\build.bat
doom\layer\build\test_vk_sim.exe doom\layer\build 2 100 1
```

The Python scripts need Python 3 with numpy (`pip install -r requirements.txt`); `demo_doom.py` also needs ffmpeg.

## In the game

With Steam running, from your own PowerShell window:

```
powershell -ExecutionPolicy Bypass -File doom\run_doom.ps1
```

F6 switches local → cloud → warp, F7 the delay (25–150 ms), F8 the blind test; End records a clip for calibration.

## License

All rights reserved. See [LICENSE](LICENSE).
