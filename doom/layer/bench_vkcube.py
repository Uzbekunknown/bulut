"""bench_vkcube.py - the layer's GPU cost on vkcube with a camera that keeps turning.

  python -I doom/layer/bench_vkcube.py <out dir> [--mode 2] [--frames 900] [--width 1920 --height 1080]
         [--speed 2000] [--extra "sim_history=0;sim_keep=1"] [--dll-dir doom/layer/build]

vkcube runs with the layer (sim_profile=1, sim_logger=0); this script writes the mouse into the
layer's shared memory (Local\\cloud-sim-vk-mouse-<pid>): a turn left and right, `speed` counts a
second at the peak, one second a period, so the warp has holes at both edges and the history
fill works. vkcube's cube has its own depth (D16), the layer follows it. At the end the layer's
log lines with the GPU time are printed.
"""
import argparse
import ctypes
import math
import mmap
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VKCUBE = Path(r"C:\VulkanSDK\1.4.357.0\Bin\vkcube.exe")
EVENTS = 65536
SHM_BYTES = 24 + EVENTS * 16

k32 = ctypes.windll.kernel32


def qpc():
    v = ctypes.c_int64()
    k32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


def qpf():
    v = ctypes.c_int64()
    k32.QueryPerformanceFrequency(ctypes.byref(v))
    return v.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--latency", type=float, default=50)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--speed", type=float, default=2000.0, help="counts a second at the turn's peak")
    ap.add_argument("--extra", default="", help="more settings, ';' between them")
    ap.add_argument("--dll-dir", default=str(HERE / "build"))
    ap.add_argument("--timeout", type=float, default=240.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings, log = out / "cloud-sim-vk.txt", out / "cloud-sim-vk.log"
    lines = ["sim=1", f"sim_mode={a.mode}", f"sim_latency={a.latency:g}", "sim_keep=0.92", "sim_slots=24", "sim_logger=0",
             "sim_cx=0 -1.0716e-03 0", "sim_cy=1.1263e-03 0 0", "sim_tau_ms=22", "sim_lag_ms=6.5", "sim_hfov=99.4",
             "sim_depth=1", "sim_depth_reversed=auto", "sim_weapon_depth=0", "sim_hole=0.028", "sim_cross=0.012",
             "sim_history=1", "sim_profile=1"]
    lines += [x.strip() for x in a.extra.split(";") if x.strip()]
    settings.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    env.update({"VK_ADD_LAYER_PATH": a.dll_dir, "VK_INSTANCE_LAYERS": "VK_LAYER_UZ_cloud_sim",
                "CLOUD_SIM_VK_SETTINGS": str(settings), "CLOUD_SIM_VK_LOG": str(log)})
    p = subprocess.Popen([str(VKCUBE), "--width", str(a.width), "--height", str(a.height), "--c", str(a.frames),
                          "--present_mode", "2"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    name = f"Local\\cloud-sim-vk-mouse-{p.pid}"
    shm = None
    t_end = time.time() + a.timeout
    freq = qpf()
    sent = 0.0
    count = 0
    last = None
    t0 = None
    while p.poll() is None and time.time() < t_end:
        if shm is None:
            time.sleep(0.05)
            if log.exists() and "the mouse from" in log.read_text(encoding="utf-8", errors="replace"):
                shm = mmap.mmap(-1, SHM_BYTES, tagname=name)
                struct.pack_into("<qii", shm, 8, freq, 1, os.getpid())
                count = struct.unpack_from("<q", shm, 0)[0]
                t0 = last = qpc()
            continue
        time.sleep(0.002)
        now = qpc()
        t = (now - t0) / freq
        # counts so far: speed * sin(2 pi t) integrated; send the difference as whole counts
        total = a.speed / (2 * math.pi) * (1 - math.cos(2 * math.pi * t))
        dx = int(round(total - sent))
        if dx:
            sent += dx
            struct.pack_into("<qii", shm, 24 + (count % EVENTS) * 16, now, dx, 0)
            count += 1
            struct.pack_into("<q", shm, 0, count)
        last = now
    if p.poll() is None:
        p.kill()
        print("vkcube killed after the timeout")
    p.wait()
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    for ln in text.splitlines():
        if "sim: on for" in ln or "GPU time" in ln or "profile" in ln or "error" in ln.lower():
            print(ln.strip())
    print(f"vkcube exit {p.returncode}; mouse events written {count}; log {log}")


if __name__ == "__main__":
    main()
