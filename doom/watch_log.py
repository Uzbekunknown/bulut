"""watch_log.py - follows the simulator layer's log while the game runs and prints the lines that
matter: the start, the queues and the swapchain, the gamepad (when it comes or goes), recordings,
F6 / F7 / F8 and blind trials, frames passing the simulator by, errors; the fps line only when the mode
or the delay changes, and once a minute. A new game writes the log anew: that is noticed too.

  python -I doom/watch_log.py [--log <path>] [--from-start]

(`tail -F` in Git Bash missed the log being written anew on 2026-11-08; this polls twice a second.)
"""
import argparse
import re
import sys
import time
from pathlib import Path

KEEP = re.compile(r"cloud_sim_vk in|device:|swapchain:|sim: on for|recording of|demo recording|snapshot|did not start|"
                  r"no input_logger|unknown key|pad's right stick|pass the simulator|reach the simulator|starts again|"
                  r"no graphics|vkGetDeviceQueue2|stops|failed|error|ERROR|sim: mode |sim: delay |sim: blind|"
                  r"device destroyed|settings \(again\)")
FPS = re.compile(r"sim: ([\d.]+) fps, mode (\w+)( \(blind\))?, delay (\d+) ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(Path(__file__).resolve().parent / "layer" / "build" / "cloud-sim-vk.log"))
    ap.add_argument("--from-start", action="store_true", help="print the matching lines already in the log too")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    path = Path(a.log)
    pos = path.stat().st_size if path.exists() and not a.from_start else 0
    rest = b""
    pad_state = None
    last_mode = None
    last_fps_print = 0.0
    while True:
        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(0.5)
            continue
        if size < pos:                                   # written anew: a new game
            print("--- the log starts again (a new game)")
            pos, rest, pad_state, last_mode = 0, b"", None, None
        if size > pos:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read(size - pos)
            pos = size
            lines = (rest + data).split(b"\n")
            rest = lines.pop()
            for raw in lines:
                line = raw.decode("utf-8", "replace").rstrip("\r")
                if "sim: gamepad" in line:
                    state = "not connected" not in line
                    if state != pad_state:
                        print(line)
                        pad_state = state
                    continue
                m = FPS.search(line)
                if m:
                    mode = (m.group(2), bool(m.group(3)), m.group(4))
                    now = time.time()
                    if mode != last_mode or now - last_fps_print > 60:
                        print(line[:160])
                        last_mode, last_fps_print = mode, now
                    continue
                if KEEP.search(line):
                    print(line[:300])
        time.sleep(0.5)


if __name__ == "__main__":
    main()
