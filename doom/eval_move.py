"""eval_move.py - what the move compensation (sim_move) adds, on a demo recording (Insert: half-size
colour, nearness, keys, mouse).

  python -I doom/eval_move.py <demo recording> [--settings doom/layer/build/cloud-sim-vk.txt]
         [--delays 50,100] [--step 4]

For frame pairs `delay` apart: the old frame as the cloud player has it, turned by the mouse (the
layer's rotation warp), and turned and moved (the splat path with the keys' move, at the settings'
sim_move_speed / tau / lag), each registered against the frame the player should see on world
patches (the weapon, the HUD rectangles and the centre left out). Reported: the median residual
shift (px at the recording's half size and at 1080p), with keys held and without.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import demo_doom as D  # noqa: E402
import fit_doom as F  # noqa: E402
from eval_doom import rodrigues  # noqa: E402
from fit_move import unit_path  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def residual(a, b, cen, p, win, mask):
    out = []
    for x, y in cen:
        x0, y0 = int(round(x - p / 2)), int(round(y - p / 2))
        if mask[y0:y0 + p, x0:x0 + p].mean() > 0.2:
            continue
        dx, dy, peak = F.phase_shift(a[y0:y0 + p, x0:x0 + p].astype(np.float32), b[y0:y0 + p, x0:x0 + p].astype(np.float32), win)
        if peak >= 0.1:
            out.append(np.hypot(dx, dy))
    return float(np.median(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--settings", default=str(HERE / "layer" / "build" / "cloud-sim-vk.txt"))
    ap.add_argument("--delays", default="50,100")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--speed", type=float, default=0.0, help="the move speed when the settings have none yet")
    a = ap.parse_args()
    rec = Path(a.rec)
    P = D.read_settings(a.settings)
    if a.speed > 0:
        P["move_speed"] = a.speed
    frames, near, qpc, mouse, freq, kind = D.load(rec, "", P["weapon"])
    if kind != "colour":
        sys.exit("a demo recording (Insert) is needed: it has the depth and the keys")
    keys = D.KEYS[: len(qpc)]
    n, h, w = near.shape
    t_ms = (qpc - qpc[0]) / freq * 1e3
    lock = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in P["locks"]:
        lock[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))] = True
    yy, xx = np.mgrid[0:h, 0:w]
    hp = P["hole"] * h
    hole = (np.abs(xx + 0.5 - w / 2) < hp) & (np.abs(yy + 0.5 - h / 2) < hp)
    wk = P["weapon"] * 65535 if P["weapon"] > 0 else -1
    mask = lock | hole | (near[0] < wk)
    cnt = F.Lagged(mouse, P["lag"] * 1e-3 * freq)
    tau = P["tau"] * 1e-3 * freq
    t1, up = unit_path(qpc, keys, freq, P["move_lag"] * 1e-3)
    mt = P["move_tau"] * 1e-3 * freq
    p = 64
    xs = np.round(np.linspace(p / 2 + 18, w - p / 2 - 18, 9))
    ys = np.round(np.linspace(p / 2 + 18, h - p / 2 - 18, 5))
    cen = np.array([(x, y) for y in ys for x in xs])
    win = F.hann2(p)
    scale = 1080.0 / h
    luma = lambda img: img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114  # noqa: E731
    print(f"{rec.name}: {n} frames {w}x{h}; move speed {P['move_speed']} near-plane units a second, tau {P['move_tau']} ms, "
          f"lag {P['move_lag']} ms; residual of world patches, median, px at 1080p")
    for lat in [float(x) for x in a.delays.split(",")]:
        rows = {"walking": [], "standing": []}
        for k in range(1, n, a.step):
            j = int(np.searchsorted(t_ms, t_ms[k] - lat, side="right") - 1)
            if j < 0:
                continue
            m = cnt(np.array([qpc[k] - tau], float)) - cnt(np.array([qpc[j] - tau], float))
            R = rodrigues(P["cx"] * m[0, 0] + P["cy"] * m[0, 1])
            pa = [np.interp(qpc[j] - mt, t1, up[:, c]) for c in range(2)]
            pb = [np.interp(qpc[k] - mt, t1, up[:, c]) for c in range(2)]
            T = P["move_speed"] * np.array([pb[0] - pa[0], 0.0, pb[1] - pa[1]])
            truth = luma(frames[k].astype(np.float64))
            none = luma(frames[j].astype(np.float64))
            rot = luma(D.screen(frames[j], near[j], R, P, lock, hole, wk))
            both = luma(D.screen(frames[j], near[j], R, P, lock, hole, wk, T)) if np.any(T) else rot
            walking = bool(np.any(keys[max(j, 0):k + 1] & 15))
            rows["walking" if walking else "standing"].append(
                [residual(none, truth, cen, p, win, mask), residual(rot, truth, cen, p, win, mask),
                 residual(both, truth, cen, p, win, mask)])
        for name, r in rows.items():
            if not r:
                continue
            r = np.array(r) * scale
            md = np.nanmedian(r, 0)
            p90 = np.nanpercentile(r, 90, 0)
            print(f"  {lat:4.0f} ms, {name:8s} ({len(r):3d} pairs): today {md[0]:6.1f} (p90 {p90[0]:6.1f}) | turn only "
                  f"{md[1]:5.1f} (p90 {p90[1]:5.1f}) | turn + move {md[2]:5.1f} (p90 {p90[2]:5.1f})")


if __name__ == "__main__":
    main()
