"""fit_move.py - the keys -> camera move model for the cloud simulator layer's splat path (sim_move),
from a demo recording (Insert in the game: half-size colour, nearness, times with the
movement keys, the mouse). Walk and strafe in it; turning is fine (the mouse model takes it out).

  python -I doom/fit_move.py <demo recording> [--settings doom/layer/build/cloud-sim-vk.txt] [--write]

1. Each frame pair: the turn the mouse model (the settings' sim_cx, sim_cy, sim_tau_ms, sim_lag_ms)
   predicts is taken out of the older frame (the layer's warp), and what is left between it and the
   newer frame is measured on patches (fit_doom's phase correlation).
2. What is left is the parallax of the camera's move D (old frame's axes, near-plane units): at a
   patch with inverse distance q = 1 - d (depth d = 1 - n/z), dx = f q (-Dx + u Dz), dy = f q (-Dy + v Dz).
   D per pair by weighted least squares (Huber); patches on the weapon or the HUD are left out.
3. The keys: D over a pair is the speed times the time each key state was held in it (each present's
   keys held since the present before, W forward, S back, D right, A left, a diagonal as fast as a
   straight line), delayed by tau: tau searched, the speed by least squares.
Prints the fit and the settings lines (sim_move=1, sim_move_speed, sim_move_tau_ms); --write puts
them into the settings file.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_doom as F  # noqa: E402
from demo_doom import read_settings  # noqa: E402
from eval_doom import rodrigues, warp  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
KEY_W, KEY_A, KEY_S, KEY_D = 1, 2, 4, 8


def unit_path(qpc, keys, freq, lag_s, step_s=0.001):
    """The camera's path at unit speed, (x, z), every 1 ms from the first present to the last: the keys
    held since each present before (W forward, S back, D right, A left, a diagonal as fast as a
    straight line), the velocity following them with a first-order lag (0: at once), as the layer's
    move_over does. Returns the times (qpc ticks) and the path."""
    fw = ((keys & KEY_W) > 0).astype(float) - ((keys & KEY_S) > 0).astype(float)
    rt = ((keys & KEY_D) > 0).astype(float) - ((keys & KEY_A) > 0).astype(float)
    nrm = np.hypot(fw, rt)
    dirs = np.stack([np.where(nrm > 0, rt / np.maximum(nrm, 1e-9), 0), np.where(nrm > 0, fw / np.maximum(nrm, 1e-9), 0)], 1)
    t = np.arange(float(qpc[0]), float(qpc[-1]), step_s * freq)
    s = np.clip(np.searchsorted(qpc.astype(float), t, side="left"), 1, len(qpc) - 1)   # (q[s-1], q[s]] -> keys[s]
    u = dirs[s]
    if lag_s > 0:
        a = 1.0 - np.exp(-step_s / lag_s)
        v = np.empty_like(u)
        cur = u[0].copy()
        for i in range(len(u)):
            cur += (u[i] - cur) * a
            v[i] = cur
    else:
        v = u
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(v * step_s, 0)])[:-1]
    return t, path


def unit_moves(t, path, a, b):
    """The path's move over each span (a_i, b_i], from the 1 ms path."""
    pa = np.stack([np.interp(a, t, path[:, c]) for c in range(2)], 1)
    pb = np.stack([np.interp(b, t, path[:, c]) for c in range(2)], 1)
    return pb - pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--settings", default=str(HERE / "layer" / "build" / "cloud-sim-vk.txt"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rec = Path(a.rec)
    P = read_settings(a.settings)
    info = F.read_kv(rec / "info.txt")
    w, h = int(info["half_width"][0]), int(info["half_height"][0])
    freq = float(info["qpc_frequency"][0])
    rows = [l.split() for l in (rec / "times.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    qpc = np.array([int(r[1]) for r in rows], np.int64)
    keys = np.array([int(r[3]) if len(r) > 3 else 0 for r in rows], np.int64)
    n = min(len(qpc), (rec / "color.half").stat().st_size // (w * h * 4))
    qpc, keys = qpc[:n], keys[:n]
    col = np.fromfile(rec / "color.half", np.uint8)[: n * w * h * 4].reshape(n, h, w, 4).astype(np.float32)
    luma = col[..., 0] * 0.114 + col[..., 1] * 0.587 + col[..., 2] * 0.299        # BGRA or RGBA: luma close enough
    near = np.fromfile(rec / "near.half", np.uint16)[: n * w * h].reshape(n, h, w).astype(np.float64) / 65535.0
    ev = [l.split() for l in (rec / "input.txt").read_text().splitlines() if l.startswith("m ")]
    mouse = np.array([[int(e[1]), int(e[2]), int(e[3])] for e in ev], np.int64).reshape(-1, 3)
    mouse = mouse[np.argsort(mouse[:, 0], kind="stable")] if len(mouse) else np.zeros((0, 3), np.int64)
    moving = (keys & (KEY_W | KEY_A | KEY_S | KEY_D)) > 0
    print(f"{rec.name}: {n} frames {w}x{h}, keys held in {moving.sum()} of them, {len(mouse)} mouse events; "
          f"turns from the settings (hfov {P['hfov']}, tau {P['tau']} ms)")
    Fq = (w / 2) / np.tan(np.radians(P["hfov"]) / 2)
    p = 64
    xs = np.round(np.linspace(p / 2 + 18, w - p / 2 - 18, 9))
    ys = np.round(np.linspace(p / 2 + 18, h - p / 2 - 18, 5))
    cen = np.array([(x, y) for y in ys for x in xs])
    lock = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in P["locks"]:
        lock[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    cnt = F.Lagged(mouse, P["lag"] * 1e-3 * freq) if len(mouse) else None
    tau = P["tau"] * 1e-3 * freq
    win = F.hann2(p)
    D = np.full((n - 1, 3), np.nan)
    for k in range(1, n):
        if cnt is not None:
            m = cnt(np.array([qpc[k] - tau], float)) - cnt(np.array([qpc[k - 1] - tau], float))
            R = rodrigues(P["cx"] * m[0, 0] + P["cy"] * m[0, 1])
        else:
            R = np.eye(3)
        old = warp(luma[k - 1].astype(np.float64), R, Fq)
        old = np.where(np.isnan(old), luma[k], old).astype(np.float32)
        sh = F.pair_shifts(np.stack([old, luma[k]]), p, cen)[0]
        rows_x, rows_y, bx, by, wts = [], [], [], [], []
        for i, (x, y) in enumerate(cen):
            x0, y0 = int(round(x - p / 2)), int(round(y - p / 2))
            pn = near[k - 1, y0:y0 + p, x0:x0 + p]
            if lock[y0:y0 + p, x0:x0 + p].mean() > 0.2 or (P["weapon"] > 0 and (pn < P["weapon"]).mean() > 0.1):
                continue
            q = float(np.median(np.clip(1.0 - pn, 0.0, 1.0)))   # inverse distance, near-plane units
            if sh[i, 2] < 0.1:
                continue
            u, v = (x - w / 2) / Fq, (y - h / 2) / Fq
            rows_x.append([-Fq * q, 0.0, Fq * q * u])
            rows_y.append([0.0, -Fq * q, Fq * q * v])
            bx.append(sh[i, 0])
            by.append(sh[i, 1])
            wts.append(sh[i, 2])
        if len(bx) < 6:
            continue
        A = np.array(rows_x + rows_y)
        b = np.array(bx + by)
        wt = np.array(wts + wts)
        x = np.zeros(3)
        for _ in range(4):
            sw = np.sqrt(wt)
            x, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
            r = b - A @ x
            wt = np.array(wts + wts) * np.where(np.abs(r) <= 0.5, 1.0, 0.5 / np.maximum(np.abs(r), 1e-9))
        D[k - 1] = x
    ok = np.isfinite(D[:, 0])
    print(f"camera move measured on {ok.sum()} of {n - 1} pairs; median |D| {np.median(np.hypot(D[ok, 0], D[ok, 2])):.4f} "
          f"near-plane units a frame")
    best = None
    Y = np.concatenate([D[ok, 0], D[ok, 2]])
    for lag in (0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0):
        t1ms, path = unit_path(qpc, keys, freq, lag * 1e-3)
        for tm in np.arange(0.0, 120.01, 2.0):
            d = tm * 1e-3 * freq
            U = unit_moves(t1ms, path, qpc[:-1] - d, qpc[1:] - d)           # unit-speed move per pair
            X = np.concatenate([U[ok, 0], U[ok, 1]])
            if not np.any(X):
                continue
            sp = float(np.dot(X, Y) / np.dot(X, X))
            r2 = 1 - np.sum((Y - sp * X) ** 2) / max(np.sum((Y - Y.mean()) ** 2), 1e-12)
            if best is None or r2 > best[0]:
                best = (r2, tm, lag, sp, U)
    if best is None:
        sys.exit("no movement keys in this recording: walk and strafe in it")
    r2, tm, lag, sp, U = best
    sx = float(np.dot(U[ok, 0], D[ok, 0]) / max(np.dot(U[ok, 0], U[ok, 0]), 1e-12))
    sz = float(np.dot(U[ok, 1], D[ok, 2]) / max(np.dot(U[ok, 1], U[ok, 1]), 1e-12))
    res = np.hypot(D[ok, 0] - sp * U[ok, 0], D[ok, 2] - sp * U[ok, 1])
    print(f"keys -> move: tau {tm:.0f} ms, lag {lag:.0f} ms, speed {sp:.3f} near-plane units a second (strafe alone {sx:.3f}, "
          f"forward alone {sz:.3f}), R^2 {r2:.3f}; per frame the model misses by {np.median(res):.4f} near-plane units median")
    lines = {"sim_move": "1", "sim_move_speed": f"{sp:.4f}", "sim_move_tau_ms": f"{tm:.0f}", "sim_move_lag_ms": f"{lag:.0f}"}
    print("settings:")
    for k_, v_ in lines.items():
        print(f"  {k_}={v_}")
    if a.write:
        path = Path(a.settings)
        text = path.read_text().splitlines()
        seen, out = set(), []
        for line in text:
            key = line.split("=", 1)[0].strip()
            if key in lines:
                out.append(f"{key}={lines[key]}")
                seen.add(key)
            else:
                out.append(line)
        out += [f"{k_}={v_}" for k_, v_ in lines.items() if k_ not in seen]
        path.write_text("\n".join(out) + "\n", encoding="ascii")
        print(f"written into {path}")


if __name__ == "__main__":
    main()
