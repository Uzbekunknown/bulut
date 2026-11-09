"""fit_doom.py - the mouse -> camera model for the cloud simulator layer, from a recording
the layer made (End in the game, or run_doom.ps1 -Record): frames.y8 (quarter-size luma), times.txt
(the present times), input.txt (the raw mouse), info.txt.

  python -I doom/fit_doom.py <recording folder> [--write <cloud-sim-vk.txt>] [--rows <share>]

1. The image motion of each frame pair, by phase correlation of patches in the upper part of the
   frame (the weapon and the HUD are at the bottom and stay put on the screen; the centre, where the
   game's crosshair is, is left out).
2. The camera's rotation per pair, w (radians about the camera's x right, y down, z forward axes,
   P_new = R P_old), from the patch shifts, for a focal f. Under a rotation a shift grows toward the
   edges as 1 + u^2, which is what tells f: f is searched over the field of view.
3. The mouse: w_k = C m_k, m_k the counts in x and y over (t_(k-1) - tau, t_k - tau] seen through a
   first-order lag; tau and the lag searched, C (3x2) by least squares.
It prints the fit and the settings lines (sim_hfov, sim_cx, sim_cy, sim_tau_ms, sim_lag_ms); --write
puts them into a settings file (a running layer picks them up within half a second).

For a recording in a game: stand still and turn the mouse left, right, up and down for 10-15 s.
Walking adds parallax that a rotation does not explain; the fit drops what does not fit, but less
of it is better.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def read_kv(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if p:
            out[p[0]] = p[1:]
    return out


def read_recording(rec):
    info = read_kv(rec / "info.txt")
    w, h, rb = int(info["luma_width"][0]), int(info["luma_height"][0]), int(info["luma_row_bytes"][0])
    raw = np.fromfile(rec / "frames.y8", np.uint8)
    n = raw.size // (rb * h)
    frames = raw[: n * rb * h].reshape(n, h, rb)[:, :, :w].astype(np.float32)
    rows = [l.split() for l in (rec / "times.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    qpc = np.array([int(r[1]) for r in rows[:n]], np.int64)
    order = np.argsort(qpc, kind="stable")
    ev = [l.split() for l in (rec / "input.txt").read_text().splitlines() if l.startswith("m ")]
    mouse = np.array([[int(e[1]), int(e[2]), int(e[3])] for e in ev], np.int64).reshape(-1, 3)
    mouse = mouse[np.argsort(mouse[:, 0], kind="stable")]
    freq = float(info["qpc_frequency"][0])
    screen = (int(info["screen"][0]), int(info["screen"][1]))
    return frames[order], qpc[order], mouse, freq, screen, info


def hann2(n):
    v = np.hanning(n).astype(np.float32)
    return np.outer(v, v)


def phase_shift(a, b, win):
    """The (dx, dy) that carries patch a onto patch b (b(x) = a(x - d)), sub-pixel, and the peak."""
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifft2(R))
    ny, nx = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    c = r[iy, ix]

    def vertex(l, m, rr):
        d = l - 2.0 * m + rr
        return 0.0 if abs(d) < 1e-12 else 0.5 * (l - rr) / d

    dx = ix + vertex(r[iy, (ix - 1) % nx], c, r[iy, (ix + 1) % nx])
    dy = iy + vertex(r[(iy - 1) % ny, ix], c, r[(iy + 1) % ny, ix])
    if dx > nx / 2:
        dx -= nx
    if dy > ny / 2:
        dy -= ny
    return dx, dy, c


def patch_grid(w, h, rows_share, cols=7, rows=3):
    """Patch size and centres: `cols` x `rows` over the upper `rows_share` of the frame, none on the centre."""
    p = 1 << int(round(np.log2(max(16, min(w, h) / 4))))
    m = p // 4 + 2                       # room for the content to move: a patch must stay in the frame
    xs = np.round(np.linspace(p / 2 + m, w - p / 2 - m, cols))
    ys = np.round(np.linspace(p / 2 + m, max(p / 2 + m, rows_share * h - p / 2 - m), rows))
    cen = []
    for y in ys:
        for x in xs:
            if abs(x - w / 2) < p / 2 + 2 and abs(y - h / 2) < p / 2 + 2:
                continue
            cen.append((x, y))
    return p, np.array(cen)


def screen_locked(sh, min_motion=0.5, min_slope=0.5):
    """Patches that do not follow the frame's motion: the HUD, the weapon, a message. Their shift,
    regressed on the patches' median shift (pairs where that moved), has a slope well under 1."""
    med = np.median(sh[:, :, :2], axis=1)                     # (n, 2)
    mag = np.hypot(med[:, 0], med[:, 1])
    use = mag > min_motion
    if use.sum() < 10:
        return np.zeros(sh.shape[1], bool), np.ones(sh.shape[1])
    m = med[use]
    slopes = np.empty(sh.shape[1])
    for i in range(sh.shape[1]):
        s = sh[use, i, :2]
        slopes[i] = float(np.sum(s * m) / max(np.sum(m * m), 1e-9))
    return slopes < min_slope, slopes


def phase_slope(a, b, win):
    """The sub-pixel (dx, dy) that carries a onto b, from the slope of the cross-spectrum's phase at
    low frequencies (|d| well under a pixel): unbiased where the correlation peak's vertex is not."""
    n = a.shape[0]
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    X = B * np.conj(A)                                  # |A|^2 exp(-i k.d)
    k = np.fft.fftfreq(n) * 2 * np.pi
    ky, kx = np.meshgrid(k, k, indexing="ij")
    band = (np.hypot(kx, ky) < 0.45 * np.pi) & (np.hypot(kx, ky) > 0)
    wgt = np.abs(X[band])
    ph = np.angle(X[band])
    M = np.stack([kx[band], ky[band]], 1)
    sw = np.sqrt(wgt)
    d, *_ = np.linalg.lstsq(M * sw[:, None], -ph * sw, rcond=None)
    return float(d[0]), float(d[1])


def sample(img, x0, y0, p):
    """img's p x p patch whose corner is at (x0, y0), sub-pixel, bilinear; None outside."""
    h, w = img.shape
    if x0 < 0 or y0 < 0 or x0 + p + 1 >= w or y0 + p + 1 >= h:
        return None
    ix, iy = int(np.floor(x0)), int(np.floor(y0))
    fx, fy = x0 - ix, y0 - iy
    q = img[iy:iy + p + 1, ix:ix + p + 1]
    top = q[:p, :p] * (1 - fx) + q[:p, 1:p + 1] * fx
    bot = q[1:p + 1, :p] * (1 - fx) + q[1:p + 1, 1:p + 1] * fx
    return top * (1 - fy) + bot * fy


def pair_shifts(frames, p, cen, rounds=3):
    """Per frame pair, per patch: dx, dy, peak. The shift by phase correlation, then refined: b's
    patch is taken again where a's content went by the estimate so far (bilinear, from the whole
    frame) and the phase slope gives what is left. A window that stays put while the content moves
    under it pulls a shift toward zero, in proportion to the shift it sees; as what is left goes to
    zero, so does that pull."""
    win = hann2(p)
    n, h, w = frames.shape
    out = np.zeros((n - 1, len(cen), 3), np.float64)
    for k in range(1, n):
        a, b = frames[k - 1], frames[k]
        for i, (x, y) in enumerate(cen):
            x0, y0 = int(round(x - p / 2)), int(round(y - p / 2))
            pa = a[y0:y0 + p, x0:x0 + p]
            dx, dy, peak = phase_shift(pa, b[y0:y0 + p, x0:x0 + p], win)
            dx, dy = float(round(dx)), float(round(dy))
            done = 0
            for _ in range(rounds):
                pb = sample(b, x0 + dx, y0 + dy, p)
                if pb is None:
                    break
                rx, ry = phase_slope(pa, pb, win)
                if not (abs(rx) < 1.5 and abs(ry) < 1.5):
                    break
                dx, dy = dx + rx, dy + ry
                done += 1
            out[k - 1, i] = (dx, dy, peak if done == rounds else 0.0)   # not refined: not used
    return out


def selftest():
    """Known sub-pixel shifts of a smooth random image, through the patch estimator."""
    rng = np.random.default_rng(1)
    n = 256
    base = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    spec = np.fft.fft2(base) * np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.08 ** 2))
    worst = 0.0
    for dx, dy in [(0.3, -0.2), (1.3, -0.7), (-2.6, 0.45), (4.2, 3.1)]:
        a = np.real(np.fft.ifft2(spec)).astype(np.float32)
        b = np.real(np.fft.ifft2(spec * np.exp(-2j * np.pi * (kx * dx + ky * dy)))).astype(np.float32)
        frames = np.stack([a, b])
        p, cen = 32, np.array([[128.0, 128.0], [70.0, 90.0]])
        sh = pair_shifts(frames, p, cen)[0]
        err = np.abs(sh[:, :2] - [dx, dy]).max()
        worst = max(worst, err)
        print(f"shift ({dx:+.2f}, {dy:+.2f}): measured ({sh[0, 0]:+.3f}, {sh[0, 1]:+.3f}), ({sh[1, 0]:+.3f}, {sh[1, 1]:+.3f})")
    print(f"selftest: worst error {worst:.3f} px")
    return worst < 0.05


def rotation_fit(sh, cen, F, cxy, iters=4):
    """Per pair, w (3) that best explains the shifts at focal F (luma px): IRLS (Huber, 0.5 px).
    Returns w (n x 3), the robust cost, and the per-pair residual (px, rms of the kept patches)."""
    u = (cen[:, 0] - cxy[0]) / F
    v = (cen[:, 1] - cxy[1]) / F
    # dx/F = wy (1 + u^2) - wx u v - wz v ; dy/F = -wx (1 + v^2) + wy u v + wz u
    Ax = np.stack([-u * v, 1 + u * u, -v], 1) * F
    Ay = np.stack([-(1 + v * v), u * v, u], 1) * F
    A = np.concatenate([Ax, Ay], 0)                       # (2m, 3)
    n = sh.shape[0]
    W = np.zeros((n, 3))
    cost = 0.0
    res = np.zeros(n)
    for k in range(n):
        y = np.concatenate([sh[k, :, 0], sh[k, :, 1]])
        conf = np.clip(np.concatenate([sh[k, :, 2], sh[k, :, 2]]), 0.0, 1.0)
        wt = conf.copy()
        x = np.zeros(3)
        for _ in range(iters):
            sw = np.sqrt(wt)
            x, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
            r = y - A @ x
            wt = conf * np.where(np.abs(r) <= 0.5, 1.0, 0.5 / np.maximum(np.abs(r), 1e-9))
        r = y - A @ x
        a = np.abs(r)
        cost += float(np.sum(conf * np.where(a <= 0.5, 0.5 * a * a, 0.5 * (a - 0.25))))
        keep = a < 2.0
        res[k] = float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else np.inf
        W[k] = x
    return W, cost, res


class Lagged:
    """C(T): sum over the mouse events t_i <= T of v_i (1 - exp(-(T - t_i) / lag)), x and y; the
    decaying part is run once per lag, so many T (many tau) cost one lookup each."""

    def __init__(self, mouse, lag_ticks):
        self.t = mouse[:, 0].astype(np.float64)
        v = mouse[:, 1:3].astype(np.float64)
        self.step = np.concatenate([np.zeros((1, 2)), np.cumsum(v, 0)], 0)   # step[j]: the first j events
        self.lag = lag_ticks
        self.L = None
        if lag_ticks > 0:
            L = np.zeros((len(self.t) + 1, 2))                               # L[j]: just after event j-1
            dec = np.exp(-np.diff(self.t, prepend=self.t[:1]) / lag_ticks)
            for i in range(len(self.t)):
                L[i + 1] = L[i] * dec[i] + v[i]
            self.L = L

    def __call__(self, times):
        j = np.searchsorted(self.t, times, side="right")
        S = self.step[j]
        if self.L is None:
            return S
        last = np.where(j > 0, self.t[np.maximum(j - 1, 0)], times)
        return S - self.L[j] * np.exp(-(times - last) / self.lag)[:, None]


def lagged_counts(mouse, times, lag_ticks):
    return Lagged(mouse, lag_ticks)(times)


def mouse_fit(W, ok, qpc, mouse, freq, taus_ms, lags_ms):
    """C (3x2), tau, lag maximising the explained rotation (yaw and pitch) over the kept pairs."""
    best = None
    t1 = qpc[:-1].astype(np.float64)
    t2 = qpc[1:].astype(np.float64)
    Y = W[ok]
    sst = np.sum((Y[:, :2] - Y[:, :2].mean(0)) ** 2, 0)
    for lag in lags_ms:
        cnt = Lagged(mouse, lag * 1e-3 * freq)
        for tau in taus_ms:
            d = tau * 1e-3 * freq
            M = cnt(t2 - d) - cnt(t1 - d)
            X = M[ok]
            C, *_ = np.linalg.lstsq(X, Y, rcond=None)             # Y ~ X C, C (2 x 3)
            sse = np.sum((Y[:, :2] - X @ C[:, :2]) ** 2, 0)
            r2 = 1 - sse / np.maximum(sst, 1e-30)
            score = float(np.sum(sse))
            if best is None or score < best[0]:
                best = (score, tau, lag, C.T.copy(), r2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="?")
    ap.add_argument("--write", help="a settings file to put the fit into")
    ap.add_argument("--rows", type=float, default=1.0, help="the upper share of the frame the patches cover")
    ap.add_argument("--cols", type=int, default=9, help="patch columns")
    ap.add_argument("--grid-rows", type=int, default=5, help="patch rows")
    ap.add_argument("--max-shift", type=float, default=6.0, help="pairs moving more (quarter px) are left out of the FOV fit")
    ap.add_argument("--hfov", type=float, default=0.0, help="the field of view, when the fit cannot tell it")
    ap.add_argument("--selftest", action="store_true", help="check the shift estimator on known shifts")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.rec:
        ap.error("a recording folder")
    rec = Path(a.rec)
    frames, qpc, mouse, freq, screen, info = read_recording(rec)
    n, h, w = frames.shape
    dt = np.diff(qpc) / freq * 1e3
    print(f"{rec.name}: {n} frames {w}x{h} (screen {screen[0]}x{screen[1]}), {np.median(dt):.1f} ms a frame "
          f"(p95 {np.percentile(dt, 95):.1f}), {len(mouse)} mouse events")
    if len(mouse) < 10:
        sys.exit("too few mouse events: was input_logger running? (sim_logger=1)")
    p, cen = patch_grid(w, h, a.rows, a.cols, a.grid_rows)
    sh = pair_shifts(frames, p, cen)
    locked, slopes = screen_locked(sh)
    if locked.any():
        print("left out, they stay put on the screen (HUD, weapon): "
              + ", ".join(f"({cen[i, 0]:.0f},{cen[i, 1]:.0f}) slope {slopes[i]:.2f}" for i in np.where(locked)[0]))
    cen, sh = cen[~locked], sh[:, ~locked]
    print(f"{len(cen)} patches of {p}x{p}; median patch shift {np.median(np.abs(sh[:, :, 0])):.2f} px x, "
          f"{np.median(np.abs(sh[:, :, 1])):.2f} px y (quarter size), median peak {np.median(sh[:, :, 2]):.2f}")
    # fast turns blur the frame (motion blur) and the shifts with them: those pairs are left out
    fast = np.median(np.hypot(sh[:, :, 0], sh[:, :, 1]), axis=1) > a.max_shift
    if fast.any():
        print(f"{fast.sum()} pairs faster than {a.max_shift} quarter px a frame left out of the field of view fit")
    cxy = (w / 2.0, h / 2.0)
    fovs = np.arange(30.0, 150.01, 1.0)
    costs = []
    for fov in fovs:
        F = (w / 2.0) / np.tan(np.radians(fov) / 2)
        costs.append(rotation_fit(sh[~fast], cen, F, cxy, iters=2)[1])
    i = int(np.argmin(costs))
    fine = np.arange(max(25.0, fovs[i] - 1.5), fovs[i] + 1.51, 0.1)
    fcost = [rotation_fit(sh[~fast], cen, (w / 2.0) / np.tan(np.radians(f) / 2), cxy)[1] for f in fine]
    hfov = float(fine[int(np.argmin(fcost))])
    if a.hfov:
        print(f"field of view fit {hfov:.1f} deg; using the given {a.hfov:.1f}")
        hfov = a.hfov
    elif i in (0, len(fovs) - 1):
        print(f"WARNING: the field of view fit ran to the edge of 30..150 ({hfov:.1f}): the shifts do not tell it; "
              "give it with --hfov (the game's setting) and the rest is still fitted")
    F = (w / 2.0) / np.tan(np.radians(hfov) / 2)
    W, _, res = rotation_fit(sh, cen, F, cxy)
    moving = np.linalg.norm(W, axis=1) * F > 0.3          # pairs where the camera turned (> 0.3 quarter px)
    ok = np.isfinite(res) & (res < 1.0) & ~fast
    print(f"field of view {hfov:.1f} deg (cost curve: {fovs[i]:.0f} best of 30..150); rotation residual median "
          f"{np.median(res[ok]):.3f} quarter px; {ok.sum()} of {len(res)} pairs kept, {moving.sum()} with a turn")
    taus = np.arange(0.0, 200.01, 1.0)
    lags = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    score, tau, lag, C, r2 = mouse_fit(W, ok, qpc, mouse, freq, taus, lags)
    # refine tau to 0.25 ms around the best
    score, tau, lag, C, r2 = mouse_fit(W, ok, qpc, mouse, freq, np.arange(max(0.0, tau - 2), tau + 2.01, 0.25),
                                       [max(0.0, lag + d) for d in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5)])
    Ffull = F * screen[0] / w
    print(f"mouse -> camera: tau {tau:.2f} ms, lag {lag:.1f} ms; R^2 pitch (x axis) {r2[0]:.4f}, yaw (y axis) {r2[1]:.4f}")
    print(f"  a count in x: w = ({C[0, 0]:.4e}, {C[1, 0]:.4e}, {C[2, 0]:.4e}) rad"
          f"  = {np.degrees(-C[1, 0]):.5f} deg yaw, {-C[1, 0] * Ffull:.3f} px at the centre of the full frame")
    print(f"  a count in y: w = ({C[0, 1]:.4e}, {C[1, 1]:.4e}, {C[2, 1]:.4e}) rad"
          f"  = {np.degrees(C[0, 1]):.5f} deg pitch")
    t1, t2 = qpc[:-1].astype(np.float64), qpc[1:].astype(np.float64)
    d = tau * 1e-3 * freq
    M = lagged_counts(mouse, t2 - d, lag * 1e-3 * freq) - lagged_counts(mouse, t1 - d, lag * 1e-3 * freq)
    miss = np.linalg.norm((W - M @ C.T)[ok][:, :2], axis=1) * Ffull
    print(f"  per frame, the camera the mouse predicts misses the measured one by {np.median(miss):.2f} px median, "
          f"{np.percentile(miss, 90):.2f} p90 (full frame, at the centre)")
    lines = {
        "sim_hfov": f"{hfov:.2f}",
        "sim_cx": f"{C[0, 0]:.6e} {C[1, 0]:.6e} {C[2, 0]:.6e}",
        "sim_cy": f"{C[0, 1]:.6e} {C[1, 1]:.6e} {C[2, 1]:.6e}",
        "sim_tau_ms": f"{tau:.2f}",
        "sim_lag_ms": f"{lag:.1f}",
    }
    print("settings:")
    for k, v in lines.items():
        print(f"  {k}={v}")
    if a.write:
        path = Path(a.write)
        text = path.read_text().splitlines() if path.exists() else []
        seen = set()
        out = []
        for line in text:
            key = line.split("=", 1)[0].strip()
            if key in lines:
                out.append(f"{key}={lines[key]}")
                seen.add(key)
            else:
                out.append(line)
        out += [f"{k}={v}" for k, v in lines.items() if k not in seen]
        path.write_text("\n".join(out) + "\n", encoding="ascii")
        print(f"written into {path}")


if __name__ == "__main__":
    main()
