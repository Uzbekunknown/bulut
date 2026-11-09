"""fit_pad.py - the gamepad's right stick -> camera model for the cloud simulator layer, and
how far off the screen is with and without the stick warp at any delay, from a recording the layer
made (End in the game): frames.y8, times.txt, input.txt (its "p" lines are the pad), info.txt.

  python -I doom/fit_pad.py <recording> [--settings <cloud-sim-vk.txt>] [--write] [--hfov <deg>]
                                          [--radial | --axial] [--truth "<yaw table>|<pitch table>"]

1. The camera's rotation per frame pair: fit_doom's patches (the screen-locked ones left out), at the
   field of view given (--hfov), else the settings' sim_hfov, else fitted.
2. The stick: the camera turns at rate(m) (sx, sy) / m, m the right stick's distance from the centre
   (radial; --axial: each axis by itself), rate a table at m = 0, 0.1 .. 1, linear between, one for
   yaw and one for pitch (the table's value at 0 is 0). The rate reaches the camera through a
   first-order lag, tau after the stick. The tables enter linearly, so for each tau and lag they
   come from least squares (with a little smoothing, so knots the stick never reached follow their
   neighbours); tau and lag are searched. Without --radial / --axial both are tried.
3. Cross-validated, like eval_doom.py: the model fitted on one half of the recording is judged on
   the other; every 5 ms, the camera's turn over the last `delay` ms (the miss a cloud player sees
   today) against what is left after the stick's predicted turn (the miss with the warp), at the
   centre and at 75 % of the half-width, in px of a 1080p screen.
It prints the fit, the sweep and the settings lines (sim_pad=1, sim_pad_radial, sim_pad_tau_ms,
sim_pad_lag_ms, sim_pad_yaw, sim_pad_pitch); --write puts them into the --settings file (a running
layer takes them within half a second).

For a recording in the game: stand still, aim assist off, and turn with the right stick alone for
10-15 s: slowly, fast, left, right, up, down and in circles, now and then letting go.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_doom as F  # noqa: E402
import eval_doom as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

KNOTS = 11                                   # the tables: m = 0, 0.1 .. 1


def read_pads(rec):
    """The pad's samples, sorted: qpc, right stick x and y (-1..1, y up), as an (n, 3) array; a
    "no pad" sample reads 0."""
    rows = []
    for line in (rec / "input.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 10 and p[0] == "p":
            gone = int(p[9]) < 0
            rows.append((int(p[1]), 0.0 if gone else int(p[4]) / 32767.0, 0.0 if gone else int(p[5]) / 32767.0))
    a = np.array(rows, np.float64).reshape(-1, 3)
    a = a[np.argsort(a[:, 0], kind="stable")]
    a[:, 1:] = np.clip(a[:, 1:], -1.0, 1.0)
    return a


def basis(pads, radial):
    """Per sample, the rate each table knot k = 1..10 adds per unit of its value: yaw (n, 10), pitch (n, 10)."""
    sx, sy = pads[:, 1], pads[:, 2]
    knots = np.arange(1, KNOTS)

    def hat(m):
        return np.clip(1.0 - np.abs(m[:, None] * (KNOTS - 1) - knots[None, :]), 0.0, None)

    if radial:
        m = np.hypot(sx, sy)
        safe = np.where(m > 1e-9, m, 1.0)
        h = hat(np.minimum(m, 1.0))
        ux, uy = np.where(m > 1e-9, sx / safe, 0.0), np.where(m > 1e-9, sy / safe, 0.0)
        return h * ux[:, None], h * uy[:, None]
    return hat(np.abs(sx)) * np.sign(sx)[:, None], hat(np.abs(sy)) * np.sign(sy)[:, None]


class Turned:
    """Theta(t): how far the rates (held from each sample to the next, 0 before the first) have
    turned the camera by t through a first-order lag L (ticks): S(t) - L y(t), S the integral of the
    rate and y the lagged rate. Many columns (the knots) at once; many t cost one lookup each."""

    def __init__(self, t, rates, lag_ticks):
        self.t = t
        self.r = rates
        self.L = lag_ticks
        n = len(t)
        dt = np.diff(t)
        self.S = np.zeros((n, rates.shape[1]))
        self.S[1:] = np.cumsum(rates[:-1] * dt[:, None], 0)
        self.y = np.zeros((n, rates.shape[1]))
        if lag_ticks > 0:
            a = np.exp(-dt / lag_ticks)
            y = self.y
            for i in range(n - 1):
                y[i + 1] = rates[i] + (y[i] - rates[i]) * a[i]

    def __call__(self, q):
        j = np.searchsorted(self.t, q, side="right") - 1
        before = j < 0
        jc = np.maximum(j, 0)
        d = (q - self.t[jc])[:, None]
        S = self.S[jc] + self.r[jc] * d
        if self.L > 0:
            y = self.r[jc] + (self.y[jc] - self.r[jc]) * np.exp(-d / self.L)
            out = S - self.L * y
        else:
            out = S
        out[before] = 0.0
        return out


def smooth_rows(lam):
    """Second differences of the table (value 0 at m = 0 fixed), as rows over knots 1..10."""
    D = np.zeros((KNOTS - 2, KNOTS - 1))
    for r in range(KNOTS - 2):                # knots r, r+1, r+2 of 0..10; column c is knot c+1
        for c, v in ((r - 1, 1.0), (r, -2.0), (r + 1, 1.0)):
            if c >= 0:
                D[r, c] = v
    return lam * D


SMOOTH = 1e-3                                # the smoothing's weight, against the data's (--smooth)


def solve(X, y, lam_rel=None):
    """Least squares with the smoothing rows, scaled to the data."""
    lam_rel = SMOOTH if lam_rel is None else lam_rel
    s = np.sqrt(lam_rel * max(np.mean(np.sum(X * X, 0)), 1e-30))
    A = np.vstack([X, smooth_rows(s)])
    b = np.concatenate([y, np.zeros(KNOTS - 2)])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


def fit(W, ok, qpc, pads, freq, radial, taus_ms, lags_ms, prev=None):
    """The tables, tau and lag that best explain the yaw (W[:, 1]) and the pitch (W[:, 0]) of the
    kept pairs: (sse, tau, lag, yaw table, pitch table, R^2 yaw, R^2 pitch)."""
    by, bp = basis(pads, radial)
    rates = np.concatenate([by, bp], 1)
    t1, t2 = qpc[:-1].astype(np.float64), qpc[1:].astype(np.float64)
    Yy, Yp = W[ok, 1], W[ok, 0]
    sst_y, sst_p = np.sum((Yy - Yy.mean()) ** 2), np.sum((Yp - Yp.mean()) ** 2)
    best = prev
    K = KNOTS - 1
    for lag in lags_ms:
        th = Turned(pads[:, 0], rates, lag * 1e-3 * freq)
        for tau in taus_ms:
            dq = tau * 1e-3 * freq
            X = (th(t2 - dq) - th(t1 - dq))[ok] / freq          # rate x seconds: the tables come out in rad/s
            cy, cp = solve(X[:, :K], Yy), solve(X[:, K:], Yp)
            sy, sp = np.sum((Yy - X[:, :K] @ cy) ** 2), np.sum((Yp - X[:, K:] @ cp) ** 2)
            if best is None or sy + sp < best[0]:
                best = (sy + sp, tau, lag, np.concatenate([[0.0], cy]), np.concatenate([[0.0], cp]),
                        1 - sy / max(sst_y, 1e-30), 1 - sp / max(sst_p, 1e-30))
    return best


def full_fit(W, ok, qpc, pads, freq, radial):
    taus = np.arange(0.0, 150.01, 2.0)
    lags = [0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    b = fit(W, ok, qpc, pads, freq, radial, taus, lags)
    tau, lag = b[1], b[2]
    b = fit(W, ok, qpc, pads, freq, radial, np.arange(max(0.0, tau - 2.5), tau + 2.51, 0.25),
            sorted({max(0.0, lag + d) for d in (-3, -2, -1, 0, 1, 2, 3)}), None)
    return b


def rotations(rec, hfov_given, max_shift):
    frames, qpc, mouse, freq, screen, info = F.read_recording(rec)
    n, h, w = frames.shape
    p, cen = F.patch_grid(w, h, 1.0, 9, 5)
    sh = F.pair_shifts(frames, p, cen)
    locked, _ = F.screen_locked(sh)
    cen, sh = cen[~locked], sh[:, ~locked]
    fast = np.median(np.hypot(sh[:, :, 0], sh[:, :, 1]), axis=1) > max_shift
    cxy = (w / 2.0, h / 2.0)
    if hfov_given:
        hfov = hfov_given
    else:
        fovs = np.arange(30.0, 150.01, 1.0)
        costs = [F.rotation_fit(sh[~fast], cen, (w / 2) / np.tan(np.radians(v) / 2), cxy, 2)[1] for v in fovs]
        i = int(np.argmin(costs))
        fine = np.arange(max(25.0, fovs[i] - 1.5), fovs[i] + 1.51, 0.1)
        hfov = float(fine[int(np.argmin([F.rotation_fit(sh[~fast], cen, (w / 2) / np.tan(np.radians(v) / 2), cxy)[1]
                                         for v in fine]))])
    Fq = (w / 2) / np.tan(np.radians(hfov) / 2)
    W, _, res = F.rotation_fit(sh, cen, Fq, cxy)
    good = np.isfinite(res) & (res < 1.0) & ~fast
    return W, good, hfov, qpc, freq, screen, n, w, int(locked.sum()), int(fast.sum())


def sweep(W, good, qpc, pads, freq, hfov, radial, models, half, delays):
    """eval_doom's sweep with the stick's prediction, each half judged by the other half's model."""
    t_ms = (qpc - qpc[0]) / freq * 1e3
    phi = np.concatenate([np.zeros((1, 3)), np.cumsum(W, 0)])
    f1080 = 960.0 / np.tan(np.radians(hfov) / 2)
    u_edge = 0.75 * np.tan(np.radians(hfov) / 2)
    by, bp = basis(pads, radial)
    rates = np.concatenate([by, bp], 1)
    K = KNOTS - 1

    def path(q):
        return np.stack([np.interp(q, t_ms, phi[:, c]) for c in range(3)], -1)

    def pair_of(q):
        return np.clip(np.searchsorted(t_ms, q, side="right") - 1, 0, len(W) - 1)

    rows = {}
    for d in delays:
        q = np.arange(t_ms[0] + d + 5.0, t_ms[-1], 5.0)
        k1, k0 = pair_of(q), pair_of(q - d)
        okspan = np.array([good[a:b + 1].all() for a, b in zip(k0, k1)])
        turn = path(q) - path(q - d)
        miss_none = f1080 * np.hypot(turn[:, 0], turn[:, 1])
        edge_none = E.edge_miss(turn, f1080, u_edge)
        miss_pred = np.full(len(q), np.nan)
        edge_pred = np.full(len(q), np.nan)
        for name, part in (("first", half), ("second", ~half)):
            _, tau, lag, ty, tp, _, _ = models[name]
            other = ~part[k1] if name == "first" else part[k1]
            th = Turned(pads[:, 0], rates, lag * 1e-3 * freq)
            tq = qpc[0] + q * 1e-3 * freq
            dq = tau * 1e-3 * freq
            X = (th(tq - dq) - th(tq - d * 1e-3 * freq - dq)) / freq
            pred = np.zeros_like(turn)
            pred[:, 1] = X[:, :K] @ ty[1:]
            pred[:, 0] = X[:, K:] @ tp[1:]
            e = turn - pred
            sel = okspan & other
            miss_pred[sel] = f1080 * np.hypot(e[sel, 0], e[sel, 1])
            edge_pred[sel] = E.edge_miss(e[sel], f1080, u_edge)
        sel = okspan & np.isfinite(miss_pred)
        if sel.sum() < 5:
            rows[d] = None
            continue
        rows[d] = (np.median(miss_none[sel]), np.percentile(miss_none[sel], 90), np.median(miss_pred[sel]),
                   np.percentile(miss_pred[sel], 90), int(sel.sum()), np.percentile(edge_none[sel], 90),
                   np.percentile(edge_pred[sel], 90))
    return rows


def settings_hfov(path):
    if not path or not Path(path).exists():
        return 0.0
    for line in Path(path).read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "sim_hfov":
            try:
                return float(v.split()[0])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def table_text(t):
    return " ".join(f"{v:.4f}" for v in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--settings", help="the layer's settings: sim_hfov is read from it, and --write writes into it")
    ap.add_argument("--write", action="store_true", help="put the fit into --settings (sim_pad=1 and the rest)")
    ap.add_argument("--hfov", type=float, default=0.0, help="the field of view (else the settings', else fitted)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--radial", action="store_true", help="the rate by the stick's distance from the centre only")
    g.add_argument("--axial", action="store_true", help="the rate by each axis by itself only")
    ap.add_argument("--max-shift", type=float, default=6.0)
    ap.add_argument("--delays", default="10,16,25,33,50,75,100,150")
    ap.add_argument("--truth", default="", help="a test's tables, 'yaw values|pitch values', to compare with")
    ap.add_argument("--smooth", type=float, default=SMOOTH, help="the tables' smoothing, against the data")
    a = ap.parse_args()
    globals()["SMOOTH"] = a.smooth
    rec = Path(a.rec)
    pads = read_pads(rec)
    hfov0 = a.hfov or settings_hfov(a.settings)
    W, good, hfov, qpc, freq, screen, n, w, nlocked, nfast = rotations(rec, hfov0, a.max_shift)
    dt = np.diff(qpc) / freq * 1e3
    moving = np.abs(pads[:, 1:]).max(1) > 0.25 if len(pads) else np.zeros(0, bool)
    print(f"{rec.name}: {n} frames, {np.median(dt):.1f} ms a frame; {len(pads)} pad samples "
          f"({moving.sum()} with the right stick past 0.25); field of view {hfov:.1f} "
          f"({'given' if hfov0 else 'fitted'}); {nlocked} screen-locked patches and {nfast} fast pairs left out; "
          f"{good.sum()} of {len(W)} pairs kept")
    if len(pads) < 20 or moving.sum() < 20:
        sys.exit("too little of the right stick in this recording: was the gamepad connected (the layer's log says "
                 "'gamepad connected'), and was the camera turned with the right stick during the recording?")
    shapes = [True] if a.radial else [False] if a.axial else [True, False]
    fits = {r: full_fit(W, good, qpc, pads, freq, r) for r in shapes}
    radial = min(fits, key=lambda r: fits[r][0])
    for r, b in fits.items():
        print(f"  {'radial' if r else 'per axis'}: tau {b[1]:.2f} ms, lag {b[2]:.1f} ms; R^2 yaw {b[5]:.4f}, pitch {b[6]:.4f}"
              f"{'  <- kept' if r == radial and len(fits) > 1 else ''}")
    sse, tau, lag, ty, tp, r2y, r2p = fits[radial]
    f1080 = 960.0 / np.tan(np.radians(hfov) / 2)
    print(f"stick -> camera ({'radial' if radial else 'per axis'}): tau {tau:.2f} ms, lag {lag:.1f} ms; "
          f"R^2 yaw {r2y:.4f}, pitch {r2p:.4f}")
    print("  m:          " + " ".join(f"{k / (KNOTS - 1):6.1f}" for k in range(KNOTS)))
    print("  yaw rad/s:  " + " ".join(f"{v:6.3f}" for v in ty) + f"   (full right: {np.degrees(-ty[-1]):.0f} deg/s to the right)")
    print("  pitch rad/s:" + " ".join(f"{v:6.3f}" for v in tp) + f"   (full up: {np.degrees(abs(tp[-1])):.0f} deg/s)")
    if a.truth:
        yt, pt = (np.array([float(v) for v in s.split()]) for s in a.truth.split("|"))
        print(f"  against the truth: yaw off by {np.abs(ty - yt).max():.4f} rad/s at most "
              f"({np.abs(ty - yt).max() / max(np.abs(yt).max(), 1e-9) * 100:.1f} % of its largest), "
              f"pitch by {np.abs(tp - pt).max():.4f} ({np.abs(tp - pt).max() / max(np.abs(pt).max(), 1e-9) * 100:.1f} %)")
    # the per-pair miss of the fitted model, at the centre of the full screen
    by, bp = basis(pads, radial)
    th = Turned(pads[:, 0], np.concatenate([by, bp], 1), lag * 1e-3 * freq)
    dq = tau * 1e-3 * freq
    X = (th(qpc[1:].astype(np.float64) - dq) - th(qpc[:-1].astype(np.float64) - dq)) / freq
    K = KNOTS - 1
    e = np.stack([W[:, 0] - X[:, K:] @ tp[1:], W[:, 1] - X[:, :K] @ ty[1:]], 1)[good]
    miss = np.hypot(e[:, 0], e[:, 1]) * f1080
    print(f"  per frame, the camera the stick predicts misses the measured one by {np.median(miss):.2f} px median, "
          f"{np.percentile(miss, 90):.2f} p90 (1080p, at the centre)")
    # cross-validation
    half = np.arange(len(W)) < len(W) // 2
    models = {name: full_fit(W, good & part, qpc, pads, freq, radial) for name, part in (("first", half), ("second", ~half))}
    print(f"  halves: first tau {models['first'][1]:.1f} lag {models['first'][2]:.0f} (R^2 yaw {models['first'][5]:.3f}), "
          f"second tau {models['second'][1]:.1f} lag {models['second'][2]:.0f} (R^2 yaw {models['second'][5]:.3f})")
    delays = [float(x) for x in a.delays.split(",")]
    rows = sweep(W, good, qpc, pads, freq, hfov, radial, models, half, delays)
    print("  cross-validated, px at 1080p; each half judged with the other half's model")
    print("  delay ms | centre, today: median / p90 | centre, stick warp: median / p90 | edge (75 %) p90: today / warp | spans")
    for d in delays:
        r = rows[d]
        if r is None:
            print(f"  {d:7.0f}  | too few spans")
            continue
        mn, pn, mp, pp, cnt, en, ep = r
        print(f"  {d:7.0f}  | {mn:7.1f} / {pn:6.1f}            | {mp:6.1f} / {pp:5.1f}                 | {en:6.1f} / {ep:5.1f}"
              f"               | {cnt}")
    lines = {
        "sim_pad": "1",
        "sim_pad_radial": "1" if radial else "0",
        "sim_pad_tau_ms": f"{tau:.2f}",
        "sim_pad_lag_ms": f"{lag:.1f}",
        "sim_pad_yaw": table_text(ty),
        "sim_pad_pitch": table_text(tp),
    }
    print(f"R^2 {min(r2y, r2p):.4f} (the lower of yaw and pitch)")
    print("settings:")
    for k, v in lines.items():
        print(f"  {k}={v}")
    if a.write:
        if not a.settings:
            sys.exit("--write needs --settings")
        path = Path(a.settings)
        text = path.read_text().splitlines() if path.exists() else []
        seen, out = set(), []
        for line in text:
            key = line.split("=", 1)[0].strip()
            if key in lines:
                out.append(f"{key}={lines[key]}")
                seen.add(key)
            else:
                out.append(line)
        missing = [k for k in lines if k not in seen]
        if missing:
            out.append("# the gamepad's right stick -> camera (fit_pad.py)")
            out += [f"{k}={lines[k]}" for k in missing]
        path.write_text("\n".join(out) + "\n", encoding="ascii")
        print(f"written into {path}")


if __name__ == "__main__":
    main()
