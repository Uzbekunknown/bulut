"""eval_doom.py - how far off is the screen in DOOM with and without the mouse warp, at any delay,
on the cloud simulator layer's recordings.

  python -I doom/eval_doom.py <recording> [<recording> ...] [--delays 10,16,25,33,50,75,100,150]

For each recording:
1. The camera's path: the per-pair rotations (fit_doom's patches, the screen-locked ones left
   out, at the fitted field of view) added up at the present times.
2. The mouse model (tau, lag, C) fitted on one half of the recording and used on the other
   (both ways), so no number here comes from fitting the frames it is judged on.
3. Every 5 ms: the camera's turn over the last `delay` ms (what a cloud player sees wrong today)
   and the turn the mouse predicts for that span; the miss at the centre of the 1080p screen is
   f * |turn - prediction| (yaw and pitch). Spans touching a pair that was too fast (blur) or
   badly fitted are left out.
4. A check on the pictures: the old frame turned by the prediction, the layer's own warp (each
   screen pixel looks up where it was: P = R^T Q), then registered against the frame the player
   should see; the residual shift of the world patches, against the same without the warp.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_doom as F  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def warp(img, R, Fq):
    """The layer's warp (sim.comp bwarp, without the locks): screen pixel q shows the old frame at
    the ray R^T Q; outside the old frame stays NaN."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    Q = np.stack([(xx + 0.5 - w / 2) / Fq, (yy + 0.5 - h / 2) / Fq, np.ones_like(xx)], -1)
    P = Q @ R                                   # rows: R^T applied to each Q
    sx = P[..., 0] / P[..., 2] * Fq + w / 2 - 0.5
    sy = P[..., 1] / P[..., 2] * Fq + h / 2 - 0.5
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1) & (P[..., 2] > 1e-3)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    v[~ok] = np.nan
    return v


def patch_residual(a, b, cen, p, win):
    """Median shift magnitude (quarter px) of b against a over the given patches (NaN-free ones)."""
    out = []
    for x, y in cen:
        x0, y0 = int(round(x - p / 2)), int(round(y - p / 2))
        pa, pb = a[y0:y0 + p, x0:x0 + p], b[y0:y0 + p, x0:x0 + p]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        dx, dy, peak = F.phase_shift(pa.astype(np.float32), pb.astype(np.float32), win)
        if peak < 0.1:
            continue
        out.append(np.hypot(dx, dy))
    return float(np.median(out)) if out else np.nan


def edge_miss(e, f, u):
    """The displacement at the screen points (+-u, 0) for a rotation error e (n x 3): dx = f e_y (1 + u^2),
    dy = f (-e_x + e_z u); the larger of the two sides."""
    out = []
    for s in (-u, u):
        dx = f * e[:, 1] * (1 + s * s)
        dy = f * (-e[:, 0] + e[:, 2] * s)
        out.append(np.hypot(dx, dy))
    return np.maximum(out[0], out[1])


def evaluate(rec, delays, max_shift, no_roll):
    frames, qpc, mouse, freq, screen, info = F.read_recording(rec)
    n, h, w = frames.shape
    p, cen = F.patch_grid(w, h, 1.0, 9, 5)
    sh = F.pair_shifts(frames, p, cen)
    locked, _ = F.screen_locked(sh)
    cen, sh = cen[~locked], sh[:, ~locked]
    fast = np.median(np.hypot(sh[:, :, 0], sh[:, :, 1]), axis=1) > max_shift
    cxy = (w / 2.0, h / 2.0)
    fovs = np.arange(30.0, 150.01, 1.0)
    costs = [F.rotation_fit(sh[~fast], cen, (w / 2) / np.tan(np.radians(v) / 2), cxy, 2)[1] for v in fovs]
    i = int(np.argmin(costs))
    fine = np.arange(max(25.0, fovs[i] - 1.5), fovs[i] + 1.51, 0.1)
    hfov = float(fine[int(np.argmin([F.rotation_fit(sh[~fast], cen, (w / 2) / np.tan(np.radians(v) / 2), cxy)[1] for v in fine]))])
    Fq = (w / 2) / np.tan(np.radians(hfov) / 2)
    W, _, res = F.rotation_fit(sh, cen, Fq, cxy)
    good = np.isfinite(res) & (res < 1.0) & ~fast
    t_ms = (qpc - qpc[0]) / freq * 1e3
    phi = np.concatenate([np.zeros((1, 3)), np.cumsum(W, 0)])
    f1080 = 960.0 / np.tan(np.radians(hfov) / 2)          # the focal of a 1080p screen, px
    taus = np.arange(0.0, 120.01, 0.5)
    lags = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    half = np.arange(len(W)) < len(W) // 2
    models = {}
    for name, part in (("first", half), ("second", ~half)):
        _, tau, lag, C, r2 = F.mouse_fit(W, good & part, qpc, mouse, freq, taus, lags)
        if no_roll:                                      # as the layer runs it: yaw and pitch alone
            C = C.copy()
            C[2, :] = 0.0
            C[0, 0] = C[1, 1] = 0.0
        models[name] = (tau, lag, C, r2)
    u_edge = 0.75 * np.tan(np.radians(hfov) / 2)
    print(f"{rec.name}: {n} frames, field of view {hfov:.1f}, {good.sum()} of {len(W)} pairs good; "
          f"models: first half tau {models['first'][0]:.1f} lag {models['first'][1]:.0f} (R^2 {models['first'][3][1]:.3f}), "
          f"second half tau {models['second'][0]:.1f} lag {models['second'][1]:.0f} (R^2 {models['second'][3][1]:.3f})")

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
        edge_none = edge_miss(turn, f1080, u_edge)
        miss_pred = np.full(len(q), np.nan)
        edge_pred = np.full(len(q), np.nan)
        for name, part in (("first", half), ("second", ~half)):
            tau, lag, C, _ = models[name]
            other = ~part[k1] if name == "first" else part[k1]          # judge on the other half only
            cnt = F.Lagged(mouse, lag * 1e-3 * freq)
            tq = qpc[0] + q * 1e-3 * freq
            dq = tau * 1e-3 * freq
            m = cnt(tq - dq) - cnt(tq - d * 1e-3 * freq - dq)
            pred = m @ C.T
            e = turn - pred
            sel = okspan & other
            miss_pred[sel] = f1080 * np.hypot(e[sel, 0], e[sel, 1])
            edge_pred[sel] = edge_miss(e[sel], f1080, u_edge)
        sel = okspan & np.isfinite(miss_pred)
        rows[d] = (np.median(miss_none[sel]), np.percentile(miss_none[sel], 90), np.median(miss_pred[sel]),
                   np.percentile(miss_pred[sel], 90), int(sel.sum()), np.percentile(edge_none[sel], 90),
                   np.percentile(edge_pred[sel], 90))
    # the pictures: frame j turned by the prediction against frame k
    tau, lag, C, _ = models["first"]
    cnt = F.Lagged(mouse, lag * 1e-3 * freq)
    win = F.hann2(p)
    pic = {}
    for d in (50.0, 100.0):
        r_none, r_warp = [], []
        for k in range(len(W) // 2 + 5, n, 7):                         # the second half, judged with the first's model
            j = int(np.searchsorted(t_ms, t_ms[k] - d, side="right") - 1)
            if j < 0 or not good[j:k].all():
                continue
            dq = tau * 1e-3 * freq
            m = cnt(np.array([qpc[k] - dq], float)) - cnt(np.array([qpc[j] - dq], float))
            R = rodrigues((m @ C.T)[0])
            warped = warp(frames[j].astype(np.float64), R, Fq)
            r_warp.append(patch_residual(warped, frames[k].astype(np.float64), cen, p, win))
            r_none.append(patch_residual(frames[j].astype(np.float64), frames[k].astype(np.float64), cen, p, win))
        pic[d] = (np.nanmedian(r_none) * screen[0] / w * 1080 / screen[1], np.nanmedian(r_warp) * screen[0] / w * 1080 / screen[1],
                  len(r_warp))
    return hfov, rows, pic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--delays", default="10,16,25,33,50,75,100,150")
    ap.add_argument("--max-shift", type=float, default=6.0)
    ap.add_argument("--with-roll", action="store_true", help="let the mouse model turn about all three axes (the layer does not)")
    a = ap.parse_args()
    delays = [float(x) for x in a.delays.split(",")]
    for r in a.recs:
        hfov, rows, pic = evaluate(Path(r), delays, a.max_shift, not a.with_roll)
        print(f"  model: {'yaw, pitch and roll' if a.with_roll else 'yaw and pitch only, as the layer'}; px at 1080p")
        print("  delay ms | centre, today: median / p90 | centre, warp: median / p90 | edge (75 %) p90: today / warp | spans")
        for d in delays:
            mn, pn, mp, pp, cnt, en, ep = rows[d]
            print(f"  {d:7.0f}  | {mn:7.1f} / {pn:6.1f}            | {mp:6.1f} / {pp:5.1f}           | {en:6.1f} / {ep:5.1f}"
                  f"               | {cnt}")
        for d, (none, warped, cnt) in pic.items():
            print(f"  picture check at {d:.0f} ms: world patches off by {none:.1f} px unwarped, {warped:.1f} px warped "
                  f"(1080p px, median over {cnt} frames)")


if __name__ == "__main__":
    main()
