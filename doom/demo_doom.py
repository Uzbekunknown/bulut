"""demo_doom.py - a video of the cloud simulator in DOOM: the frame a cloud player has
today (`latency` ms old), the same old frame turned by the mouse (the layer's warp), and the frame
with no delay, from a recording the layer made:
  - a demo recording (Insert in the game: demo-*, half-size colour and nearness), or
  - a calibration recording (End: rec-*, quarter-size luma, no depth; the weapon's mask then comes
    from a snapshot's depth, --weapon-from <snap folder>, and stays where it is on the screen).

  python -I doom/demo_doom.py <recording> --settings doom/layer/build/cloud-sim-vk.txt
         [--latency 100] [--out doom/demo_doom_100ms.mp4] [--weapon-from <snapshot>]
         [--today "7.5 / 65"] [--ours "0.5 / 2.6"] [--seconds 12]

All three panels go through the same steps as the layer's screen: the game's crosshair cut out and
the layer's drawn, the HUD rectangles and the weapon kept where they are on the screen, holes
filled from the farthest surface around; only the turn differs. The video runs at 60 frames a
second on the recording's own clock, then the same stretch again at half speed.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_doom as F  # noqa: E402
from eval_doom import rodrigues  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"  # FFMPEG, else the PATH
FONT = "C\\:/Windows/Fonts/arial.ttf"
BOLD = "C\\:/Windows/Fonts/arialbd.ttf"


def ffpath(p):
    return str(p).replace("\\", "/").replace(":", "\\:")


def read_settings(path):
    s = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        s[k.strip()] = v.strip()
    nums = lambda k, d: [float(x) for x in s.get(k, d).split()]  # noqa: E731
    out = {"hfov": float(s.get("sim_hfov", 90)), "cx": np.array(nums("sim_cx", "0 -5e-4 0")),
           "cy": np.array(nums("sim_cy", "5e-4 0 0")), "tau": float(s.get("sim_tau_ms", 30)),
           "lag": float(s.get("sim_lag_ms", 0)), "weapon": float(s.get("sim_weapon_depth", 0)),
           "hole": float(s.get("sim_hole", 0.02)), "cross": float(s.get("sim_cross", 0.012)),
           "keep": float(s.get("sim_keep", 0.92)), "locks": [],
           "history": int(s.get("sim_history", "1") or 0), "slots": int(s.get("sim_slots", "24") or 24),
           "move": int(s.get("sim_move", "0") or 0), "move_speed": float(s.get("sim_move_speed", 0) or 0),
           "move_tau": float(s.get("sim_move_tau_ms", -1) or -1), "move_lag": float(s.get("sim_move_lag_ms", 0) or 0)}
    if out["move_tau"] < 0:
        out["move_tau"] = out["tau"]
    for i in range(1, 5):
        r = nums(f"sim_lock{i}", "0 0 0 0")
        if len(r) == 4 and r[2] > r[0] and r[3] > r[1]:
            out["locks"].append(r)
    return out


def load(rec, weapon_from, weapon_t):
    """Frames (n, h, w, 3) uint8, nearness (n, h, w) float 0 near .. 1 far, present qpc, mouse, freq."""
    info = F.read_kv(rec / "info.txt")
    freq = float(info["qpc_frequency"][0])
    rows = [l.split() for l in (rec / "times.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    qpc = np.array([int(r[1]) for r in rows], np.int64)
    global KEYS
    KEYS = np.array([int(r[3]) if len(r) > 3 else 0 for r in rows], np.int64)
    ev = [l.split() for l in (rec / "input.txt").read_text().splitlines() if l.startswith("m ")]
    mouse = np.array([[int(e[1]), int(e[2]), int(e[3])] for e in ev], np.int64).reshape(-1, 3)
    mouse = mouse[np.argsort(mouse[:, 0], kind="stable")]
    if (rec / "color.half").exists():
        w, h = int(info["half_width"][0]), int(info["half_height"][0])
        n = min(len(qpc), (rec / "color.half").stat().st_size // (w * h * 4))
        col = np.fromfile(rec / "color.half", np.uint8)[: n * w * h * 4].reshape(n, h, w, 4)
        rgb = col[..., [2, 1, 0]] if int(info.get("bgra", ["1"])[0]) else col[..., :3]
        near = np.fromfile(rec / "near.half", np.uint16)[: n * w * h].reshape(n, h, w)
        return rgb, near, qpc[:n], mouse, freq, "colour"
    w, h, rb = int(info["luma_width"][0]), int(info["luma_height"][0]), int(info["luma_row_bytes"][0])
    raw = np.fromfile(rec / "frames.y8", np.uint8)
    n = min(len(qpc), raw.size // (rb * h))
    y = raw[: n * rb * h].reshape(n, h, rb)[:, :, :w]
    rgb = np.repeat(y[..., None], 3, -1)
    near = np.full((h, w), 65535, np.uint16)
    if weapon_from:
        si = F.read_kv(Path(weapon_from) / "snapshot.txt")
        RW, RH = int(si["depth"][0]), int(si["depth"][1])
        fl = int(si["depth_flags"][0])
        if fl & 1:
            d = (np.fromfile(Path(weapon_from) / "depth.raw", np.uint32)[: RW * RH].reshape(RH, RW) & 0xFFFFFF) / 16777215.0
        elif fl & 4:
            d = np.fromfile(Path(weapon_from) / "depth.raw", np.uint16)[: RW * RH].reshape(RH, RW) / 65535.0
        else:
            d = np.fromfile(Path(weapon_from) / "depth.raw", np.float32)[: RW * RH].reshape(RH, RW).astype(np.float64)
        nr = 1.0 - d if fl & 2 else d
        fy, fx = RH // h, RW // w
        nr = nr[: h * fy, : w * fx].reshape(h, fy, w, fx).min((1, 3))      # the nearest of each block
        near = (np.clip(nr, 0, 1) * 65535).astype(np.uint16)
    return rgb, np.broadcast_to(near, (n, h, w)), qpc[:n], mouse, freq, "luma"


def push_pull_fill(img, key, valid):
    """Holes (not valid) from the farthest surface around: a pyramid whose 2x2 keeps the valid
    pixel with the largest key (farthest), then each level's holes from the coarser one."""
    levels = [(img, key, valid)]
    while min(levels[-1][0].shape[:2]) > 4:
        im, k, v = levels[-1]
        h, w = k.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pad = lambda a, fill: np.pad(a, ((0, 2 * h2 - h), (0, 2 * w2 - w)) + ((0, 0),) * (a.ndim - 2), constant_values=fill)  # noqa: E731
        kk = np.where(pad(v, False), pad(k, -1.0), -1.0)
        imp = pad(im, 0.0)
        blocks = kk.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)
        pick = np.argmax(blocks, -1)
        cimg = imp.reshape(h2, 2, w2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h2, w2, 4, 3)
        nim = np.take_along_axis(cimg, pick[..., None, None], 2)[:, :, 0]
        nk = np.take_along_axis(blocks, pick[..., None], 2)[..., 0]
        levels.append((nim, nk, nk >= 0))
    im, k, v = levels[-1]
    if not v.all() and v.any():
        idx = np.argwhere(v)
        for y, x in np.argwhere(~v):
            j = np.argmin(np.sum((idx - [y, x]) ** 2, 1))
            im[y, x] = im[tuple(idx[j])]
        v = np.ones_like(v)
    up = im
    for im, k, v in reversed(levels[:-1]):
        h, w = k.shape
        hc, wc = up.shape[:2]
        yy = np.clip((np.arange(h) + 0.5) * hc / h - 0.5, 0, hc - 1)
        xx = np.clip((np.arange(w) + 0.5) * wc / w - 0.5, 0, wc - 1)
        y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
        y1, x1 = np.minimum(y0 + 1, hc - 1), np.minimum(x0 + 1, wc - 1)
        fy, fx = (yy - y0)[:, None, None], (xx - x0)[None, :, None]
        coarse = (up[y0][:, x0] * (1 - fy) * (1 - fx) + up[y0][:, x1] * (1 - fy) * fx + up[y1][:, x0] * fy * (1 - fx)
                  + up[y1][:, x1] * fy * fx)
        up = np.where(v[..., None], im, coarse)
    return up


KEYS = None


def splat(img, near, R, Fq, T, stay, hole):
    """The layer's splat path (sim_move): each old pixel to where it is now, P' = R (P - T), its
    distance z = 1 / (1 - d) in near-plane units, each over 2x2, the nearest winning; the weapon
    and the HUD stay. Returns the picture and its keys (distance; -1 a hole) for the fill."""
    h, w = near.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 1.0 / np.maximum(1.0 - near.astype(np.float64) / 65535.0, 1e-6)
    Pv = np.stack([(xx + 0.5 - w / 2) / Fq * z, (yy + 0.5 - h / 2) / Fq * z, z], -1) - T
    Q = Pv @ R.T
    ok = (Q[..., 2] > 1e-3) & ~stay & ~hole
    tx = Q[..., 0] / np.where(ok, Q[..., 2], 1.0) * Fq + w / 2
    ty = Q[..., 1] / np.where(ok, Q[..., 2], 1.0) * Fq + h / 2
    x0, y0 = np.floor(tx - 0.5).astype(np.int64), np.floor(ty - 0.5).astype(np.int64)
    src = np.arange(h * w).reshape(h, w)
    idx_l, key_l, src_l = [np.flatnonzero(stay)], [np.zeros(int(stay.sum()))], [np.flatnonzero(stay)]
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            v = ok & (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
            idx_l.append((yi * w + xi)[v])
            key_l.append(Q[..., 2][v])
            src_l.append(src[v])
    idx, key, sp = np.concatenate(idx_l), np.concatenate(key_l), np.concatenate(src_l)
    order = np.lexsort((key, idx))
    idx, key, sp = idx[order], key[order], sp[order]
    first = np.ones(len(idx), bool)
    first[1:] = idx[1:] != idx[:-1]
    out = np.zeros((h * w, 3))
    kk = np.full(h * w, -1.0)
    out[idx[first]] = img.reshape(-1, 3)[sp[first]]
    kk[idx[first]] = key[first]
    return out.reshape(h, w, 3), kk.reshape(h, w)


def bwarp(img, near, R, Fq, bad):
    """Each screen pixel from where it was in this frame before the turn R (P = R^T Q), bilinear over
    the neighbours that are world (not `bad`). Returns the picture, where it is valid, and the farthest
    nearness among the neighbours taken (the fill's key)."""
    h, w = near.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    Q = np.stack([(xx + 0.5 - w / 2) / Fq, (yy + 0.5 - h / 2) / Fq, np.ones_like(xx)], -1)
    S = Q @ R                                                # R^T applied to each ray
    sx = S[..., 0] / S[..., 2] * Fq + w / 2 - 0.5
    sy = S[..., 1] / S[..., 2] * Fq + h / 2 - 0.5
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    inside = (sx > -0.5) & (sy > -0.5) & (sx < w - 0.5) & (sy < h - 0.5) & (S[..., 2] > 1e-3)
    acc = np.zeros((h, w, 3))
    wsum = np.zeros((h, w))
    kmax = np.zeros((h, w))
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)
            wt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            okk = inside & ~bad[yi, xi] & (wt > 0)
            acc += np.where(okk[..., None], img[yi, xi] * wt[..., None], 0)
            wsum += np.where(okk, wt, 0)
            kmax = np.maximum(kmax, np.where(okk, near[yi, xi], 0))
    valid = wsum >= 0.5
    return np.where(valid[..., None], acc / np.maximum(wsum, 1e-9)[..., None], 0), valid, kmax


def screen(frame, near, R, P, lockmask, hole, weapon_key, T=None, history=()):
    """The layer's screen for one panel: frame (h, w, 3), nearness (h, w), rotation R, the move T
    (near-plane units, old frame's axes) when sim_move is on, and older frames (frame, nearness, R'
    to now) for what this one had turned away from (sim_history: the first that has a pixel fills it)."""
    h, w = near.shape
    Fq = (w / 2) / np.tan(np.radians(P["hfov"]) / 2)
    img = frame.astype(np.float64)
    weapon = near < weapon_key
    stay = (lockmask | weapon) & ~hole                       # on the screen, not in the world
    if T is not None and np.any(T):
        out, kk = splat(img, near, R, Fq, T, stay, hole)
        valid = kk >= 0
        kk = np.where(valid & ~stay, kk / (1.0 + np.maximum(kk, 0.0)), np.where(stay, 0.0, -1.0))   # farther: larger, in (0, 1)
        return push_pull_fill(out, kk, valid)
    bad = lockmask | weapon | hole                            # what the world sample may not take
    out, valid, kmax = bwarp(img, near, R, Fq, bad)
    out = np.where(stay[..., None], img, out)
    key = np.where(stay, 0.0, np.where(valid, kmax, -1.0)).astype(np.float64)
    filled = valid | stay
    for hf, hn, hR in history:
        if filled.all():
            break
        hbad = lockmask | (hn < weapon_key) | hole
        ho, hv, hk = bwarp(hf.astype(np.float64), hn, hR, Fq, hbad)
        take = hv & ~filled
        out = np.where(take[..., None], ho, out)
        key = np.where(take, hk, key)
        filled = filled | take
    return push_pull_fill(out, key / 65535.0, filled)


def zoom(img, keep):
    h, w = img.shape[:2]
    yy = np.clip((np.arange(h) + 0.5 - h / 2) * keep + h / 2 - 0.5, 0, h - 1)
    xx = np.clip((np.arange(w) + 0.5 - w / 2) * keep + w / 2 - 0.5, 0, w - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (yy - y0)[:, None, None], (xx - x0)[None, :, None]
    return img[y0][:, x0] * (1 - fy) * (1 - fx) + img[y0][:, x1] * (1 - fy) * fx + img[y1][:, x0] * fy * (1 - fx) + img[y1][:, x1] * fy * fx


def crosshair(img, cross_px):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    arm = max(4, cross_px)
    th = max(1, arm // 8)
    gap = max(2, int(arm * 0.3))
    for colour, grow in ((0, 1), (255, 0)):
        t, a, g = th + grow, arm + grow, gap - grow
        img[cy - t:cy + t, cx - a:cx - g] = colour
        img[cy - t:cy + t, cx + g:cx + a] = colour
        img[cy - a:cy - g, cx - t:cx + t] = colour
        img[cy + g:cy + a, cx - t:cx + t] = colour
        img[cy - t:cy + t, cx - t:cx + t] = colour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--settings", default=str(HERE / "layer" / "build" / "cloud-sim-vk.txt"))
    ap.add_argument("--latency", type=float, default=100.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--weapon-from", default="")
    ap.add_argument("--today", default="", help="the miss today at this delay, for the caption (median / p90 px)")
    ap.add_argument("--ours", default="", help="the miss with the warp")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the recording")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality (lower = bigger file)")
    a = ap.parse_args()
    rec = Path(a.rec)
    P = read_settings(a.settings)
    frames, near, qpc, mouse, freq, kind = load(rec, a.weapon_from, P["weapon"])
    n, h, w = near.shape
    t_ms = (qpc - qpc[0]) / freq * 1e3
    print(f"{rec.name}: {n} {kind} frames {w}x{h} over {t_ms[-1] / 1e3:.1f} s; hfov {P['hfov']}, tau {P['tau']} ms, lag {P['lag']} ms; "
          f"delay {a.latency:.0f} ms")
    lockmask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in P["locks"]:
        lockmask[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))] = True
    yy, xx = np.mgrid[0:h, 0:w]
    hp = P["hole"] * h
    hole = (np.abs(xx + 0.5 - w / 2) < hp) & (np.abs(yy + 0.5 - h / 2) < hp)
    weapon_key = P["weapon"] * 65535 if P["weapon"] > 0 else -1
    cnt = F.Lagged(mouse, P["lag"] * 1e-3 * freq)
    tau = P["tau"] * 1e-3 * freq
    use_move = P["move"] and P["move_speed"] > 0 and kind == "colour" and KEYS is not None and np.any(KEYS & 15)
    if use_move:
        from fit_move import unit_path
        t1ms, upath = unit_path(qpc, KEYS[:n], freq, P["move_lag"] * 1e-3)
        mtau = P["move_tau"] * 1e-3 * freq
        print(f"the move too: speed {P['move_speed']} near-plane units a second, tau {P['move_tau']} ms, lag {P['move_lag']} ms")
    # the video's clock: 60 a second from `start` + latency
    t0 = max(a.latency + 20.0, a.start * 1e3)
    ticks = np.arange(t0, min(t_ms[-1], t0 + a.seconds * 1e3), 1000.0 / 60.0)
    pw, ph = 960, 540
    cw = int(round(w * P["keep"] / 2)) * 2
    cache = {}

    def panel(k, j, turn):
        key = (k, j, turn)
        if key in cache:
            return cache[key]
        T = None
        if turn:
            m = cnt(np.array([qpc[k] - tau], float)) - cnt(np.array([qpc[j] - tau], float))
            R = rodrigues(P["cx"] * m[0, 0] + P["cy"] * m[0, 1])
            if use_move:
                pa = [np.interp(qpc[j] - mtau, t1ms, upath[:, c]) for c in range(2)]
                pb = [np.interp(qpc[k] - mtau, t1ms, upath[:, c]) for c in range(2)]
                T = P["move_speed"] * np.array([pb[0] - pa[0], 0.0, pb[1] - pa[1]])
        else:
            R = np.eye(3)
        hist = []
        if turn and P["history"]:
            oldest = max(0, k - (P["slots"] - 1))
            for hidx in sorted({j - 4, j - 10, oldest}, reverse=True):
                if 0 <= hidx < j and hidx >= oldest:
                    mh = cnt(np.array([qpc[k] - tau], float)) - cnt(np.array([qpc[hidx] - tau], float))
                    hist.append((frames[hidx], near[hidx], rodrigues(P["cx"] * mh[0, 0] + P["cy"] * mh[0, 1])))
        img = screen(frames[j], near[j], R, P, lockmask, hole, weapon_key, T, hist)
        img = zoom(img, P["keep"])
        # to the panel's size
        ys = np.clip(((np.arange(ph) + 0.5) * h / ph - 0.5), 0, h - 1)
        xs = np.clip(((np.arange(pw) + 0.5) * w / pw - 0.5), 0, w - 1)
        y0_, x0_ = np.floor(ys).astype(int), np.floor(xs).astype(int)
        y1_, x1_ = np.minimum(y0_ + 1, h - 1), np.minimum(x0_ + 1, w - 1)
        fy, fx = (ys - y0_)[:, None, None], (xs - x0_)[None, :, None]
        big = img[y0_][:, x0_] * (1 - fy) * (1 - fx) + img[y0_][:, x1_] * (1 - fy) * fx + img[y1_][:, x0_] * fy * (1 - fx) + img[y1_][:, x1_] * fy * fx
        out = np.clip(big + 0.5, 0, 255).astype(np.uint8)
        crosshair(out, int(round(P["cross"] * ph * 1.0 / P["keep"])))
        if len(cache) > 400:
            cache.clear()
        cache[key] = out
        return out

    out = Path(a.out) if a.out else HERE / f"demo_doom_{a.latency:.0f}ms.mp4"
    work = out.parent
    texts = {
        "tl": f"Cloud today: the picture is {a.latency:.0f} ms old",
        "tr": "Compensated by the mouse: the same picture, turned",
        "bl": "No delay: what the player should see",
        "title": "DOOM (2016), cloud delay " + f"{a.latency:.0f} ms",
        "body": ("The server's frame comes late. The client knows the player's\n"
                 "mouse right now and turns the old frame to where the camera\n"
                 "already is. The weapon and the HUD stay in place (by the\n"
                 "game's own depth); the edges that open up are filled from\n"
                 "the background.\n\n"
                 + (f"Miss at the centre of a 1080p screen, median / p90:\n"
                    f"today {a.today} px, compensated {a.ours} px.\n\n" if a.today else "")
                 + ("A grey quarter-size calibration recording;\n"
                    "the weapon's mask comes from a depth snapshot." if kind == "luma" else
                    "A half-size recording: colour and depth from the game.")),
        "speed1": "real speed",
        "speed2": "half speed",
    }
    files = {}
    for key, text in texts.items():
        files[key] = work / f"demo_text_{key}.txt"
        files[key].write_text(text, encoding="utf-8", newline="\n")

    def label(key, x, y, size=26, font=BOLD, box=True, enable=None):
        opts = [f"fontfile='{font}'", f"textfile='{ffpath(files[key])}'", "expansion=none", f"fontsize={size}",
                "fontcolor=white", f"x={x}", f"y={y}", "line_spacing=8"]
        if box:
            opts += ["box=1", "boxcolor=black@0.6", "boxborderw=8"]
        if enable:
            opts.append(f"enable='{enable}'")
        return "drawtext=" + ":".join(opts)

    switch = len(ticks) / 60.0
    vf = ",".join([label("tl", 16, 14), label("tr", 976, 14), label("bl", 16, 554),
                   label("title", 990, 560, size=34, box=False), label("body", 990, 606, size=22, font=FONT, box=False),
                   label("speed1", 990, 1040, size=26, box=False, enable=f"lt(t,{switch:.3f})"),
                   label("speed2", 990, 1040, size=26, box=False, enable=f"gte(t,{switch:.3f})")])
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
           "-r", "60", "-i", "-", "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canvas = np.full((1080, 1920, 3), 22, np.uint8)
    plan = []
    for tq in ticks:
        k = int(np.searchsorted(t_ms, tq, side="right") - 1)
        j = int(np.searchsorted(t_ms, t_ms[k] - a.latency, side="right") - 1)
        plan.append((k, max(j, 0)))
    slow = [plan[i // 2] for i in range(2 * len(plan))]
    last = None
    for i, (k, j) in enumerate(plan + slow):
        if (k, j) != last:
            canvas[:540, :960] = panel(k, j, False)
            canvas[:540, 960:] = panel(k, j, True)
            canvas[540:, :960] = panel(k, k, False)
            canvas[540:, 960:] = 22
            canvas[:, 958:962] = 0
            canvas[538:542, :] = 0
            frame = canvas.tobytes()
            last = (k, j)
        proc.stdin.write(frame)
        if i % 120 == 0:
            print(f"  {i} of {len(plan) + len(slow)} video frames", flush=True)
    proc.stdin.close()
    if proc.wait():
        sys.exit(f"ffmpeg failed ({proc.returncode})")
    for f_ in files.values():
        f_.unlink(missing_ok=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {(len(plan) + len(slow)) / 60:.1f} s)")


if __name__ == "__main__":
    main()
