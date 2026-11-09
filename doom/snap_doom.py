"""snap_doom.py - a snapshot the cloud simulator layer took (Home in the game, or run_doom.ps1
-Snapshot): the game's frame and its depth buffer, as images, with the depth's spread by region and
the threshold that tells the weapon from the world (sim_weapon_depth).

  python -I doom/snap_doom.py <snapshot folder>

Writes into the folder: color.png, depth.png (nearer is brighter, log scale), and near-<t>.png for a
few thresholds: the pixels nearer than t in red over the frame. The weapon is at the bottom of the
screen and nearer than anything in the world there; the suggestion is the largest threshold that
leaves the upper part of the frame (world only) almost untouched while it takes in the weapon.
"""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def write_png(path, img):
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    kind = {1: 0, 3: 2, 4: 6}[ch]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, kind, 0, 0, 0))
                          + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_kv(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if p:
            out[p[0]] = p[1:]
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    snap = Path(sys.argv[1])
    info = read_kv(snap / "snapshot.txt")
    W, H = int(info["screen"][0]), int(info["screen"][1])
    bgra = int(info["bgra"][0])
    col = np.fromfile(snap / "color.raw", np.uint8)[: W * H * 4].reshape(H, W, 4)
    rgb = col[:, :, [2, 1, 0]] if bgra else col[:, :, :3]
    write_png(snap / "color.png", rgb)
    if int(info["depth_ok"][0]) == 0 or not (snap / "depth.raw").exists():
        print(f"{snap.name}: {W}x{H}, no depth in this snapshot")
        return
    RW, RH = int(info["depth"][0]), int(info["depth"][1])
    flags = int(info["depth_flags"][0])
    if flags & 4:
        d = np.fromfile(snap / "depth.raw", np.uint16)[: RW * RH].reshape(RH, RW).astype(np.float64) / 65535.0
        kind = "16-bit"
    elif flags & 1:
        d = (np.fromfile(snap / "depth.raw", np.uint32)[: RW * RH].reshape(RH, RW) & 0xFFFFFF).astype(np.float64) / 16777215.0
        kind = "24-bit"
    else:
        d = np.fromfile(snap / "depth.raw", np.float32)[: RW * RH].reshape(RH, RW).astype(np.float64)
        kind = "float"
    rev = bool(flags & 2)
    near = 1.0 - d if rev else d                  # small = near, whatever the convention
    print(f"{snap.name}: frame {W}x{H}, depth {RW}x{RH} {kind}, {'reversed (1 near)' if rev else 'standard (0 near)'}, "
          f"format {info.get('depth_format', ['?'])[0]}, store op {info.get('depth_store_op', ['?'])[0]}, "
          f"layout {info.get('depth_layout', ['?'])[0]}")
    # nearer is brighter, log scale over the used range
    key = np.clip(near, 1e-7, 1.0)
    lo, hi = np.percentile(key, 0.5), np.percentile(key, 99.5)
    vis = 1.0 - (np.log(key) - np.log(lo)) / max(np.log(hi) - np.log(lo), 1e-9)
    write_png(snap / "depth.png", (np.clip(vis, 0, 1) * 255).astype(np.uint8))
    regions = {
        "top third": (slice(0, RH // 3), slice(0, RW)),
        "middle third": (slice(RH // 3, 2 * RH // 3), slice(0, RW)),
        "bottom third, left": (slice(2 * RH // 3, RH), slice(0, RW // 3)),
        "bottom third, centre": (slice(2 * RH // 3, RH), slice(RW // 3, 2 * RW // 3)),
        "bottom third, right": (slice(2 * RH // 3, RH), slice(2 * RW // 3, RW)),
    }
    print("depth by region (raw value; small = near): p1 / p10 / p50 / p90")
    for name, (ys, xs) in regions.items():
        v = d[ys, xs].ravel()
        print(f"  {name:22s} {np.percentile(v, 1):.6f} / {np.percentile(v, 10):.6f} / {np.percentile(v, 50):.6f} / "
              f"{np.percentile(v, 90):.6f}")
    # thresholds: the share of the top third (world) and of the bottom third (weapon there) nearer than t
    top = near[: RH // 3].ravel()
    bottom = near[2 * RH // 3:].ravel()
    qs = np.unique(np.clip(np.percentile(bottom, [1, 2, 5, 10, 15, 20, 30, 40, 50]), 1e-7, 1.0))
    cands = sorted(set(np.concatenate([qs, np.sqrt(qs[:-1] * qs[1:])]).tolist()))   # and the points between
    rows = []
    for q in cands:
        rows.append((q, float(np.mean(top < q)), float(np.mean(bottom < q))))
    print("threshold (as nearness)  top third nearer  bottom third nearer")
    for q, ft, fb in rows:
        print(f"  {q:.6f}                {ft * 100:6.2f} %          {fb * 100:6.2f} %")
    # the weapon is the near end of the bottom third, cut off from the world by the widest gap in
    # distance: with depth d = 1 - n/z (infinite far plane), z is n/(1 - d), so the gap is read on
    # log z = -log(1 - d). The world's floor at the bottom is nearer than anything at the top, so
    # "the most of the bottom while the top stays clean" would take the floor too (DOOM, 2026-11-08).
    near_b = np.sort(bottom)
    near_b = near_b[: max(2, int(0.6 * near_b.size))]                 # the nearer 60 %: the weapon and some floor
    sub = near_b[:: max(1, near_b.size // 4000)]
    logz = -np.log(np.clip(1.0 - sub, 1e-9, 1.0))
    gaps = np.diff(logz)
    pick = None
    if gaps.size:
        g = int(np.argmax(gaps))
        if gaps[g] > 0.7:                                                  # a factor of 2 in distance
            zt = np.sqrt(np.exp(logz[g]) * np.exp(logz[g + 1]))            # halfway in distance
            t_near = 1.0 - 1.0 / zt
            share_top = float(np.mean(top < t_near))
            share_bottom = float(np.mean(bottom < t_near))
            pick = (t_near, share_top, share_bottom)
            print(f"widest gap in the bottom third's near end: {sub[g]:.6f} -> {sub[g + 1]:.6f} "
                  f"(x{np.exp(gaps[g]):.1f} in distance)")
    if pick:
        t_raw = 1.0 - pick[0] if rev else pick[0]
        print(f"suggested: sim_weapon_depth={t_raw:.6f} (takes {pick[2] * 100:.1f} % of the bottom third, "
              f"{pick[1] * 100:.2f} % of the top)")
    else:
        print("no clear gap: the weapon is not set apart by depth here (or no weapon on screen)")
    for q, ft, fb in rows:
        m = near < q
        big = np.repeat(np.repeat(m, max(1, H // RH), 0), max(1, W // RW), 1)[:H, :W] if (RW, RH) != (W, H) else m
        over = rgb.copy()
        over[big] = (0.5 * over[big] + [127, 0, 0]).astype(np.uint8)
        write_png(snap / f"near-{q:.6f}.png", over)
    print(f"images: {snap / 'color.png'}, depth.png, near-*.png")


if __name__ == "__main__":
    main()
