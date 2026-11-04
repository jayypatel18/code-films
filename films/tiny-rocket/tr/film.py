"""Film framework: shot registry, deterministic frame rendering, ffmpeg streaming, contact sheets.

Every frame is rendered alone from (shot, frame) so stills and partial renders work.
Cels are drawn on twos or threes: the drawing index is frame // step, and a drawing is reused
between those frames (a tiny boil comes from the drawing seed).
"""

import os
import subprocess
import sys
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from . import raster as R

FPS = 24
W, H = 1920, 1080
W2, H2 = 3840, 2160
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f"{ROOT}/build/cache"


class Shot:
    def __init__(self, name, start, end, render, bg=None):
        self.name = name
        self.start = start      # seconds
        self.end = end
        self.render = render    # fn(shot, f_local, t_local) -> float32 (H2, W2, 3) frame at 2x
        self.bg = bg
        self.frames = int(round((end - start) * FPS))

    def frame_range(self):
        return int(round(self.start * FPS)), int(round(self.end * FPS))


class Film:
    def __init__(self):
        self.shots = []

    def add(self, shot):
        self.shots.append(shot)

    @property
    def total_frames(self):
        return int(round(self.shots[-1].end * FPS))

    def locate(self, f):
        for s in self.shots:
            a, b = s.frame_range()
            if a <= f < b:
                return s, f - a
        return self.shots[-1], f - self.shots[-1].frame_range()[0]


# ----------------------------------------------------------------------------
# camera and finishing (per frame)
# ----------------------------------------------------------------------------

def camera(img2x, zoom=1.0, cx=0.5, cy=0.5):
    """Crop the 2x frame with a zoom about (cx, cy) in 0..1 and resample to 1920x1080."""
    if abs(zoom - 1.0) < 1e-4:
        return R.downsample(np.ascontiguousarray(img2x, np.float32), 2)
    cw, ch = W2 / zoom, H2 / zoom
    x0 = int(round(np.clip(cx * W2 - cw / 2, 0, W2 - cw)))
    y0 = int(round(np.clip(cy * H2 - ch / 2, 0, H2 - ch)))
    crop = np.ascontiguousarray(img2x[y0:y0 + int(ch), x0:x0 + int(cw)], np.float32)
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        im = Image.fromarray(crop[..., c], mode="F")
        out[..., c] = np.asarray(im.resize((W, H), Image.BICUBIC), np.float32)
    return out


def finish1x(img, seed, grain=0.03, vig=0.3, bloom_th=0.72, bloom_sigma=18, bloom_str=0.3, fade=1.0, fade_to=(0, 0, 0)):
    """1x float frame to uint8 with bloom, vignette, fade, grain and dither."""
    if bloom_str > 0:
        lum = img.mean(axis=2)
        hi = np.clip((lum - bloom_th) / (1 - bloom_th), 0, 1)[..., None] * img
        b = np.stack([gaussian_filter(hi[..., c], bloom_sigma) for c in range(3)], axis=2)
        img = img + b * bloom_str
    if vig > 0:
        img = img * _vignette(vig)[..., None]
    if fade < 1.0:
        img = img * fade + np.asarray(fade_to, np.float32) * (1 - fade)
    img = np.clip(img, 0, 1).astype(np.float32)
    return R.grain_dither(np.ascontiguousarray(img), seed, grain)


_VIG = {}


def _vignette(strength):
    if strength not in _VIG:
        yy, xx = np.mgrid[0:H, 0:W]
        dx = (xx - W / 2) / (W / 2)
        dy = (yy - H / 2) / (H / 2)
        d = np.sqrt(dx * dx + dy * dy) / 1.2
        _VIG[strength] = (1 - strength * np.clip(d, 0, 1) ** 2.2).astype(np.float32)
    return _VIG[strength]


# ----------------------------------------------------------------------------
# timing helpers
# ----------------------------------------------------------------------------

def ease(t, kind="smooth"):
    t = float(np.clip(t, 0, 1))
    if kind == "smooth":
        return t * t * (3 - 2 * t)
    if kind == "in":
        return t * t
    if kind == "out":
        return 1 - (1 - t) ** 2
    if kind == "snap":  # anime snap: anticipation then fast
        return 0 if t < 0.25 else min(1, (t - 0.25) / 0.35) ** 0.7
    return t


def seg(t, a, b, kind="smooth"):
    """0 before a, 1 after b, eased between."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return ease((t - a) / (b - a), kind)


def hold_frames(f, step):
    """Drawing index for animation on twos/threes."""
    return f // step


def keys(t, kf, kind="smooth"):
    """Piecewise keyframes: kf = [(t0, v0), (t1, v1), ...]."""
    if t <= kf[0][0]:
        return kf[0][1]
    for (t0, v0), (t1, v1) in zip(kf[:-1], kf[1:]):
        if t0 <= t < t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0), kind)
    return kf[-1][1]


def overshoot(t, amount=0.15):
    """1 with a small overshoot and settle, for snappy moves (t in 0..1)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 1 + amount * np.sin(np.pi * t) * (1 - t) ** 0.5 if t > 0.5 else ease(t * 2, "in") * (1 + amount)


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------

def cache_bg(name, painter):
    path = f"{CACHE}/{name}.npy"
    if os.path.exists(path):
        return np.load(path)
    img = painter()
    np.save(path, img)
    return img


def encode(film, path, f0=0, f1=None, crf=14, log=print):
    f1 = film.total_frames if f1 is None else f1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(f0, f1):
        frame = render_frame(film, f)
        proc.stdin.write(frame.tobytes())
        if f % 24 == 0:
            log(f"frame {f}/{f1}")
    proc.stdin.close()
    proc.wait()


def render_frame(film, f):
    shot, fl = film.locate(f)
    return shot.render(shot, fl, fl / FPS)


def still(film, f, path):
    Image.fromarray(render_frame(film, f)).save(path)


def contact_sheet(film, shot, path, cols=6, every=None):
    a, b = shot.frame_range()
    n = b - a
    every = every or max(1, n // 24)
    fs = list(range(a, b, every))
    rows = (len(fs) + cols - 1) // cols
    tw, th = 480, 270
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    for i, f in enumerate(fs):
        im = Image.fromarray(render_frame(film, f)).resize((tw, th), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path)
