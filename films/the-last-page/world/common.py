"""Shared plumbing for the world department: plate cache, camera sampling,
parallax, and saving review images."""

import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(FILM))
for p in (REPO, FILM):
    if p not in sys.path:
        sys.path.insert(0, p)

import timeline as T  # noqa: E402

PLATES = os.path.join(FILM, "build", "plates")
REVIEW = os.path.join(FILM, "build", "review")
W, H = T.W, T.H

# Every plate is painted with this margin on all sides so camera moves and
# parallax never run off the painted area. Plate coordinate (x, y) lives at
# array index (y + MARGIN, x + MARGIN).
MARGIN = 96


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def plate_path(name, ext="npy"):
    os.makedirs(PLATES, exist_ok=True)
    return os.path.join(PLATES, f"{name}.{ext}")


def save_plate(name, canvas):
    """Cache a painted layer: float16 npy (used by render) and an 8 bit PNG
    preview (straight alpha) for inspection."""
    np.save(plate_path(name), canvas.astype(np.float16))
    a = canvas[..., 3:4]
    rgb = np.where(a > 1e-4, canvas[..., :3] / np.maximum(a, 1e-4), 0.0)
    rgba = np.concatenate([rgb, a], -1)
    Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(plate_path(name, "png"))


_CACHE = {}


def load_plate(name):
    if name not in _CACHE:
        p = plate_path(name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"plate {name} not painted yet: run the world paint script")
        _CACHE[name] = np.load(p).astype(np.float32)
    return _CACHE[name]


def have_plate(name):
    return os.path.exists(plate_path(name))


def parallax_cam(cut_id, f, factor, drift=(0.0, 0.0)):
    """Camera for a layer that moves `factor` times as much as the stage.
    drift adds a slow pan in plate px per second (clouds)."""
    f0, _ = T.cut_range(cut_id)
    z0, x0, y0 = T.camera(cut_id, f0)
    z, x, y = T.camera(cut_id, f)
    t = (f - f0) / T.FPS
    return (z0 * (z / z0) ** factor, x0 + (x - x0) * factor + drift[0] * t,
            y0 + (y - y0) * factor + drift[1] * t)


def sample(plate, cam, out_w=W, out_h=H):
    """Resample a plate (with MARGIN) through camera (zoom, cx, cy) to screen.
    Uses a proper reducing filter so a 0.83x view stays crisp without aliasing."""
    zoom, cx, cy = cam
    x0 = cx - (out_w / 2) / zoom + MARGIN
    y0 = cy - (out_h / 2) / zoom + MARGIN
    x1 = x0 + out_w / zoom
    y1 = y0 + out_h / zoom
    out = np.empty((out_h, out_w, plate.shape[2]), np.float32)
    for c in range(plate.shape[2]):
        im = Image.fromarray(np.ascontiguousarray(plate[..., c]), "F")
        out[..., c] = np.asarray(im.resize((out_w, out_h), Image.BICUBIC, box=(x0, y0, x1, y1),
                                           reducing_gap=None), np.float32)
    return np.clip(out, 0.0, 1.0, out=out)


def screen_grid(cam, x0, y0, x1, y1):
    """Plate coordinates (array index space, margin included) for the screen
    rectangle [x0, x1) x [y0, y1)."""
    zoom, cx, cy = cam
    sy, sx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    px = (sx - W / 2) / zoom + cx + MARGIN
    py = (sy - H / 2) / zoom + cy + MARGIN
    return px, py


def to_screen(cam, x, y):
    zoom, cx, cy = cam
    return (x - cx) * zoom + W / 2, (y - cy) * zoom + H / 2


def over(dst, src):
    a = src[..., 3:4]
    dst *= (1.0 - a)
    dst += src
    return dst


def save_review(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def flatten(layers, bg=(0.0, 0.0, 0.0)):
    """Quick composite of a render() dict for review (no fox)."""
    out = np.empty((H, W, 3), np.float32)
    out[:] = bg
    b = layers["bg"]
    out = out * (1 - b[..., 3:4]) + b[..., :3]
    if layers.get("fg") is not None:
        fg = layers["fg"]
        out = out * (1 - fg[..., 3:4]) + fg[..., :3]
    if layers.get("glow") is not None:
        out = out + layers["glow"]
    return np.clip(out, 0, 1)
