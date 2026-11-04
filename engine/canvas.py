"""Supersampled drawing canvas for cel characters.

A Canvas covers a screen rectangle at ss times the resolution. Characters draw
in canvas coordinates (canvas = ss * (screen - origin)); resolve() box-filters
down to a premultiplied RGBA tile that comp.place() puts into a full frame.
"""

import numpy as np

from . import raster
from .geom import Xform


class Canvas:
    def __init__(self, x0, y0, w, h, ss=3):
        self.x0, self.y0 = int(x0), int(y0)
        self.w, self.h = int(w), int(h)
        self.ss = int(ss)
        self.W, self.H = self.w * self.ss, self.h * self.ss
        self.rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.a = np.zeros((self.H, self.W), dtype=np.float32)

    # coordinates -----------------------------------------------------------
    def xform(self):
        """Xform screen3d -> canvas3d (z is scaled too, so normals keep their meaning)."""
        s = float(self.ss)
        return Xform(np.eye(3) * s, (-self.x0 * s, -self.y0 * s, 0.0))

    def to_c(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - np.array([self.x0, self.y0])) * self.ss

    def bbox(self, polys, pad=2):
        """Integer canvas bbox (y0, y1, x0, x1) of polygons, clipped. None if empty."""
        if isinstance(polys, np.ndarray) and polys.ndim == 2:
            polys = [polys]
        allp = np.vstack([np.asarray(p) for p in polys if len(p)])
        x0 = int(np.floor(allp[:, 0].min())) - pad
        x1 = int(np.ceil(allp[:, 0].max())) + pad
        y0 = int(np.floor(allp[:, 1].min())) - pad
        y1 = int(np.ceil(allp[:, 1].max())) + pad
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (y0, y1, x0, x1)

    def mask(self, polys, box=None):
        """Bool mask of polygons inside box (defaults to their bbox). Returns (box, mask)."""
        if box is None:
            box = self.bbox(polys)
            if box is None:
                return None, None
        y0, y1, x0, x1 = box
        return box, raster.poly_mask(polys, y1 - y0, x1 - x0, ox=x0, oy=y0)

    def grid(self, box):
        """Canvas pixel-centre coordinates (xs, ys) over a box."""
        y0, y1, x0, x1 = box
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        return xs + 0.5, ys + 0.5

    # painting --------------------------------------------------------------
    def paint(self, box, cov, colour, opacity=1.0):
        """Paint colour (3,) or (h, w, 3) straight colour over the canvas with coverage cov."""
        y0, y1, x0, x1 = box
        c = np.asarray(cov, dtype=np.float32)
        if opacity != 1.0:
            c = c * np.float32(opacity)
        col = np.asarray(colour, dtype=np.float32)
        rgb = self.rgb[y0:y1, x0:x1]
        a = self.a[y0:y1, x0:x1]
        inv = 1.0 - c
        if col.ndim == 1:
            rgb *= inv[..., None]
            rgb += c[..., None] * col
        else:
            rgb *= inv[..., None]
            rgb += c[..., None] * col
        a *= inv
        a += c

    def paint_index(self, box, mask, table, idx):
        """Paint per-pixel colours table[idx] where mask is set (hard, supersampled)."""
        y0, y1, x0, x1 = box
        sel = mask
        rgb = self.rgb[y0:y1, x0:x1]
        a = self.a[y0:y1, x0:x1]
        rgb[sel] = table[idx[sel]]
        a[sel] = 1.0

    def stroke_buffer(self, box):
        y0, y1, x0, x1 = box
        return np.zeros((y1 - y0, x1 - x0), dtype=np.float32)

    def resolve(self):
        """Premultiplied RGBA tile (h, w, 4) at 1x."""
        rgb = raster.downsample(self.rgb, self.ss)
        a = raster.downsample(self.a, self.ss)
        return np.concatenate([rgb, a[..., None]], axis=2).astype(np.float32)
