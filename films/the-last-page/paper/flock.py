"""Shared machinery for the paper cuts: a camera locked to the plate camera,
and flocks of birds or pages drawn by drawing key.

CutCam is a pinhole camera fixed at the cut's first frame; per frame the plate
camera's zoom and pan are applied as a 2D affine to the projected points, so
paper at the wall depth stays locked to the painted plate exactly and nearer
paper follows it closely.

A flock is a set of birds (bird.specs) with a drawing key per bird:
    0, 1, 2   fold stages (flat page, first crease, half fold)
    3         the smear between half fold and bird
    10 + i    flap key i (bird.FLAP_KEYS; 11 is the down smear)
    20        glide
Birds are grouped by key and level of detail and drawn with drawbird.
"""

import math

import numpy as np

import timeline as TL
from . import bird as BD
from . import drawbird as DB

W, H = TL.W, TL.H
KEY_GLIDE = 20


class CutCam:
    def __init__(self, cut, pos=(0, 0, 0), look=(0, 0, -1), up=(0, 1, 0), focal=1400.0):
        self.cut = cut
        self.base = BD.Camera(pos=pos, look=look, up=up, focal=focal, cx=W / 2, cy=H / 2)
        self.f0 = TL.cut_range(cut)[0]
        self.z0, self.cx0, self.cy0 = TL.camera(cut, self.f0)
        self.focal = focal
        self.set_frame(self.f0)
        self.R = self.base.R
        self.pos = self.base.pos
        self.ortho = False

    def set_frame(self, f):
        z, cx, cy = TL.camera(self.cut, f)
        self.k = z / self.z0
        self.tx = (self.cx0 - cx) * z
        self.ty = (self.cy0 - cy) * z
        return self

    def project(self, X):
        S = self.base.project(X)
        S[..., 0] = (S[..., 0] - W / 2) * self.k + W / 2 + self.tx
        S[..., 1] = (S[..., 1] - H / 2) * self.k + H / 2 + self.ty
        return S

    def view_dir(self, X):
        return self.base.view_dir(X)

    def unproject0(self, sx, sy, Z):
        """World point at camera depth Z under screen point (sx, sy) of the first frame."""
        b = self.base
        d = b.R.T @ np.array([(sx - b.cx) / b.focal, (sy - b.cy) / b.focal, 1.0])
        return b.pos + d * (Z / (b.R[2] @ d))

    def plate_to_world(self, px, py, Z):
        sx = (px - self.cx0) * self.z0 + W / 2
        sy = (py - self.cy0) * self.z0 + H / 2
        return self.unproject0(sx, sy, Z)

    def screen_to_screen0(self, f, sx, sy):
        """A screen point of frame f to the same point in first frame screen coords."""
        self.set_frame(f)
        x0 = (sx - W / 2 - self.tx) / self.k + W / 2
        y0 = (sy - H / 2 - self.ty) / self.k + H / 2
        return x0, y0

    def depth(self, X):
        return (np.asarray(X) - self.base.pos) @ self.base.R[2]


def tumble(axis, angle):
    """Rotation matrices (B, 3, 3) about unit axes (B, 3) by angles (B,)."""
    a = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12)
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    I = np.eye(3)[None]
    return I + s * K + (1 - c) * (K @ K)


def smear_mix(Xa, Xb, T):
    X = Xb.copy()
    back = np.array(T.Wb + T.Ri + T.Ro + T.Go)
    X[:, :, back] = 0.35 * Xb[:, :, back] + 0.65 * Xa[:, :, back]
    return X


def _state(s, T, key):
    if key in (0, 1, 2):
        return BD.fold_state(s, T, key)
    if key == 3:
        return BD.bird_state(s, T, BD.GLIDE)
    if key == KEY_GLIDE:
        return BD.bird_state(s, T, BD.GLIDE)
    return BD.bird_state(s, T, BD.FLAP_KEYS[key - 10])


def draw(dls, s, keys, pos, R, scale, cam, look, drawing, split=None, alpha=1.0, flutter=None,
         lod_px=60.0, ss_near=3, page_override=None):
    """Draw a flock. dls: dict with 'back' (and 'front') DrawLists; split: depth
    below which birds go to 'front'. flutter: optional (B,) phase for pages."""
    B = s["B"]
    if B == 0:
        return 0
    keys = np.asarray(keys)
    depth = cam.depth(pos)
    ok = depth > 0.5
    # projected size estimate for level of detail and line weight
    approx = scale * cam.focal * cam.k / np.maximum(depth, 0.5) * 1.5
    ok &= approx > 0.8
    alpha = np.broadcast_to(np.asarray(alpha, np.float64), (B,))
    ok &= alpha > 0.01
    count = 0
    for key in np.unique(keys[ok]):
        for lod in (0, 1):
            m = ok & (keys == key) & ((approx >= lod_px) if lod else (approx < lod_px))
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            T = BD.topo(8 if lod else 4)
            sk = BD.take(s, idx)
            st = _state(sk, T, int(key))
            if flutter is not None and key in (0, 1):
                ph = np.asarray(flutter)[idx]
                fl = BD.wing_bend(T, len(idx), np.radians(6) * np.sin(ph), np.radians(22) * np.sin(ph * 1.3 + 1),
                                  np.radians(14) * np.sin(ph * 0.7 + 2))
                st["bend"] = st["bend"] + fl
                st["ridge"] = np.asarray(st.get("ridge", 0.0)) + np.radians(10) * np.sin(ph * 0.9)
            X, UV = BD.build(sk, st, st, T)
            if key == 3:
                st2 = BD.fold_state(sk, T, 2)
                X2, _ = BD.build(sk, st2, st2, T)
                X = smear_mix(X2, X, T)
            elif key == 11:
                st0 = BD.bird_state(sk, T, BD.FLAP_KEYS[0])
                X0, _ = BD.build(sk, st0, st0, T)
                X = smear_mix(X0, X, T)
            for which in ("back", "front"):
                if which == "front" and (split is None or "front" not in dls):
                    continue
                if split is None or "front" not in dls:
                    mm = np.ones(len(idx), bool)
                elif which == "front":
                    mm = depth[idx] < split
                else:
                    mm = depth[idx] >= split
                if not mm.any():
                    continue
                sel = np.flatnonzero(mm)
                s2 = BD.take(sk, sel)
                st_sel = dict(st)
                for kk in ("bend", "ridge", "head", "neck", "tail", "spread", "curl"):
                    v = st_sel.get(kk)
                    if isinstance(v, np.ndarray) and v.shape[:1] == (len(idx),):
                        st_sel[kk] = v[sel]
                DB.bird_batch(dls[which], s2, X[sel], UV[sel], st_sel, look, cam, R[idx][sel], pos[idx][sel],
                              scale[idx][sel], T=T, drawing=drawing + int(key), ss=ss_near if lod else 2,
                              N=40 if lod else 24, alpha=alpha[idx][sel])
                count += len(sel)
    return count


def flap_key(drawing, phase, rate=1):
    """Flap key per bird for a drawing number (six drawings a cycle)."""
    return 10 + ((drawing * rate + phase) % 6)
