"""Cel colour tones from a scene light.

A character material has one albedo (its colour in neutral light). Each scene
light turns it into a small set of flat tones, the way a colour designer
writes a colour script: lit, shadow (hue shifted toward the ambient, never
grey), warm (lantern side), warm shadow, and rim.
"""

import colorsys

import numpy as np


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _norm(c):
    c = np.asarray(c, dtype=np.float32)
    return c / max(float(c.max()), 1e-6)


def _sat(c, amount):
    """Scale saturation of an RGB colour in HSV."""
    h, s, v = colorsys.rgb_to_hsv(*[float(x) for x in np.clip(c, 0, 1)])
    return np.array(colorsys.hsv_to_rgb(h, min(1.0, s * amount), v), dtype=np.float32)


EXPOSURE = {"night_out": 0.80, "hall": 0.86, "gold_burst": 1.0, "sunrise": 1.0, "morning": 1.04}


def tones(albedo, L, name="hall", shade_depth=0.58, warm_amount=0.55):
    """Return dict of RGB tones for one material under light L (a LIGHTS entry)."""
    alb = np.asarray(albedo, dtype=np.float32)
    k = _norm(L["key"])
    amb = _norm(L["ambient"])
    exp = EXPOSURE.get(name, 1.0)
    lit = alb * (0.45 + 0.55 * k) * exp
    lit = _sat(lit, 1.06)
    # shadow: albedo pushed toward the ambient hue, a little more saturated
    sh_tint = 0.5 * amb + 0.5 * np.array([0.62, 0.55, 0.92], dtype=np.float32)
    shade = alb * sh_tint * shade_depth * (exp / 0.86) ** 0.5
    shade = _sat(shade, 1.12)
    out = {"lit": lit, "shade": shade}
    if L.get("warm") is not None:
        w = np.asarray(L["warm"], dtype=np.float32)
        out["warm"] = np.clip(lit * (1 - warm_amount) + alb * w * (warm_amount + 0.25), 0, 1)
        out["warm_shade"] = np.clip(shade * (1 - warm_amount) + alb * w * warm_amount * 0.85, 0, 1)
    else:
        out["warm"] = lit
        out["warm_shade"] = shade
    if L.get("rim") is not None:
        r = np.asarray(L["rim"], dtype=np.float32)
        out["rim"] = np.clip(lit * 0.55 + r * 0.55 * (0.5 + 0.5 * alb.mean()), 0, 1)
    else:
        out["rim"] = lit
    return out


def light3(L, z=0.55):
    """3D unit vector toward the key light in screen space (z toward viewer)."""
    d = np.array([L["key_dir"][0], L["key_dir"][1], z], dtype=np.float64)
    return d / np.linalg.norm(d)
