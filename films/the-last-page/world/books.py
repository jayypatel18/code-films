"""Books, painted the way a background painter paints them: one confident
stroke per spine in the book's own colour, then accents (shadow edge, lit
edge, raised bands, gilt, a title label, page edges, moss). Every book is
synthesized from its own seed, so no two are alike and nothing is a stamp.

Shared by every hall cut (2, 3, 4a, 5a, 5b, 5c, 5d, 6a, 6b).
"""

import math

import numpy as np

from engine import paint as P
from engine.noise import rng

BOOK_COLS = np.array([
    (0.56, 0.17, 0.14), (0.42, 0.10, 0.12), (0.62, 0.26, 0.16), (0.18, 0.33, 0.23), (0.30, 0.37, 0.18),
    (0.13, 0.20, 0.37), (0.20, 0.33, 0.43), (0.41, 0.27, 0.15), (0.30, 0.20, 0.14), (0.64, 0.47, 0.20),
    (0.63, 0.53, 0.39), (0.33, 0.16, 0.29), (0.11, 0.10, 0.11), (0.74, 0.68, 0.54), (0.48, 0.38, 0.30),
    (0.24, 0.24, 0.36), (0.52, 0.30, 0.32), (0.26, 0.40, 0.38),
], np.float32)
GILT = np.array([0.85, 0.70, 0.35], np.float32)


def books_for_segment(x0, x1, base_y, rh, key, scale=1.0, flat_rate=0.06, gap_rate=0.05):
    """Book list for one shelf run from x0 to x1 standing on y = base_y, with
    room rh above the board. scale multiplies spine thickness."""
    r = rng("books", key)
    out = []
    sag = (1.0 + 3.5 * r.random()) * min(1.0, (x1 - x0) / 160) * scale
    x = x0 + r.random() * 4
    lean_next = 0.0
    while x < x1 - 4 * scale:
        u = (x - x0) / max(x1 - x0, 1)
        yb = base_y + sag * math.sin(math.pi * min(max(u, 0), 1))
        p = r.random()
        col = BOOK_COLS[int(r.integers(0, len(BOOK_COLS)))].copy() * (0.85 + 0.3 * r.random())
        faded = r.random() < 0.22
        if faded:
            col = col * 0.55 + np.array([0.60, 0.56, 0.50], np.float32) * 0.45
        mossy = r.random() < 0.08
        if p < gap_rate:
            x += (6 + r.random() * 22) * scale
            continue
        if p < gap_rate + flat_rate and x1 - x > 70 * scale:
            n = int(2 + r.integers(0, 4))
            y = yb
            bw = rh * (0.55 + 0.35 * r.random())
            for j in range(n):
                th = (5 + r.random() * 8) * scale
                if y - th < yb - rh + 6 * scale:
                    break
                c = BOOK_COLS[int(r.integers(0, len(BOOK_COLS)))] * (0.85 + 0.3 * r.random())
                dx = (r.random() - 0.5) * 6 * scale
                out.append(dict(kind="flat", x=x + bw / 2 + dx, y=y - th / 2, w=bw * (0.85 + 0.15 * r.random()),
                                h=th, angle=0.0, col=c, faded=False, mossy=False, seed=int(r.integers(1 << 30))))
                y -= th
            x += bw + (2 + r.random() * 3) * scale
            continue
        w = (6 + r.random() * 9 if r.random() > 0.12 else 14 + r.random() * 8) * scale
        if r.random() < 0.08:
            w = (3.5 + r.random() * 2.5) * scale
        h = rh * (0.58 + 0.34 * r.random())
        if r.random() < 0.1:
            h = rh * (0.88 + 0.06 * r.random())
        ang = 0.0
        if lean_next:
            ang, lean_next = lean_next, 0.0
        elif r.random() < 0.09 and x1 - x > 40 * scale:
            ang = 0.12 + 0.3 * r.random()
        elif r.random() < 0.05:
            ang = -(0.08 + 0.2 * r.random())
        out.append(dict(kind="stand", x=x + w / 2, y=yb, w=w, h=h, angle=ang, col=col, faded=faded,
                        mossy=mossy, seed=int(r.integers(1 << 30))))
        x += w * math.cos(ang) + h * abs(math.sin(ang)) * (0.85 if ang > 0 else 0.4) + (0.4 + r.random() * 1.2) * scale
    return out


def book_geom(b):
    """Returns P_(u, v) mapping spine coords to plate coords for a standing book."""
    x, y, w, h, a = b["x"], b["y"], b["w"], b["h"], b["angle"]
    ca, sa = math.cos(a), math.sin(a)
    if a == 0:
        bx, by = x, y
    else:
        piv = x + (w / 2 if a > 0 else -w / 2)
        bx = piv + (-(w / 2) if a > 0 else (w / 2)) * ca
        by = y - (w / 2) * abs(sa)

    def P_(u, v):
        px = bx + u * (w / 2) * ca + v * h * sa
        py = by + u * (w / 2) * sa * (1 if a else 0) - v * h * ca
        return (px, py)
    return P_


PAGE = np.array([0.80, 0.74, 0.58], np.float32)


def open_book_strokes(books, light, amount=1.0, offset=(0.0, 0.0), seed_salt=0):
    """Books flung open where they stand: the two covers swing out into a V
    and the page block fans between them. amount 0.5 is the snapped half
    open drawing, 1.0 fully open."""
    body = P.Strokes(max_pts=6)
    ox, oy = offset
    for b in books:
        r = np.random.default_rng(b["seed"] + seed_salt)
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        L = np.asarray(light(x, y - h / 2), np.float32)
        col = np.clip(b["col"] * L, 0, 1)
        pc = np.clip(PAGE * L * (0.62 + 0.18 * r.random()), 0, 1)
        if b["kind"] == "flat":
            # a lying book pops its top cover up
            a = (0.5 + 0.6 * r.random()) * amount
            x0 = x - w / 2
            body.add([(x0 + ox, y + oy), (x0 + w * math.cos(a) + ox, y - w * math.sin(a) * 0.5 + oy)], max(1.0, h * 0.25),
                     col, seed=b["seed"], opacity=0.95, dry=0.2, ragged=0.08, bristles=2, pool=0.1, streak=0.1,
                     taper0=0, taper1=0, cap=1.0)
            for k in range(4):
                aa = a * (k + 1) / 5
                body.add([(x0 + ox, y + oy), (x0 + w * 0.95 * math.cos(aa) + ox, y - w * 0.95 * math.sin(aa) * 0.5 + oy)],
                         0.7, pc, seed=b["seed"] + 5 + k, opacity=0.8, dry=0.3, ragged=0.15, bristles=2, pool=0,
                         streak=0.05, taper0=0, taper1=0.4)
            continue
        # spread angle of the covers from vertical
        spread = (0.55 + 0.5 * r.random()) * amount
        base = (x + ox, y + oy)
        L_c = h * (0.95 + 0.05 * r.random())
        lean = b["angle"] * 0.6 + (r.random() - 0.5) * 0.25 * amount
        for side in (-1, 1):
            a = lean + side * spread
            tip = (base[0] + math.sin(a) * L_c, base[1] - math.cos(a) * L_c)
            body.add([base, tip], max(1.5, w * 0.30), np.clip(col * (0.8 if side < 0 else 1.1), 0, 1),
                     seed=b["seed"] + 1 + (side > 0), opacity=0.97, dry=0.2, ragged=0.08, bristles=2, pool=0.1,
                     streak=0.1, taper0=0, taper1=0, cap=1.0)
        # the page block: a solid pale fan between the covers, a few pages lifting free
        npg = 5
        fan_w = max(1.2, L_c * spread * 2 / npg * 0.42)
        for k in range(npg):
            u = (k + 0.5) / npg * 2 - 1
            a = lean + u * spread * 0.85
            ln = L_c * 0.92
            tip = (base[0] + math.sin(a) * ln * 0.55, base[1] - math.cos(a) * ln * 0.55)
            shade = 0.80 + 0.28 * (1 - abs(u))
            body.add([base, tip], fan_w * 0.6, np.clip(pc * shade, 0, 1), seed=b["seed"] + 10 + k, opacity=0.95,
                     dry=0.2, ragged=0.1, bristles=3, pool=0.08, streak=0.06, taper0=0.0, taper1=0.0)
            tip2 = (base[0] + math.sin(a) * ln, base[1] - math.cos(a) * ln)
            body.add([tip, tip2], fan_w * 0.45, np.clip(pc * shade * 1.04, 0, 1), seed=b["seed"] + 20 + k,
                     opacity=0.95, dry=0.25, ragged=0.12, bristles=3, pool=0.06, streak=0.06, taper0=0.0, taper1=0.3)
        for k in range(3):
            a = lean + (r.random() * 2 - 1) * spread * 0.9
            ln = L_c * (0.85 + 0.25 * r.random())
            tip = (base[0] + math.sin(a) * ln, base[1] - math.cos(a) * ln)
            body.add([base, tip], 0.6, np.clip(pc * 0.6, 0, 1), seed=b["seed"] + 30 + k, opacity=0.6, dry=0.4,
                     ragged=0.15, bristles=2, pool=0, streak=0, taper0=0, taper1=0.4)
    return body.arrays()


def book_strokes(books, light, haze=None, eye_y=560.0, moon=(0.62, 0.71, 0.90), offset=(0.0, 0.0),
                 light_side=1.0):
    """Strokes for a list of books.

    light(x, y) -> RGB multiplier at a plate point (already scaled so that
    colour * light is the displayed colour). haze: None or (colour, amount
    function (x, y) -> 0..1) to sink distant books into the air.
    offset is added to every point (e.g. the plate margin). light_side: +1 when
    the light comes from the right of the spines, -1 from the left."""
    moon = np.asarray(moon, np.float32)
    body = P.Strokes(max_pts=4)
    acc = P.Strokes(max_pts=4)
    ox, oy = offset
    for b in books:
        r = np.random.default_rng(b["seed"])
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        L = np.asarray(light(x, y - h / 2), np.float32)
        col = np.clip(b["col"] * L, 0, 1)
        hz = 0.0
        if haze is not None:
            hz = float(haze[1](x, y - h / 2))
            col = col * (1 - hz) + np.asarray(haze[0], np.float32) * hz
        lum = float(col.mean())

        def sh(c, k):
            c = np.clip(c * k, 0, 1)
            if haze is not None:
                c = c * (1 - hz) + np.asarray(haze[0], np.float32) * hz
            return c
        if b["kind"] == "flat":
            p0 = (x - w / 2 + ox, y + oy)
            p1 = (x + w / 2 + ox, y + oy)
            body.add([p0, ((p0[0] + p1[0]) / 2, y + oy + (r.random() - 0.5) * 0.6), p1], h / 2 * 1.02, col,
                     seed=b["seed"], opacity=0.97, dry=0.15, ragged=0.06, bristles=max(2, h / 3), pool=0.10,
                     streak=0.10, taper0=0.0, taper1=0.0, pickup=0.0, paper_k=0.15, cap=1.0)
            acc.add([(x - w / 2 + ox, y + h / 2 - 0.8 + oy), (x + w / 2 + ox, y + h / 2 - 0.8 + oy)], 0.7,
                    sh(b["col"] * L, 0.5), seed=b["seed"] + 1, opacity=0.7, dry=0.3, ragged=0.1, bristles=2, pool=0,
                    streak=0, taper0=0.05, taper1=0.1, pickup=0)
            if r.random() < 0.6:
                acc.add([(x - w * 0.25 + ox, y + oy), (x + w * 0.1 + ox, y + oy)], max(0.6, h * 0.2),
                        np.clip(col * 1.5 + 0.03, 0, 1), seed=b["seed"] + 2, opacity=0.5, dry=0.5, ragged=0.2,
                        bristles=2, pool=0, streak=0, pickup=0)
            continue
        P0 = book_geom(b)

        def P_(u, v):
            px, py = P0(u, v)
            return (px + ox, py + oy)
        bend = (r.random() - 0.5) * 0.8
        mid = P_(0, 0.5)
        body.add([P_(0, 0.02), (mid[0] + bend, mid[1]), P_(0, 0.98)], w / 2 * 1.03, col, seed=b["seed"],
                 opacity=0.97, dry=0.18, ragged=0.05, bristles=max(2.0, w / 2.5), pool=0.12, streak=0.12,
                 taper0=0.0, taper1=0.0, pickup=0.0, paper_k=0.15, cap=1.0)
        s = light_side
        acc.add([P_(-0.78 * s, 0.01), P_(-0.78 * s, 0.99)], max(0.6, w * 0.13), sh(b["col"] * L, 0.48),
                seed=b["seed"] + 3, opacity=0.65, dry=0.35, ragged=0.15, bristles=2, pool=0, streak=0.05,
                taper0=0.0, taper1=0.1, pickup=0, cap=1.0)
        if lum > 0.05:
            acc.add([P_(0.42 * s, 0.05), P_(0.42 * s, 0.95)], max(0.5, w * 0.12),
                    np.clip(col * 1.45 + moon * 0.03, 0, 1), seed=b["seed"] + 4, opacity=0.45 + 0.3 * r.random(),
                    dry=0.65, ragged=0.2, bristles=2, pool=0, streak=0.05, taper0=0.1, taper1=0.3, pickup=0)
        nb = int(r.integers(0, 4))
        gilt = r.random() < 0.45
        bcol = sh(GILT * L * 2.0, 1.0) if gilt else np.clip(col * 1.6, 0, 1)
        for k in range(nb):
            v = 0.12 + 0.76 * (k + 0.5 + (r.random() - 0.5) * 0.4) / max(nb, 1)
            acc.add([P_(-0.85, v), P_(0.85, v)], (0.55 + 0.35 * r.random()) * max(1.0, w / 12), bcol,
                    seed=b["seed"] + 10 + k, opacity=0.55, dry=0.4, ragged=0.2, bristles=2, pool=0, streak=0,
                    taper0=0.1, taper1=0.1, pickup=0)
        if r.random() < 0.22 and w > 7:
            v0 = 0.55 + 0.25 * r.random()
            lc = sh(np.array([0.62, 0.57, 0.46]) * L * 1.6, 1.0) if r.random() < 0.6 else bcol
            acc.add([P_(0, v0), P_(0, v0 + 0.08 + 0.06 * r.random())], w * 0.3, lc, seed=b["seed"] + 20,
                    opacity=0.45, dry=0.3, ragged=0.15, bristles=2, pool=0.05, streak=0.05, taper0=0.1,
                    taper1=0.1, pickup=0)
        if y - h > eye_y - 40 and r.random() < 0.5:
            acc.add([P_(-0.85, 0.995), P_(0.85, 0.995)], 0.6 * max(1.0, w / 12), np.clip(col * 1.35, 0, 1),
                    seed=b["seed"] + 30, opacity=0.5, dry=0.5, ragged=0.2, bristles=2, pool=0, streak=0, pickup=0,
                    cap=1.0, taper0=0.0, taper1=0.0)
        if b["mossy"]:
            for k in range(int(3 + w)):
                u = r.random() * 2 - 1
                v = r.random() ** 2 * 0.35
                q = P_(u, v)
                mc = sh(b["col"] * L * 0.5 + np.array([0.10, 0.20, 0.10]) * L * 2.0 * (0.6 + 0.6 * r.random()), 1.0)
                acc.add([q, (q[0] + (r.random() - 0.5) * 2, q[1] - 1 - r.random() * 2)], 0.7 + 0.6 * r.random(), mc,
                        seed=b["seed"] + 40 + k, opacity=0.85, dry=0.3, ragged=0.4, bristles=2, pool=0.1, streak=0,
                        pickup=0)
    return body.arrays(), acc.arrays()
