"""The fox: an original small red fox, built from simple 3D masses and drawn as cels.

Construction (fox space: units of one head radius, x = the fox's left, y down,
z forward, ground at y = 0): the head is a cranium, a cheek ruff and a short
muzzle; the eyes sit on curved planes on the face; the ears are cupped cards
that swivel and flatten on their own. The body is a neck ruff, chest, barrel
and hips, with tube legs solved by two-bone IK and a fluffy spline tail.

Every drawing the masses are posed, projected and painted as flat cel shapes
through one z-buffer: a lit tone, one hard shadow tone hue-shifted toward the
ambient, a warm lantern tone, a cool rim, then pressure ink lines traced from
the visible silhouette of each group (head, body, each leg, tail, each ear).

Pose parameters live in FoxPose; animation code keys them on twos and threes.
"""

import math
import zlib
from dataclasses import dataclass, field, replace

import numpy as np

from engine import cel, contour, geom, ink, raster, vnoise
from engine.canvas import Canvas
from engine.geom import Ellipsoid, Xform, deg, euler
from engine.light import hexrgb, light3
from engine.tube import Tube

ALBEDO = {
    "orange": hexrgb("#DB7A2E"),
    "cream": hexrgb("#F6E9D3"),
    "sock": hexrgb("#3A2A2A"),
    "inner": hexrgb("#6A4440"),
    "nose": hexrgb("#2A1D1B"),
    "mouth": hexrgb("#5E2A2C"),
    "eyedark": hexrgb("#3A2216"),
}
INK = hexrgb("#3B2618")

# Hand-picked colour scripts per scene light (lit, shade, warm, warm_shade, rim).
# Shadows lean violet in moonlight; the lantern side goes warm; the rim is cool.
DESIGNED = {
    "hall": {
        "orange": dict(lit="#BD6638", shade="#6A3A58", warm="#D87A3E", warm_shade="#94504A", rim="#D99A8E"),
        "cream": dict(lit="#E6E1E4", shade="#938DB6", warm="#F6DEBE", warm_shade="#E4C4A2", rim="#EEEEFF"),
        "sock": dict(lit="#41303A", shade="#2B2033", warm="#5C3C34", warm_shade="#3F2B2E", rim="#6E6888"),
        "inner": dict(lit="#7A4A58", shade="#4E3044", warm="#8E5648", warm_shade="#603A40", rim="#7A4A58"),
        "nose": dict(lit="#2A1E26", shade="#1E1620", warm="#3A2424", warm_shade="#2A1C20", rim="#4A4058"),
        "mouth": dict(lit="#5A2A38", shade="#3E2030", warm="#7A3434", warm_shade="#502630", rim="#5A2A38"),
        "eyedark": dict(lit="#3A2420", shade="#2C1C1E", warm="#4A2C20", warm_shade="#3A2420", rim="#3A2420"),
    },
    "morning": {
        "orange": dict(lit="#DD7B30", shade="#A64D3E", warm="#E58A3C", warm_shade="#B0583E", rim="#F4B27C"),
        "cream": dict(lit="#F7EBD6", shade="#BFAFC4", warm="#FBEBD2", warm_shade="#C9B2BE", rim="#FFFDF6"),
        "sock": dict(lit="#3C2C2C", shade="#2C1E28", warm="#463230", warm_shade="#30222A", rim="#6A5654"),
        "inner": dict(lit="#8E5C56", shade="#623E44", warm="#8E5C56", warm_shade="#623E44", rim="#8E5C56"),
        "nose": dict(lit="#2A1D1B", shade="#20161A", warm="#2A1D1B", warm_shade="#20161A", rim="#5A4A4A"),
        "mouth": dict(lit="#6E3036", shade="#4E2430", warm="#6E3036", warm_shade="#4E2430", rim="#6E3036"),
        "eyedark": dict(lit="#3E2418", shade="#2E1C18", warm="#3E2418", warm_shade="#2E1C18", rim="#3E2418"),
    },
    "sunrise": {
        "orange": dict(lit="#EE8A3C", shade="#9E4E54", warm="#EE8A3C", warm_shade="#9E4E54", rim="#FFE3A2"),
        "cream": dict(lit="#FFEBD0", shade="#C3A7BE", warm="#FFEBD0", warm_shade="#C3A7BE", rim="#FFFDEE"),
        "sock": dict(lit="#48302E", shade="#30202C", warm="#48302E", warm_shade="#30202C", rim="#D2A070"),
        "inner": dict(lit="#93605A", shade="#663E48", warm="#93605A", warm_shade="#663E48", rim="#93605A"),
        "nose": dict(lit="#2E1E1C", shade="#22161C", warm="#2E1E1C", warm_shade="#22161C", rim="#6A5048"),
        "mouth": dict(lit="#70323A", shade="#502632", warm="#70323A", warm_shade="#502632", rim="#70323A"),
        "eyedark": dict(lit="#402618", shade="#301C18", warm="#402618", warm_shade="#301C18", rim="#402618"),
    },
}
DESIGNED["night_out"] = DESIGNED["hall"]
DESIGNED["gold_burst"] = DESIGNED["sunrise"]


@dataclass
class FoxPose:
    x: float = 960.0           # screen position of the root (head centre for a bust, ground point for full)
    y: float = 540.0
    scale: float = 100.0       # screen px per head radius
    yaw: float = 0.55          # whole fox: + turns the nose toward screen right
    pitch: float = 0.0
    roll: float = 0.0
    cam_tilt: float = 0.0      # + the camera looks down on the fox (shows the ground plane)
    head_yaw: float = 0.0      # head relative to the body
    head_pitch: float = 0.0    # + lifts the nose
    head_roll: float = 0.0     # + tilts clockwise on screen
    ear_swivel: tuple = (0.0, 0.0)   # (fox right, fox left): + opens the ear outward
    ear_flat: tuple = (0.0, 0.0)     # 0 up, 1 pinned back
    eye_open: float = 1.0
    gaze: tuple = (0.0, 0.0)         # iris offset in head space, about -1..1
    worry: float = 0.3               # lifts the inner ends of the upper lids
    mouth_open: float = 0.0
    sniff: float = 0.0               # 0..1 nose lift, whisker twitch
    body: str = "bust"               # "bust" (head and chest) or "full"
    stance: str = "stand"            # "stand" or "sit" (full body)
    walk: float = None               # walk phase 0..1, None = standing still
    stride: float = 1.0
    body_bob: float = 0.0            # extra body drop (units)
    tail: tuple = (0.0, 0.0, 0.0)    # (lift, sideways, curl)
    tail_wind: float = 0.0           # streaming in wind, 0..1
    wind_dir: tuple = (1.0, 0.0)     # screen direction of the wind (for fur)
    lantern: bool = False            # paper lantern held in the mouth
    lantern_swing: float = 0.0       # swing of the held lantern (radians)
    squash: tuple = (1.0, 1.0)       # screen space squash and stretch about the root
    reach: tuple = None              # (side, (x, y, z) fox space): one front paw reaches a target
    fur_wind: float = 0.0            # wind combing the fur tufts toward wind_dir
    seed: int = 0                    # drawing number, for line boil
    line: float = None               # nominal line width in screen px (auto if None)
    glint: bool = True               # warm lantern glint in the eyes


# ----------------------------------------------------------------- head design
# All in head-local space (cranium radius 1).
CRANIUM = dict(c=(0.0, 0.0, 0.0), r=(0.96, 0.86, 0.90))
JOWL = dict(c=(0.0, 0.40, 0.02), r=(1.10, 0.50, 0.72))
MUZZLE = dict(c=(0.0, 0.25, 0.74), r=(0.30, 0.245, 0.68), pitch=deg(-14))
CHIN = dict(c=(0.0, 0.47, 0.62), r=(0.19, 0.10, 0.38))
NECK_PIVOT = np.array([0.0, 0.6, -0.4])

EYE_X, EYE_Y = 0.375, 0.03
EYE_SCALE = 1.08
EYE_TILT = deg(10)

UPPER_OPEN = np.array([(-0.23, 0.05), (-0.17, -0.10), (-0.04, -0.185), (0.11, -0.175), (0.22, -0.075), (0.25, -0.02)])
UPPER_CLOSED = np.array([(-0.23, 0.05), (-0.13, 0.095), (0.0, 0.11), (0.12, 0.085), (0.21, 0.035), (0.25, -0.02)])
LOWER = np.array([(0.25, -0.02), (0.20, 0.085), (0.08, 0.16), (-0.07, 0.155), (-0.18, 0.11), (-0.23, 0.05)])

# ----------------------------------------------------------------- body layouts
# fox space, ground at y = 0. Masses: name -> (centre, radii, pitch).
LAYOUTS = {
    "bust": dict(
        head=(0.0, 0.0, 0.0),
        masses={
            "neck": ((0.0, 1.05, -0.30), (0.98, 0.62, 0.80), 0.0),
            "chest": ((0.0, 2.05, -0.50), (0.96, 1.30, 0.90), 0.0),
        },
    ),
    "stand": dict(
        head=(0.0, -3.75, 1.95),
        masses={
            "neck": ((0.0, -3.12, 1.30), (0.62, 0.88, 0.80), deg(-35)),
            "chest": ((0.0, -2.38, 1.00), (0.76, 0.90, 0.92), 0.0),
            "barrel": ((0.0, -2.42, -0.08), (0.68, 0.76, 1.08), deg(2)),
            "hips": ((0.0, -2.52, -1.08), (0.72, 0.82, 0.84), 0.0),
        },
        shoulder=(0.40, -2.48, 1.18), hip=(0.44, -2.55, -1.12),
        tail_root=(0.0, -2.92, -1.76),
    ),
    "sit": dict(
        head=(0.0, -4.42, 0.80),
        masses={
            "neck": ((0.0, -3.70, 0.48), (0.68, 0.84, 0.72), deg(-30)),
            "chest": ((0.0, -2.85, 0.42), (0.84, 1.02, 0.90), deg(-70)),
            "barrel": ((0.0, -1.85, -0.18), (0.80, 0.98, 0.90), deg(-60)),
            "hips": ((0.0, -0.86, -0.58), (0.88, 0.82, 0.94), 0.0),
            "haunchL": ((0.58, -0.80, -0.30), (0.40, 0.72, 0.90), deg(-15)),
            "haunchR": ((-0.58, -0.80, -0.30), (0.40, 0.72, 0.90), deg(-15)),
        },
        shoulder=(0.38, -2.72, 0.70), hip=(0.62, -0.60, -0.50),
        tail_root=(0.0, -0.62, -1.40),
    ),
}

def muzzle_point(u):
    """Head-local point on the muzzle surface from a unit direction u in muzzle space."""
    u = geom.normalize(np.asarray(u, dtype=np.float64))
    R = geom.rot_x(MUZZLE["pitch"]) @ np.diag(MUZZLE["r"])
    return np.asarray(MUZZLE["c"]) + u @ R.T


def _ell(c, r, pitch=0.0):
    rot = geom.rot_x(pitch) if pitch else None
    return Ellipsoid(c, r, rot)


def ik2(A, B, a, b, hint):
    """Two-bone IK: joint position between A and B with bone lengths a, b, bending toward hint."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d = B - A
    L = float(np.linalg.norm(d))
    L = min(max(L, abs(a - b) + 1e-3), a + b - 1e-3)
    dn = d / (np.linalg.norm(d) + 1e-12)
    x = (a * a - b * b + L * L) / (2 * L)
    y = math.sqrt(max(a * a - x * x, 0.0))
    h = np.asarray(hint, dtype=np.float64)
    h = h - dn * np.dot(h, dn)
    h = h / (np.linalg.norm(h) + 1e-12)
    return A + dn * x + h * y


class FoxDrawing:
    """Builds one drawing of the fox for a pose and paints it into a canvas."""

    def __init__(self, pose, light_name, L, lantern_screen=None, lantern_reach=0.0, ss=3):
        self.p = pose
        self.lname = light_name
        self.L = L
        self.lantern_screen = lantern_screen
        self.lantern_reach = lantern_reach
        self.ss = ss
        self.glow = []      # (x, y, radius, rgb, strength) in screen px for the compositor
        self.hard_warm = 0.0  # > 0: the lantern also cuts a hard warm tone (set per shot)
        self.pal_override = {}   # {material: {slot: '#hex'}} on top of the designed palette
        self.rim_override = None  # (dir2, width in head units) for a special rim light

    # -------------------------------------------------------------- frames
    def _layout(self):
        p = self.p
        if p.body == "bust":
            return LAYOUTS["bust"]
        return LAYOUTS["sit" if p.stance == "sit" else "stand"]

    def _xforms(self, Xs2c):
        p = self.p
        lay = self._layout()
        Rv = geom.rot_x(-p.cam_tilt) @ euler(p.yaw, p.pitch, p.roll)
        Sq = np.diag([p.squash[0], p.squash[1], 1.0])
        Xs = Xform(Sq @ Rv * p.scale, (p.x, p.y, 0.0))
        Xc = Xs2c @ Xs
        Hc = np.asarray(lay["head"], dtype=np.float64) + np.array([0, p.body_bob, 0])
        Rh = euler(p.head_yaw, p.head_pitch, p.head_roll)
        Xhf = Xform(Rh, Hc + NECK_PIVOT - Rh @ NECK_PIVOT)
        return Xc, Xc @ Xhf

    @staticmethod
    def _facing_ranges(P, t, N, d2, thr, **kw):
        """Fur ranges covering the stretches of an outline whose normal faces d2."""
        ok = (N @ d2) > thr
        out = []
        n = len(P)
        i = 0
        while i < n:
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i > 4:
                out.append(dict(t0=t[i], t1=t[j - 1], **kw))
            i = j
        return out

    def _windy(self, ranges, key):
        """Comb fur ranges toward the wind and lengthen them; tufts re-drawn every drawing."""
        w = self.p.fur_wind
        if w <= 0:
            return ranges, key
        wd = np.asarray(self.p.wind_dir, dtype=np.float64)
        wd = wd / (np.linalg.norm(wd) + 1e-9)
        out = []
        for r in ranges:
            r = dict(r)
            lean = r.get("lean", (0.0, 1.0))
            if callable(lean):
                base = lean
                r["lean"] = (lambda tq, base=base: geom.normalize(np.asarray(base(tq)) + wd * 1.4 * w))
            else:
                r["lean"] = geom.normalize(np.asarray(lean, dtype=np.float64) + wd * 1.4 * w)
            r["length"] = r["length"] * (1.0 + 0.45 * w)
            r["comb"] = r.get("comb", 0.8) + 0.6 * w
            out.append(r)
        return out, key[:-1] + (self.p.seed,)

    def _organic(self, P, t, key, amp):
        """Break a perfect projected ellipse into a drawn shape: a slow, fixed
        wobble of the silhouette, attached to the surface parameter so it does
        not swim, plus a tiny boil from drawing to drawing."""
        N = geom.outward_normals(P)
        k = zlib.crc32(key.encode()) % 1000  # stable across runs (str hash is salted)
        d = 0.75 * vnoise.value1(t * 1.6 + k, k) + 0.25 * vnoise.value1(t * 4.1 + k * 0.3, k + 1) \
            + 0.08 * vnoise.value1(t * 3.0 + self.p.seed * 0.7, k + 2)
        return P + N * (d * amp * self.unit)[:, None]

    @staticmethod
    def _dir2(X, v):
        d = X.A @ np.asarray(v, dtype=np.float64)
        d2 = d[:2]
        return d2 / (np.linalg.norm(d2) + 1e-9)

    # -------------------------------------------------------------- geometry
    def build(self, Xs2c):
        """Project every part into canvas px (or screen px when Xs2c is identity)."""
        p = self.p
        Xc, Xh = self._xforms(Xs2c)
        self.Xc, self.Xh = Xc, Xh
        unit = float(np.linalg.norm(Xh.A[:, 0]))
        self.unit = unit
        seed = p.seed
        pi = math.pi
        groups = []

        # ---------------- head
        head = dict(name="head", parts=[], depth=float(Xh.apply(np.zeros((1, 3)))[0, 2]))
        cr = _ell(CRANIUM["c"], CRANIUM["r"]).project(Xh)
        P, z, t = cr.outline(240)
        P = self._organic(P, t, "cranium", 0.014)
        N = geom.outward_normals(P)
        up2 = self._dir2(Xh, (0, -1, 0))
        back2 = self._dir2(Xh, (0, -0.4, -1))
        crown = [dict(t0=2 * pi - 0.30, t1=2 * pi - 0.03, length=0.13, width=0.10, lean=back2, comb=0.5,
                      env=lambda u: math.sin(math.pi * u) ** 0.6),
                 dict(t0=0.03, t1=0.20, length=0.08, width=0.09, lean=back2, comb=0.5,
                      env=lambda u: math.sin(math.pi * u) ** 0.7)]
        crown, ck = self._windy(crown, ("crown", seed // 6))
        Pc, _ = cel.fur_edge(P, t, N, crown, ck, unit)
        head["parts"].append(dict(kind="ell", pe=cr, poly=Pc, pattern=self._head_pattern))
        jw = _ell(JOWL["c"], JOWL["r"]).project(Xh)
        P, z, t = jw.outline(300)
        P = self._organic(P, t, "jowl", 0.018)
        N = geom.outward_normals(P)
        down2 = self._dir2(Xh, (0, 1, -0.35))
        # one continuous run of tufts from cheek to cheek: long at the sides,
        # short under the chin where the cheek fur lies over the neck ruff
        fur = [dict(t0=0.34 * pi, t1=1.66 * pi, length=0.22, width=0.24, lean=down2, comb=0.9,
                    env=lambda u: 0.30 + 0.70 * (abs(math.cos(math.pi * u)) ** 0.8) * (1 - 0.6 * max(0, 0.15 - min(u, 1 - u)) / 0.15))]
        fur, jk = self._windy(fur, ("jowl", seed // 6))
        Pj, _ = cel.fur_edge(P, t, N, fur, jk, unit)
        head["parts"].append(dict(kind="ell", pe=jw, poly=Pj, pattern=self._head_pattern))
        mz = _ell(MUZZLE["c"], MUZZLE["r"], MUZZLE["pitch"]).project(Xh)
        mP, _, mt = mz.outline(160)
        head["parts"].append(dict(kind="ell", pe=mz, poly=self._organic(mP, mt, "muzzle", 0.008), pattern=self._muzzle_pattern, name="muzzle", bias=-0.80))
        ch = _ell(CHIN["c"], CHIN["r"]).project(Xh)
        head["parts"].append(dict(kind="ell", pe=ch, poly=ch.outline(120)[0], pattern=self._cream_pattern))
        self.muzzle_pe = mz
        groups.append(head)

        # ---------------- ears (each its own group)
        for s in (-1, 1):
            e = self._ear(Xh, s)
            groups.append(dict(name="ear", ear=e, parts=[], depth=e["depth"]))

        # ---------------- body
        lay = self._layout()
        body = dict(name="body", parts=[], depth=0.0)
        zs = []
        for name, (c, r, pt) in lay["masses"].items():
            c = np.asarray(c, dtype=np.float64) + np.array([0, p.body_bob, 0])
            pe = _ell(c, r, pt).project(Xc)
            P, z, t = pe.outline(220)
            P = self._organic(P, t, "body" + name, 0.022)
            N = geom.outward_normals(P)
            poly = P
            if name == "neck":
                out2 = self._dir2(Xc, (0, 1, -0.2))
                varied = lambda u: 0.45 + 0.55 * abs(math.sin(math.pi * u * 2.6 + 0.4))
                ruff = [dict(t0=0.18 * pi, t1=0.92 * pi, length=0.26, width=0.38, lean=out2, comb=0.7, env=varied),
                        dict(t0=1.08 * pi, t1=1.82 * pi, length=0.26, width=0.38, lean=out2, comb=0.7, env=varied)]
                if p.body == "full" and p.fur_wind <= 0:
                    # throat ruff only: tufts on the outline that faces down and forward
                    front2 = self._dir2(Xc, (0, 0.7, 1.0))
                    ruff = self._facing_ranges(P, t, N, front2, -0.15, length=0.24, width=0.36, lean=out2,
                                               comb=0.7, env=varied)
                ruff, rk = self._windy(ruff, ("ruff", seed // 6))
                poly, _ = cel.fur_edge(P, t, N, ruff, rk, unit)
            elif name == "chest" and p.body == "bust":
                out2 = self._dir2(Xc, (0, 1, -0.3))
                ruff = [dict(t0=0.20 * pi, t1=0.55 * pi, length=0.16, width=0.30, lean=out2, comb=0.6),
                        dict(t0=1.45 * pi, t1=1.80 * pi, length=0.16, width=0.30, lean=out2, comb=0.6)]
                poly, _ = cel.fur_edge(P, t, N, ruff, ("chestf", seed // 6), unit)
            elif name in ("hips", "haunchL", "haunchR"):
                back2 = self._dir2(Xc, (0, 0.6, -1))
                tuft = [dict(t0=0.50 * pi, t1=0.90 * pi, length=0.16, width=0.24, lean=back2, comb=0.7)]
                tuft, hk = self._windy(tuft, ("hipf", name, seed // 6))
                poly, _ = cel.fur_edge(P, t, N, tuft, hk, unit)
            # a bust's ruff is cream white in the light: only its far side, the head's cast
            # shadow and each clump's underside go into shade (designed shapes, not a mass)
            bias = {"neck": -0.46, "chest": -0.36}.get(name, 0.0) if p.body == "bust" else 0.0
            body["parts"].append(dict(kind="ell", pe=pe, poly=poly, pattern=self._body_pattern(name), name=name, bias=bias))
            zs.append(pe.c[2])
        body["depth"] = float(np.mean(zs))
        groups.append(body)
        ruff = self._ruff_clumps(lay, Xc, unit, seed)
        if ruff is not None:
            groups.append(ruff)

        # ---------------- legs and tail
        if p.body == "full":
            for leg in self._legs(lay):
                pt = leg["tube"].project(Xc)
                lpoly = self._leg_fur(pt, ("legfur", leg["side"], leg["front"], seed // 6))
                parts = [dict(kind="tube", pt=pt, poly=lpoly, pattern=self._leg_pattern(leg["sock"]))]
                if leg.get("paw") is not None:
                    pc, pr = leg["paw"]
                    pe = _ell(pc, pr).project(Xc)
                    pP, _, pt_ = pe.outline(60)
                    parts.append(dict(kind="ell", pe=pe, poly=pP, pattern=self._sock_pattern))
                    if p.scale > 60:
                        pc = np.asarray(pc, dtype=np.float64)
                        for side in (-0.55, 0.0, 0.55):
                            tc = pc + np.array([side * pr[0] * 0.85, 0.02, pr[2] * 0.82 - abs(side) * 0.08])
                            te = _ell(tc, (0.075, 0.07, 0.08)).project(Xc)
                            parts.append(dict(kind="ell", pe=te, poly=te.outline(40)[0], pattern=self._sock_pattern))
                groups.append(dict(name="leg", parts=parts,
                                   depth=pt.depth() + (0.02 * unit if leg["near"] else 0.0), leg=leg))
            tail = self._tail(lay)
            pt = tail.project(Xc)
            P = pt.outline(cap_samples=6)
            poly, _ = self._tail_fur(P, pt, ("tail", seed // 6))
            groups.append(dict(name="tail", parts=[dict(kind="tube", pt=pt, poly=poly, pattern=self._tail_pattern)],
                               depth=pt.depth()))

        # ---------------- face features
        self.feat = dict(eyes=[self._eye(Xh, s) for s in (-1, 1)], nose=self._nose(Xh), mouth=self._mouth(Xh),
                         whiskers=self._whiskers(Xh))
        # ---------------- lantern
        self.lantern_geo = self._lantern(Xh) if p.lantern else None
        if self.lantern_geo is not None:
            groups.append(dict(name="lantern", parts=[], depth=self.lantern_geo["depth"]))
        self.groups = groups
        return groups

    def bounds_screen(self):
        groups = self.build(Xform(np.eye(3), (0, 0, 0)))
        pts = []
        for g in groups:
            for part in g["parts"]:
                pts.append(part["poly"])
            if g["name"] == "ear":
                pts.append(g["ear"]["outline"])
        if self.lantern_geo is not None:
            pts.append(self.lantern_geo["body"])
        allp = np.vstack(pts)
        pad = 16
        return (allp[:, 0].min() - pad, allp[:, 1].min() - pad, allp[:, 0].max() + pad, allp[:, 1].max() + pad)

    # ---------------- legs
    def _legs(self, lay):
        p = self.p
        out = []
        sx, sy, sz = lay["shoulder"]
        hx, hy, hz = lay["hip"]
        bob = p.body_bob
        if p.stance == "sit":
            for s in (-1, 1):
                S = np.array([s * sx, sy + bob, sz])
                paw = np.array([s * 0.34, -0.12, 1.02])
                reaching = p.reach is not None and p.reach[0] == s
                if reaching:
                    paw = np.asarray(p.reach[1], dtype=np.float64)
                    # a real reach: the elbow lifts and bends back, the wrist breaks over the paw
                    wrist = paw + np.array([0, -0.36, -0.30])
                    E = ik2(S, wrist, 1.15, 1.08, (0, -0.8, -1))
                    pts = [S, E, wrist, paw + np.array([0, -0.04, 0.06])]
                    rad = [0.42, 0.28, 0.20, 0.20]
                    sock = 0.74
                else:
                    wrist = paw + np.array([0, -0.26, -0.08])
                    E = ik2(S, wrist, 1.32, 1.30, (0, 0, -1))
                    pts = [S, E, wrist, paw + np.array([0, 0, 0.12])]
                    rad = [0.32, 0.21, 0.155, 0.175]
                    sock = 0.60
                out.append(dict(tube=Tube(pts, rad, 6), sock=sock, near=False, front=True, side=s,
                                paw=(paw + np.array([0, -0.11, 0.10]), (0.21, 0.13, 0.29)), toes=True))
                # hind: the long foot lies flat on the ground in front of the haunch
                Hp = np.array([s * hx, hy + bob, hz])
                hock = np.array([s * 0.66, -0.16, -0.62])
                hpaw = np.array([s * 0.60, -0.12, 0.42])
                pts = [Hp, hock, (hock + hpaw) * 0.5 + np.array([0, -0.02, 0]), hpaw]
                rad = [0.30, 0.16, 0.15, 0.175]
                out.append(dict(tube=Tube(pts, rad, 6), sock=0.30, near=False, front=False, side=s,
                                paw=(hpaw + np.array([0, -0.03, 0.06]), (0.20, 0.13, 0.28))))
            return out
        ph = p.walk
        offs = {("H", 1): 0.0, ("F", 1): 0.25, ("H", -1): 0.5, ("F", -1): 0.75}
        for kind in ("F", "H"):
            for s in (-1, 1):
                if kind == "F":
                    root = np.array([s * sx, sy + bob, sz])
                    home = np.array([s * 0.36, 0.0, sz + 0.12])
                else:
                    root = np.array([s * hx, hy + bob, hz])
                    home = np.array([s * 0.40, 0.0, hz + 0.05])
                paw = home.copy()
                lift = 0.0
                if ph is not None:
                    q = (ph + offs[(kind, s)]) % 1.0
                    st = 0.62
                    if q < st:   # stance: paw slides back under the body
                        u = q / st
                        paw[2] = home[2] + p.stride * (0.5 - u)
                    else:        # swing: lift and reach forward
                        u = (q - st) / (1 - st)
                        e = 0.5 - 0.5 * math.cos(math.pi * u)
                        paw[2] = home[2] + p.stride * (-0.5 + e)
                        lift = math.sin(math.pi * u) * 0.42
                        paw[1] = -lift
                if kind == "F" and p.reach is not None and p.reach[0] == s:
                    paw = np.asarray(p.reach[1], dtype=np.float64)
                    lift = max(lift, 0.15)
                if kind == "F":
                    wrist = paw + np.array([0, -0.26, -0.10 + 0.25 * lift])
                    E = ik2(root, wrist, 1.18, 1.12, (0, 0.1, -1))
                    pts = [root, E, wrist, paw + np.array([0, -0.10 - 0.04 * lift, 0.13])]
                    rad = [0.38, 0.21, 0.15, 0.18]
                    sock = 0.52
                else:
                    hock = paw + np.array([0, -0.62 + 0.2 * lift, -0.30 - 0.15 * lift])
                    K = ik2(root, hock, 1.05, 1.02, (0, 0.2, 1))
                    pts = [root, K, hock, paw + np.array([0, -0.10, 0.12])]
                    rad = [0.54, 0.25, 0.15, 0.18]
                    sock = 0.66
                toe = paw + np.array([0, -0.11 - 0.02 * lift, 0.14])
                out.append(dict(tube=Tube(pts, rad, 6), sock=sock, near=False, front=kind == "F", side=s,
                                paw=(toe, (0.195, 0.125, 0.27))))
        return out

    def _ruff_clumps(self, lay, Xc, unit, seed):
        """The chest ruff broken into fur clumps: small flattened masses on the neck
        surface, each with its own form shadow and a ragged, tufted lower edge."""
        p = self.p
        if p.body != "bust" and p.stance != "sit":
            return None
        c, r, pt = lay["masses"]["neck"]
        neck = _ell(np.asarray(c) + np.array([0, p.body_bob, 0]), r, pt)
        R = neck.rot @ np.diag(neck.r)
        rnd = vnoise.rng("clumps", p.stance, p.body)
        rows = [(-0.25, [-0.62, -0.22, 0.20, 0.60]), (0.20, [-0.48, -0.05, 0.40]), (0.60, [-0.30, 0.22])] \
            if p.body == "bust" else [(-0.05, [-0.45, 0.0, 0.45]), (0.40, [-0.28, 0.25])]
        parts = []
        down2 = self._dir2(Xc, (0, 1, -0.25))
        for uy, xs in rows:
            for ux in xs:
                ux2 = ux + rnd.uniform(-0.05, 0.05)
                uz = math.sqrt(max(0.05, 1 - ux2 * ux2 - uy * uy))
                u = np.array([ux2, uy, uz])
                base = neck.c + R @ u
                nrm = geom.normalize(neck.rot @ (u / neck.r))
                tang_x = geom.normalize(np.cross(np.array([0, 1.0, 0]), nrm))
                tang_y = np.cross(nrm, tang_x)
                rot = np.stack([tang_x, tang_y, nrm], axis=1)
                w = rnd.uniform(0.30, 0.40) * (1.0 if p.body == "bust" else 0.85)
                cen = base + nrm * 0.04
                e = Ellipsoid(cen, (w, w * 0.62, 0.16), rot)
                pe = e.project(Xc)
                P, z, t = pe.outline(90)
                N = geom.outward_normals(P)
                fur = self._facing_ranges(P, t, N, down2, 0.15, length=0.13, width=0.30, lean=down2, comb=1.0,
                                          env=lambda q: 0.5 + 0.5 * math.sin(math.pi * q))
                poly, _ = cel.fur_edge(P, t, N, fur, ("clump", ux, uy, seed // 6), unit)
                parts.append(dict(kind="ell", pe=pe, poly=poly, pattern=self._cream_pattern, name="clump", bias=-0.12))
        zs = [pt_["pe"].c[2] for pt_ in parts]
        return dict(name="ruff", parts=parts, depth=float(np.mean(zs)), kz=0.012)

    def _leg_fur(self, pt, key):
        """Leg outline with a few tufts on the upper third (fur mass where the leg meets the body)."""
        P = pt.outline(cap_samples=6)
        n = len(P)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        N = geom.outward_normals(P)
        m = len(pt.C)
        # outline order: left side (m points), end cap, right side reversed, start cap
        down2 = pt.C[-1, :2] - pt.C[0, :2]
        down2 = down2 / (np.linalg.norm(down2) + 1e-9)
        k1 = int(0.30 * m)
        cap = 6
        left = (t[1], t[k1])
        r0 = m + cap + (m - k1)
        r1 = min(m + cap + m - 1, n - 1)
        ranges = [dict(t0=left[0], t1=left[1], length=0.13, width=0.20, lean=down2, comb=1.2),
                  dict(t0=t[min(r0, n - 2)], t1=t[r1], length=0.13, width=0.20, lean=down2, comb=1.2)]
        poly, _ = cel.fur_edge(P, t, N, ranges, key, self.unit)
        return poly

    # ---------------- tail
    def _tail(self, lay):
        p = self.p
        lift, side, curl = p.tail
        R = np.asarray(lay["tail_root"], dtype=np.float64) + np.array([0, p.body_bob, 0])
        if p.stance == "sit" and p.body == "full":
            sg = -1.0 if side <= 0 else 1.0
            pts = [R, R + (sg * 0.25, 0.25, -0.40), (sg * 0.95, -0.40, -1.30), (sg * 1.35, -0.38, -0.30),
                   (sg * (1.05 + curl * 0.2), -0.36, 0.65), (sg * 0.45, -0.36 - 0.3 * lift, 1.15)]
            rad = [0.24, 0.40, 0.52, 0.52, 0.40, 0.08]
        else:
            w = p.tail_wind
            base = [R, R + (0, 0.05 - 0.25 * lift, -0.65), R + (side * 0.35, 0.55 - 0.9 * lift, -1.35),
                    R + (side * 0.8, 1.10 - 1.4 * lift + 0.3 * curl, -1.85), R + (side * 1.1, 1.45 - 1.6 * lift + 0.6 * curl, -2.35)]
            if w:
                # stream with the wind (wind_dir in screen space is handled by side/lift from the caller)
                base = [b + np.array([0, -0.4 * w * k / 4, -0.3 * w * k / 4]) for k, b in enumerate(base)]
            pts = base
            rad = [0.24, 0.48, 0.62, 0.55, 0.12]
        return Tube(np.array(pts, dtype=np.float64), np.array(rad), 10)

    def _tail_fur(self, P, pt, key):
        """Fur tufts all along the tail outline, combed toward the tip."""
        rnd = vnoise.rng(*key)
        n = len(P)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        N = geom.outward_normals(P)
        C = pt.C[:, :2]
        tip = C[-1]
        ranges = []
        # skip the stretch next to the root (first and last 8% of the outline)
        k0, k1 = int(0.06 * n), int(0.94 * n)

        def lean_at(i):
            # direction toward the tail tip along the centre line
            q = P[i]
            j = int(np.argmin(np.sum((C - q) ** 2, axis=1)))
            j2 = min(j + 3, len(C) - 1)
            d = C[j2] - C[j] if j2 > j else tip - C[max(j - 1, 0)]
            return d / (np.linalg.norm(d) + 1e-9)

        idx_of_t = lambda tq: int(round((tq % (2 * math.pi)) / (2 * math.pi) * n)) % n
        ranges.append(dict(t0=t[k0], t1=t[k1 - 1], length=0.22, width=0.34, lean=lambda tq: lean_at(idx_of_t(tq)),
                           comb=1.3, env=lambda u: 0.55 + 0.45 * abs(math.sin(math.pi * u * 3.0 + 0.7))))
        ranges, key = self._windy(ranges, key)
        return cel.fur_edge(P, t, N, ranges, key, self.unit)

    # ---------------- ears
    def _ear(self, Xh, s):
        p = self.p
        k = 0 if s < 0 else 1
        sw = p.ear_swivel[k]
        fl = p.ear_flat[k]
        B = np.array([s * 0.50, -0.60, -0.12])
        U = geom.normalize(np.array([s * 0.34, -1.0, -0.10]))
        Nn = np.array([0.0, 0.0, 1.0])
        Nn = geom.normalize(Nn - U * np.dot(Nn, U))
        Nn = geom.rot_axis(U, s * deg(14)) @ Nn
        A = geom.normalize(np.cross(U, Nn))
        if np.dot(A, [s, 0, 0]) < 0:
            A = -A
        Rs = geom.rot_axis(U, -s * sw)
        Nn, A = Rs @ Nn, Rs @ A
        # anime cheat: an ear never goes edge-on to the camera; it turns just
        # enough about its own axis to keep a readable shape
        nz = (Xh.A @ Nn)[2] / np.linalg.norm(Xh.A @ Nn)
        zmin = 0.42
        if abs(nz) < zmin and not fl:
            sign = 1.0 if nz >= -0.05 else -1.0
            best = None
            for dd in np.linspace(0, math.pi / 2, 19):
                for sg in (1, -1):
                    Rt = geom.rot_axis(U, sg * dd)
                    n2 = Xh.A @ (Rt @ Nn)
                    z2 = n2[2] / np.linalg.norm(n2)
                    if z2 * sign >= zmin:
                        best = Rt
                        break
                if best is not None:
                    break
            if best is not None:
                Nn, A = best @ Nn, best @ A
        if fl:
            # pinned back and out to the sides, still readable beside the head
            Rf = geom.rot_axis(A, s * deg(58) * fl)
            U, Nn = Rf @ U, Rf @ Nn
            Rd = geom.rot_axis(np.array([0, 0, 1.0]), s * deg(48) * fl)
            U, Nn, A = Rd @ U, Rd @ Nn, Rd @ A
            B = B + np.array([s * 0.12, 0.12, -0.08]) * fl

        def lift(uv, cup=0.15):
            uv = np.asarray(uv, dtype=np.float64)
            u, v = uv[:, 0] * 1.08, uv[:, 1] * 0.93
            c = cup * np.clip(1 - (u / 0.45) ** 2, 0, 1) * np.clip(v, 0, 1.1)
            P3 = B + u[:, None] * A + v[:, None] * U - c[:, None] * Nn
            return Xh.apply(P3)

        inner_edge = geom.catmull_rom([(-0.40, -0.30), (-0.41, 0.12), (-0.36, 0.52), (-0.23, 0.90), (-0.06, 1.18), (0.03, 1.29)], 10)
        outer_edge = geom.catmull_rom([(0.03, 1.29), (0.13, 1.14), (0.28, 0.84), (0.39, 0.46), (0.43, 0.10), (0.42, -0.30)], 10)
        outline_uv = np.vstack([inner_edge, outer_edge[1:]])
        tip_idx = len(inner_edge) - 1
        O3 = lift(outline_uv)
        tip_uv = self._clip_above(outline_uv, 0.80, ("eartip", s, p.seed // 6))
        T3 = lift(tip_uv)
        inner_uv = np.vstack([
            geom.catmull_rom([(-0.28, -0.05), (-0.27, 0.33), (-0.19, 0.66), (-0.08, 0.88), (0.01, 0.97)], 8),
            geom.catmull_rom([(0.01, 0.97), (0.10, 0.88), (0.21, 0.64), (0.29, 0.34), (0.30, -0.05)], 8)[1:],
        ])
        I3 = lift(inner_uv, cup=0.17)
        tufts = [lift(t, cup=0.10)[:, :2] for t in self._ear_tufts(("eartuft", s, p.seed // 6))]
        nrm = Xh.A @ Nn
        nrm = nrm / np.linalg.norm(nrm)
        # plane for per-pixel depth: z = z0 + gx * (x - x0) + gy * (y - y0)
        c0 = O3.mean(axis=0)
        gx = -nrm[0] / nrm[2] if abs(nrm[2]) > 0.05 else 0.0
        gy = -nrm[1] / nrm[2] if abs(nrm[2]) > 0.05 else 0.0
        return dict(outline=O3[:, :2], tip=T3[:, :2], inner=I3[:, :2], tufts=tufts, front=nrm[2] > 0.02,
                    normal=nrm, depth=float(np.mean(O3[:, 2])), side=s, tip_idx=tip_idx, flat=fl,
                    plane=(c0, gx, gy), base=float(O3[0, 2]))

    @staticmethod
    def _clip_above(uv, v0, key):
        rnd = vnoise.rng(*key)
        top = uv[uv[:, 1] >= v0]
        xs = np.linspace(0.44, -0.44, 9)
        ys = v0 + rnd.uniform(-0.04, 0.05, len(xs)) + np.where(np.arange(len(xs)) % 2 == 0, -0.035, 0.045)
        return np.vstack([top, np.stack([xs, ys], 1)])

    @staticmethod
    def _ear_tufts(key):
        """Inner-ear fur: one fluffy clump at the base with a ragged top, plus two long wisps."""
        rnd = vnoise.rng(*key)
        out = []
        # clump: from the inner base, a ragged edge of 5 points curling toward the ear centre
        pts = [np.array([0.33, -0.15])]
        n = 4
        for i in range(n + 1):
            u = 0.30 - 0.60 * i / n
            v_base = 0.14 + 0.12 * math.sin(math.pi * i / n) + rnd.uniform(-0.03, 0.03)
            pts.append(np.array([u, v_base]))
            if i < n:
                um = u - 0.075 + rnd.uniform(-0.02, 0.02)
                peak = 0.36 + 0.24 * math.sin(math.pi * (i + 0.5) / n) + rnd.uniform(-0.06, 0.06)
                # each lock curls toward the middle of the ear
                curl = -0.06 * (um / 0.3)
                pts.append(np.array([um + curl * 0.5, (v_base + peak) * 0.5 + 0.02]))
                pts.append(np.array([um + curl, peak]))
        pts.append(np.array([-0.33, -0.15]))
        out.append(np.array(pts))
        # two wisps reaching up the ear
        for (b, h, lean) in ((-0.10, 0.80, 0.05), (0.08, 0.70, -0.06)):
            w = 0.045
            p0 = np.array([b - w, 0.25])
            p1 = np.array([b + w, 0.25])
            tip = np.array([b + lean, h + rnd.uniform(-0.04, 0.04)])
            ctrl_l = np.array([b - w * 0.4, (0.25 + h) * 0.55])
            ctrl_r = np.array([b + w * 0.9, (0.25 + h) * 0.5])
            ss = np.linspace(0, 1, 9)
            left = [(1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl_l + s * s * tip for s in ss]
            right = [(1 - s) ** 2 * tip + 2 * (1 - s) * s * ctrl_r + s * s * p1 for s in ss[1:]]
            out.append(np.array(left + right))
        return out

    # ---------------- eyes
    def _eye_frame(self, s, Xh):
        # anime cheat: as the head turns to profile, the near eye slides toward
        # the side and its plane turns with the face so it stays readable
        f = Xh.A @ np.array([0, 0, 1.0])
        f = f / np.linalg.norm(f)
        prof = 1.0 - abs(f[2])
        cr = CRANIUM["r"]
        x, y = s * (EYE_X + 0.14 * prof), EYE_Y
        z = cr[2] * math.sqrt(max(0.0, 1 - (x / cr[0]) ** 2 - (y / cr[1]) ** 2)) + 0.015
        E = np.array([x, y, z])
        ns = geom.normalize(np.array([x / cr[0] ** 2, y / cr[1] ** 2, z / cr[2] ** 2]))
        wf = 0.45 * max(f[2], 0.0) ** 0.5
        n = geom.normalize((1 - wf) * ns + wf * np.array([0, 0, 1.0]))
        u = np.array([1.0, 0, 0])
        u = geom.normalize(u - n * np.dot(u, n))
        v = np.cross(n, u)
        return E, n, u, v

    def _eye(self, Xh, s):
        p = self.p
        E, n, u, v = self._eye_frame(s, Xh)
        o = float(np.clip(p.eye_open, 0, 1))
        lower = LOWER.copy()
        lower[:, 1] -= (1 - o) * 0.045 * np.sin(np.linspace(0, math.pi, len(lower)))
        lo_d = geom.catmull_rom(lower, 10)
        if 0.08 < o < 0.98:
            # a half-closed lid keeps the soft arch of the open lid and just comes down,
            # evenly from both corners (a flat lid line reads as a glare in a still)
            up_d = geom.catmull_rom(UPPER_OPEN.copy(), 10)
            c0, c1 = up_d[0], up_d[-1]
            tt = (up_d[:, 0] - c0[0]) / (c1[0] - c0[0])
            chord = c0[1] + (c1[1] - c0[1]) * tt
            k = o ** 0.8
            up_d[:, 1] = chord + (up_d[:, 1] - chord) * k
            up_d[:, 1] -= p.worry * o * 0.02 * np.clip(-up_d[:, 0] / 0.23, 0, 1)
            lo_y = np.interp(up_d[:, 0], lo_d[::-1, 0], lo_d[::-1, 1])
            up_d[:, 1] = np.minimum(up_d[:, 1], lo_y - 0.015)
        else:
            upper = UPPER_CLOSED + (UPPER_OPEN - UPPER_CLOSED) * (1.0 if o >= 0.98 else 0.0)
            w = p.worry * (1.0 if o >= 0.98 else 0.0)
            upper = upper.copy()
            upper[:3, 1] -= w * np.array([0.035, 0.03, 0.012])
            up_d = geom.catmull_rom(upper, 10)
        ca, sa = math.cos(EYE_TILT), math.sin(EYE_TILT)

        def to3(ab, push=0.0):
            ab = np.asarray(ab, dtype=np.float64)
            a = ab[:, 0] * ca + ab[:, 1] * sa
            b = -ab[:, 0] * sa + ab[:, 1] * ca
            a = a * s * EYE_SCALE
            b = b * EYE_SCALE
            wrap = 0.55 * (a ** 2 + b ** 2)
            P3 = E + a[:, None] * u + b[:, None] * v - (wrap - push)[:, None] * n
            S = Xh.apply(P3)
            return S[:, :2], S[:, 2]

        opening, _ = to3(np.vstack([up_d, lo_d[1:-1]]))
        upper2, _ = to3(up_d)
        lower2, _ = to3(lo_d)
        gx, gy = p.gaze
        ic = np.array([gx * 0.075 * s, 0.0 + gy * 0.05])
        th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        ri = 0.188
        iris_ab = np.stack([ic[0] + ri * np.cos(th), ic[1] + ri * 1.10 * np.sin(th)], 1)
        iris, _ = to3(iris_ab, push=0.004)
        pr = (0.080, 0.096)
        pc = ic + np.array([0.0, 0.01])
        pupil, _ = to3(np.stack([pc[0] + pr[0] * np.cos(th), pc[1] + pr[1] * np.sin(th)], 1), push=0.006)
        cres, _ = to3(np.stack([ic[0] + ri * 0.86 * np.cos(th), ic[1] + 0.035 + ri * 0.80 * np.sin(th)], 1), push=0.005)
        band, _ = to3(np.vstack([up_d + np.array([0, -0.02]), (up_d + np.array([0, 0.085]))[::-1]]), push=0.007)
        centre, _ = to3(ic[None, :])
        nrm = Xh.A @ n
        nrm = nrm / np.linalg.norm(nrm)
        ri_px = float(np.mean(np.linalg.norm(iris - centre, axis=1)))
        return dict(opening=opening, upper=upper2, lower=lower2, iris=iris, pupil=pupil, cres=cres,
                    band=band, centre=centre[0], ri=ri_px, normal=nrm, open=o, side=s,
                    z=float(Xh.apply(E[None, :])[0, 2]))

    def _nose(self, Xh):
        p = self.p
        lift = 0.035 * p.sniff
        c = muzzle_point((0, -0.32, 1.0)) + np.array([0.0, -lift, 0.012])
        n = geom.normalize(np.array([0, -0.55, 1.0]))
        u = np.array([1.0, 0, 0])
        v = np.cross(n, u)
        sq = 1.0 + 0.10 * p.sniff
        pts = np.array([(-0.125, -0.035), (-0.095, -0.072), (0.0, -0.082), (0.095, -0.072), (0.125, -0.035),
                        (0.07, 0.035), (0.0, 0.07), (-0.07, 0.035)]) * np.array([sq, 1.0 / sq])
        d = geom.catmull_rom(pts, 8, closed=True)
        S = Xh.apply(c + d[:, 0:1] * u + d[:, 1:2] * v)
        hl = geom.catmull_rom(np.array([(-0.075, -0.05), (0.0, -0.068), (0.05, -0.055), (-0.01, -0.04)]), 6, closed=True)
        H = Xh.apply(c + hl[:, 0:1] * u + hl[:, 1:2] * v + n * 0.005)
        nrm = Xh.A @ n
        return dict(poly=S[:, :2], hl=H[:, :2], z=float(Xh.apply(c[None])[0, 2]), normal=nrm / np.linalg.norm(nrm))

    def _mouth(self, Xh):
        p = self.p
        lift = 0.03 * p.sniff
        philtrum = np.array([muzzle_point((0, -0.05, 1.0)) + [0, -lift, 0.01], muzzle_point((0, 0.30, 1.0)) + [0, 0, 0.01],
                             muzzle_point((0, 0.62, 0.8)) + [0, 0, 0.01]])
        mo = p.mouth_open
        left = np.array([muzzle_point((0, 0.62, 0.8)), muzzle_point((0.28, 0.70 + 0.1 * mo, 0.66)),
                         muzzle_point((0.52, 0.66, 0.52))]) + [0, 0, 0.01]
        right = left * np.array([-1, 1, 1])
        jaw = None
        if mo > 0.05:
            # open mouth: the lower jaw drops, a dark mouth shape shows
            jaw = np.array([muzzle_point((0.45, 0.70, 0.55)), muzzle_point((0.2, 0.95, 0.5)) + [0, 0.10 * mo, 0],
                            muzzle_point((0, 1.0, 0.6)) + [0, 0.16 * mo, 0], muzzle_point((-0.2, 0.95, 0.5)) + [0, 0.10 * mo, 0],
                            muzzle_point((-0.45, 0.70, 0.55))])
            jaw = Xh.apply(jaw)
        return dict(philtrum=Xh.apply(philtrum), left=Xh.apply(left), right=Xh.apply(right), jaw=jaw)

    def _whiskers(self, Xh):
        p = self.p
        rnd = vnoise.rng("whisk", p.seed // 4)
        out = []
        tw = 0.10 * p.sniff
        for s in (-1, 1):
            for k, (a, l) in enumerate([(0.12, 0.70), (-0.06, 0.80), (-0.26, 0.62)]):
                base = muzzle_point((s * (0.62 + 0.04 * k), 0.12 + 0.13 * k, 0.75))
                ang = a + tw + rnd.uniform(-0.03, 0.03)
                d1 = np.array([s * math.cos(ang), -math.sin(ang) * 0.6 + 0.12, 0.35])
                d1 = d1 / np.linalg.norm(d1)
                pts = np.array([base + d1 * l * f + np.array([0, 0.10 * f * f, -0.15 * f * f]) for f in np.linspace(0, 1, 8)])
                out.append(Xh.apply(pts))
        dots = []
        for s in (-1, 1):
            for k in range(3):
                dots.append(Xh.apply(muzzle_point((s * (0.45 + 0.12 * k), 0.10 + 0.10 * (k == 1), 0.85 - 0.1 * k))[None, :] + [0, 0, 0.01])[0])
        return dict(lines=out, dots=dots)

    # ---------------- lantern
    def _lantern(self, Xh):
        """A small paper lantern on a bamboo stick held in the mouth."""
        p = self.p
        bite = muzzle_point((0.0, 0.75, 0.55))
        stick_end = bite + np.array([0.0, 0.42, 0.95])
        stick = Xh.apply(np.array([bite + [0, 0, -0.05], stick_end]))
        top = stick_end + np.array([0, 0.30, 0.0])
        cord = Xh.apply(np.array([stick_end, top]))
        centre = top + np.array([0, 0.48, 0.0])
        C = Xh.apply(centre[None, :])[0]
        unit = self.unit
        # the lantern hangs straight down in screen space, whatever the head does
        rx, ry = 0.40 * unit, 0.48 * unit
        t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        body = np.stack([C[0] + rx * np.cos(t), C[1] + ry * np.sin(t) * (1 - 0.08 * np.cos(t) ** 2)], 1)
        T = Xh.apply(top[None, :])[0]
        return dict(stick=stick, cord=cord, centre=C, rx=rx, ry=ry, body=body, top=T,
                    depth=float(C[2]))

    # -------------------------------------------------------------- patterns
    def _head_pattern(self, pl, u, nl=None):
        x, y, z = pl
        pal = self.ctx.pal
        ax = np.abs(x)
        s = np.clip((ax - 0.36) / 0.62, 0, 1)
        s = s * s * (3 - 2 * s)
        yb = 0.25 - 0.33 * s
        q = ax * 6.5 + 0.35
        t = q % 1.0
        k = np.floor(q)
        amp = 0.045 + 0.05 * (np.sin(k * 12.9898 + 4.1) * 43758.5453 % 1.0)
        tooth = np.where(t < 0.7, t / 0.7, (1.0 - t) / 0.3)
        yb = yb + amp * tooth - 0.03
        back = np.clip(-z - 0.05, 0, None)
        cream = y > yb + 0.9 * back
        cream |= (y > 0.66)
        return np.where(cream, pal["cream"], pal["orange"]).astype(np.int16)

    def _muzzle_pattern(self, pl, u, nl=None):
        pal = self.ctx.pal
        ux, uy, uz = u
        orange = uy < (0.08 - 0.30 * (uz + 1) * 0.5)
        return np.where(orange, pal["orange"], pal["cream"]).astype(np.int16)

    def _cream_pattern(self, pl, u, nl=None):
        return np.full(u[0].shape, self.ctx.pal["cream"], np.int16)

    def _body_pattern(self, name):
        """Body markings decided on the fox-space surface normal, so they stay put in any stance."""
        sit = self.p.stance == "sit"

        def f(pl, u, nl):
            pal = self.ctx.pal
            ux, uy, uz = u
            nx, ny, nz = nl
            # ragged fur edge: a sawtooth along the boundary
            saw = ((np.abs(nx) * 4.0 + ny * 3.0 + 10.0) % 1.0)
            tooth = np.where(saw < 0.7, saw / 0.7, (1.0 - saw) / 0.3) * 0.12
            if name == "neck":
                if self.p.body == "bust":
                    cream = uz > -0.55 + 0.1 * ((np.abs(ux) * 5) % 1.0)
                else:
                    cream = nz > -0.25 - tooth
            elif name == "chest":
                if self.p.body == "bust":
                    tri = np.abs(((uy * 5.0 + np.abs(ux) * 1.5) % 1.0) - 0.5) * 2.0
                    lim = 0.56 - 0.20 * np.clip(uy, 0, 1) + 0.07 * tri ** 2
                    cream = (np.abs(ux) < lim) & (uz > -0.35)
                else:
                    cream = (nz > 0.32 - tooth) & (np.abs(nx) < 0.72 + tooth)
            elif name == "barrel":
                cream = (ny > 0.55 - tooth) | ((nz > 0.62 - tooth) & (np.abs(nx) < 0.55) if sit else False)
            else:
                cream = ny > 0.78 - tooth
            return np.where(cream, pal["cream"], pal["orange"]).astype(np.int16)
        return f

    def _leg_pattern(self, sock_s):
        def f(s, side):
            pal = self.ctx.pal
            b = sock_s + 0.04 * ((np.abs(side) * 3.0 + 0.2) % 1.0) - 0.03 * np.abs(side)
            return np.where(s > b, pal["sock"], pal["orange"]).astype(np.int16)
        return f

    def _sock_pattern(self, pl, u, nl=None):
        return np.full(u[0].shape, self.ctx.pal["sock"], np.int16)

    def _tail_pattern(self, s, side):
        pal = self.ctx.pal
        a = np.clip(np.abs(side), 0, 1)
        b = 0.80 - 0.05 * (1 - a) + 0.02 * np.abs(np.sin(side * 4.5))
        return np.where(s > b, pal["cream"], pal["orange"]).astype(np.int16)

    # -------------------------------------------------------------- painting
    def paint(self, cv):
        p = self.p
        groups = self.build(cv.xform())
        ss = cv.ss
        unit = self.unit
        L = self.L
        kz = 0.30 if self.lname in ("hall", "night_out") else 0.55
        if self.lname in ("sunrise", "gold_burst") and math.cos(p.yaw) < -0.3:
            kz = -0.40      # the fox faces the sun: backlit, a dark shape with a gold rim
        key3 = light3(L, z=kz)
        kd = np.array(L["key_dir"], dtype=np.float64)
        rim_dir = np.array([-kd[0], kd[1] * 0.6 - 0.4])
        rim_w = 0.011
        if self.lname in ("sunrise", "gold_burst"):
            # the sun is the rim: a gold edge on the side toward it, wider when seen from behind
            rim_dir = kd.copy()
            rim_w = 0.03 + 0.09 * max(0.0, -math.cos(p.yaw))
        if self.rim_override is not None:
            rim_dir, rim_w = np.asarray(self.rim_override[0], dtype=np.float64), self.rim_override[1]
        rim_dir /= np.linalg.norm(rim_dir)
        self.rim_w = rim_w
        lantern = None
        if self.lantern_geo is not None:
            C = self.lantern_geo["centre"]
            lantern = (C[0], C[1], C[2] + 0.3 * unit)
            self.lantern_reach = self.lantern_reach or 9.0 * unit / ss
        elif self.lantern_screen is not None:
            lx, ly, lz = self.lantern_screen
            lantern = ((lx - cv.x0) * ss, (ly - cv.y0) * ss, lz * ss)
        designed = {k: dict(v) for k, v in DESIGNED.get(self.lname, {}).items()}
        for k, v in self.pal_override.items():
            designed.setdefault(k, {}).update(v)
        pal = cel.Palette(ALBEDO, L, self.lname, designed=designed)
        self.ctx = cel.LightCtx(L=L, name=self.lname, key3=key3, rim_dir=rim_dir, lantern=lantern,
                                lantern_reach=self.lantern_reach * ss, pal=pal, hard_warm=self.hard_warm)
        self.lw = (p.line if p.line else 0.235 * math.sqrt(p.scale) + 0.35) * ss
        H, W = cv.H, cv.W
        self.zbuf = np.full((H, W), -1e9, dtype=np.float32)
        self.warmbuf = np.zeros((H, W), dtype=np.float32)
        self.matbuf = np.full((H, W), -1, dtype=np.int16)
        self.gid = np.full((H, W), -1, dtype=np.int16)

        # masks of every part (for the rim and cast shadows)
        total = np.zeros((H, W), dtype=bool)
        for g in groups:
            for part in g["parts"]:
                part["box"], part["m"] = cv.mask(part["poly"])
                if part["box"] is not None:
                    y0, y1, x0, x1 = part["box"]
                    total[y0:y1, x0:x1] |= part["m"]
            if g["name"] == "ear":
                e = g["ear"]
                e["box"], e["m"] = cv.mask(e["outline"])
                if e["box"] is not None:
                    y0, y1, x0, x1 = e["box"]
                    total[y0:y1, x0:x1] |= e["m"]
        self.rim = cel.rim_mask(total, rim_dir, max(2.0, self.rim_w * unit))
        head = groups[0]
        hm = np.zeros((H, W), dtype=bool)
        if head["name"] == "head":
            for part in head["parts"]:
                if part["box"] is not None:
                    y0, y1, x0, x1 = part["box"]
                    hm[y0:y1, x0:x1] |= part["m"]
        co = 0.20 if p.body == "bust" else 0.16
        self.cast = cel.cast_shadow(hm, -kd[0] * co * unit, -kd[1] * co * unit + 0.08 * unit)

        # ears behind the head go first, ears in front (back views) after it
        hd = head["depth"]
        order = sorted(range(len(groups)), key=lambda i: self._group_depth(groups[i], hd))
        for gi in order:
            g = groups[gi]
            if g["name"] == "head":
                self._paint_head(cv, g, gi)
            elif g["name"] == "ear":
                self._paint_ear(cv, g["ear"], gi)
            elif g["name"] == "lantern":
                self._paint_lantern(cv)
            else:
                self._paint_group(cv, g, gi)
        self._warm_pass(cv)
        return cv

    def _warm_pass(self, cv):
        """Photography pass: the top of the head falls off away from the light (a soft
        gradient over the cel), then the lantern adds a broad warm wash on its side."""
        if self.groups and self.groups[0]["name"] == "head" and self.p.body in ("bust", "full"):
            hc = self.Xh.apply(np.zeros((1, 3)))[0]
            up = self.Xh.A @ np.array([0.0, -1.0, 0.0])
            up2 = up[:2] / (np.linalg.norm(up[:2]) + 1e-9)
            heads = np.isin(self.gid, [i for i, g in enumerate(self.groups) if g["name"] in ("head", "ear")])
            rows = np.flatnonzero(heads.any(axis=1))
            if len(rows):
                y0, y1 = rows[0], rows[-1] + 1
                ys, xs = np.mgrid[y0:y1, 0:cv.W].astype(np.float32)
                h = ((xs - hc[0]) * up2[0] + (ys - hc[1]) * up2[1]) / self.unit
                kd = np.array(self.L["key_dir"], np.float32)
                side = ((xs - hc[0]) * kd[0] + (ys - hc[1]) * kd[1]) / self.unit
                g = np.clip((h - 0.15) / 1.2, 0, 1) ** 1.3 * (0.75 - 0.35 * np.clip(side, -1, 1))
                dark = 1.0 - 0.17 * g * heads[y0:y1]
                cv.rgb[y0:y1] *= dark[..., None]
        if self.ctx.lantern is None or not self.warmbuf.any():
            return
        from scipy import ndimage
        sig = 0.10 * self.unit
        small = self.warmbuf[::4, ::4]
        wb = ndimage.gaussian_filter(small, sig / 4.0)
        wb = np.repeat(np.repeat(wb, 4, axis=0), 4, axis=1)[: cv.H, : cv.W]
        warm = np.asarray(self.L.get("warm") or (1.0, 0.72, 0.4), dtype=np.float32)
        k = (wb ** 1.3 * (0.12 if self.p.body == "bust" else 0.40))[..., None] * cv.a[..., None]
        # screen blend toward the lantern colour, multiplied by the paint so dark lines stay dark
        base = cv.rgb
        lift = base * warm * 1.6 - base
        cv.rgb[...] = base + np.clip(lift, 0, None) * k + warm * k * 0.10

    def _group_depth(self, g, hd):
        if g["name"] == "ear":
            e = g["ear"]
            # the ear base is buried in the head; it goes over the head only when
            # the head is seen from behind
            return hd + (0.05 if e["depth"] > hd + 0.1 * self.unit else -0.6) * self.unit
        if g["name"] == "head":
            return hd
        if g["name"] == "body":
            return min(g["depth"], hd - 0.3 * self.unit) if self.p.body == "bust" else g["depth"]
        if g["name"] == "lantern":
            return g["depth"]
        if g["name"] == "ruff":
            body = [gg for gg in self.groups if gg["name"] == "body"][0]
            return self._group_depth(body, hd) + 0.01 * self.unit
        return g["depth"]

    # painting helpers -----------------------------------------------------
    def _eval_part(self, cv, part, box, m):
        """Surface samples of one part at its mask pixels: z, n, mat, noise key, xs, ys."""
        xs, ys = cv.grid(box)
        xs, ys = xs[m], ys[m]
        if part["kind"] == "ell":
            pe = part["pe"]
            valid, z, n, u = pe.surface(xs, ys)
            ul = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) + 1e-12
            u = [u[0] / ul, u[1] / ul, u[2] / ul]
            n[2] = np.where(valid, n[2], 0.0)
            pl = pe.local_point(u)
            R = pe.src.rot
            rr = pe.src.r
            g0, g1, g2 = u[0] / rr[0], u[1] / rr[1], u[2] / rr[2]
            nl = [R[k, 0] * g0 + R[k, 1] * g1 + R[k, 2] * g2 for k in range(3)]
            nn = np.sqrt(nl[0] ** 2 + nl[1] ** 2 + nl[2] ** 2) + 1e-12
            nl = [nl[0] / nn, nl[1] / nn, nl[2] / nn]
            mat = part["pattern"](pl, u, nl)
            nkey = pl
        else:
            z, n, sp, side = part["pt"].surface(xs, ys)
            mat = part["pattern"](sp, side)
            nkey = [sp * 4.0, side, np.zeros_like(sp)]
        return z, n, mat, nkey, xs, ys

    def _fill_group(self, cv, g, gi, extra_shadow=True):
        """Z-tested cel fill of a whole group as one form.

        Where masses meet, their normals blend by depth (a smooth union), so the
        shadow shape runs across the join the way an animator draws one form,
        instead of switching surface by surface.
        """
        parts = [pt for pt in g["parts"] if pt.get("box") is not None]
        if not parts:
            return
        Y0 = min(pt["box"][0] for pt in parts)
        Y1 = max(pt["box"][1] for pt in parts)
        X0 = min(pt["box"][2] for pt in parts)
        X1 = max(pt["box"][3] for pt in parts)
        hh, ww = Y1 - Y0, X1 - X0
        zmax = np.full((hh, ww), -1e9, np.float32)
        mat = np.zeros((hh, ww), np.int16)
        bias = np.zeros((hh, ww), np.float32)
        key = np.zeros((3, hh, ww), np.float32)
        samples = []
        for pt in parts:
            y0, y1, x0, x1 = pt["box"]
            m = pt["m"]
            z, n, mt, nk, xs, ys = self._eval_part(cv, pt, pt["box"], m)
            sub = (slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
            zm = zmax[sub]
            cur = zm[m]
            front = z > cur
            idx_y, idx_x = np.nonzero(m)
            fy, fx = idx_y[front] + (y0 - Y0), idx_x[front] + (x0 - X0)
            zmax[fy, fx] = z[front]
            mat[fy, fx] = mt[front]
            bias[fy, fx] = pt.get("bias", 0.0)
            for k in range(3):
                key[k, fy, fx] = nk[k][front]
            samples.append((idx_y + (y0 - Y0), idx_x + (x0 - X0), z, n))
        kz = g.get("kz", 0.10) * self.unit
        acc = np.zeros((3, hh, ww), np.float32)
        wsum = np.zeros((hh, ww), np.float32)
        for (iy, ix, z, n) in samples:
            # pixel indices are unique within one part, so plain fancy-index adds are exact
            w = np.exp(np.clip((z - zmax[iy, ix]) / kz, -30, 0)).astype(np.float32)
            for k in range(3):
                acc[k][iy, ix] += w * n[k]
            wsum[iy, ix] += w
        cov = wsum > 0
        nb = acc / np.maximum(wsum, 1e-9)
        nl = np.sqrt((nb ** 2).sum(0)) + 1e-9
        nb /= nl
        zbw = self.zbuf[Y0:Y1, X0:X1]
        vis = cov & (zmax >= zbw - 0.5)
        if not vis.any():
            return
        iy, ix = np.nonzero(vis)
        n = [nb[0][vis], nb[1][vis], nb[2][vis]]
        if g["name"] == "head" and self.p.body == "bust":
            # the white cheek fur under the eyes takes the same designed top light
            cm = (mat[vis] == self.ctx.pal["cream"]) & (key[1][vis] > 0.30)
            ny = np.where(cm, n[1] * 0.35, n[1])
            nl = np.sqrt(n[0] ** 2 + ny ** 2 + n[2] ** 2) + 1e-9
            n = [n[0] / nl, ny / nl, n[2] / nl]
        if self.p.body == "bust" and g["name"] in ("body", "ruff"):
            # designed lighting for the ruff: long fur catches the top light, so the
            # downward lean of the surface counts for little; the far side still turns away
            ny = n[1] * 0.35
            nl = np.sqrt(n[0] ** 2 + ny ** 2 + n[2] ** 2) + 1e-9
            n = [n[0] / nl, ny / nl, n[2] / nl]
        z = zmax[vis]
        xs = ix + X0 + 0.5
        ys = iy + Y0 + 0.5
        kk = [key[0][vis], key[1][vis], key[2][vis]]
        # the terminator noise is smooth, so evaluate it on a coarse grid and interpolate
        st = 6
        ks = key[:, ::st, ::st]
        ncoarse = 0.05 * vnoise.value3(ks[0] * 2.3, ks[1] * 2.3, ks[2] * 2.3, seed=11) \
            + 0.02 * vnoise.value3(ks[0] * 3.1, ks[1] * 3.1, ks[2] * 3.1 + self.p.seed * 0.37, seed=7)
        from scipy import ndimage as _nd
        nfull = _nd.zoom(ncoarse.astype(np.float32), (hh / ncoarse.shape[0], ww / ncoarse.shape[1]), order=1)
        nfull = nfull[:hh, :ww]
        if nfull.shape != (hh, ww):
            nfull = np.pad(nfull, ((0, hh - nfull.shape[0]), (0, ww - nfull.shape[1])), mode="edge")
        thr = 0.02 + nfull[vis] + bias[vis]
        if g["name"] == "head":
            # designed shadows: a sweep from the forehead back toward the ears, and a
            # band under the cheek fur. kk holds head-local positions here.
            hx, hy, hz = kk
            back = np.clip((0.42 - hz) / 0.55, 0, 1) * np.clip((0.25 - hy) / 0.6, 0, 1)
            sweep = 0.10 * np.sin(np.abs(hx) * 6.0 + hy * 4.0)
            cheek = np.clip((hy - 0.64) / 0.10, 0, 1) * np.clip((np.abs(hx) - 0.30) / 0.3, 0, 1)
            thr = thr + 0.42 * np.clip(back + sweep * back, 0, 1) + 0.45 * cheek
        # the hard warm tone is for the face only; on the ruff the lantern reads
        # better as the soft compositing wash (hard cuts there made blobs)
        hw = self.ctx.hard_warm
        if g["name"] not in ("head", "ruff"):
            self.ctx.hard_warm = 0.0
        elif g["name"] == "ruff" and hw > 0:
            self.ctx.hard_warm = hw - 0.22   # clumps facing the lantern catch it sooner, as hard shapes
        tone = cel.tone_index(n, (xs, ys, z), self.ctx, thr=thr)
        self.ctx.hard_warm = hw
        ndw, fall = cel.lantern_terms(n, (xs, ys, z), self.ctx)
        if extra_shadow:
            sh = self.cast[Y0:Y1, X0:X1][vis]
            tone = np.where(sh & (tone == cel.LIT), cel.SHADE, tone)
            tone = np.where(sh & (tone == cel.WARM), cel.WARM_SHADE, tone)
        r = self.rim[Y0:Y1, X0:X1][vis]
        tone = np.where(r, cel.RIM, tone)
        idx = mat[vis].astype(np.int32) * cel.NTONE + tone
        cols = self.ctx.pal.table[idx]
        if g["name"] == "ruff":
            # each clump's underside falls into a deeper shade, so the clumps read as forms
            under = (n[1] > 0.38) & ((tone == cel.SHADE) | (tone == cel.WARM_SHADE))
            cols = np.where(under[:, None], cols * np.array([0.80, 0.80, 0.86], np.float32), cols)
        cv.rgb[Y0:Y1, X0:X1][vis] = cols
        cv.a[Y0:Y1, X0:X1][vis] = 1.0
        self.matbuf[Y0:Y1, X0:X1][vis] = mat[vis]
        zbw[vis] = z
        self.gid[Y0:Y1, X0:X1][vis] = gi
        if ndw is not None:
            self.warmbuf[Y0:Y1, X0:X1][vis] = (np.clip(ndw * 0.7 + 0.3, 0, 1) * fall).astype(np.float32)

    def _group_mask(self, cv, gi):
        return self.gid == gi

    def _pen(self, cv, mult, **kw):
        args = dict(width=self.lw * mult, light=tuple(self.ctx.key3[:2]), seed=self.p.seed,
                    min_len=70 * cv.ss, max_len=230 * cv.ss, wobble=0.35 * cv.ss, taper=0.18,
                    shadow_gain=0.85, lit_thin=0.62, break_lit=0.05)
        args.update(kw)
        return ink.Pen(**args)

    def _ink_mask(self, cv, m, gi, mult=1.0, colour=None, depth=None, soft_against=None, keep_tips=False):
        """Ink the silhouette of bool mask m (canvas size) as hand-drawn strokes.

        soft_against: group ids; where this group's cream fur lies over cream fur
        of those groups, the line drops out except at the tuft tips, the way an
        animator draws fur over fur.
        """
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if len(rows) == 0:
            return
        pad = int(self.lw * 2) + 4
        bx = (max(0, rows[0] - pad), min(cv.H, rows[-1] + pad + 1), max(0, cols[0] - pad), min(cv.W, cols[-1] + pad + 1))
        y0, y1, x0, x1 = bx
        polys = contour.mask_contours(m[y0:y1, x0:x1], bx, blur=0.9, min_area=(0.03 * self.unit) ** 2 + 30)
        buf = cv.stroke_buffer(bx)
        pen = self._pen(cv, mult)
        cream = self.ctx.pal["cream"]
        H, W = self.gid.shape
        for P in polys:
            P = geom.resample(P, 1.2 * cv.ss, closed=True)
            if len(P) < 6:
                continue
            tips = self._find_tips(P)
            N = geom.outward_normals(P)
            qo = P + N * 3.0 * cv.ss
            qi = P - N * 3.0 * cv.ss
            ox_ = np.clip(qo[:, 0].astype(int), 0, W - 1)
            oy_ = np.clip(qo[:, 1].astype(int), 0, H - 1)
            ix_ = np.clip(qi[:, 0].astype(int), 0, W - 1)
            iy_ = np.clip(qi[:, 1].astype(int), 0, H - 1)
            g_out = self.gid[oy_, ox_]
            # a line is drawn where the depth jumps: against empty paper, or where
            # what lies beyond is clearly behind. Where two masses meet at the same
            # depth (a leg growing out of the chest) the line drops out.
            z_in = self.zbuf[iy_, ix_]
            z_out = self.zbuf[oy_, ox_]
            gap = (z_in - z_out) > 0.10 * self.unit
            vis = (g_out < 0) | (g_out == gi) | gap
            if soft_against:
                other = np.isin(g_out, soft_against)
                soft = other & (self.matbuf[oy_, ox_] == cream) & (self.matbuf[iy_, ix_] == cream)
                vis &= ~soft
            # tufts always keep their little strokes
            near_tip = np.zeros(len(P), dtype=bool)
            for t in tips:
                for d in range(-int(5 * cv.ss / 1.2), int(3 * cv.ss / 1.2) + 1):
                    near_tip[(t + d) % len(P)] = True
            if soft_against and not keep_tips:
                vis |= near_tip & ~soft
            else:
                vis |= near_tip
            ink.ink_polyline(buf, P, pen, ox=x0, oy=y0, closed=True, tips=tips, visible=vis)
        # hide line pixels that fall on a nearer group drawn earlier
        if depth is not None:
            zb = self.zbuf[y0:y1, x0:x1]
            own = self.gid[y0:y1, x0:x1] == gi
            ok = own | (zb < depth)
            buf *= ok
        cv.paint(bx, buf, INK if colour is None else colour)

    @staticmethod
    def _find_tips(P, ang=0.9):
        a = np.roll(P, 3, axis=0) - P
        b = np.roll(P, -3, axis=0) - P
        cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        sharp = cosang > math.cos(math.pi - ang * 1.6)
        out = []
        for i in np.flatnonzero(sharp):
            if cosang[i] >= cosang[(i - 1) % len(P)] and cosang[i] >= cosang[(i + 1) % len(P)]:
                out.append(i)
        return out

    def _paint_group(self, cv, g, gi):
        self._fill_group(cv, g, gi)
        if g["name"] == "ruff":
            self._clump_lines(cv, g, gi)
            return
        self._ink_mask(cv, self._group_mask(cv, gi), gi, depth=g["depth"] + 0.15 * self.unit)
        if g["name"] == "body" and self.p.body == "bust":
            self._chest_fur_lines(cv, g, gi)

    def _paint_head(self, cv, g, gi):
        self._fill_group(cv, g, gi, extra_shadow=False)
        hm = self._group_mask(cv, gi)
        self.head_vis = hm
        eyes = sorted(self.feat["eyes"], key=lambda e: e["z"])
        for e in eyes:
            self._paint_eye(cv, e)
        body_ids = [i for i, gg in enumerate(self.groups) if gg["name"] == "body"]
        self._ink_mask(cv, hm, gi, depth=g["depth"] + 0.3 * self.unit, soft_against=body_ids)
        self._muzzle_line(cv, gi)
        self._fur_strands(cv, gi)
        for e in eyes:
            self._paint_lids(cv, e)
        self._paint_nose_mouth(cv)
        self._paint_whiskers(cv)

    def _clump_lines(self, cv, g, gi):
        """The lower, tufted edge of each ruff clump in a soft coloured line."""
        down2 = self._dir2(self.Xc, (0, 1, -0.25))
        mine = self.gid == gi
        allP = [pt["poly"] for pt in g["parts"] if pt.get("box") is not None]
        if not allP:
            return
        box = cv.bbox(allP, pad=int(self.lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        buf = cv.stroke_buffer(box)
        H, W = self.gid.shape
        pen = self._pen(cv, 0.5, taper=0.35, min_len=25 * cv.ss, max_len=90 * cv.ss, break_lit=0.0)
        for pt in g["parts"]:
            P = pt["poly"]
            if len(P) < 6:
                continue
            P = geom.resample(P, 1.2 * cv.ss, closed=True)
            N = geom.outward_normals(P)
            q = P - N * 2.0 * cv.ss
            ix = np.clip(q[:, 0].astype(int), 0, W - 1)
            iy = np.clip(q[:, 1].astype(int), 0, H - 1)
            vis = ((N @ down2) > 0.25) & mine[iy, ix]
            if vis.sum() < 4:
                continue
            ink.ink_polyline(buf, P, pen, ox=x0, oy=y0, closed=True, visible=vis, tips=self._find_tips(P))
        col = self.ctx.pal.colour("cream", cel.SHADE) * 0.62 + INK * 0.38
        cv.paint(box, buf * 0.85, col)

    def _fur_strands(self, cv, gi):
        """A few small fur tufts on the forehead and cheeks: two short strokes meeting
        in a point, combed back, in a coloured line a step darker than the fur."""
        if self.unit < 60 * cv.ss:
            return
        Xh = self.Xh
        cr = CRANIUM["r"]
        jc, jr = np.asarray(JOWL["c"]), np.asarray(JOWL["r"])

        def on_cranium(px, py):
            return cr[2] * math.sqrt(max(0.0, 1 - (px / cr[0]) ** 2 - (py / cr[1]) ** 2)) + 0.012

        def on_jowl(px, py):
            q = 1 - ((px - jc[0]) / jr[0]) ** 2 - ((py - jc[1]) / jr[1]) ** 2
            return jc[2] + jr[2] * math.sqrt(max(0.0, q)) + 0.012

        tufts = []
        # forehead: three tufts above the eyes, pointing back over the crown
        for (x, y, w, dx, dy) in [(-0.16, -0.50, 0.045, -0.02, -0.13), (0.02, -0.56, 0.05, 0.0, -0.14),
                                  (0.19, -0.49, 0.045, 0.03, -0.12)]:
            tufts.append((x, y, w, dx, dy, on_cranium, "orange"))
        for s in (-1, 1):
            for (x, y, w, dx, dy) in [(0.72, 0.32, 0.045, 0.07, 0.12), (0.86, 0.20, 0.04, 0.08, 0.10)]:
                tufts.append((s * x, y, w, s * dx, dy, on_jowl, "cream"))
        strokes = []
        for (x, y, w, dx, dy, surf, mat) in tufts:
            tip = (x + dx, y + dy)
            for side in (-1, 1):
                b0 = (x + side * w, y)
                ctrl = (x + side * w * 0.35 + dx * 0.45, y + dy * 0.45)
                pts = []
                for f in np.linspace(0, 1, 6):
                    px = (1 - f) ** 2 * b0[0] + 2 * (1 - f) * f * ctrl[0] + f * f * tip[0]
                    py = (1 - f) ** 2 * b0[1] + 2 * (1 - f) * f * ctrl[1] + f * f * tip[1]
                    pts.append((px, py, surf(px, py)))
                S = Xh.apply(np.array(pts))
                if self._z_ok(S, 0.05 * self.unit).all():
                    strokes.append((S, mat))
        if not strokes:
            return
        box = cv.bbox([S[:, :2] for S, _ in strokes], pad=int(self.lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        pal = self.ctx.pal
        for mat in ("orange", "cream"):
            buf = cv.stroke_buffer(box)
            for S, m in strokes:
                if m != mat:
                    continue
                ink.ink_designed(buf, S[:, :2], self.lw * 0.40,
                                 profile=lambda u: (0.35 + 0.65 * np.clip(u / 0.3, 0, 1)) * np.clip((1 - u) / 0.45, 0, 1) ** 0.8,
                                 ox=x0, oy=y0)
            col = pal.colour(mat, cel.SHADE) * 0.75 + INK * 0.25
            cv.paint(box, buf * self.head_vis[y0:y1, x0:x1] * 0.9, col)

    def _z_ok(self, S, eps):
        """Which 3D canvas points are not hidden behind what is already painted."""
        H, W = self.zbuf.shape
        ix = np.clip(S[:, 0].astype(int), 0, W - 1)
        iy = np.clip(S[:, 1].astype(int), 0, H - 1)
        return S[:, 2] >= self.zbuf[iy, ix] - eps

    def _paint_ear(self, cv, e, gi):
        lw = self.lw
        pal = self.ctx.pal
        box, m = e["box"], e["m"]
        if box is None:
            return
        y0, y1, x0, x1 = box
        xs, ys = cv.grid(box)
        c0, gx, gy = e["plane"]
        zpl = c0[2] + gx * (xs - c0[0]) + gy * (ys - c0[1])
        zb = self.zbuf[y0:y1, x0:x1]
        m = m & (zpl >= zb - 0.5)
        if not m.any():
            return
        tone = cel.LIT if float(np.dot(e["normal"] if e["front"] else -e["normal"], self.ctx.key3)) > -0.05 else cel.SHADE
        cv.paint(box, m.astype(np.float32), pal.colour("orange", tone))
        if e["front"]:
            sh = self._ear_half_shadow(cv, e, box)
            cv.paint(box, (m & sh).astype(np.float32), pal.colour("orange", cel.SHADE))
        r = self.rim[y0:y1, x0:x1] & m
        cv.paint(box, r.astype(np.float32), pal.colour("orange", cel.RIM))
        full = np.zeros(self.zbuf.shape, dtype=bool)
        full[y0:y1, x0:x1] = m
        self._fill_poly(cv, e["tip"], pal.colour("sock", tone), clip=full)
        if e["front"]:
            self._fill_poly(cv, e["inner"], pal.colour("inner", cel.SHADE), clip=full)
            for k, t in enumerate(e["tufts"]):
                # fur inside the ear sits in the ear's own shade; it only catches a little light
                lit_amt = 0.45 if (tone != cel.SHADE and k == 0) else 0.15
                col = pal.colour("cream", cel.SHADE) * (1 - lit_amt) + pal.colour("cream", cel.LIT) * lit_amt
                self._fill_poly(cv, t, col, clip=full)
            buf = cv.stroke_buffer(box)
            ink.ink_polyline(buf, e["inner"], self._pen(cv, 0.5, taper=0.3, min_len=40 * cv.ss, max_len=140 * cv.ss),
                             ox=x0, oy=y0, closed=True)
            clump = e["tufts"][0]
            ink.ink_polyline(buf, clump[1:-1], self._pen(cv, 0.42, taper=0.3, min_len=30 * cv.ss, max_len=90 * cv.ss, break_lit=0),
                             ox=x0, oy=y0, closed=False, tips=list(range(2, len(clump) - 2, 2)))
            for t in e["tufts"][1:]:
                half = t[: len(t) // 2 + 1]
                ink.ink_designed(buf, half, lw * 0.32, profile=lambda u: np.sin(np.clip(u, 0, 1) * math.pi * 0.95) ** 0.8,
                                 ox=x0, oy=y0)
            cv.paint(box, buf * m, INK * 0.85 + pal.colour("inner", cel.SHADE) * 0.15)
        zb[m] = zpl[m]
        self.gid[y0:y1, x0:x1][m] = gi
        buf = cv.stroke_buffer(box)
        ink.ink_polyline(buf, e["outline"], self._pen(cv, 1.0), ox=x0, oy=y0, closed=True, tips=[e["tip_idx"]])
        # only where the ear is visible (its base is buried in the head)
        vis = self.gid[y0:y1, x0:x1] == gi
        grow = vis | raster.shift(vis, 2, 0) | raster.shift(vis, -2, 0) | raster.shift(vis, 0, 2) | raster.shift(vis, 0, -2)
        free = (self.gid[y0:y1, x0:x1] < 0) | grow
        cv.paint(box, buf * free, INK)

    def _ear_half_shadow(self, cv, e, box):
        P = e["outline"]
        base = 0.5 * (P[0] + P[-1])
        tip = P[e["tip_idx"]]
        axis = tip - base
        nrm = np.array([axis[1], -axis[0]])
        nrm /= np.linalg.norm(nrm) + 1e-9
        if np.dot(nrm, self.ctx.key3[:2]) > 0:
            nrm = -nrm
        xs, ys = cv.grid(box)
        d = (xs - base[0]) * nrm[0] + (ys - base[1]) * nrm[1]
        return d > -0.02 * self.unit

    def _fill_poly(self, cv, poly, rgb, clip=None, opacity=1.0):
        box, m = cv.mask(poly)
        if box is None:
            return None, None
        y0, y1, x0, x1 = box
        if clip is not None:
            m = m & clip[y0:y1, x0:x1]
        cv.paint(box, m.astype(np.float32), rgb, opacity)
        return box, m

    def _muzzle_line(self, cv, gi):
        """The far edge of the muzzle against the face: the bridge line."""
        mz = self.muzzle_pe
        P, zr, t = mz.outline(160)
        N = geom.outward_normals(P)
        Xh = self.Xh
        tip = Xh.apply(np.array([[0, 0.38, 1.3]]))[0, :2]
        base = Xh.apply(np.array([[0, 0.40, 0.2]]))[0, :2]
        f2 = tip - base
        if np.linalg.norm(f2) < 0.08 * self.unit:
            return
        f2 /= np.linalg.norm(f2)
        up2 = self._dir2(Xh, (0, -1, 0))
        H, W = self.zbuf.shape
        q = P - N * 1.5 * cv.ss
        ix = np.clip(q[:, 0].astype(int), 0, W - 1)
        iy = np.clip(q[:, 1].astype(int), 0, H - 1)
        q2 = P + N * 2.5 * cv.ss
        jx = np.clip(q2[:, 0].astype(int), 0, W - 1)
        jy = np.clip(q2[:, 1].astype(int), 0, H - 1)
        # visible muzzle edge, with head behind it (not background)
        front = np.abs(self.zbuf[iy, ix] - zr) < 0.08 * self.unit
        behind = self.head_vis[jy, jx]
        sel = front & behind & ((N @ f2) > 0.0) & ((N @ up2) > -0.3)
        if sel.sum() < 4:
            return
        box = cv.bbox(P, pad=int(self.lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        buf = cv.stroke_buffer(box)
        ink.ink_polyline(buf, P, self._pen(cv, 0.75, taper=0.4, break_lit=0.0), ox=x0, oy=y0, closed=True, visible=sel)
        cv.paint(box, buf, INK)

    def _chest_fur_lines(self, cv, g, gi):
        """Interior fur flicks in the ruff: short curved strokes combed down and out,
        in a coloured line a step darker than the fur, so the big shadow shape reads soft."""
        rnd = vnoise.rng("ruffl", self.p.seed // 6)
        for part in g["parts"]:
            pe = part["pe"]
            box = part["box"]
            if box is None:
                continue
            y0, y1, x0, x1 = box
            buf = cv.stroke_buffer(box)
            nt = 14 if part.get("name") == "chest" else 30
            for k in range(nt):
                a = rnd.uniform(-0.75, 0.75)
                b = rnd.uniform(-0.1, 0.75) if part.get("name") == "chest" else rnd.uniform(-0.65, 0.75)
                if a * a + b * b > 0.92:
                    continue
                cz = math.sqrt(max(0.05, 1 - a * a - b * b))
                side = 1.0 if a >= 0 else -1.0
                ln = rnd.uniform(0.10, 0.20)
                pts = []
                for f in np.linspace(0, 1, 7):
                    # a comma: starts going down, curls outward
                    q = np.array([a + side * ln * 0.55 * f * f, b + ln * f, cz - 0.05 * f])
                    pts.append(q / np.linalg.norm(q))
                S = pe.c + np.array(pts) @ pe.M.T
                if not self._z_ok(S, 0.05 * self.unit).all():
                    continue
                ink.ink_designed(buf, S[:, :2], self.lw * rnd.uniform(0.38, 0.55),
                                 profile=lambda u: np.clip(u / 0.25, 0, 1) ** 0.7 * np.clip((1 - u) / 0.6, 0, 1) ** 1.2,
                                 ox=x0, oy=y0)
            mine = (self.gid[y0:y1, x0:x1] == gi).astype(np.float32)
            col = self.ctx.pal.colour("cream", cel.SHADE) * 0.62 + INK * 0.38
            cv.paint(box, np.minimum(buf, 1.0) * mine * 0.85, col)

    def _paint_eye(self, cv, e):
        if e["normal"][2] < 0.06:
            return
        p = self.p
        ctx = self.ctx
        lw = self.lw
        box = cv.bbox([e["opening"], e["upper"]], pad=int(lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        xs, ys = cv.grid(box)
        zb = self.zbuf[y0:y1, x0:x1]
        clipb = self.head_vis[y0:y1, x0:x1] & (zb <= e["z"] + 0.06 * self.unit)
        self._eye_clip = (box, clipb)
        if e["open"] <= 0.08:
            return
        _, mo = cv.mask(e["opening"], box)
        mo &= clipb
        pal = ctx.pal
        cv.paint(box, mo.astype(np.float32), pal.colour("eyedark", cel.LIT))
        _, mi = cv.mask(e["iris"], box)
        mi &= mo
        night = self.lname in ("hall", "night_out")
        iris_mid = hexrgb("#D9942F") if night else hexrgb("#E39A33")
        iris_lo = hexrgb("#F2C764") if night else hexrgb("#F7CF6A")
        iris_top = hexrgb("#7E4220") if night else hexrgb("#8C4A1C")
        cv.paint(box, mi.astype(np.float32), iris_mid)
        _, mc = cv.mask(e["cres"], box)
        cv.paint(box, (mi & ~mc).astype(np.float32), iris_lo)
        _, mb = cv.mask(e["band"], box)
        cv.paint(box, (mi & mb).astype(np.float32), iris_top)
        buf = cv.stroke_buffer(box)
        ink.ink_designed(buf, np.vstack([e["iris"], e["iris"][:1]]), lw * 0.30, profile=lambda u: np.ones_like(u) * 0.9,
                         ox=x0, oy=y0)
        cv.paint(box, buf * mo, hexrgb("#5A2E14"))
        _, mp = cv.mask(e["pupil"], box)
        cv.paint(box, (mp & mo).astype(np.float32), hexrgb("#1A100D"))
        c = e["centre"]
        r = e["ri"]
        kd = np.array(ctx.L["key_dir"])
        kd = kd / np.linalg.norm(kd)
        big = self._oval(c + np.array([kd[0] * 0.36, kd[1] * 0.40]) * r, r * 0.30, r * 0.25, -0.5)
        _, mh = cv.mask(big, box)
        cv.paint(box, (mh & mo).astype(np.float32), np.array([1.0, 0.99, 0.97]))
        small = self._oval(c + np.array([-kd[0] * 0.42, -kd[1] * 0.40]) * r, r * 0.11, r * 0.10, 0)
        _, ms = cv.mask(small, box)
        cv.paint(box, (ms & mo).astype(np.float32), np.array([0.95, 0.96, 1.0]))
        if p.glint and ctx.lantern is not None:
            ld = np.array(ctx.lantern[:2]) - c
            ld = ld / (np.linalg.norm(ld) + 1e-9)
            gl = self._oval(c + ld * r * 0.55, r * 0.14, r * 0.09, math.atan2(ld[1], ld[0]))
            _, mg = cv.mask(gl, box)
            cv.paint(box, (mg & mo & ~mh).astype(np.float32), hexrgb("#FFD9A0"))
        buf = cv.stroke_buffer(box)
        lo = e["lower"]
        n = len(lo)
        ink.ink_designed(buf, lo[: int(n * 0.62)], lw * 0.42, profile=lambda u: np.sin(np.clip(u, 0, 1) * math.pi) ** 0.7,
                         ox=x0, oy=y0)
        cv.paint(box, buf * clipb, INK * 0.85 + ALBEDO["orange"] * 0.15)

    def _paint_lids(self, cv, e):
        if e["normal"][2] < 0.06:
            return
        lw = self.lw
        box = cv.bbox([e["opening"], e["upper"]], pad=int(lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        clipb = self.head_vis[y0:y1, x0:x1] & (self.zbuf[y0:y1, x0:x1] <= e["z"] + 0.06 * self.unit)
        o = e["open"]
        buf = cv.stroke_buffer(box)
        up = e["upper"]
        ext = up[-1] + (up[-1] - up[-3]) * 0.9
        upx = np.vstack([up, ext[None, :]])
        th = 1.0 if o >= 0.98 else (0.80 if o > 0.08 else 0.85)

        def lidprof(u):
            return np.clip(0.25 + 1.1 * u, 0, 1) ** 0.8 * np.clip((1.0 - u) / 0.12, 0, 1) ** 0.7 * th + 0.05
        ink.ink_designed(buf, upx, lw * 1.55, profile=lidprof, ox=x0, oy=y0, wobble=0.2 * cv.ss, seed=self.p.seed)
        cv.paint(box, buf * clipb, INK * 0.95)

    @staticmethod
    def _oval(c, rx, ry, ang, n=36):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ca, sa = math.cos(ang), math.sin(ang)
        x = rx * np.cos(t)
        y = ry * np.sin(t)
        return np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], 1)

    def _paint_nose_mouth(self, cv):
        lw = self.lw
        pal = self.ctx.pal
        nose = self.feat["nose"]
        box = cv.bbox(nose["poly"], pad=int(lw * 4))
        if box is not None and nose["normal"][2] > -0.3:
            y0, y1, x0, x1 = box
            vis = self.zbuf[y0:y1, x0:x1] <= nose["z"] + 0.08 * self.unit
            _, m = cv.mask(nose["poly"], box)
            m &= vis
            if m.any():
                cv.paint(box, m.astype(np.float32), pal.colour("nose", cel.LIT))
                _, mh = cv.mask(nose["hl"], box)
                hl = np.array([0.62, 0.64, 0.78]) if self.lname in ("hall", "night_out") else np.array([0.88, 0.84, 0.82])
                cv.paint(box, (mh & m).astype(np.float32), hl)
                buf = cv.stroke_buffer(box)
                ink.ink_polyline(buf, nose["poly"], self._pen(cv, 0.6, min_len=400 * cv.ss, max_len=600 * cv.ss, break_lit=0.0),
                                 ox=x0, oy=y0, closed=True)
                cv.paint(box, buf * vis, INK * 0.8)
        mo = self.feat["mouth"]
        allp = np.vstack([mo["philtrum"][:, :2], mo["left"][:, :2], mo["right"][:, :2]])
        box = cv.bbox(allp, pad=int(lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        buf = cv.stroke_buffer(box)
        if mo["jaw"] is not None:
            J = mo["jaw"]
            ok = self._z_ok(J, 0.1 * self.unit)
            if ok.sum() >= 3:
                poly = np.vstack([J[ok][:, :2], mo["philtrum"][-1:, :2]])
                bx2, mm = cv.mask(poly)
                if bx2 is not None:
                    cv.paint(bx2, mm.astype(np.float32), pal.colour("mouth", cel.LIT))
        for key, w in (("philtrum", 0.55), ("left", 0.6), ("right", 0.6)):
            S = mo[key]
            vis = self._z_ok(S, 0.05 * self.unit)
            if vis.sum() < 2:
                continue
            P = geom.catmull_rom(S[vis][:, :2], 8)
            ink.ink_designed(buf, P, lw * w, profile=lambda u: (np.sin(np.clip(u, 0, 1) * math.pi) ** 0.5) * 0.9 + 0.1, ox=x0, oy=y0)
        cv.paint(box, buf * self.head_vis[y0:y1, x0:x1], INK * 0.9)

    def _paint_whiskers(self, cv):
        lw = self.lw
        wh = self.feat["whiskers"]
        allp = np.vstack([l[:, :2] for l in wh["lines"]])
        box = cv.bbox(allp, pad=int(lw * 3))
        if box is None:
            return
        y0, y1, x0, x1 = box
        buf = cv.stroke_buffer(box)
        for S in wh["lines"]:
            if not self._z_ok(S[:1], 0.04 * self.unit).all():
                continue
            ink.ink_designed(buf, S[:, :2], lw * 0.22, profile=lambda u: np.clip(1.15 - u, 0, 1) ** 0.7, ox=x0, oy=y0)
        cv.paint(box, buf * 0.7, self.ctx.pal.colour("cream", cel.LIT) * 0.9)
        buf = cv.stroke_buffer(box)
        for d in wh["dots"]:
            if not self._z_ok(d[None, :], 0.04 * self.unit).all():
                continue
            raster.stroke(buf, np.array([d[:2], d[:2] + 0.3]), np.array([lw * 0.18, lw * 0.18]), x0, y0)
        cv.paint(box, buf * self.head_vis[y0:y1, x0:x1] * 0.7, INK)

    def _paint_lantern(self, cv):
        from actors import lantern as LT
        lg = self.lantern_geo
        g = LT.paint(cv, lg["centre"], lg["rx"], lg["ry"], self.lw, stick=lg["stick"], cord=lg["cord"],
                     swing=self.p.lantern_swing)
        if g is not None:
            self.glow.append(g)


def draw(pose, light_name, L, lantern_screen=None, lantern_reach=0.0, ss=3, frame_wh=(1920, 1080), hard_warm=0.0):
    """Draw the fox. Returns (tile premultiplied RGBA at 1x, x0, y0, FoxDrawing)."""
    fd = FoxDrawing(pose, light_name, L, lantern_screen, lantern_reach, ss)
    fd.hard_warm = hard_warm
    x0, y0, x1, y1 = fd.bounds_screen()
    x0 = int(max(math.floor(x0), -300))
    y0 = int(max(math.floor(y0), -300))
    x1 = int(min(math.ceil(x1), frame_wh[0] + 300))
    y1 = int(min(math.ceil(y1), frame_wh[1] + 300))
    cv = Canvas(x0, y0, x1 - x0, y1 - y0, ss)
    fd.paint(cv)
    return cv.resolve(), x0, y0, fd
