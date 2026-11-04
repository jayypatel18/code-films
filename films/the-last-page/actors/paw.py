"""Cut 4b: only the fox's forepaw, pressing the blank page.

Drawn with the fox's own cel machinery (palette, ink, z buffer) but built in
screen space: the foreleg comes in from the bottom left, orange fur above and
the black stocking below, bends at the wrist, and the paw with its four toes
spreads a little as it presses down.
"""

import math

import numpy as np

import timeline as T
from actors import fox as F
from engine import geom
from engine.canvas import Canvas
from engine.geom import Ellipsoid, Xform
from engine.tube import Tube


class PawDrawing(F.FoxDrawing):
    def __init__(self, x, y, lift, seed):
        pose = F.FoxPose(scale=190.0, seed=seed, body="paw")
        super().__init__(pose, "hall", T.LIGHTS["hall"], lantern_screen=(240.0, 1180.0, 320.0), lantern_reach=1700.0)
        self.px, self.py, self.lift = x, y, lift
        # on a lit page the stocking is dark brown, not black, and the lantern warms its edge
        self.pal_override = {"sock": dict(lit="#6E4C40", shade="#46323A", warm="#93623F", warm_shade="#634236",
                                          rim="#D29A66")}
        self.rim_override = ((-0.75, 0.66), 0.035)

    def build(self, Xs2c):
        """A close crop: only the front of the paw and its toes come in from the bottom
        left corner; the leg runs off frame. (x, y) is the front edge of the toes."""
        from engine import cel
        Xc = Xs2c
        self.Xc = self.Xh = Xc
        self.unit = 210.0 * float(np.linalg.norm(Xc.A[:, 0]))
        S = 1.45
        ang = math.atan2(-0.62, 1.0)          # the paw points up and to the right, into the page
        dv = np.array([math.cos(ang), math.sin(ang), 0.0])
        pv = np.array([-math.sin(ang), math.cos(ang), 0.0])
        lift = max(self.lift, 0.0)
        press = max(0.0, 1.0 - lift * 6.0)
        spread = 1.0 + 0.10 * press
        rx, ry, rz = 132.0 * S, 84.0 * S * spread, 34.0 * S * (1.0 - 0.12 * press)
        front = np.array([self.px, self.py, 60.0]) - dv * 150.0 * lift
        pc = front - dv * rx * 1.02 + np.array([0, 0, 0.0])
        rot = np.stack([dv, pv, np.array([0, 0, 1.0])], axis=1)
        pe = Ellipsoid(pc, (rx, ry, rz), rot).project(Xc)
        P, z, t = pe.outline(120)
        N = geom.outward_normals(P)
        back2 = -dv[:2] / np.linalg.norm(dv[:2])
        fur = self._facing_ranges(P, t, N, back2, -0.35, length=0.10, width=0.16, lean=back2, comb=1.0)
        ppoly, _ = cel.fur_edge(P, t, N, fur, ("padfur", self.p.seed // 4), self.unit)
        # the lower leg, foreshortened, running off the bottom left corner
        legd = geom.normalize(np.array([-0.78, 0.62, 0.0]))
        root = pc + legd * 520.0 + np.array([0, 0, 200.0])
        mid = pc + legd * 250.0 + np.array([0, 0, 110.0])
        near = pc + legd * 70.0 + np.array([0, 0, 50.0])
        leg = Tube(np.array([root, mid, near]), np.array([250.0, 175.0, 110.0]), 8)
        pt = leg.project(Xc)
        parts = [dict(kind="tube", pt=pt, poly=self._leg_fur_long(pt, ("pawfur", self.p.seed // 6)),
                      pattern=self._paw_leg_pattern(0.22)),
                 dict(kind="ell", pe=pe, poly=ppoly, pattern=self._sock_pattern)]
        # four small toe ovals along the front edge, pressed flat
        for k, side in enumerate((-0.66, -0.22, 0.22, 0.66)):
            tcen = pc + dv * rx * (0.86 - 0.24 * abs(side)) + pv * side * ry * 0.98 + np.array([0, 0, rz * 0.55])
            te = Ellipsoid(tcen, (46 * S * 0.8, 40 * S * 0.8 * spread, 22 * S), rot).project(Xc)
            tP, _, tt = te.outline(48)
            parts.append(dict(kind="ell", pe=te, poly=tP, pattern=self._sock_pattern))
        self.paw_axes = (pc, dv, pv, rx, ry)
        groups = [dict(name="leg", parts=parts, depth=0.0)]
        self.feat = None
        self.lantern_geo = None
        self.groups = groups
        return groups

    def _paw_leg_pattern(self, sock_s):
        def f(sp, side):
            pal = self.ctx.pal
            # the stocking edge is a row of orange fur tufts poking down into the brown
            q = (side + 1.0) * 3.5
            tooth = np.abs((q % 1.0) - 0.5) * 2.0
            b = sock_s - 0.035 * (1.0 - tooth) ** 1.5 + 0.01 * np.sin(q * 2.1)
            return np.where(sp > b, pal["sock"], pal["orange"]).astype(np.int16)
        return f

    def paint(self, cv):
        super().paint(cv)
        self._toe_lines(cv)
        return cv

    def _toe_lines(self, cv):
        """Clear divisions between the toes, seen from above: short tapered strokes from
        the front edge of the paw back between each pair of toes."""
        from engine import ink
        from actors.fox import INK
        g = self.groups[0]
        toes = [pt for pt in g["parts"][2:]]
        if len(toes) < 2:
            return
        pc = g["parts"][1]["pe"].c
        cs = [t["pe"].c for t in toes]
        allp = [t["poly"] for t in toes]
        box = cv.bbox(allp, pad=int(self.lw * 4))
        if box is None:
            return
        y0, y1, x0, x1 = box
        buf = cv.stroke_buffer(box)
        for a, b in zip(cs[:-1], cs[1:]):
            mid = 0.5 * (a + b)
            front = mid + (mid - pc) * 0.10
            back = mid - (mid - pc) * 0.16
            pts = np.stack([np.linspace(front[0], back[0], 6), np.linspace(front[1], back[1], 6)], 1)
            ink.ink_designed(buf, pts, self.lw * 0.6, profile=lambda u: np.clip(1.1 - u, 0, 1) ** 0.8, ox=x0, oy=y0)
        cv.paint(box, buf, INK * 0.85 + np.array([0.06, 0.04, 0.03], np.float32))

    def _leg_fur_long(self, pt, key):
        """Tufts along the orange upper leg on both sides, combed toward the paw."""
        from engine import cel
        P = pt.outline(cap_samples=8)
        n = len(P)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        N = geom.outward_normals(P)
        m = len(pt.C)
        down2 = pt.C[-1, :2] - pt.C[0, :2]
        down2 = down2 / (np.linalg.norm(down2) + 1e-9)
        k1 = int(0.55 * m)
        cap = 8
        r0 = m + cap + (m - k1)
        r1 = min(m + cap + m - 1, n - 1)
        ranges = [dict(t0=t[1], t1=t[k1], length=0.16, width=0.22, lean=down2, comb=1.3),
                  dict(t0=t[min(r0, n - 2)], t1=t[r1], length=0.16, width=0.22, lean=down2, comb=1.3)]
        poly, _ = cel.fur_edge(P, t, N, ranges, key, self.unit)
        return poly

    def bounds_screen(self):
        g = self.build(Xform(np.eye(3), (0, 0, 0)))
        allp = np.vstack([p["poly"] for p in g[0]["parts"]])
        return allp[:, 0].min() - 20, allp[:, 1].min() - 20, allp[:, 0].max() + 20, allp[:, 1].max() + 20


def draw(x, y, lift=0.0, seed=0, ss=3):
    pd = PawDrawing(x, y, lift, seed)
    pd.hard_warm = 0.45      # the lantern below left warms the near side of the leg
    x0, y0, x1, y1 = pd.bounds_screen()
    x0, y0 = int(max(math.floor(x0), -200)), int(max(math.floor(y0), -200))
    x1, y1 = int(min(math.ceil(x1), T.W + 200)), int(min(math.ceil(y1), T.H + 200))
    cv = Canvas(x0, y0, x1 - x0, y1 - y0, ss)
    pd.paint(cv)
    return cv.resolve(), x0, y0
