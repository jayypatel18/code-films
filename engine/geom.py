"""Geometry for drawn characters: rotations, projection, splines, silhouettes.

Screen space is x right, y down, z toward the viewer. A character is built from
simple 3D masses (ellipsoids, tubes, cards) in its own space, posed with
rotations, projected orthographically and then drawn as flat 2D cel shapes.
"""

import math

import numpy as np


# ---------------------------------------------------------------- rotations

def rot_x(a):
    """Pitch. Positive a lifts +z (forward) toward -y (up, since y is down)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    """Yaw. Positive a turns +z (forward) toward +x (screen right)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    """Roll. Positive a turns clockwise on screen (y is down)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler(yaw=0.0, pitch=0.0, roll=0.0):
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def rot_axis(axis, a):
    """Rotation by angle a about a unit axis (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def deg(d):
    return d * math.pi / 180.0


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class Xform:
    """Affine map local -> screen3d: s = A @ p + t (A includes scale)."""

    def __init__(self, A, t):
        self.A = np.asarray(A, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)

    def __matmul__(self, other):
        return Xform(self.A @ other.A, self.A @ other.t + self.t)

    def apply(self, p):
        p = np.asarray(p, dtype=np.float64)
        return p @ self.A.T + self.t

    def inverse(self):
        Ai = np.linalg.inv(self.A)
        return Xform(Ai, -Ai @ self.t)

    @staticmethod
    def make(R=None, scale=1.0, t=(0, 0, 0)):
        R = np.eye(3) if R is None else R
        return Xform(np.asarray(R) * scale, t)


# ---------------------------------------------------------------- splines

def catmull_rom(points, samples=12, closed=False, alpha=0.5):
    """Centripetal Catmull-Rom through points (n, d). Returns dense polyline."""
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    if n < 2:
        return P.copy()
    if closed:
        ext = np.vstack([P[-1:], P, P[:2]])
        segs = n
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        segs = n - 1
    out = []
    tt = np.linspace(0, 1, samples, endpoint=False)
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** alpha, 1e-6)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** alpha, 1e-6)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** alpha, 1e-6)
        t = t1 + (t2 - t1) * tt[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    if not closed:
        out.append(P[-1:])
    return np.vstack(out)


def arclen(poly, closed=False):
    P = np.asarray(poly)
    if closed:
        P = np.vstack([P, P[:1]])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(poly, spacing, closed=False, extra=None):
    """Resample a polyline at even arc length. extra: (n,) or (n,k) values to carry."""
    P = np.asarray(poly, dtype=np.float64)
    if closed:
        P2 = np.vstack([P, P[:1]])
        E2 = None if extra is None else np.concatenate([extra, extra[:1]], axis=0)
    else:
        P2, E2 = P, extra
    s = arclen(P2)
    L = s[-1]
    if L < 1e-9:
        return (P[:1].copy(), None if extra is None else np.asarray(extra)[:1]) if extra is not None else P[:1].copy()
    n = max(2, int(math.ceil(L / spacing)) + (0 if closed else 1))
    if closed:
        q = np.linspace(0, L, n, endpoint=False)
    else:
        q = np.linspace(0, L, n)
    out = np.stack([np.interp(q, s, P2[:, k]) for k in range(P.shape[1])], axis=1)
    if extra is None:
        return out
    E2 = np.asarray(E2, dtype=np.float64)
    if E2.ndim == 1:
        ev = np.interp(q, s, E2)
    else:
        ev = np.stack([np.interp(q, s, E2[:, k]) for k in range(E2.shape[1])], axis=1)
    return out, ev


def normals2d(poly, closed=False):
    """Unit left normals of a 2D polyline (rotate tangent by -90 deg in y-down)."""
    P = np.asarray(poly, dtype=np.float64)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.gradient(P, axis=0)
    T = normalize(T)
    return np.stack([T[:, 1], -T[:, 0]], axis=1)


def signed_area(poly):
    P = np.asarray(poly)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def outward_normals(poly):
    """Outward unit normals of a closed polygon in y-down screen space."""
    n = normals2d(poly, closed=True)
    # In y-down coordinates, positive signed area means clockwise on screen,
    # and for a clockwise loop the left normal (t_y, -t_x) points outward.
    if signed_area(poly) > 0:
        return n
    return -n


def smooth_closed(poly, iters=1):
    P = np.asarray(poly, dtype=np.float64)
    for _ in range(iters):
        P = 0.25 * np.roll(P, 1, axis=0) + 0.5 * P + 0.25 * np.roll(P, -1, axis=0)
    return P


# ---------------------------------------------------------------- ellipsoids

class Ellipsoid:
    """Ellipsoid in a local space: centre, radii along the columns of rot."""

    def __init__(self, centre, radii, rot=None):
        self.c = np.asarray(centre, dtype=np.float64)
        self.r = np.asarray(radii, dtype=np.float64)
        self.rot = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)

    def project(self, X):
        """X: Xform local -> screen3d. Returns a ProjEllipsoid."""
        c = X.apply(self.c)
        M = X.A @ self.rot @ np.diag(self.r)
        return ProjEllipsoid(c, M, X, self)


class ProjEllipsoid:
    """An ellipsoid placed in screen space: s = c + M u, |u| = 1."""

    def __init__(self, c, M, X, src):
        self.c = c
        self.M = M
        self.Mi = np.linalg.inv(M)
        self.X = X
        self.src = src
        self.d = self.Mi[:, 2].copy()      # u-space direction of screen +z

    def rim_basis(self, ref_local=(0.0, -1.0, 0.0)):
        """Orthonormal basis (e1, e2) of the rim plane in u space.

        The rim is the great circle of u perpendicular to the view direction.
        e1 follows a fixed local axis so fur spikes stay attached to the body.
        """
        d = self.d / np.linalg.norm(self.d)
        ref = np.linalg.inv(self.src.rot @ np.diag(self.src.r)) @ np.asarray(ref_local, dtype=np.float64)
        ref = ref - d * np.dot(ref, d)
        if np.linalg.norm(ref) < 1e-6:
            ref = np.array([1.0, 0, 0]) - d * d[0]
        e1 = ref / np.linalg.norm(ref)
        e2 = np.cross(d, e1)
        return e1, e2

    def outline(self, n=180, ref_local=(0.0, -1.0, 0.0)):
        """Silhouette points (n, 2), their depths (n,) and params t (n,).

        t = 0 points along ref_local (default local up), increasing with the
        rotation from e1 toward e2.
        """
        e1, e2 = self.rim_basis(ref_local)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        U = np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2
        S = self.c + U @ self.M.T
        return S[:, :2], S[:, 2], t

    def surface(self, xs, ys):
        """Front surface at screen points. Returns valid, z, normal (3,..), u (3,..)."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        Mi = self.Mi
        dx = xs - self.c[0]
        dy = ys - self.c[1]
        u0 = [Mi[k, 0] * dx + Mi[k, 1] * dy - Mi[k, 2] * 0.0 for k in range(3)]
        d = self.d
        dd = float(d @ d)
        b = u0[0] * d[0] + u0[1] * d[1] + u0[2] * d[2]
        cc = u0[0] ** 2 + u0[1] ** 2 + u0[2] ** 2 - 1.0
        disc = b * b - dd * cc
        valid = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        z1 = (-b + sq) / dd
        z2 = (-b - sq) / dd
        # the front hit is the one with the larger screen z
        zr = np.maximum(z1, z2)
        u = [u0[k] + zr * d[k] for k in range(3)]
        # normal = Mi^T u
        n = [Mi[0, k] * u[0] + Mi[1, k] * u[1] + Mi[2, k] * u[2] for k in range(3)]
        nl = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) + 1e-12
        n = [n[k] / nl for k in range(3)]
        z = self.c[2] + zr
        return valid, z, n, u

    def local_point(self, u):
        """u (3, ...) on the unit sphere -> point in the ellipsoid's local space."""
        R = self.src.rot @ np.diag(self.src.r)
        return [self.src.c[k] + R[k, 0] * u[0] + R[k, 1] * u[1] + R[k, 2] * u[2] for k in range(3)]


# ---------------------------------------------------------------- tubes

def tube_outline(centres, radii, cap_samples=10):
    """Closed outline of a 2D swept circle along a polyline (envelope of circles).

    centres: (n, 2), radii: (n,). Returns (m, 2) closed polygon, clockwise
    starting at the first left side point.
    """
    C = np.asarray(centres, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    n = len(C)
    if n == 1:
        t = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        return C[0] + r[0] * np.stack([np.cos(t), np.sin(t)], axis=1)
    T = np.gradient(C, axis=0)
    seg = np.linalg.norm(T, axis=1)
    T = T / np.maximum(seg, 1e-9)[:, None]
    dr = np.gradient(r) / np.maximum(seg, 1e-9)
    dr = np.clip(dr, -0.95, 0.95)
    Nl = np.stack([T[:, 1], -T[:, 0]], axis=1)
    ca = np.sqrt(1 - dr ** 2)
    left = C + r[:, None] * (Nl * ca[:, None] - T * dr[:, None])
    right = C + r[:, None] * (-Nl * ca[:, None] - T * dr[:, None])
    # caps
    a_end = math.atan2(Nl[-1, 1], Nl[-1, 0])
    a_beg = math.atan2(-Nl[0, 1], -Nl[0, 0])
    te = a_end - np.linspace(0, math.pi, cap_samples + 2)[1:-1]
    tb = a_beg - np.linspace(0, math.pi, cap_samples + 2)[1:-1]
    cap_e = C[-1] + r[-1] * np.stack([np.cos(te), np.sin(te)], axis=1)
    cap_b = C[0] + r[0] * np.stack([np.cos(tb), np.sin(tb)], axis=1)
    return np.vstack([left, cap_e, right[::-1], cap_b])


def point_in_poly(pts, poly):
    """Even odd test, vectorised over pts (m, 2)."""
    P = np.asarray(poly)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = P[:, 0][None, :], P[:, 1][None, :]
    x2, y2 = np.roll(P[:, 0], -1)[None, :], np.roll(P[:, 1], -1)[None, :]
    cond = ((y1 > y) != (y2 > y))
    xint = (x2 - x1) * (y - y1) / np.where(np.abs(y2 - y1) < 1e-12, 1e-12, (y2 - y1)) + x1
    hit = cond & (x < xint)
    return (np.sum(hit, axis=1) % 2) == 1
