"""Marching squares contour tracing into closed, oriented polylines.

Used to ink the outline of a union of shapes (head plus cheeks plus muzzle)
as one clean silhouette, and for paper and lettering outlines later.
"""

import numpy as np
from numba import njit
from scipy import ndimage

# corners: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left
# edges:   0 top (0-1), 1 right (1-2), 2 bottom (2-3), 3 left (3-0)
_CORNER_XY = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
_EDGE_CORNERS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _build_table():
    """Oriented segment table: for each case, up to 2 (from_edge, to_edge) pairs.

    Orientation keeps the inside on the left in y-down screen space, so outer
    contours run counter-clockwise on screen and every crossing edge has
    exactly one outgoing segment.
    """
    table = -np.ones((16, 2, 2), dtype=np.int64)
    for case in range(16):
        inside = [(case >> k) & 1 for k in range(4)]
        edges = [e for e, (a, b) in enumerate(_EDGE_CORNERS) if inside[a] != inside[b]]
        if not edges:
            continue
        if len(edges) == 2:
            pairs = [(edges[0], edges[1])]
        else:
            # saddle: separate the inside corners (they stay disconnected)
            pairs = []
            for k in range(4):
                if inside[k]:
                    # the two edges touching corner k
                    e_in = [e for e in edges if k in _EDGE_CORNERS[e]]
                    pairs.append((e_in[0], e_in[1]))
        out = []
        for (ea, eb) in pairs:
            pa = _CORNER_XY[list(_EDGE_CORNERS[ea])].mean(axis=0)
            pb = _CORNER_XY[list(_EDGE_CORNERS[eb])].mean(axis=0)
            mid = 0.5 * (pa + pb)
            dx, dy = pb - pa
            left = np.array([dy, -dx])
            # centroid of the inside corners touching these edges
            ins = [k for k in set(_EDGE_CORNERS[ea] + _EDGE_CORNERS[eb]) if inside[k]]
            cen = _CORNER_XY[ins].mean(axis=0)
            if np.dot(cen - mid, left) >= 0:
                out.append((ea, eb))
            else:
                out.append((eb, ea))
        for i, (ea, eb) in enumerate(out[:2]):
            table[case, i, 0] = ea
            table[case, i, 1] = eb
    return table


_TABLE = _build_table()


@njit(cache=True)
def _march(f, level, table):
    H, W = f.shape
    nH = H * W            # horizontal edges indexed by (i, j): between (i,j) and (i,j+1)
    nE = 2 * H * W
    nxt = -np.ones(nE, dtype=np.int64)
    px = np.zeros(nE, dtype=np.float64)
    py = np.zeros(nE, dtype=np.float64)
    for i in range(H - 1):
        for j in range(W - 1):
            c0 = f[i, j] > level
            c1 = f[i, j + 1] > level
            c2 = f[i + 1, j + 1] > level
            c3 = f[i + 1, j] > level
            case = (1 if c0 else 0) | (2 if c1 else 0) | (4 if c2 else 0) | (8 if c3 else 0)
            if case == 0 or case == 15:
                continue
            for s in range(2):
                ea = table[case, s, 0]
                if ea < 0:
                    continue
                eb = table[case, s, 1]
                ids = np.empty(2, dtype=np.int64)
                for q in range(2):
                    e = ea if q == 0 else eb
                    if e == 0:      # top: H edge (i, j)
                        eid = i * W + j
                        a, b = f[i, j], f[i, j + 1]
                        t = (level - a) / (b - a)
                        px[eid] = j + t
                        py[eid] = i
                    elif e == 2:    # bottom: H edge (i+1, j)
                        eid = (i + 1) * W + j
                        a, b = f[i + 1, j], f[i + 1, j + 1]
                        t = (level - a) / (b - a)
                        px[eid] = j + t
                        py[eid] = i + 1
                    elif e == 3:    # left: V edge (i, j)
                        eid = nH + i * W + j
                        a, b = f[i, j], f[i + 1, j]
                        t = (level - a) / (b - a)
                        px[eid] = j
                        py[eid] = i + t
                    else:           # right: V edge (i, j+1)
                        eid = nH + i * W + j + 1
                        a, b = f[i, j + 1], f[i + 1, j + 1]
                        t = (level - a) / (b - a)
                        px[eid] = j + 1
                        py[eid] = i + t
                    ids[q] = eid
                nxt[ids[0]] = ids[1]
    return nxt, px, py


@njit(cache=True)
def _walk(nxt, px, py):
    n = nxt.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    out_x = np.empty(n, dtype=np.float64)
    out_y = np.empty(n, dtype=np.float64)
    starts = np.empty(n + 1, dtype=np.int64)
    k = 0
    nl = 0
    for e0 in range(n):
        if nxt[e0] < 0 or visited[e0]:
            continue
        starts[nl] = k
        nl += 1
        e = e0
        while not visited[e] and e >= 0:
            visited[e] = True
            out_x[k] = px[e]
            out_y[k] = py[e]
            k += 1
            e = nxt[e]
    starts[nl] = k
    return out_x[:k], out_y[:k], starts[:nl + 1]


def trace(field, level=0.5, min_points=8, offset=(0.0, 0.0)):
    """Closed contours of field at level. Returns list of (n, 2) arrays (x, y).

    Coordinates are in pixel-centre units of the field plus offset, so a field
    covering canvas box (y0, y1, x0, x1) passes offset=(x0 + 0.5, y0 + 0.5).
    Outer boundaries run counter-clockwise on screen, holes clockwise.
    """
    f = np.pad(np.asarray(field, dtype=np.float64), 1, constant_values=-1e9)
    nxt, px, py = _march(f, float(level), _TABLE)
    xs, ys, starts = _walk(nxt, px, py)
    out = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= min_points:
            P = np.stack([xs[a:b] - 1 + offset[0], ys[a:b] - 1 + offset[1]], axis=1)
            out.append(P)
    return out


def mask_contours(mask, box, blur=0.8, min_area=40.0):
    """Contours of a bool mask placed at canvas box, smoothed to subpixel."""
    y0, y1, x0, x1 = box
    f = ndimage.gaussian_filter(mask.astype(np.float32), blur) if blur > 0 else mask.astype(np.float32)
    polys = trace(f, 0.5, offset=(x0 + 0.5, y0 + 0.5))
    out = []
    for P in polys:
        x, y = P[:, 0], P[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area >= min_area:
            out.append(P)
    return out
