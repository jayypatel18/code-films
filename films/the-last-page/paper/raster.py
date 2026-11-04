"""The paper department's batch rasterizer.

Thousands of small paper shapes per frame, drawn the way the engine's Canvas
draws a character (hard fills on a supersampled grid, box filtered down,
pressure ink lines with analytic coverage) but in one numba pass.

An object (one bird, one page, one tile of the great bird) owns a list of
textured cel triangles and pressure polylines. Each object is drawn into its
own supersampled tile with its own depth buffer, so a bird's wing hides the
flap folded under it and hides the ink lines behind it. Tiles render in
parallel, then composite over the layer in object order (far to near).

Triangles carry a flat cel tone, an opacity, a page of the texture atlas with
per vertex texel coordinates (the page's ink, stains and fibres multiply the
tone), and an ink strength. Lines carry x, y, depth and radius per point.
"""

import math

import numpy as np
from numba import njit, prange

TILE = 224          # big objects are split into screen tiles of this size (px)


# ---------------------------------------------------------------- texture

@njit(cache=True, fastmath=True, inline="always")
def _texel(tex, off, w, h, page, x, y, out):
    # bilinear sample at level stored at off (pages of w*h*3 bytes)
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if x > w - 1.001:
        x = w - 1.001
    if y > h - 1.001:
        y = h - 1.001
    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy
    base = off + page * (w * h * 3)
    i00 = base + (iy * w + ix) * 3
    i10 = i00 + 3
    i01 = i00 + w * 3
    i11 = i01 + 3
    for c in range(3):
        a = tex[i00 + c] * (1.0 - fx) + tex[i10 + c] * fx
        b = tex[i01 + c] * (1.0 - fx) + tex[i11 + c] * fx
        out[c] = (a * (1.0 - fy) + b * fy) * (1.0 / 255.0)


# ---------------------------------------------------------------- one object

@njit(cache=True, fastmath=True)
def _draw_object(tile, zb, scratch, ss, bx0, by0,
                 t0, t1, txyz, tuv, tcol, tk, tpage,
                 s0, s1, sp, spts, scol, bias,
                 tex, lev_off, lev_w, lev_h, nlev):
    th = tile.shape[0]
    tw = tile.shape[1]
    sm = np.zeros(3, np.float32)
    sm2 = np.zeros(3, np.float32)
    fss = float(ss)
    # -------------------------------------------------------- triangles
    for t in range(t0, t1):
        al = tcol[t, 3]
        if al <= 0.002:
            continue
        X0 = (txyz[t, 0] - bx0) * fss - 0.5
        Y0 = (txyz[t, 1] - by0) * fss - 0.5
        Z0 = txyz[t, 2]
        X1 = (txyz[t, 3] - bx0) * fss - 0.5
        Y1 = (txyz[t, 4] - by0) * fss - 0.5
        Z1 = txyz[t, 5]
        X2 = (txyz[t, 6] - bx0) * fss - 0.5
        Y2 = (txyz[t, 7] - by0) * fss - 0.5
        Z2 = txyz[t, 8]
        U0 = tuv[t, 0]
        V0 = tuv[t, 1]
        U1 = tuv[t, 2]
        V1 = tuv[t, 3]
        U2 = tuv[t, 4]
        V2 = tuv[t, 5]
        area = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0)
        if abs(area) < 1e-9:
            continue
        if area < 0.0:
            X1, X2 = X2, X1
            Y1, Y2 = Y2, Y1
            Z1, Z2 = Z2, Z1
            U1, U2 = U2, U1
            V1, V2 = V2, V1
            area = -area
        xmin = int(math.ceil(min(X0, min(X1, X2))))
        xmax = int(math.floor(max(X0, max(X1, X2))))
        ymin = int(math.ceil(min(Y0, min(Y1, Y2))))
        ymax = int(math.floor(max(Y0, max(Y1, Y2))))
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > tw - 1:
            xmax = tw - 1
        if ymax > th - 1:
            ymax = th - 1
        if xmin > xmax or ymin > ymax:
            continue
        # top left rule for each edge (y down, positive area = clockwise)
        e0x = X2 - X1
        e0y = Y2 - Y1
        e1x = X0 - X2
        e1y = Y0 - Y2
        e2x = X1 - X0
        e2y = Y1 - Y0
        tl0 = (e0y < 0.0) or (e0y == 0.0 and e0x > 0.0)
        tl1 = (e1y < 0.0) or (e1y == 0.0 and e1x > 0.0)
        tl2 = (e2y < 0.0) or (e2y == 0.0 and e2x > 0.0)
        inv = 1.0 / area
        page = tpage[t]
        k = tk[t]
        cr = tcol[t, 0]
        cg = tcol[t, 1]
        cb = tcol[t, 2]
        # mip level from texels per sample
        lf = 0.0
        if page >= 0:
            uva = abs((U1 - U0) * (V2 - V0) - (V1 - V0) * (U2 - U0))
            r = math.sqrt(uva / area + 1e-12)
            if r > 1.0:
                lf = math.log2(r)
            if lf > nlev - 1.001:
                lf = nlev - 1.001
        l0 = int(lf)
        lfr = lf - l0
        l1 = l0 + 1 if l0 + 1 < nlev else l0
        sc0 = 1.0 / (1 << l0)
        sc1 = 1.0 / (1 << l1)
        for y in range(ymin, ymax + 1):
            py = float(y)
            for x in range(xmin, xmax + 1):
                px = float(x)
                w0 = e0x * (py - Y1) - e0y * (px - X1)
                if w0 < 0.0 or (w0 == 0.0 and not tl0):
                    continue
                w1 = e1x * (py - Y2) - e1y * (px - X2)
                if w1 < 0.0 or (w1 == 0.0 and not tl1):
                    continue
                w2 = e2x * (py - Y0) - e2y * (px - X0)
                if w2 < 0.0 or (w2 == 0.0 and not tl2):
                    continue
                b0 = w0 * inv
                b1 = w1 * inv
                b2 = w2 * inv
                z = b0 * Z0 + b1 * Z1 + b2 * Z2
                if z >= zb[y, x]:
                    continue
                if al >= 0.999:
                    zb[y, x] = z
                r_ = cr
                g_ = cg
                bl_ = cb
                if page >= 0 and k > 0.0:
                    u = b0 * U0 + b1 * U1 + b2 * U2
                    v = b0 * V0 + b1 * V1 + b2 * V2
                    _texel(tex, lev_off[l0], lev_w[l0], lev_h[l0], page, u * sc0, v * sc0, sm)
                    if lfr > 0.02 and l1 != l0:
                        _texel(tex, lev_off[l1], lev_w[l1], lev_h[l1], page, u * sc1, v * sc1, sm2)
                        for c in range(3):
                            sm[c] = sm[c] * (1.0 - lfr) + sm2[c] * lfr
                    r_ = cr * (1.0 - k * (1.0 - sm[0]))
                    g_ = cg * (1.0 - k * (1.0 - sm[1]))
                    bl_ = cb * (1.0 - k * (1.0 - sm[2]))
                ia = 1.0 - al
                tile[y, x, 0] = r_ * al + tile[y, x, 0] * ia
                tile[y, x, 1] = g_ * al + tile[y, x, 1] * ia
                tile[y, x, 2] = bl_ * al + tile[y, x, 2] * ia
                tile[y, x, 3] = al + tile[y, x, 3] * ia
    # -------------------------------------------------------- ink lines
    for s in range(s0, s1):
        op = scol[s, 3]
        if op <= 0.002:
            continue
        p0 = sp[s]
        p1 = sp[s + 1]
        if p1 - p0 < 2:
            continue
        # bbox of the line in samples
        gx0 = 1e30
        gy0 = 1e30
        gx1 = -1e30
        gy1 = -1e30
        for i in range(p0, p1):
            rr = spts[i, 3] * fss + 1.5
            X = (spts[i, 0] - bx0) * fss - 0.5
            Y = (spts[i, 1] - by0) * fss - 0.5
            if X - rr < gx0:
                gx0 = X - rr
            if X + rr > gx1:
                gx1 = X + rr
            if Y - rr < gy0:
                gy0 = Y - rr
            if Y + rr > gy1:
                gy1 = Y + rr
        ax0 = int(math.floor(gx0))
        ay0 = int(math.floor(gy0))
        ax1 = int(math.ceil(gx1))
        ay1 = int(math.ceil(gy1))
        if ax0 < 0:
            ax0 = 0
        if ay0 < 0:
            ay0 = 0
        if ax1 > tw - 1:
            ax1 = tw - 1
        if ay1 > th - 1:
            ay1 = th - 1
        if ax0 > ax1 or ay0 > ay1:
            continue
        for i in range(p0, p1 - 1):
            ax = (spts[i, 0] - bx0) * fss - 0.5
            ay = (spts[i, 1] - by0) * fss - 0.5
            az = spts[i, 2]
            ar = spts[i, 3] * fss
            bx = (spts[i + 1, 0] - bx0) * fss - 0.5
            by = (spts[i + 1, 1] - by0) * fss - 0.5
            bz = spts[i + 1, 2]
            br = spts[i + 1, 3] * fss
            if ar <= 0.02 and br <= 0.02:
                continue
            mr = max(ar, br) + 1.5
            x0 = int(math.floor(min(ax, bx) - mr))
            x1 = int(math.ceil(max(ax, bx) + mr))
            y0 = int(math.floor(min(ay, by) - mr))
            y1 = int(math.ceil(max(ay, by) + mr))
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > tw - 1:
                x1 = tw - 1
            if y1 > th - 1:
                y1 = th - 1
            vx = bx - ax
            vy = by - ay
            vv = vx * vx + vy * vy
            for y in range(y0, y1 + 1):
                py = float(y)
                for x in range(x0, x1 + 1):
                    px = float(x)
                    if vv > 1e-12:
                        tt = ((px - ax) * vx + (py - ay) * vy) / vv
                        if tt < 0.0:
                            tt = 0.0
                        elif tt > 1.0:
                            tt = 1.0
                    else:
                        tt = 0.0
                    qx = ax + vx * tt - px
                    qy = ay + vy * tt - py
                    d = math.sqrt(qx * qx + qy * qy)
                    rad = ar + (br - ar) * tt
                    c = rad - d + 0.5
                    if c <= 0.0:
                        continue
                    if c > 1.0:
                        c = 1.0
                    if rad < 0.5:
                        c = c * (rad / 0.5)
                    zz = az + (bz - az) * tt - bias
                    if zz > zb[y, x]:
                        continue
                    if c > scratch[y, x]:
                        scratch[y, x] = c
        cr = scol[s, 0]
        cg = scol[s, 1]
        cb = scol[s, 2]
        for y in range(ay0, ay1 + 1):
            for x in range(ax0, ax1 + 1):
                c = scratch[y, x]
                if c <= 0.0:
                    continue
                scratch[y, x] = 0.0
                a = c * op
                ia = 1.0 - a
                tile[y, x, 0] = cr * a + tile[y, x, 0] * ia
                tile[y, x, 1] = cg * a + tile[y, x, 1] * ia
                tile[y, x, 2] = cb * a + tile[y, x, 2] * ia
                tile[y, x, 3] = a + tile[y, x, 3] * ia


@njit(cache=True, parallel=True, fastmath=True)
def _render_all(out, boxes, oss, oalpha, obias, otr, osr,
                txyz, tuv, tcol, tk, tpage,
                sp, spts, scol,
                tex, lev_off, lev_w, lev_h, nlev, pool_off, pool):
    n = boxes.shape[0]
    for o in prange(n):
        x0 = boxes[o, 0]
        y0 = boxes[o, 1]
        x1 = boxes[o, 2]
        y1 = boxes[o, 3]
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        ss = oss[o]
        tile = np.zeros((h * ss, w * ss, 4), np.float32)
        zb = np.full((h * ss, w * ss), np.inf, np.float32)
        scratch = np.zeros((h * ss, w * ss), np.float32)
        _draw_object(tile, zb, scratch, ss, float(x0), float(y0),
                     otr[o, 0], otr[o, 1], txyz, tuv, tcol, tk, tpage,
                     osr[o, 0], osr[o, 1], sp, spts, scol, obias[o],
                     tex, lev_off, lev_w, lev_h, nlev)
        inv = oalpha[o] / (ss * ss)
        base = pool_off[o]
        for y in range(h):
            for x in range(w):
                for c in range(4):
                    acc = 0.0
                    for j in range(ss):
                        for i in range(ss):
                            acc += tile[y * ss + j, x * ss + i, c]
                    pool[base + (y * w + x) * 4 + c] = acc * inv
    # composite in object order, rows in parallel
    H = out.shape[0]
    for y in prange(H):
        for o in range(n):
            x0 = boxes[o, 0]
            y0 = boxes[o, 1]
            x1 = boxes[o, 2]
            y1 = boxes[o, 3]
            if y < y0 or y >= y1 or x1 <= x0:
                continue
            w = x1 - x0
            base = pool_off[o] + ((y - y0) * w) * 4
            for x in range(x0, x1):
                i = base + (x - x0) * 4
                a = pool[i + 3]
                if a <= 0.0:
                    continue
                ia = 1.0 - a
                out[y, x, 0] = pool[i] + out[y, x, 0] * ia
                out[y, x, 1] = pool[i + 1] + out[y, x, 1] * ia
                out[y, x, 2] = pool[i + 2] + out[y, x, 2] * ia
                out[y, x, 3] = a + out[y, x, 3] * ia


# ---------------------------------------------------------------- python side

class Atlas:
    """Texture pyramid of page maps, flattened for the kernel."""

    def __init__(self, levels):
        # levels: list of uint8 arrays (n_pages, h, w, 3), level 0 first
        self.n = levels[0].shape[0]
        self.w0 = levels[0].shape[2]
        self.h0 = levels[0].shape[1]
        offs, ws, hs, flat = [], [], [], []
        o = 0
        for L in levels:
            offs.append(o)
            hs.append(L.shape[1])
            ws.append(L.shape[2])
            flat.append(np.ascontiguousarray(L).ravel())
            o += L.size
        self.flat = np.concatenate(flat)
        self.off = np.array(offs, np.int64)
        self.w = np.array(ws, np.int64)
        self.h = np.array(hs, np.int64)
        self.nlev = len(levels)


_EMPTY_ATLAS = None


def empty_atlas():
    global _EMPTY_ATLAS
    if _EMPTY_ATLAS is None:
        _EMPTY_ATLAS = Atlas([np.full((1, 2, 2, 3), 255, np.uint8)])
    return _EMPTY_ATLAS


class DrawList:
    """Objects in draw order. Add batches of objects that share a structure.

    add(tri_xyz (B,T,3,3), tri_uv (B,T,3,2), tri_col (B,T,4), tri_k (B,T),
        tri_page (B,T), str_pts (B,S,N,4), str_col (B,S,4), ss (B,), depth (B,),
        alpha (B,), bias (B,))
    Any of the stroke or triangle parts may be None. Objects are sorted far to
    near by depth when drawn (stable, so equal depths keep insertion order).
    """

    def __init__(self, w=1920, h=1080):
        self.w, self.h = w, h
        self.batches = []

    def add(self, tri_xyz=None, tri_uv=None, tri_col=None, tri_k=None, tri_page=None,
            str_pts=None, str_col=None, ss=3, depth=0.0, alpha=1.0, bias=0.0, order=None):
        B = None
        if tri_xyz is not None:
            B = tri_xyz.shape[0]
        elif str_pts is not None:
            B = str_pts.shape[0]
        if B is None or B == 0:
            return
        bc = lambda v, dt: np.broadcast_to(np.asarray(v, dt), (B,)).copy()
        self.batches.append(dict(
            tri_xyz=tri_xyz, tri_uv=tri_uv, tri_col=tri_col, tri_k=tri_k, tri_page=tri_page,
            str_pts=str_pts, str_col=str_col, ss=bc(ss, np.int64), depth=bc(depth, np.float64),
            alpha=bc(alpha, np.float32), bias=bc(bias, np.float32),
            order=bc(0 if order is None else order, np.float64)))

    def render(self, out=None, atlas=None, tile=TILE):
        W, H = self.w, self.h
        if out is None:
            out = np.zeros((H, W, 4), np.float32)
        if not self.batches:
            return out
        atlas = atlas or empty_atlas()
        # flatten
        txyz, tuv, tcol, tk, tpage = [], [], [], [], []
        spts, scol, slen = [], [], []
        obj = []  # (order, depth, seq, tri0, tri1, str0, str1, box, ss, alpha, bias)
        nt = 0
        ns = 0
        seq = 0
        for b in self.batches:
            B = len(b["ss"])
            T = 0 if b["tri_xyz"] is None else b["tri_xyz"].shape[1]
            S = 0 if b["str_pts"] is None else b["str_pts"].shape[1]
            N = 0 if b["str_pts"] is None else b["str_pts"].shape[2]
            lo = np.full((B, 2), np.inf)
            hi = np.full((B, 2), -np.inf)
            if T:
                xyz = b["tri_xyz"].reshape(B, T * 3, 3)
                vis = (b["tri_col"][..., 3] > 0.002)
                visv = np.repeat(vis, 3, axis=1)
                xs = np.where(visv, xyz[..., 0], np.nan)
                ys = np.where(visv, xyz[..., 1], np.nan)
                with np.errstate(all="ignore"):
                    lo[:, 0] = np.fmin(lo[:, 0], np.nanmin(np.where(np.isnan(xs), np.inf, xs), axis=1))
                    lo[:, 1] = np.fmin(lo[:, 1], np.nanmin(np.where(np.isnan(ys), np.inf, ys), axis=1))
                    hi[:, 0] = np.fmax(hi[:, 0], np.nanmax(np.where(np.isnan(xs), -np.inf, xs), axis=1))
                    hi[:, 1] = np.fmax(hi[:, 1], np.nanmax(np.where(np.isnan(ys), -np.inf, ys), axis=1))
                txyz.append(b["tri_xyz"].reshape(B * T, 9))
                tuv.append(np.zeros((B * T, 6), np.float32) if b["tri_uv"] is None else b["tri_uv"].reshape(B * T, 6))
                tcol.append(b["tri_col"].reshape(B * T, 4))
                tk.append(np.zeros(B * T, np.float32) if b["tri_k"] is None else b["tri_k"].reshape(B * T))
                tpage.append(np.full(B * T, -1, np.int64) if b["tri_page"] is None else b["tri_page"].reshape(B * T))
            if S:
                P = b["str_pts"]
                vis = (b["str_col"][..., 3] > 0.002)[..., None] & (P[..., 3] > 0.0)
                r = P[..., 3] + 2.0
                xs0 = np.where(vis, P[..., 0] - r, np.inf).reshape(B, -1).min(axis=1)
                ys0 = np.where(vis, P[..., 1] - r, np.inf).reshape(B, -1).min(axis=1)
                xs1 = np.where(vis, P[..., 0] + r, -np.inf).reshape(B, -1).max(axis=1)
                ys1 = np.where(vis, P[..., 1] + r, -np.inf).reshape(B, -1).max(axis=1)
                lo[:, 0] = np.minimum(lo[:, 0], xs0)
                lo[:, 1] = np.minimum(lo[:, 1], ys0)
                hi[:, 0] = np.maximum(hi[:, 0], xs1)
                hi[:, 1] = np.maximum(hi[:, 1], ys1)
                spts.append(P.reshape(B * S * N, 4))
                scol.append(b["str_col"].reshape(B * S, 4))
                slen.append(np.full(B * S, N, np.int64))
            for i in range(B):
                if not np.isfinite(lo[i]).all() or not np.isfinite(hi[i]).all():
                    continue
                x0 = max(0, int(math.floor(lo[i, 0])) - 1)
                y0 = max(0, int(math.floor(lo[i, 1])) - 1)
                x1 = min(W, int(math.ceil(hi[i, 0])) + 2)
                y1 = min(H, int(math.ceil(hi[i, 1])) + 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                obj.append((b["order"][i], -b["depth"][i], seq, nt + i * T, nt + (i + 1) * T,
                            ns + i * S, ns + (i + 1) * S, (x0, y0, x1, y1), int(b["ss"][i]),
                            float(b["alpha"][i]), float(b["bias"][i])))
                seq += 1
            nt += B * T
            ns += B * S
        if not obj:
            return out
        obj.sort(key=lambda o: (o[0], o[1], o[2]))
        # split big objects into screen tiles
        boxes, oss, oal, obi, otr, osr = [], [], [], [], [], []
        for o in obj:
            x0, y0, x1, y1 = o[7]
            for ty in range(y0, y1, tile):
                for tx in range(x0, x1, tile):
                    boxes.append((tx, ty, min(x1, tx + tile), min(y1, ty + tile)))
                    oss.append(o[8])
                    oal.append(o[9])
                    obi.append(o[10])
                    otr.append((o[3], o[4]))
                    osr.append((o[5], o[6]))
        boxes = np.array(boxes, np.int64)
        f32 = lambda L, shape: np.ascontiguousarray(np.concatenate(L).astype(np.float32)) if L else np.zeros(shape, np.float32)
        TXYZ = f32(txyz, (1, 9))
        TUV = f32(tuv, (1, 6))
        TCOL = f32(tcol, (1, 4))
        TK = f32(tk, (1,))
        TPAGE = np.ascontiguousarray(np.concatenate(tpage).astype(np.int64)) if tpage else np.zeros(1, np.int64)
        SPTS = f32(spts, (1, 4))
        SCOL = f32(scol, (1, 4))
        if slen:
            L = np.concatenate(slen)
            SP = np.zeros(len(L) + 1, np.int64)
            SP[1:] = np.cumsum(L)
        else:
            SP = np.zeros(2, np.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * 4
        pool_off = np.zeros(len(boxes), np.int64)
        pool_off[1:] = np.cumsum(areas)[:-1]
        pool = np.zeros(int(areas.sum()), np.float32)
        _render_all(out, boxes, np.array(oss, np.int64), np.array(oal, np.float32),
                    np.array(obi, np.float32), np.array(otr, np.int64).reshape(-1, 2),
                    np.array(osr, np.int64).reshape(-1, 2),
                    TXYZ, TUV, TCOL, TK, TPAGE, SP, SPTS, SCOL,
                    atlas.flat, atlas.off, atlas.w, atlas.h, atlas.nlev, pool_off, pool)
        return out
