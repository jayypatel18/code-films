"""The shots of TINY ROCKET: acting, timing and compositing per frame (2x canvas in, 1x uint8 out)."""

import numpy as np
from . import film as F
from . import scenes as S
from . import bg as B
from . import raster as R
from . import post
from .film import seg, keys, ease, Shot, Film, FPS
from . import world as WO

_memo = {}


def memo(key, fn):
    if key not in _memo:
        if len(_memo) > 6:
            _memo.clear()
        _memo[key] = fn()
    return _memo[key]


# ----------------------------------------------------------------------------
# shot 1: the last knot (0 to 5 s)
# ----------------------------------------------------------------------------

def blink_curve(t, t0, dur=0.25):
    """Eye openness for a blink starting at t0: close fast, open slower."""
    u = (t - t0) / dur
    if u < 0 or u > 1:
        return 1.0
    return 1 - np.sin(np.pi * min(u / 0.8, 1.0)) ** 0.8 if u < 0.8 else 1.0


def flick(t, t0, dur=0.3):
    u = (t - t0) / dur
    if u < 0 or u > 1:
        return 0.0
    return float(np.sin(np.pi * u) ** 0.6)


def shot1_render(shot, f, t):
    bg = memo("s1bg", lambda: F.cache_bg("shot1_bg", lambda: S.paint_closeup_bg(1)))
    step = 2
    d = f // step
    td = d * step / FPS
    # acting timeline
    pull = keys(td, [(0.0, 0.35), (0.6, 0.35), (0.85, 0.15), (0.95, 1.1), (1.1, 0.95), (1.4, 1.0)], "smooth")
    smear = 1.0 if 0.85 <= td < 1.0 else 0.0
    pitch = keys(td, [(0.0, 0.40), (0.6, 0.40), (0.85, 0.5), (0.95, 0.28), (1.3, 0.42), (3.6, 0.42), (4.2, 0.25)], "smooth")
    head_dy = keys(td, [(0.0, 0), (0.85, 10), (0.95, -28), (1.3, 0)], "smooth")
    ear_r = 0.25 * flick(td, 1.8) - 0.1 * flick(td, 2.9, 0.35)
    ear_l = -0.2 * flick(td, 2.9, 0.35)
    tail_swish = float(np.sin(np.pi * np.clip((td - 2.2) / 1.0, 0, 1)))
    blink = min(blink_curve(td, 3.2), blink_curve(td, 4.35))
    released = td >= 3.8
    mouth = "smile" if released else "teeth"
    look = (keys(td, [(3.6, -0.35), (4.0, -0.15)]), keys(td, [(3.6, 0.45), (4.0, 0.2)]))
    brow = keys(td, [(3.6, 0.4), (4.0, 0.0)])
    cel = memo(("s1", d), lambda: S.shot1_cels(seed=1 + (d % 3), pull=pull, blink=blink, head_dy=head_dy, pitch=pitch,
                                               look=look, mouth=mouth, ear_l=ear_l, ear_r=ear_r, smear=smear,
                                               string_in_mouth=not released, tail_swish=tail_swish, brow=brow))
    frame = bg.copy()
    cel.over(frame, tint=(1.0, 0.94, 0.90))
    S.rim_light(frame, cel, (1.0, 0.70, 0.30), 0.8, -0.5, strength=0.9, width=16, ink_w=7, seed=3 + d % 3)
    frame = post.grade(frame, lift=(0.02, 0.0, 0.0), gain=(1.02, 0.97, 0.92), sat=1.05)
    zoom = 1.0 + 0.06 * ease(t / 5.0, "linear")
    img = F.camera(frame, zoom, 0.55, 0.5)
    return F.finish1x(img, 1000 + f, grain=0.03, vig=0.32, bloom_th=0.72, bloom_sigma=20, bloom_str=0.35)


# ----------------------------------------------------------------------------
# shot 2: the hill (5 to 12 s)
# ----------------------------------------------------------------------------

def shot2_render(shot, f, t):
    bg = memo("s2bg", lambda: F.cache_bg("shot2_bg", lambda: S.paint_hill(2, stars=False, windows=False)[0]))
    glows = memo("s2glows", lambda: S.village_glow_list(2 + 7))
    props = memo("s2props", lambda: S.shot2_props())
    frame = bg.copy()
    # stars come out one by one; village lights come on along the river
    B.stars(frame, 260, 14, (0, 0, S.W2, 900), rmin=1.2, rmax=3.0, strength=0.8,
            fade=lambda xs, ys: np.clip(1 - ys / 900, 0, 1) ** 1.5 * np.clip((t / 7.0) * 1.4 - (xs / S.W2) * 0.4 - 0.1, 0, 1))
    S.village_lights(frame, glows, min(1.0, 0.15 + t / 6.0))
    # panda: two steps back (walk cycle), admires, turns to the map, looks at it
    step = 2
    d = f // step
    td = d * step / FPS
    walking = td < 1.6
    stepping = 4.2 <= td < 5.0
    px = keys(td, [(0.0, 2010), (1.6, 1840), (4.2, 1840), (5.0, 1720)], "linear")
    stride = float(np.sin(2 * np.pi * td * 1.5)) if walking else (float(np.sin(2 * np.pi * (td - 4.2) * 1.5)) if stepping else 0.0)
    bob = abs(np.sin(2 * np.pi * td * 1.5)) * 6 if walking or stepping else 0.0
    turned = td >= 4.0
    yaw = 0.65 if not turned else 0.55
    pitch = keys(td, [(0.0, -0.45), (1.6, -0.55), (3.8, -0.55), (4.0, -0.1), (5.0, -0.1), (5.4, 0.45)], "smooth")
    tail_swish = float(np.sin(np.pi * np.clip((td - 2.2) / 1.2, 0, 1))) + float(np.sin(np.pi * np.clip((td - 6.0) / 1.0, 0, 1)))
    ear_r = 0.25 * flick(td, 3.0) + 0.2 * flick(td, 6.2)
    blink = min(blink_curve(td, 2.4), blink_curve(td, 5.9))
    look = (0.3, -0.3) if not turned else (-0.2, 0.4)
    cel = memo(("s2p", d), lambda: S.shot2_panda(seed=2 + (d % 3), px=px, yaw=yaw, pitch=pitch, stride=stride, bob=bob,
                                                 flip=turned, look=look, ear_r=ear_r, eye_open=blink, tail_swish=tail_swish))
    props.over(frame, tint=(1.0, 0.92, 0.88))
    cel.over(frame, tint=(1.0, 0.92, 0.88))
    S.rim_light(frame, cel, (0.9, 0.55, 0.2), -0.8, -0.5, strength=0.6, width=5, ink_w=2, seed=5 + d % 3)
    frame = post.grade(frame, lift=(0.01, 0.0, 0.03), gain=(1.0, 0.97, 0.98), sat=1.08)
    zoom = 1.0 + 0.03 * (t / 7.0)
    img = F.camera(frame, zoom, 0.5, 0.52)
    return F.finish1x(img, 2000 + f, grain=0.03, vig=0.3, bloom_th=0.7, bloom_sigma=16, bloom_str=0.3)


# ----------------------------------------------------------------------------
# shot 3: waiting for morning (12 to 19 s)
# ----------------------------------------------------------------------------

def shot3_render(shot, f, t):
    bg = memo("s3bg", lambda: F.cache_bg("shot3_bg", lambda: S.paint_hill(3, stars=False, windows=False, palette="night")[0]))
    glows = memo("s2glows", lambda: S.village_glow_list(2 + 7))
    cat = memo("s3stars", lambda: S.star_field(3))
    props = memo("s3props", lambda: S.shot2_props(light=(-0.3, -0.9), seed=2))
    frame = bg.copy()
    S.draw_stars(frame, cat, 12.0 + t, strength=1.5)
    S.shooting_star(frame, t, 3.6)
    S.village_lights(frame, glows, 1.0)
    step = 3
    d = f // step
    td = d * step / FPS
    ear_r = 0.3 * flick(td, 3.0, 0.4) + 0.2 * flick(td, 5.6, 0.35)
    cel = memo(("s3p", d), lambda: S.shot3_panda(seed=3 + (d % 3), ear_r=ear_r))
    night = (0.42, 0.48, 0.70)
    props.over(frame, tint=night)
    cel.over(frame, tint=night)
    S.rim_light(frame, cel, (0.55, 0.65, 0.95), -0.3, -0.8, strength=0.5, width=5, ink_w=2, seed=7 + d % 3)
    S.fireflies(frame, t)
    frame = post.grade(frame, lift=(0.02, 0.02, 0.05), gain=(1.08, 1.1, 1.2), sat=1.0)
    zoom = 1.55 + 0.18 * ease(t / 7.0, "linear")
    img = F.camera(frame, zoom, 0.575, 0.62)
    return F.finish1x(img, 3000 + f, grain=0.04, vig=0.4, bloom_th=0.45, bloom_sigma=14, bloom_str=0.5)


# ----------------------------------------------------------------------------
# shot 4: first light (19 to 25 s)
# ----------------------------------------------------------------------------

def shot4_render(shot, f, t):
    bg = memo("s4bg", lambda: F.cache_bg("shot4_bg", lambda: S.paint_hill(4, stars=False, windows=False, palette="predawn")[0]))
    cat = memo("s3stars", lambda: S.star_field(3))
    light = (0.5, -0.8)
    props = memo("s4props", lambda: S.shot2_props(light=light, seed=2))
    frame = bg.copy()
    S.draw_stars(frame, cat, 19.0 + t, strength=0.35 * max(0.0, 1 - t / 3.5))
    step = 2
    d = f // step
    td = d * step / FPS
    # phases
    if td < 0.6:
        phase, u = "sleep", td / 0.6
    elif td < 1.2:
        phase, u = "sit", (td - 0.6) / 0.6
    elif td < 1.5:
        phase, u = "stand", (td - 1.2) / 0.3
    elif td < 2.1:
        phase, u = "walk", (td - 1.5) / 0.6
    elif td < 2.4:
        phase, u = "crouch", (td - 2.1) / 0.3
    elif td < 2.8:
        # keyed: launch stretch (held), tuck at the top (held), reach for the rim (held)
        uu = (td - 2.4) / 0.4
        phase, u = "leap", (0.18 if uu < 0.3 else (0.55 if uu < 0.65 else 0.9))
    else:
        phase, u = "inside", min(1.0, (td - 2.8) / 0.3)
    px = 2110 + 120 * seg(td, 1.5, 2.1, "linear")
    ear_r = 0.3 * flick(td, 0.2, 0.4)
    blink = blink_curve(td, 3.1)
    cel = memo(("s4p", d), lambda: S.shot4_panda(phase, u, light=light, seed=4 + (d % 3), px=px, ear_r=ear_r, blink=blink))
    tint = (0.72, 0.76, 0.95)
    if phase == "inside":
        gm = memo("s4glass", lambda: S.glass_mask())
        props.over(frame, tint=tint)
        cel.alpha *= gm
        cel.over(frame, tint=(0.6, 0.64, 0.8))
    else:
        props.over(frame, tint=tint)
        cel.over(frame, tint=tint)
        S.rim_light(frame, cel, (0.75, 0.80, 1.0), 0.5, -0.8, strength=0.5, width=5, ink_w=2, seed=9 + d % 3)
    frame = post.grade(frame, lift=(0.0, 0.0, 0.03), gain=(0.96, 0.98, 1.06), sat=0.95)
    zoom = 1.7 - 0.08 * ease(t / 3.5, "smooth")
    img = F.camera(frame, zoom, 0.585, 0.56)
    return F.finish1x(img, 4000 + f, grain=0.035, vig=0.35, bloom_th=0.6, bloom_sigma=14, bloom_str=0.3)


def shot4b_render(shot, f, t):
    step = 2
    d = f // step
    td = d * step / FPS
    goggles_up = td < 0.8
    pitch = keys(td, [(0.0, 0.0), (0.75, 0.15), (0.85, -0.05), (1.0, 0.0), (1.95, 0.0), (2.1, 0.45), (2.3, -0.15), (2.5, 0.0)], "smooth")
    head_dy = keys(td, [(0.0, 0), (1.35, 0), (1.45, 10), (1.6, -4), (1.75, 0)], "smooth")
    swallow = float(np.sin(np.pi * np.clip((td - 1.35) / 0.4, 0, 1)))
    brow = keys(td, [(0.0, -0.5), (1.8, -0.5), (2.1, 0.9)], "smooth")
    mouth = "line" if td < 2.05 else "smile"
    blink = min(blink_curve(td, 0.5), blink_curve(td, 1.5, 0.2))
    look = (keys(td, [(0.0, 0.1), (0.6, -0.2), (1.2, 0.0)]), keys(td, [(0.0, -0.1), (1.2, 0.1), (2.0, -0.05)]))
    frame, cel = memo(("s4b", d), lambda: S.shot4b_frame(goggles_up=goggles_up, pitch=pitch, brow=brow, mouth=mouth, look=look,
                                                         head_dy=head_dy, blink=blink, swallow=swallow, seed=4 + (d % 3)))
    frame = frame.copy()
    frame = post.grade(frame, lift=(0.0, 0.0, 0.02), gain=(0.9, 0.92, 1.0), sat=0.95)
    zoom = 1.25 + 0.05 * (t / 2.5)
    img = F.camera(frame, zoom, 0.5, 0.52)
    return F.finish1x(img, 4500 + f, grain=0.035, vig=0.4, bloom_th=0.7, bloom_sigma=14, bloom_str=0.25)


# ----------------------------------------------------------------------------
# shot 5: the spark (25 to 31 s)
# ----------------------------------------------------------------------------

def shot5a_render(shot, f, t):
    bg = memo("s5abg", lambda: F.cache_bg("shot5a_bg", lambda: S.paper_wall_bg(5)))
    step = 2
    d = f // step
    td = d * step / FPS
    # paw comes in from the right (0 to 0.25), strikes fast (0.3 to 0.42), match lit after
    u = keys(td, [(0.0, -0.1), (0.25, 0.05), (0.32, 0.05), (0.42, 0.95), (0.55, 0.85)], "smooth")
    lit = td >= 0.42
    frame = bg.copy()
    cel, head = memo(("s5a", d), lambda: S.shot5a_cel(u, lit, 0.0, seed=5 + (d % 3)))
    cel.over(frame, tint=(0.85, 0.85, 0.95))
    if 0.36 <= td < 0.5:
        S.sparks(frame, head, 50 + d, n=30, size=1.3)
    if lit:
        gl = np.array([[head[0], head[1] - 50, 420 + 40 * np.sin(t * 31), 1.0, 0.6, 0.2, 0.35 + 0.1 * np.sin(t * 23)],
                       [head[0], head[1] - 50, 140, 1.0, 0.8, 0.4, 0.8]])
        R.add_glows(frame, gl)
        S.rim_light(frame, cel, (1.0, 0.7, 0.3), -0.3, -0.6, strength=0.8, width=12, ink_w=6, seed=11 + d % 3)
    frame = post.grade(frame, lift=(0.0, 0.0, 0.02), gain=(0.82, 0.84, 0.92), sat=0.95)
    img = F.camera(frame, 1.15, 0.5, 0.5)
    return F.finish1x(img, 5000 + f, grain=0.035, vig=0.45, bloom_th=0.65, bloom_sigma=18, bloom_str=0.5)


def shot5b_render(shot, f, t):
    bg = memo("s5bbg", lambda: F.cache_bg("shot5b_bg", lambda: S.fuse_bg(5)))
    step = 3
    d = f // step
    td = d * step / FPS
    # the spark runs the fuse on hard holds: it jumps every three frames
    spark_t = min(1.0, td / 1.9)
    cel, pos = memo(("s5b", d), lambda: S.shot5b_cels(spark_t, seed=5 + (d % 3)))
    frame = bg.copy()
    cel.over(frame, tint=(0.72, 0.76, 0.95))
    S.sparks(frame, pos, 70 + d, n=22, size=1.0)
    frame = post.grade(frame, lift=(0.0, 0.0, 0.03), gain=(0.9, 0.93, 1.05), sat=0.95)
    zoom = 1.12
    img = F.camera(frame, zoom, 0.5 + 0.08 * spark_t, 0.6)
    return F.finish1x(img, 5300 + f, grain=0.04, vig=0.45, bloom_th=0.6, bloom_sigma=16, bloom_str=0.5)


def shot5c_render(shot, f, t):
    step = 2
    d = f // step
    td = d * step / FPS
    # eyes squeeze shut (0 to 0.3), breath held, total stillness 0.8 to 1.9 (one drawing, no boil), one eye opens at 2.0
    if td < 0.3:
        kw = dict(squeeze=0.0, blink=max(0.05, 1 - td / 0.25), brow=-0.6, mouth="line")
    elif td < 2.0:
        kw = dict(squeeze=1.0, blink=0.0, brow=-0.3, mouth="line", cheek=0.6)
    else:
        kw = dict(squeeze=0.0, blink=1.0, eye_open_r=0.0, eye_open_l=min(1.0, 0.3 + (td - 2.0) * 1.5), brow=-0.4, mouth="line", look=(-0.4, -0.2))
    still_hold = 0.8 <= td < 1.9
    seed = 5 if still_hold else 5 + (d % 3)
    frame, cel = memo(("s5c", d if not still_hold else -1), lambda: S.shot4b_frame(goggles_up=False, pitch=0.1, head_dy=20, seed=seed, light=(0.3, -0.9), **kw))
    frame = frame.copy()
    flick_ = 0.5 + 0.5 * np.sin(t * 37) * np.sin(t * 23)
    R.add_glows(frame, np.array([[600, 2300, 1900, 1.0, 0.6, 0.25, 0.25 + 0.2 * flick_]]))
    frame = post.grade(frame, lift=(0.0, 0.0, 0.02), gain=(0.86, 0.86, 0.96), sat=0.92)
    img = F.camera(frame, 1.75, 0.46, 0.5)
    return F.finish1x(img, 5600 + f, grain=0.035, vig=0.5, bloom_th=0.65, bloom_sigma=16, bloom_str=0.3)


# ----------------------------------------------------------------------------
# shot 6: lift (31 to 37 s)
# ----------------------------------------------------------------------------

def shot6a_render(shot, f, t):
    bg = memo("s4bg", lambda: F.cache_bg("shot4_bg", lambda: S.paint_hill(4, stars=False, windows=False, palette="predawn")[0]))
    light = (0.5, -0.8)
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    # flash, sparks, smoke, shudder, then the leap
    S.grass_ring(frame, td - 0.05)
    shud = 9 * np.sin(td * 70) * seg(td, 0.25, 0.35) * max(0.0, 1 - (td - 0.35) / 0.9) if td < 1.25 else 0.0
    lift = -2600 * max(0.0, td - 1.2) ** 2
    ghost = 60 if 1.35 <= td < 1.7 else 0.0
    cel = memo(("s6a", d), lambda: S.rocket_cel(dy=lift, shudder=shud, light=light, seed=2 + (d % 3), ghost=ghost))
    # smoke column behind the rocket once it lifts
    if td > 1.2:
        col = [(2330 + 20 * np.sin(k * 1.3), S.hill_crest(2330) + 20 + lift * (1 - k / 14) + 40, 60 + 30 * k, 0.8 * (1 - k / 16)) for k in range(14)]
        S.smoke(frame, col, 60 + d, base=(0.70, 0.66, 0.70), lit=(1.0, 0.78, 0.5), light_dir=(0.0, 1.0))
    props_tint = (0.8, 0.82, 0.98)
    cel.over(frame, tint=props_tint)
    if td > 1.2:
        fl = memo(("s6fl", d), lambda: S.flame_cel(2330, S.hill_crest(2330) + 20 + lift - 0.2 * 640 + 0.06 * 640, 180.0, 600 + d))
        fl.over(frame)
    S.smoke(frame, S.ignition_puffs(td), 50 + d % 3, base=(0.74, 0.70, 0.74), lit=(1.0, 0.76, 0.46), light_dir=(0.0, 1.0))
    if td < 0.45:
        S.sparks(frame, (2330, S.hill_crest(2330) + 10), 80 + d, n=40, size=2.2 * (1 - td / 0.5) + 0.4)
    gl = np.array([[2330, S.hill_crest(2330), 700 * (1 + td), 1.0, 0.7, 0.3, max(0.0, 1.2 - td * 1.5)]])
    R.add_glows(frame, gl)
    frame = post.grade(frame, lift=(0.0, 0.0, 0.03), gain=(0.98, 0.98, 1.05), sat=1.0)
    zoom = 1.25
    shake = (np.sin(td * 53) * 0.004 + np.sin(td * 31) * 0.003) * seg(td, 0.0, 0.1) * max(0.0, 1 - td / 1.8)
    img = F.camera(frame, zoom, 0.595 + shake, 0.55 + shake * 0.7)
    return F.finish1x(img, 6000 + f, grain=0.035, vig=0.35, bloom_th=0.6, bloom_sigma=18, bloom_str=0.5)


def rocket6b_path(u):
    return (2350 - 300 * u, 2500 - 3100 * u ** 1.1)


def shot6b_render(shot, f, t):
    bg = memo("s6bbg", lambda: F.cache_bg("shot6b_bg", lambda: S.low_angle_bg(6)))
    light = (0.2, 0.9)  # lit from below by its own flame and the dawn
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    u = td / 3.6
    x, y = rocket6b_path(u)
    puffs = [(px_, py_, r_ * 0.45, a_ * 0.35) for (px_, py_, r_, a_) in S.trail_puffs(lambda tb: rocket6b_path(tb / 3.6), td, 0.0, 70, life=1.6, every=0.1, r0=22, grow=70, drift=140)]
    S.smoke(frame, puffs, 61 + d % 3, base=(0.82, 0.76, 0.78), lit=(1.0, 0.84, 0.58), light_dir=(0.0, 1.0))
    fl = memo(("s6bfl", d), lambda: S.flame_cel(x, y + 60, 150.0, 700 + d))
    fl.over(frame)
    rc = memo(("s6br", d), lambda: S.small_rocket_cel(x, y, 420.0, light, 2 + d % 3, rot=-0.08, fin_flap=12 * np.sin(td * 40)))
    rc.over(frame, tint=(0.95, 0.9, 0.95))
    birds = memo(("s6bb", f // 3), lambda: S.birds_cel(f // 3 * 3 / FPS, 0.6, 6))
    birds.over(frame)
    R.add_glows(frame, np.array([[x, y + 80, 300, 1.0, 0.7, 0.3, 0.6]]))
    frame = post.grade(frame, lift=(0.0, 0.0, 0.02), gain=(1.0, 0.97, 1.0), sat=1.05)
    zoom = 1.08
    img = F.camera(frame, zoom, 0.5, 0.5 - 0.06 * ease(t / 4.0, "smooth"))
    return F.finish1x(img, 6500 + f, grain=0.035, vig=0.35, bloom_th=0.68, bloom_sigma=18, bloom_str=0.4)


# ----------------------------------------------------------------------------
# shot 7: through the clouds (37 to 44 s)
# ----------------------------------------------------------------------------

def shot7a_render(shot, f, t):
    bg = memo("s7bg", lambda: F.cache_bg("shot7_bg", lambda: S.sky_climb_bg(7)))
    layers = memo("s7layers", lambda: [
        S.cloud_layer(71, 1900, 260, 7, (0.70, 0.60, 0.68), (1.0, 0.86, 0.62), 0.95),
        S.cloud_layer(72, 1500, 420, 6, (0.78, 0.70, 0.76), (1.0, 0.90, 0.70), 0.95),
        S.cloud_layer(73, 2100, 520, 5, (0.84, 0.78, 0.82), (1.0, 0.94, 0.78), 1.0),
    ])
    light = (0.3, 0.9)
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    # clouds stream down past the climbing rocket at three speeds (layer 0 far, 2 near)
    S.over_layer(frame, layers[0], 420 * t + 300, 1.0)
    rc = memo(("s7r", d), lambda: S.small_rocket_cel(1920, 1250, 620.0, light, 2 + d % 3, rot=0.04 * np.sin(td * 9), fin_flap=16 * np.sin(td * 37)))
    fl = memo(("s7fl", d), lambda: S.flame_cel(1920 + 12 * np.sin(td * 9), 1250 - 0.2 * 620 + 0.06 * 620, 230.0, 710 + d))
    S.over_layer(frame, layers[1], 800 * t + 900, 1.0)
    fl.over(frame)
    rc.over(frame, tint=(1.0, 0.95, 0.95))
    S.over_layer(frame, layers[2], 1500 * t + 400, 0.95)
    R.add_glows(frame, np.array([[1920, 1300, 500, 1.0, 0.75, 0.4, 0.4]]))
    frame = post.grade(frame, lift=(0.01, 0.0, 0.0), gain=(1.02, 0.98, 0.95), sat=1.05)
    zoom = 1.1
    img = F.camera(frame, zoom, 0.5 + 0.004 * np.sin(td * 21), 0.55 + 0.004 * np.sin(td * 17))
    return F.finish1x(img, 7000 + f, grain=0.035, vig=0.3, bloom_th=0.7, bloom_sigma=22, bloom_str=0.45)


def shot7b_render(shot, f, t):
    step = 2
    d = f // step
    td = d * step / FPS
    wind = keys(td, [(0.0, 1.0), (2.0, 1.0), (2.5, 0.0)], "smooth")
    fear = td < 2.2
    kw = dict(cheek=wind, brow=(-1.0 if fear else 0.0), mouth=("line" if fear else "o"), mouth_open=(0.0 if fear else 1.0),
              look=((0.0, 0.1) if fear else (0.0, -0.5)), blink=(0.55 if fear else 1.0), pitch=(0.15 if fear else -0.35))
    frame, cel = memo(("s7b", d), lambda: S.shot4b_frame(goggles_up=False, head_dy=10 + 6 * np.sin(td * 40) * wind, seed=7 + (d % 3), light=(0.2, 0.9), **kw))
    frame = frame.copy()
    # the sky outside brightens from cloud grey to gold as it breaks into open sky
    gold = seg(td, 2.1, 2.9)
    R.add_glows(frame, np.array([[1500, 700, 1500, 1.0, 0.85, 0.55, 0.25 + 0.6 * gold]]))
    frame = post.grade(frame, lift=(0.0, 0.0, 0.0), gain=(0.9 + 0.15 * gold, 0.9 + 0.1 * gold, 0.95), sat=0.95 + 0.1 * gold)
    img = F.camera(frame, 1.12, 0.52 + 0.003 * np.sin(td * 29) * wind, 0.5 + 0.004 * np.sin(td * 23) * wind)
    return F.finish1x(img, 7500 + f, grain=0.035, vig=0.4, bloom_th=0.65, bloom_sigma=18, bloom_str=0.3 + 0.3 * gold)


# ----------------------------------------------------------------------------
# shot 8: the shooting star (44 to 52 s)
# ----------------------------------------------------------------------------

def shot8_render(shot, f, t):
    bg = memo("s8bg", lambda: F.cache_bg("shot8_bg", lambda: WO.world_bg(8)))
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    # the rocket is a speck climbing from the lower left; at 2 s its trail ignites into the star
    u = td / 7.2
    ux = min(u, 0.28)
    if td < 2.0:
        x, y = WO.star_arc(u)
        rc = memo(("s8r", d), lambda: S.small_rocket_cel(x, y, 90.0, (0.3, 0.9), 2 + d % 3, rot=-0.9 + 0.5 * u))
        puffs = [(px_, py_, r_ * 0.4, a_ * 0.5) for (px_, py_, r_, a_) in S.trail_puffs(lambda tb: WO.star_arc(tb / 7.2), td, 0.0, 80, life=1.6, every=0.06, r0=14, grow=40)]
        S.smoke(frame, puffs, 81 + d % 3, base=(0.9, 0.84, 0.84), lit=(1.0, 0.9, 0.7))
        rc.over(frame)
        R.add_glows(frame, np.array([[x, y, 60, 1.0, 0.8, 0.5, 0.8]]))
    else:
        WO.star_streak(frame, u, 80 + d % 3, head=1.0 + 0.12 * np.sin(td * 9))
    frame = post.grade(frame, lift=(0.0, 0.0, 0.01), gain=(1.02, 0.98, 0.98), sat=1.08)
    zoom = 1.18 - 0.18 * ease(t / 8.0, "smooth")
    img = F.camera(frame, zoom, 0.5, 0.52)
    return F.finish1x(img, 8000 + f, grain=0.03, vig=0.3, bloom_th=0.68, bloom_sigma=26, bloom_str=0.55)


# ----------------------------------------------------------------------------
# shot 9: home (52 to 60 s), title (60 to 63.5), fade (63.5 to 65)
# ----------------------------------------------------------------------------

def shot9_panda(phase, u, light, seed, px=2150, **kw):
    from .panda import pose_stand, pose_sit_front, draw_head, Pose as P_, Transform as T_, draw_leg
    cel = S.Cel(S.W2, S.H2)
    rx = 2330
    base_y = S.hill_crest(px) - 100
    wx, wy = rx - 10, S.hill_crest(rx) + 20 - 0.84 * 640 + 0.3 * 640
    if phase == "window":
        Tp = T_(wx - 5, wy + 8 + 30 * (1 - u), 48.0)
        pose = P_(yaw=0.0, pitch=0.1, eye_open=1.0, mouth="smile", goggles=True, goggles_up=False, seed=seed + 24)
        draw_head(cel, Tp, 0, 0, pose, light)
        S.soot_and_goggles(cel, Tp, pose, light, seed)
    elif phase == "hop":
        x = wx - 60 + (px - wx + 60) * u
        y = wy + 40 + (base_y - wy - 40) * u ** 1.4 - 200 * np.sin(np.pi * u)
        Tp = T_(x, y, 58.0, flip=True, rot=0.5 - 0.6 * u)
        pose_stand(cel, Tp, P_(yaw=0.4, pitch=0.3, eye_open=0.6, mouth="line", goggles=True, goggles_up=False, seed=seed + 25), light, stride=0.8, head_up=0.3)
    elif phase == "land":
        Tp = T_(px, base_y + 14 * (1 - u), 58.0, flip=True)
        pose_stand(cel, Tp, P_(yaw=0.3, pitch=0.3 * (1 - u), eye_open=1.0, mouth="line", goggles=True, goggles_up=False, seed=seed + 26), light, crouch=1 - u)
    elif phase == "lookup":
        Tp = T_(px, base_y, 58.0, flip=True)
        pose_stand(cel, Tp, P_(yaw=0.35, pitch=-0.75, eye_open=kw.get("blink", 1.0), mouth="line", goggles=True, goggles_up=False, look=(0.0, -0.5), seed=seed + 27), light, head_up=0.5)
    elif phase == "map":
        Tp = T_(px, base_y - 20, 58.0)
        pose_sit_front(cel, Tp, P_(yaw=0.0, pitch=0.5, eye_open=kw.get("blink", 1.0), mouth="line", goggles=True, goggles_up=True, look=(0.0, 0.6), seed=seed + 28), light)
        # the star map held in front, a paw painting a new star
        card = np.array([(-0.9, 1.0), (0.9, 0.95), (0.95, 2.1), (-0.85, 2.15)])
        cm = cel.fill(Tp(S.wobble(card, 0.02, seed + 2, closed=True)), S.PAPER)
        S.mult_shade(cel, cm, S.SHADOW_TINT, -light[0] * 10, -light[1] * 10, 0.6)
        cel.ink_closed(Tp(card), 2.6, light=light, seed=seed + 3)
        g = S.B.rng(seed + 4)
        for k in range(9):
            cel.fill(Tp(S.ellipse(g.uniform(-0.7, 0.7), g.uniform(1.15, 1.9), 0.05, 0.05, n=8)), (0.25, 0.30, 0.50))
        paint = kw.get("paint", 0.0)
        if paint > 0:
            # the new star: a tiny brushed asterisk growing with the stroke
            sx_, sy_ = 0.35, 1.3
            for k in range(3):
                a = k * np.pi / 3 + 0.3
                r_ = 0.1 * min(1.0, paint * 3 - k)
                if r_ > 0:
                    cel.ink(Tp([(sx_ - r_ * np.cos(a), sy_ - r_ * np.sin(a)), (sx_ + r_ * np.cos(a), sy_ + r_ * np.sin(a))]), 3.0, color=(0.75, 0.22, 0.18), seed=seed + 40 + k)
        # the painting paw over the card
        pu = kw.get("paw_u", 0.0)
        draw_leg(cel, Tp, [(0.6, 1.0), (0.55, 1.1 + 0.3 * pu), (0.38 + 0.08 * np.sin(pu * 12), 1.25 + 0.1 * pu)], 0.3, light, seed + 50)
    elif phase == "look_at_us":
        Tp = T_(px, base_y - 20, 58.0)
        pose = P_(yaw=0.0, pitch=0.0, eye_open=kw.get("blink", 1.0), mouth="smile", goggles=True, goggles_up=True, look=(0.0, 0.0), seed=seed + 29)
        pose_sit_front(cel, Tp, pose, light, tail_curl=kw.get("tail_curl", 0.0))
        S.soot_and_goggles(cel, T_(px, base_y - 20, 58.0), pose, light, seed)
    return cel


def shot9_render(shot, f, t):
    bg = memo("s9bg", lambda: F.cache_bg("shot9_bg", lambda: S.paint_hill(9, stars=False, windows=False, palette="morning")[0]))
    light = (0.6, -0.75)
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    rx = 2330
    # one star still fading in the morning sky
    R.add_glows(frame, np.array([[2600, 380, 26, 1.0, 1.0, 0.95, 2.2 * max(0.0, 1 - td / 9.0)], [2600, 380, 90, 1.0, 1.0, 0.95, 0.5 * max(0.0, 1 - td / 9.0)]]))
    # the rocket drifts down under the parachute and settles
    drop = keys(td, [(0.0, -720), (3.0, 0), (3.15, 18), (3.4, 0)], "smooth")
    drop = keys(td, [(0.0, -720), (3.0, 0)], "linear") if td < 3.0 else drop
    swing = 0.25 * np.sin(td * 2.1) * max(0.0, 1 - td / 3.2)
    open_u = keys(td, [(0.0, 1.0), (3.0, 1.0), (3.2, 0.7), (4.2, 0.0)], "smooth")
    rc = memo(("s9r", d), lambda: S.rocket_cel(dy=drop, shudder=0.0, light=light, seed=2 + d % 3))
    if open_u > 0.02:
        pc = memo(("s9pc", d), lambda: S.parachute_cel(rx, S.hill_crest(rx) + 20 + drop, open_u, swing, light, 9 + d % 3, drop))
        pc.over(frame)
    rc.over(frame, tint=(1.0, 0.97, 0.94))
    # the panda
    if td < 3.8:
        cel = None
    elif td < 4.4:
        cel = memo(("s9p", d), lambda: shot9_panda("window", min(1.0, (td - 3.8) / 0.4), light, 9 + d % 3))
    elif td < 5.0:
        cel = memo(("s9p", d), lambda: shot9_panda("hop", ease((td - 4.4) / 0.6, "linear"), light, 9 + d % 3))
    elif td < 5.3:
        cel = memo(("s9p", d), lambda: shot9_panda("land", (td - 5.0) / 0.3, light, 9 + d % 3))
    elif td < 6.3:
        cel = memo(("s9p", d), lambda: shot9_panda("lookup", 0, light, 9 + d % 3, blink=blink_curve(td, 5.9)))
    elif td < 7.3:
        cel = memo(("s9p", d), lambda: shot9_panda("map", 0, light, 9 + d % 3, paint=seg(td, 6.5, 6.9, "linear"), paw_u=seg(td, 6.4, 6.9, "linear")))
    else:
        cel = memo(("s9p", d), lambda: shot9_panda("look_at_us", 0, light, 9 + d % 3, blink=blink_curve(td, 7.9), tail_curl=seg(td, 7.3, 7.9, "smooth")))
    if cel is not None:
        if td < 4.4:
            gm = memo("s4glass", lambda: S.glass_mask())
            cel.alpha *= gm
        cel.over(frame, tint=(1.0, 0.97, 0.94))
        S.rim_light(frame, cel, (1.0, 0.9, 0.6), 0.6, -0.75, strength=0.5, width=5, ink_w=2, seed=13 + d % 3)
    frame = post.grade(frame, lift=(0.01, 0.01, 0.0), gain=(1.02, 1.0, 0.96), sat=1.05)
    zoom = 1.2 + 0.05 * ease((t - 3.0) / 1.5, "smooth")
    img = F.camera(frame, zoom, 0.585, 0.36 + 0.19 * ease((t - 3.0) / 1.5, "smooth"))
    return F.finish1x(img, 9000 + f, grain=0.03, vig=0.28, bloom_th=0.75, bloom_sigma=18, bloom_str=0.3)


def title_render(shot, f, t):
    bg = memo("s9bg", lambda: F.cache_bg("shot9_bg", lambda: S.paint_hill(9, stars=False, windows=False, palette="morning")[0]))
    light = (0.6, -0.75)
    frame = bg.copy()
    step = 2
    d = f // step
    td = d * step / FPS
    rc = memo(("s9r", 9999), lambda: S.rocket_cel(dy=0.0, shudder=0.0, light=light, seed=2))
    rc.over(frame, tint=(1.0, 0.97, 0.94))
    cel = memo(("s9p", 9999), lambda: shot9_panda("look_at_us", 0, light, 9, tail_curl=1.0))
    cel.over(frame, tint=(1.0, 0.97, 0.94))
    frame = post.grade(frame, lift=(0.01, 0.01, 0.0), gain=(1.02, 1.0, 0.96), sat=1.05)
    k = ease(t / 1.5, "smooth")
    img = F.camera(frame, 1.25 + 0.15 * k, 0.585 - 0.085 * k, 0.55)
    # lettering over the sky, drawn after the crop so the card composes at frame scale
    txt = memo(("title", d), lambda: title_cel(td, 9 + d % 3))
    txt.over(img)
    fade_paper = seg(t, 3.5, 4.4)
    fade_black = seg(t, 4.5, 5.0)
    out = F.finish1x(img, 9900 + f, grain=0.03, vig=0.28, bloom_th=0.75, bloom_sigma=18, bloom_str=0.3, fade=1 - fade_paper, fade_to=(0.93, 0.89, 0.80))
    if fade_black > 0:
        out = (out.astype(np.float32) * (1 - fade_black)).astype(np.uint8)
    return out


def title_cel(td, seed):
    cel = S.Cel(1920, 1080)
    reveal = seg(td, 0.0, 1.6, "linear")
    S.brush_text(cel, "Tiny Rocket", 110, 60, 128, reveal=reveal, seed=seed, weight=0.13)
    if td >= 1.7:
        press = min(1.0, (td - 1.7) / 0.15)
        S.hanko(cel, 1760, 150, 92 * (1.15 - 0.15 * press), press=press, seed=seed + 7)
    if td >= 2.3:
        S.brush_text(cel, "Fable 5.1", 130, 228, 60, reveal=seg(td, 2.3, 3.1, "linear"), seed=seed + 20, weight=0.12)
    return cel


def build():
    film = Film()
    film.add(Shot("s1_knot", 0.0, 5.0, shot1_render))
    film.add(Shot("s2_hill", 5.0, 12.0, shot2_render))
    film.add(Shot("s3_night", 12.0, 19.0, shot3_render))
    film.add(Shot("s4_first_light", 19.0, 22.5, shot4_render))
    film.add(Shot("s4b_cockpit", 22.5, 25.0, shot4b_render))
    film.add(Shot("s5a_match", 25.0, 26.6, shot5a_render))
    film.add(Shot("s5b_fuse", 26.6, 28.6, shot5b_render))
    film.add(Shot("s5c_eyes", 28.6, 31.0, shot5c_render))
    film.add(Shot("s6a_ignition", 31.0, 33.0, shot6a_render))
    film.add(Shot("s6b_low_angle", 33.0, 37.0, shot6b_render))
    film.add(Shot("s7a_clouds", 37.0, 40.5, shot7a_render))
    film.add(Shot("s7b_face", 40.5, 44.0, shot7b_render))
    film.add(Shot("s8_star", 44.0, 52.0, shot8_render))
    film.add(Shot("s9_home", 52.0, 60.0, shot9_render))
    film.add(Shot("title", 60.0, 65.0, title_render))
    return film
