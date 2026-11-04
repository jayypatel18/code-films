"""The fox's acting, cut by cut: key poses on twos and threes, holds, overshoot.

poses(cut_id, f) returns a Shot: the fox pose (or None), its light, the lantern
when it is not in the mouth, and other actors for the compositor. Every value
comes from keys sampled on the frame grid of the drawing (twos for acting, ones
for the fastest action), so in-betweens are new drawings, never smooth tweens
running every frame. events() lists the exact times of every beat for sound.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import timeline as T
from actors.fox import FoxPose

FPS = T.FPS


def held(f, f0, step=2):
    """The frame a drawing on `step`s was made for (drawings change every `step` frames)."""
    return f0 + ((f - f0) // step) * step


def ease(u, kind="io"):
    u = min(max(u, 0.0), 1.0)
    if kind == "l":
        return u
    if kind == "i":
        return u * u
    if kind == "o":
        return 1 - (1 - u) * (1 - u)
    if kind == "h":
        return 0.0
    return u * u * (3 - 2 * u)


def K(t, keys, kind="io"):
    """Sample a key list [(t, value), ...] (seconds) at t. Values are floats or tuples.
    A key may carry its own easing as a third item (applied on the way into it)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a[0] <= t < b[0]:
            k = b[2] if len(b) > 2 else kind
            u = ease((t - a[0]) / max(b[0] - a[0], 1e-9), k)
            if isinstance(a[1], tuple):
                return tuple(x + (y - x) * u for x, y in zip(a[1], b[1]))
            return a[1] + (b[1] - a[1]) * u
    return keys[-1][1]


@dataclass
class Shot:
    fox: FoxPose = None
    light: str = "hall"
    lantern_screen: tuple = None      # warm point light for the fox (screen x, y, z)
    lantern_reach: float = 0.0
    hard_warm: float = 0.0
    lantern_prop: dict = None         # dict(x, y, size, swing, flame_lean) when set down
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

STAND_H = 5.65   # ground to ear tip, head radii
SIT_H = 6.25


def plate_fox(cut_id, f, x, y, height_plate, height_units=STAND_H):
    """Screen position and scale of a fox standing at plate (x, y), glued to the camera."""
    zoom, cx, cy = T.camera(cut_id, f)
    sx, sy = T.plate_to_screen(cut_id, f, x, y)
    return sx, sy, height_plate / height_units * zoom


def screen_anchor(cut_id, f, sx, sy):
    """A screen-space anchor given at the cut's first frame, kept glued to the plate as the camera moves."""
    f0, _ = T.cut_range(cut_id)
    px, py = T.screen_to_plate(cut_id, f0, sx, sy)
    zoom0 = T.camera(cut_id, f0)[0]
    nx, ny = T.plate_to_screen(cut_id, f, px, py)
    return nx, ny, T.camera(cut_id, f)[0] / zoom0


# ---------------------------------------------------------------- cut 2

C2 = dict(x0=float(T.ANCHORS["2"]["fox_enter_x"]), t_walk0=T.CUES["fox_enter"] - 0.05, t_stop=T.CUES["fox_stop"], stride=3.0)


def _walk2(t):
    """Distance walked (plate px) and walk phase at time t in cut 2."""
    a = T.ANCHORS["2"]
    x_stop = a["fox_stop_x"]
    t0, t1 = C2["t_walk0"], C2["t_stop"]
    dist = x_stop - C2["x0"]
    unit = a["fox_height"] / STAND_H
    # constant speed with a short ease out into the stop
    u = (t - t0) / (t1 - t0)
    if u <= 0:
        d = 0.0
    elif u >= 1:
        d = dist
    else:
        e = 0.12
        if u < 1 - e:
            d = dist * u / (1 - e / 2)
        else:
            v = (u - (1 - e)) / e
            d = dist * ((1 - e) + e * (v - v * v / 2)) / (1 - e / 2)
    phase = (d / unit) / C2["stride"]
    return C2["x0"] + d, phase


def cut2(f):
    f0, f1 = T.cut_range("2")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["2"]
    C = T.CUES
    ts, td, tu = C["fox_stop"], C["lantern_down"], C["look_up"]
    x, phase = _walk2(t)
    walking = t < ts
    # stop: the body overshoots a touch forward, then settles back
    over = K(t, [(ts, 0.0), (ts + 0.12, 6.0, "o"), (ts + 0.32, -2.0), (ts + 0.5, 0.0)])
    sx, sy, sc = plate_fox("2", fh, x + over, a["floor_y"], a["fox_height"])
    lantern_down = t >= td
    head_pitch = K(t, [(7.0, -0.06), (ts, -0.06), (ts + 0.2, -0.10), (td - 0.05, -0.55, "io"), (td + 0.05, -0.58),
                       (td + 0.25, -0.30), (tu, 0.42, "o"), (tu + 0.15, 0.52), (tu + 0.3, 0.48)])
    if walking:
        head_pitch += 0.03 * math.sin(4 * math.pi * (phase - 0.12))
    bob = 0.045 * math.cos(4 * math.pi * phase) if walking else K(t, [(ts, 0.0), (ts + 0.15, 0.10), (ts + 0.4, 0.02),
                                                                      (td - 0.05, 0.18), (td + 0.2, 0.05), (tu, -0.04)])
    tail = (0.22 + 0.10 * math.sin(2 * math.pi * (phase - 0.2)), 0.08 * math.sin(2 * math.pi * phase), 0.15) \
        if walking else (K(t, [(ts, 0.22), (ts + 0.3, 0.10), (tu, 0.30, "o"), (tu + 0.3, 0.26)]), 0.0, 0.2)
    swing = 0.20 * math.sin(4 * math.pi * phase - 0.9) if walking else K(t, [(ts, 0.20), (ts + 0.15, -0.16),
                                                                            (ts + 0.3, 0.09), (ts + 0.45, -0.04), (ts + 0.6, 0.0)])
    ears = K(t, [(7.0, (0.0, 0.0)), (tu - 0.1, (0.0, 0.0)), (tu + 0.1, (0.25, 0.30), "o"), (tu + 0.25, (0.18, 0.22)),
                 (13.5, (0.18, 0.22)), (13.6, (0.45, 0.22), "o"), (13.72, (0.35, 0.22))])
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=math.pi / 2 - 0.12, body="full", walk=phase if walking else None,
                   stride=C2["stride"], body_bob=bob, head_pitch=head_pitch, tail=tail, lantern=not lantern_down,
                   lantern_swing=swing, ear_swivel=ears, gaze=(0.3, -0.2 if t > tu else 0.1),
                   worry=0.35, seed=fh // 2, cam_tilt=0.12)
    shot = Shot(fox=pose, light="hall")
    if lantern_down:
        lx, ly = a["lantern_set"]
        lsx, lsy = T.plate_to_screen("2", fh, lx, ly)
        settle = K(t, [(td, 0.12), (td + 0.15, -0.06), (td + 0.3, 0.02), (td + 0.45, 0.0)])
        shot.lantern_prop = dict(x=lsx, y=lsy, size=0.88 * sc, swing=settle, flame_lean=settle * 0.6)
        shot.lantern_screen = (lsx, lsy - 0.5 * sc, 1.2 * sc)
    shot.lantern_reach = 9.0 * sc
    return shot


# ---------------------------------------------------------------- cut 3 (the close-up)

def cut3(f):
    f0, f1 = T.cut_range("3")
    a = T.ANCHORS["3"]
    hx, hy, zr = screen_anchor("3", f, *a["head_centre_screen"])
    scale = a["head_height_screen"] / 1.30 * zr
    fh = held(f, f0, 2)
    t = fh / FPS
    C = T.CUES
    # ears swivel one at a time: anticipation, overshoot, settle (on twos)
    el = C["ear_left"]
    er = C["ear_right"]
    ear_r = K(t, [(14.0, 0.0), (el, 0.0), (el + 0.08, -0.15), (el + 0.17, 0.82, "o"), (el + 0.29, 0.58), (el + 0.42, 0.66)])
    ear_l = K(t, [(14.0, 0.0), (er, 0.0), (er + 0.08, -0.12), (er + 0.17, 0.86, "o"), (er + 0.29, 0.60), (er + 0.42, 0.68)])
    # the head follows the second sound a little
    head_yaw = K(t, [(14.0, 0.0), (er + 0.08, 0.0), (er + 0.33, 0.07, "o"), (er + 0.5, 0.05),
                     (C["eye_glint"] - 0.1, 0.05), (C["eye_glint"] + 0.2, 0.12, "o"), (C["eye_glint"] + 0.4, 0.10)])
    # two quick sniffs: the nose lifts, whiskers twitch (on ones, they are fast)
    t1 = f / FPS
    s1, s2 = C["sniff_1"], C["sniff_2"]
    sniff = K(t1, [(14.0, 0.0), (s1 - 0.04, 0.0), (s1, 1.0, "o"), (s1 + 0.12, 0.15), (s2, 1.0, "o"), (s2 + 0.16, 0.0)])
    head_pitch = 0.10 + 0.05 * K(t, [(14.0, 0.0), (s1 - 0.1, 0.0), (s1, 1.0), (s2 + 0.2, 1.0), (s2 + 0.45, 0.0)]) \
        - K(t, [(14.0, 0.0), (C["eye_glint"] - 0.1, 0.0), (C["eye_glint"] + 0.25, 0.08)])
    # the blink, on ones: open, closed held four frames, half, nearly open, open. The half
    # drawing on the closing side read as a glare in a still, so the lids snap shut and only
    # the opening side has in-betweens (a soft, rounded lid).
    fb = T.sec(C["blink_close"])
    seq = {fb: 0.0, fb + 1: 0.0, fb + 2: 0.0, fb + 3: 0.0, fb + 4: 0.55, fb + 5: 0.88}
    eye_open = seq.get(f, 1.0)
    # after the blink the eyes find the lantern: gaze down and right, the glint
    gaze = K(t, [(14.0, (0.35, -0.40)), (el, (0.35, -0.40)), (el + 0.25, (-0.25, -0.35), "o"), (er + 0.1, (-0.20, -0.35)),
                 (er + 0.3, (0.55, -0.30), "o"), (C["blink_open"], (0.45, -0.25)), (C["eye_glint"] + 0.15, (0.55, 0.30), "o")])
    worry = K(t, [(14.0, 0.45), (er + 0.3, 0.6), (C["eye_glint"], 0.55), (C["eye_glint"] + 0.3, 0.25)])
    pose = FoxPose(x=hx, y=hy, scale=scale, yaw=0.48, head_yaw=head_yaw, head_pitch=head_pitch, head_roll=-0.05,
                   ear_swivel=(ear_r, ear_l), eye_open=eye_open, gaze=gaze, worry=worry, sniff=sniff, seed=fh // 2)
    lx, ly = a["lantern_glow_from_screen"]
    return Shot(fox=pose, light="hall", lantern_screen=(lx, ly, 380.0), lantern_reach=1500.0, hard_warm=0.55)


# ---------------------------------------------------------------- cut 4a: the hop

def cut4a(f):
    f0, f1 = T.cut_range("4a")
    a = T.ANCHORS["4a"]
    C = T.CUES
    tk, tl = C["hop_takeoff"], C["hop_land"]
    t1 = f / FPS
    fast = tk - 0.05 <= t1 <= tl + 0.1
    fh = f if fast else held(f, f0, 2)
    t = fh / FPS
    x0, y0 = a["fox_start"]
    x1, y1 = a["fox_land"]
    if t < tk:
        x, y = x0, y0
    elif t < tl:
        u = (t - tk) / (tl - tk)
        x = x0 + (x1 - x0) * u
        y = y0 + (y1 - y0) * u - 230 * math.sin(math.pi * u) * (1.0 - 0.3 * u)
    else:
        x, y = x1, y1
    sx, sy, sc = plate_fox("4a", fh, x, y, a["fox_height"])
    # anticipation crouch, stretch at takeoff, tuck in the air, squash on landing, settle
    bob = K(t, [(18.0, 0.0), (18.45, 0.0), (tk - 0.02, 0.62, "io"), (tk + 0.06, -0.25, "o"), (tl - 0.1, 0.05),
                (tl + 0.02, 0.55, "o"), (tl + 0.2, 0.0), (tl + 0.35, 0.08), (tl + 0.5, 0.02)])
    pitch = K(t, [(18.0, 0.0), (tk - 0.1, 0.10), (tk + 0.04, 0.42, "o"), (tk + 0.25, 0.05), (tl - 0.05, -0.30),
                  (tl + 0.06, -0.08), (tl + 0.3, 0.0)])
    squash = K(t, [(18.0, (1.0, 1.0)), (tk - 0.04, (1.08, 0.90)), (tk + 0.04, (0.86, 1.18), "o"),
                   (tk + 0.2, (1.0, 1.0)), (tl - 0.02, (0.94, 1.07)), (tl + 0.04, (1.09, 0.91), "o"), (tl + 0.2, (0.98, 1.02)),
                   (tl + 0.34, (1.0, 1.0))])
    head_pitch = K(t, [(18.0, 0.25), (18.4, 0.35), (tk - 0.05, 0.0), (tl, -0.15), (tl + 0.3, -0.30), (20.2, -0.45),
                       (21.0, -0.48)])
    ears = K(t, [(18.0, (0.2, 0.2)), (tk, (-0.3, -0.3)), (tl + 0.05, (0.15, 0.15), "o"), (tl + 0.3, (0.25, 0.25))])
    tail = K(t, [(18.0, (0.2, 0.0, 0.2)), (tk - 0.05, (0.0, 0.0, 0.1)), (tk + 0.15, (0.9, 0.0, -0.2), "o"),
                 (tl + 0.05, (0.4, 0.0, 0.1)), (tl + 0.2, (0.15, 0.0, 0.3)), (tl + 0.5, (0.22, 0.0, 0.2))])
    walk = None
    if tk - 0.02 <= t < tl:
        walk = 0.62   # legs reach and tuck through the leap
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=math.pi / 2 - 0.35, pitch=pitch, body="full", walk=walk, stride=1.8,
                   body_bob=bob, squash=squash, head_pitch=head_pitch, ear_swivel=ears, tail=tail, gaze=(0.4, 0.3),
                   worry=0.3, seed=fh // 2, cam_tilt=0.10)
    lx, ly = a["lantern"]
    lsx, lsy = T.plate_to_screen("4a", fh, lx, ly)
    zoom = T.camera("4a", fh)[0]
    unit = a["fox_height"] / STAND_H * zoom
    return Shot(fox=pose, light="hall", lantern_prop=dict(x=lsx, y=lsy, size=0.88 * unit, swing=0.0, flame_lean=0.0),
                lantern_screen=(lsx, lsy - 0.5 * unit, 1.2 * unit), lantern_reach=12.0 * unit)


# ---------------------------------------------------------------- cut 4b: the paw and the ink bird

def cut4b(f):
    f0, f1 = T.cut_range("4b")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["4b"]
    px, py = a["paw_press_screen"]
    pd = T.CUES["paw_down"]
    down = K(t, [(21.0, 1.0), (pd - 0.12, 0.35, "io"), (pd, 0.0, "i"), (pd + 0.08, -0.06, "o"), (pd + 0.2, 0.0)])
    return Shot(fox=None, light="hall", extras=dict(paw=dict(x=px, y=py, lift=down, seed=fh // 2), inkbird=True))


# ---------------------------------------------------------------- cut 5c, 5d: the library stirs

def cut5c(f):
    f0, f1 = T.cut_range("5c")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["5c"]
    hx, hy = a["fox_head_screen"]
    ef = T.CUES["ears_flat"]
    # two hard held poses: alert, then ears pinned back and the body low; a fast snap between
    flat = K(t, [(29.0, 0.0), (ef - 0.05, 0.0), (ef + 0.08, 1.08, "o"), (ef + 0.2, 0.95), (ef + 0.3, 1.0)])
    head_pitch = K(t, [(29.0, 0.22), (ef - 0.05, 0.25), (ef + 0.1, -0.18, "o"), (ef + 0.25, -0.12)])
    low = K(t, [(29.0, 0.0), (ef - 0.05, 0.0), (ef + 0.1, 30.0, "o"), (ef + 0.25, 24.0)])
    worry = K(t, [(29.0, 0.5), (ef, 0.6), (ef + 0.1, 1.0)])
    gaze = K(t, [(29.0, (-0.3, -0.45)), (ef, (-0.3, -0.45)), (ef + 0.1, (-0.5, -0.1), "o")])
    pose = FoxPose(x=hx, y=hy + low, scale=205.0, yaw=-0.55, head_pitch=head_pitch, head_roll=0.04,
                   ear_flat=(flat, flat), worry=worry, gaze=gaze, seed=fh // 2, glint=True)
    lx, ly = a["lantern_screen"]
    lean = K(f / FPS, [(29.0, 0.0), (T.CUES["flame_lean"], 0.0), (T.CUES["flame_lean"] + 0.4, 0.9, "o"), (31.0, 0.8)])
    return Shot(fox=pose, light="hall", lantern_prop=dict(x=lx, y=ly + 120, size=230.0, swing=0.0, flame_lean=lean),
                lantern_screen=(lx, ly, 300.0), lantern_reach=1700.0, hard_warm=0.6)


def cut5d(f):
    a = T.ANCHORS["5d"]
    f0, _ = T.cut_range("5d")
    x, y = a["fox"]
    sx, sy, sc = plate_fox("5d", f0, x, y, T.ANCHORS["2"]["fox_height"])
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=math.pi / 2 - 0.2, body="full", body_bob=0.28, head_pitch=0.15,
                   ear_flat=(1.0, 1.0), tail=(-0.15, 0.0, 0.0), worry=1.0, seed=f0 // 2, cam_tilt=0.12)
    lx, ly = a["lantern"]
    lsx, lsy = T.plate_to_screen("5d", f0, lx, ly)
    return Shot(fox=pose, light="hall", lantern_prop=dict(x=lsx, y=lsy, size=0.88 * sc, swing=0.0, flame_lean=0.7),
                lantern_screen=(lsx, lsy - 0.5 * sc, 1.2 * sc), lantern_reach=9.0 * sc)


# ---------------------------------------------------------------- cut 7, 8: sunrise, the great bird

def cut7(f):
    f0, f1 = T.cut_range("7")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["7"]
    x, y = a["fox_paws"]
    sx, sy, sc = plate_fox("7", fh, x, y, a["fox_height"])
    mo = T.CUES["fox_mouth_open"]
    head_pitch = K(t, [(39.0, 0.15), (mo - 0.5, 0.18), (mo, 0.62, "o"), (mo + 0.2, 0.56)])
    mouth = K(t, [(39.0, 0.0), (mo - 0.1, 0.0), (mo + 0.1, 0.4)])
    # wind: tail and fur whip on twos, two overlapping rhythms so it never loops visibly
    w1 = math.sin(t * 9.0) * 0.5 + math.sin(t * 13.7 + 1.0) * 0.3
    tail = (0.55 + 0.15 * w1, 0.9 + 0.25 * math.sin(t * 7.3), 0.3 + 0.2 * w1)
    ears = (0.05 + 0.08 * w1, 0.10 + 0.08 * math.sin(t * 11.0))     # ears up, trembling in the gusts
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=2.45, body="full", head_pitch=head_pitch, mouth_open=mouth, tail=tail,
                   tail_wind=1.0, ear_swivel=ears, seed=fh // 2, cam_tilt=0.10, fur_wind=0.8 + 0.2 * w1,
                   wind_dir=(-1.0, -0.15))
    return Shot(fox=pose, light="sunrise")


def cut8(f):
    f0, f1 = T.cut_range("8")
    fh = held(f, f0, 3)
    t = fh / FPS
    a = T.ANCHORS["8"]
    x, y = a["fox_paws"]
    sx, sy, sc = plate_fox("8", fh, x, y, a["fox_height"])
    tail = (0.5 + 0.15 * math.sin(t * 5.0), 0.7 + 0.2 * math.sin(t * 3.3), 0.3)
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=2.6, body="full", head_pitch=0.65, tail=tail, tail_wind=0.7,
                   seed=fh // 3, cam_tilt=0.10, line=0.8, fur_wind=0.6, wind_dir=(-1.0, -0.15))
    return Shot(fox=pose, light="sunrise")


# ---------------------------------------------------------------- cut 9a, 9b: the last page

def cut9a(f):
    f0, f1 = T.cut_range("9a")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["9a"]
    sx, sy, zr = screen_anchor("9a", fh, *a["fox_seat_screen"])
    sc = a["fox_height_screen"] / SIT_H * zr
    C = T.CUES
    lp, lu = C["fox_looks_page"], C["fox_looks_us"]
    head_yaw = K(t, [(52.0, 0.0), (lp, 0.0), (lp + 0.25, 0.30, "o"), (lp + 0.4, 0.26), (lu, 0.26), (lu + 0.08, 0.22),
                     (lu + 0.3, -0.72, "o"), (lu + 0.45, -0.62)])
    head_pitch = K(t, [(52.0, 0.15), (53.2, 0.25), (lp, 0.20), (lp + 0.25, -0.38, "o"), (lp + 0.4, -0.32), (lu, -0.32),
                       (lu + 0.3, 0.02, "o"), (lu + 0.45, -0.02)])
    gaze = K(t, [(52.0, (0.3, -0.5)), (53.3, (0.1, -0.6)), (lp, (0.2, -0.2)), (lp + 0.25, (0.4, 0.5), "o"),
                 (lu, (0.4, 0.5)), (lu + 0.25, (-0.55, 0.0), "o")])
    ears = K(t, [(52.0, (0.2, 0.2)), (lp + 0.2, (0.35, 0.05), "o"), (lu + 0.3, (0.1, 0.1))])
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=0.55, body="full", stance="sit", head_yaw=head_yaw, head_pitch=head_pitch,
                   gaze=gaze, ear_swivel=ears, worry=0.2, seed=fh // 2, cam_tilt=0.22)
    return Shot(fox=pose, light="morning", extras=dict(page9a=True))


def cut9b(f):
    f0, f1 = T.cut_range("9b")
    fh = held(f, f0, 2)
    t = fh / FPS
    a = T.ANCHORS["9b"]
    sx, sy, zr = screen_anchor("9b", fh, *a["fox_seat_screen"])
    sc = a["fox_height_screen"] / SIT_H * zr
    C = T.CUES
    fp = C["final_paw"]
    # the paw goes down on the page's near left edge (fox space: forward and to the right)
    reach_u = K(t, [(55.0, 0.0), (fp - 0.3, 0.0), (fp - 0.12, 0.7, "io"), (fp, 1.0, "i"), (fp + 0.05, 1.03), (fp + 0.15, 1.0)])
    lift = math.sin(math.pi * min(max(reach_u, 0), 1)) * 0.6
    home = np.array([-0.34, -0.12, 1.02])
    target = np.array([0.20, -0.10, 2.55])
    paw = home + (target - home) * reach_u + np.array([0, -lift, 0])
    tf = C["tail_flick"]
    curl = K(t, [(55.0, 0.0), (tf, 0.0), (tf + 0.12, 1.4, "o"), (tf + 0.25, -0.3), (tf + 0.4, 0.15), (tf + 0.55, 0.0)])
    head_yaw = K(t, [(55.0, -0.62), (fp - 0.4, -0.62), (fp - 0.1, -0.2, "io"), (C["title_ink_start"] + 0.3, 0.10),
                     (C["hanko"], 0.12), (C["line_ink_end"], -0.45, "io"), (C["line_ink_end"] + 0.3, -0.55)])
    head_pitch = K(t, [(55.0, -0.02), (fp - 0.1, -0.30), (C["title_ink_end"], -0.34), (C["line_ink_end"], 0.0, "io")])
    gaze = K(t, [(55.0, (-0.55, 0.0)), (fp - 0.3, (-0.55, 0.0)), (fp - 0.05, (0.5, 0.6), "o"),
                 (C["line_ink_end"], (0.5, 0.6)), (C["line_ink_end"] + 0.25, (-0.5, 0.0), "o")])
    pose = FoxPose(x=sx, y=sy, scale=sc, yaw=0.55, body="full", stance="sit", head_yaw=head_yaw, head_pitch=head_pitch,
                   gaze=gaze, worry=0.1, tail=(0.0, -1.0, curl), reach=(-1, tuple(paw)), seed=fh // 2, cam_tilt=0.22)
    return Shot(fox=pose, light="morning", extras=dict(title=True))


def shot(cut_id, f):
    fn = {"2": cut2, "3": cut3, "4a": cut4a, "4b": cut4b, "5c": cut5c, "5d": cut5d, "7": cut7, "8": cut8,
          "9a": cut9a, "9b": cut9b}.get(cut_id)
    if fn is None:
        return Shot(fox=None, light=T.CUT_LIGHT[cut_id])
    return fn(f)


# ---------------------------------------------------------------- events for sound

def events():
    """Exact times (seconds) of every acted beat, for the sound department."""
    ev = {}
    # paw steps in cut 2: a paw lands when its phase wraps to the start of stance
    steps = []
    offs = {"HL": 0.0, "FL": 0.25, "HR": 0.5, "FR": 0.75}
    f0, f1 = T.cut_range("2")
    prev = None
    for f in range(f0, f1):
        t = held(f, f0, 2) / FPS
        if t >= C2["t_stop"]:
            break
        _, ph = _walk2(t)
        if prev is not None:
            for name, o in offs.items():
                a = (prev + o) % 1.0
                b = (ph + o) % 1.0
                if b < a:       # wrapped: stance begins, the paw touches down
                    steps.append(round(f / FPS, 3))
        prev = ph
    ev["paw_steps"] = sorted(set(steps))
    ev["fox_stop"] = C2["t_stop"]
    ev["lantern_down"] = T.CUES["lantern_down"]
    ev["look_up"] = T.CUES["look_up"]
    c = T.CUES
    ev["ear_left"] = round(c["ear_left"] + 0.17, 3)
    ev["ear_right"] = round(c["ear_right"] + 0.17, 3)
    ev["sniffs"] = [c["sniff_1"], c["sniff_2"]]
    fb = T.sec(c["blink_close"])
    ev["blink_close"] = round(fb / FPS, 3)
    ev["blink_open"] = round((fb + 5) / FPS, 3)
    ev["eye_glint"] = round(c["eye_glint"] + 0.15, 3)
    ev["hop_crouch"] = 18.45
    ev["hop_takeoff"] = c["hop_takeoff"]
    ev["hop_land"] = c["hop_land"]
    ev["paw_down"] = c["paw_down"]
    ev["ears_flat"] = round(c["ears_flat"] + 0.08, 3)
    ev["flame_lean"] = c["flame_lean"]
    ev["fox_mouth_open"] = c["fox_mouth_open"]
    ev["fox_looks_page"] = c["fox_looks_page"]
    ev["fox_looks_us"] = c["fox_looks_us"]
    ev["final_paw"] = c["final_paw"]
    ev["tail_flick"] = round(c["tail_flick"] + 0.12, 3)
    return ev
