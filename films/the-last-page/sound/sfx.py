"""Sound design for THE LAST PAGE. Every effect is synthesized from noise, oscillators and
resonators, and placed at a cue from timeline.CUES (or a time derived from one).

Each function receives the Mixer `mx` and calls mx.add(bus, signal, t0, gain, pan, space, send).
Spaces: 'out' (exterior air), 'hall' (the stone library), 'close' (intimate, tiny room).
"""

import numpy as np

from dsp import (EV, SR, TAU, TL, C, band_noise, brown, crackle_env, curve, edge_fade, env_ar, env_bell,
                 filt, modal, onepole_lp, place_stereo, rng_for, smooth_rand, smoothstep, sos_bp,
                 sos_hp, sos_lp, sos_reson, svf, swept_noise, taxis)


def seg_n(t0, t1):
    return max(8, int(round((t1 - t0) * SR)))


def tl(t0, n):
    return t0 + np.arange(n) / SR


# ====================================================================== building blocks
def wind_noise(n, rng, fc=380.0, gust_rate=0.2, depth=0.45, bright=0.25, whistle=None):
    """Stereo wind: low body (two cascaded lowpasses), brighter air on gusts, optional whistle
    (a narrow resonance wandering in pitch, only on strong gusts). Returns (wind, gust curve)."""
    nz = rng.standard_normal((2, n))
    body = onepole_lp(onepole_lp(nz, fc), fc * 1.6) * (6.0 * np.sqrt(380.0 / fc))
    body = filt(sos_hp(55.0), body)
    gust = np.clip(0.7 + depth * smooth_rand(n, gust_rate, rng), 0.12, 1.8)
    hi = filt(sos_bp(650.0, 2600.0), rng.standard_normal((2, n))) * bright
    y = body * gust + hi * gust ** 2.2
    if whistle:
        f_lo, f_hi, lev = whistle
        wf = f_lo * (f_hi / f_lo) ** np.clip(0.5 + 0.45 * smooth_rand(n, 0.13, rng), 0.0, 1.0)
        w = svf(rng.standard_normal((2, n)), wf, 24.0, "bp") * 3.0
        y += w * lev * np.clip(gust - 0.55, 0.0, None) ** 1.5
    return y, gust


def rustle(n, sway, rng, density=1.0, lo=1800.0, hi=6500.0):
    """Grass or fur rustle following a sway curve (0..2): crackly high grains plus a soft swish."""
    crk = crackle_env(n, (250.0 + 2600.0 * np.clip(sway, 0, None) ** 2) * density, rng, tau=0.004, sigma=0.7)
    y = band_noise(n, lo, hi, rng) * crk * 0.35
    y += band_noise(n, 450.0, 2000.0, rng) * np.clip(sway, 0, None) ** 1.5 * 0.25
    return y


def paw_step(rng, weight=1.0, claw_p=0.6):
    """Soft pad on stone with a felt-like thump and, sometimes, one or two tiny claw ticks."""
    L = int(0.15 * SR)
    t = taxis(L)
    pad = filt(sos_bp(120.0, 900.0), rng.standard_normal(L)) * np.clip(t / 0.0025, 0, 1) * np.exp(-t / 0.016)
    thump = np.sin(TAU * rng.uniform(95.0, 140.0) * t) * np.clip(t / 0.003, 0, 1) * np.exp(-t / 0.02)
    fur = filt(sos_bp(1500.0, 5000.0), rng.standard_normal(L)) * np.exp(-t / 0.03) * smoothstep(t / 0.01) * 0.06
    y = pad * 0.7 + thump * 0.45 + fur
    if rng.random() < claw_p:
        for _ in range(int(rng.integers(1, 3))):
            d = int(rng.uniform(0.004, 0.02) * SR)
            nb = int(0.004 * SR)
            click = filt(sos_bp(2200.0, 6500.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0006)
            y[d:d + nb] += click * rng.uniform(0.25, 0.5)
    return edge_fade(y, 0.0005, 0.01) * weight


def fur_rustle(rng, dur=0.16, level=1.0, lo=900.0, hi=4200.0):
    L = int(dur * SR)
    y = filt(sos_bp(lo, hi), rng.standard_normal(L)) * env_bell(L, 0.35, 0.2)
    y *= 0.6 + 0.4 * np.clip(crackle_env(L, 900.0, rng, tau=0.003) / 2.0, 0, 1.5)
    return edge_fade(y, 0.002, 0.01) * level


def crinkle(rng, dur=0.15, rate=2500.0, lo=1500.0, hi=6000.0, tau=0.0015):
    L = int(dur * SR)
    e = crackle_env(L, rate, rng, tau=tau, sigma=0.6) * env_bell(L, 0.3, 0.25)
    y = filt(sos_bp(lo, hi), rng.standard_normal(L)) * e
    y = np.tanh(2.5 * y / (np.abs(y).max() + 1e-12)) / np.tanh(2.5)   # grains are dense, not spiky
    return edge_fade(y, 0.001, 0.01)


def stick_slip(dur, rng, rate_curve, res, amp_env=None, jitter=0.25):
    """Friction creak: slip impulses at a wandering rate exciting resonances.
    res: list of (freq or per-sample freq array, Q, gain)."""
    n = int(dur * SR)
    rate = np.clip(rate_curve, 2.0, 400.0) * (1.0 + jitter * smooth_rand(n, 3.0, rng) * 0.3)
    ph = np.cumsum(rate) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    exc = np.zeros(n)
    exc[idx] = rng.lognormal(0.0, 0.35, len(idx))
    exc += rng.standard_normal(n) * 0.015
    out = np.zeros(n)
    for f, q, g in res:
        out += svf(exc, f, q, "bp") * g
    out /= np.abs(out).max() + 1e-12      # peak 1: callers set the level
    if amp_env is not None:
        out *= amp_env
    return out


def brush_stroke(speed, rng, rise=None, wet=1.0):
    """Wet brush on paper. speed: per-sample 0..1 (stroke speed and pressure).
    Bristle scratch grains, a wet drag, tiny wet pops and optional rising resonance (f0, f1, level)."""
    n = len(speed)
    bristle = band_noise(n, 1400.0, 6500.0, rng) * crackle_env(n, 600.0 + 5500.0 * speed, rng, tau=0.0012, sigma=0.6)
    drag = band_noise(n, 280.0, 1800.0, rng) * speed
    pops = band_noise(n, 600.0, 3000.0, rng) * crackle_env(n, 6.0 + 20.0 * speed, rng, tau=0.004, sigma=0.5)
    y = bristle * 0.32 + drag * 0.55 * wet + pops * 0.25 * wet
    if rise is not None:
        f0, f1, lev = rise
        x = np.linspace(0.0, 1.0, n)
        fc = f0 * (f1 / f0) ** (x ** 1.3)
        y += svf(rng.standard_normal(n), fc, 9.0, "bp") * speed * lev
        y += svf(rng.standard_normal(n), fc * 2.0, 30.0, "bp") * speed * lev * 0.6
    return y


def flutter_mass(n, rng, nvoices, f_lo, f_hi, rate_lo, rate_hi, sharp=(3.0, 6.0), pans=None,
                 rate_target=None, conv=None, starts=None, crinkle_amt=0.25):
    """Many sheets of paper flapping: each voice is band-limited noise gated by a flapping wave
    with its own rate and drift. Optional per-sample pan arrays, rates converging to rate_target
    along conv (0..1), and staggered start times (seconds)."""
    out = np.zeros((2, n))
    t = np.arange(n) / SR
    for v in range(nvoices):
        fc = float(np.exp(rng.uniform(np.log(f_lo), np.log(f_hi))))
        src = filt(sos_bp(fc / 1.7, min(fc * 1.7, 9000.0)), rng.standard_normal(n))
        r0 = rng.uniform(rate_lo, rate_hi)
        rate = r0 * (1.0 + 0.15 * smooth_rand(n, 0.8, rng))
        if rate_target is not None:
            rate = rate * (1 - conv) + rate_target * (1.0 + 0.03 * rng.standard_normal()) * conv
        ph = np.cumsum(rate) / SR + rng.random()
        p = rng.uniform(*sharp)
        flap = (0.5 + 0.5 * np.cos(TAU * ph)) ** p
        amp = np.clip(0.65 + 0.35 * smooth_rand(n, 1.5, rng), 0.1, 1.3) * rng.lognormal(0.0, 0.35)
        sig = src * flap * amp
        if crinkle_amt:
            sig += filt(sos_bp(1500.0, 6000.0), rng.standard_normal(n)) * crackle_env(n, 25.0, rng, tau=0.002) * flap * crinkle_amt
        if starts is not None:
            sig *= smoothstep((t - starts[v]) / 0.08)
        pan = pans[v] if pans is not None else rng.uniform(-0.95, 0.95)
        out += place_stereo(sig, pan)
    return out / np.sqrt(nvoices)


def whoosh(dur, rng, f_lo=250.0, f_hi=2400.0, peak=0.5, width=0.2, pan0=0.0, pan1=0.0):
    n = int(dur * SR)
    x = np.linspace(0.0, 1.0, n)
    bell = np.exp(-0.5 * ((x - peak) / width) ** 2)
    fc = f_lo * (f_hi / f_lo) ** bell
    w = swept_noise(n, fc, rng, 0.8, channels=1)[0] * bell ** 1.3
    w = filt(sos_lp(7000.0, 2), w)
    return edge_fade(place_stereo(w, pan0 + (pan1 - pan0) * x), 0.01, 0.02)


# ============================================================================ 0.0 to 7.0
def before_dawn(mx):
    """Wind from black, then pampas grass once the picture is up, a distant pine creak."""
    rng = rng_for("wind-dawn")
    t0, t1 = C("wind_in"), C("cut_hall")
    n = seg_n(t0, t1 + 0.02)
    w, gust = wind_noise(n, rng, fc=360.0, gust_rate=0.22, depth=0.5, bright=0.22, whistle=(520.0, 980.0, 0.32))
    tt = tl(t0, n)
    lev = curve(tt, [(t0, 0.0), (t0 + 1.6, 0.85), (C("fade_up_end"), 1.0), (t1, 1.0)])
    mx.add("amb", edge_fade(w * lev, 0.01, 0.02), t0, gain=0.085, space="out", send=0.15, cue="wind_in")
    gr = np.vstack([rustle(n, gust * 1.1, rng_for("grass-a")), rustle(n, np.roll(gust, int(0.35 * SR)), rng_for("grass-b"))])
    glev = curve(tt, [(t0, 0.0), (C("fade_up_start"), 0.0), (C("fade_up_end"), 1.0), (t1, 1.0)])
    gr = np.vstack([gr[0] * 0.9 + gr[1] * 0.2, gr[1] * 0.9 + gr[0] * 0.2]) * glev
    mx.add("amb", edge_fade(gr, 0.01, 0.02), t0, gain=0.05, space="out", send=0.1)
    # one old pine creaks far away while the light crosses the window
    r = rng_for("pine")
    d = 1.3
    rc = 9.0 + 6.0 * np.sin(np.linspace(0, np.pi, int(d * SR)))
    cr = stick_slip(d, r, rc, [(240.0, 14.0, 1.0), (410.0, 16.0, 0.6), (690.0, 18.0, 0.3)],
                    amp_env=env_bell(int(d * SR), 0.45, 0.22))
    mx.add("amb", filt(sos_lp(1800.0), cr), C("window_light_start") + 1.3, gain=0.018, pan=0.55, space="out", send=0.5)


# ========================================================================= interior room
def room_tone(mx):
    """The sleeping library: low stone rumble, faint air, the wind muffled by thick walls.
    Lives on its own bus so the stillness beat keeps it (a held breath of room tone)."""
    rng = rng_for("room")
    t0, t1 = C("cut_hall"), C("cut_sunrise")
    n = seg_n(t0, t1)
    tt = tl(t0, n)
    rum = filt(sos_lp(130.0, 2), brown(n, rng, 40.0, channels=2)) * 0.02
    air = band_noise(n, 300.0, 2500.0, rng, channels=2) * 0.0035
    w, gust = wind_noise(n, rng, fc=200.0, gust_rate=0.12, depth=0.6, bright=0.0)
    w = filt(sos_lp(260.0, 2), w) * 0.5
    y = rum + air + w * 0.12
    lev = curve(tt, [(t0, 1.0), (C("burst"), 1.0), (C("burst") + 0.4, 0.4), (t1, 0.4)])
    mx.add("room", edge_fade(y * lev, 0.01, 0.02), t0, gain=1.0, space="hall", send=0.2)
    # the stillness: the hall seems to draw a breath (a faint swell of air), one grain of dust ticks
    r = rng_for("still-air")
    st0, st1 = C("stillness_start"), C("stillness_end")
    m = seg_n(st0, st1)
    xs = np.linspace(0.0, 1.0, m)
    breath = band_noise(m, 250.0, 2200.0, r, channels=2) * (0.4 + 0.6 * smoothstep(xs / 0.9)) * 0.004
    breath[:, -int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    mx.add("room", edge_fade(breath, 0.05, 0.01), st0, gain=1.0, space="hall", send=0.3)
    # the stillness: one grain of dust ticks on stone, nothing else
    r = rng_for("still-grain")
    nb = int(0.01 * SR)
    g = filt(sos_bp(2500.0, 6000.0), r.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0007)
    mx.add("room", g, C("stillness_start") + 0.55, gain=0.004, pan=0.3, space="hall", send=0.6)


# ===================================================================== 7.0 to 14.0: the hall
def screen_pan(x):
    return float(np.clip((x - 960.0) / 960.0, -1.0, 1.0))


def plate_pan(cut, t, x):
    """Pan of plate x in a cut at time t, through the timeline camera (follows the picture)."""
    sx, _ = TL.plate_to_screen(cut, int(round(t * TL.FPS)), x, 0.0)
    return screen_pan(sx)


def fox_x_hall(t):
    """Plate x of the fox walking in (cut 2): fox_enter_x at fox_enter, fox_stop_x at fox_stop."""
    a = TL.ANCHORS["2"]
    te, ts = C("fox_enter"), C("fox_stop")
    u = np.clip((np.asarray(t, dtype=float) - te) / (ts - te), 0.0, 1.0)
    return a["fox_enter_x"] + (a["fox_stop_x"] - a["fox_enter_x"]) * (1.0 - (1.0 - u) ** 1.25)


def fox_screen_pan_hall(t, ahead=0.0):
    """Screen pan of the walking fox (ahead: plate px in front of the body, e.g. the lantern)."""
    t = np.asarray(t, dtype=float)
    f0, f1 = int(C("cut_hall") * TL.FPS), int(C("cut_closeup") * TL.FPS)
    fr = np.arange(f0, f1 + 1)
    pans = np.array([screen_pan(TL.plate_to_screen("2", int(f), float(fox_x_hall(f / TL.FPS)) + ahead, 0.0)[0])
                     for f in fr])
    out = np.interp(t * TL.FPS, fr, pans)
    return out if out.ndim else float(out)


def walk_times(t0, t1, rng, base=0.21):
    ts = []
    t = t0
    k = 0
    while t < t1 - 0.12:
        ts.append(t)
        x = (t - t0) / (t1 - t0)
        a = base * (1.0 + 0.45 * x ** 3)
        t += a * (0.72 if k % 2 == 0 else 1.28) + rng.normal(0.0, 0.01)
        k += 1
    ts.append(t1)
    return ts


def fox_walk(mx):
    """Paw steps at the animation's exact contact frames (events.json paw_steps), then the settle."""
    rng = rng_for("paws-hall")
    steps = [float(v) for v in (EV("paw_steps") or walk_times(C("fox_enter"), C("fox_stop"), rng))]
    for k, ts in enumerate(steps):
        w = (0.8 if k % 2 else 1.0) * rng.uniform(0.9, 1.05)
        mx.add("foley", paw_step(rng, w, 0.45), ts, gain=0.8, pan=float(fox_screen_pan_hall(ts)), space="hall", send=0.45,
               cue="first_paw_step" if k == 0 else None)
        if k % 2 == 1:   # the carried lantern's bamboo frame creaks with the stride
            cr = modal([rng.uniform(1050, 1250), rng.uniform(1800, 2100), rng.uniform(2700, 3000)],
                       [1.0, 0.5, 0.25], [0.03, 0.02, 0.012], 0.05, rng, 0.0008)
            mx.add("foley", cr, ts + 0.05, gain=0.03, pan=float(fox_screen_pan_hall(ts, 110.0)), space="hall", send=0.4)
    # the stop: weight settles onto all four paws, the fur settles; the head lifts to look up
    tsp = C("fox_stop")
    mx.add("foley", paw_step(rng, 0.55, 0.0), tsp, gain=0.8, pan=float(fox_screen_pan_hall(tsp)), space="hall", send=0.45,
           cue="fox_stop")
    mx.add("foley", fur_rustle(rng, 0.22, 1.0, 700.0, 3500.0), tsp + 0.08, gain=0.05,
           pan=float(fox_screen_pan_hall(tsp)), space="hall", send=0.35)
    mx.add("foley", fur_rustle(rng, 0.25, 1.0, 600.0, 3000.0), C("look_up"), gain=0.05,
           pan=float(fox_screen_pan_hall(C("look_up"))), space="hall", send=0.35, cue="look_up")
    return steps


def flame(n, rng):
    """A small candle flame in a paper lantern: a soft breathy flutter with rare tiny pops."""
    roar = filt(sos_bp(170.0, 1100.0), rng.standard_normal(n))
    flick = np.clip(0.55 + 0.35 * smooth_rand(n, 7.0, rng) + 0.2 * smooth_rand(n, 19.0, rng), 0.08, 1.5)
    pops = filt(sos_bp(800.0, 4000.0), rng.standard_normal(n)) * crackle_env(n, 2.5, rng, tau=0.002) * 0.5
    return roar * flick + pops, flick


def lantern(mx):
    """The flame while it is near (level and pan per shot), the wooden tap at lantern_down,
    and the flame leaning with no wind (cut 5c)."""
    rng = rng_for("flame")
    t0, t1 = C("fox_enter"), C("burst") + 0.3
    n = seg_n(t0, t1)
    tt = tl(t0, n)
    y, flick = flame(n, rng)
    ld, cl, lec, pg, sh, lt, fl, st = (C("lantern_down"), C("cut_closeup"), C("cut_lectern"), C("cut_page"),
                                       C("cut_shelf"), C("cut_lantern"), C("flame_lean"), C("stillness_start"))
    h = 0.01
    gk = [(t0, 0.0), (t0 + 0.3, 0.8), (ld, 0.8), (ld + 0.05, 1.0), (cl - h, 1.0), (cl, 0.85), (lec - h, 0.85),
          (lec, 0.55), (pg - h, 0.55), (pg, 0.25), (sh - h, 0.25), (sh, 0.22), (lt - h, 0.22), (lt, 3.5),
          (st, 4.0), (st + 0.02, 0.0), (t1, 0.0)]
    p_set = plate_pan("2", ld, TL.ANCHORS["2"]["lantern_set"][0])
    p_cl = screen_pan(TL.ANCHORS["3"]["lantern_glow_from_screen"][0])
    p_lec = plate_pan("4a", lec, TL.ANCHORS["4a"]["lantern"][0])
    p_lt = screen_pan(TL.ANCHORS["5c"]["lantern_screen"][0])
    pk = [(t0, -0.95), (ld, p_set), (cl - h, p_set), (cl, p_cl), (lec - h, p_cl), (lec, p_lec), (pg - h, p_lec),
          (pg, -0.5), (lt - h, -0.5), (lt, p_lt), (t1, p_lt)]
    pan = np.where(tt < ld, fox_screen_pan_hall(tt, 110.0), np.interp(tt, *zip(*pk)))
    g = np.interp(tt, *zip(*gk))
    # the lean: the flutter stops, the flame tone drops and swells, as if pulled by something
    lean = svf(rng.standard_normal(n), curve(tt, [(t0, 420.0), (fl, 420.0), (fl + 0.6, 250.0), (t1, 240.0)]), 3.0, "bp")
    le = (smoothstep((tt - fl) / 0.025) * (0.55 + 0.45 * smoothstep((tt - fl - 0.025) / 0.2))
          + 0.6 * np.exp(-np.maximum(tt - fl, 0) / 0.06) * (tt >= fl)) * (tt < st)
    steady = 1.0 - 0.75 * smoothstep((tt - fl) / 0.3)
    y = y * (steady + (1 - steady) * 1.0 / np.maximum(flick, 0.3) * 0.6) + lean * le * 6.0
    sig = place_stereo(y * g, pan)
    mx.add("foley", edge_fade(sig, 0.01, 0.01), t0, gain=0.035, space="hall", send=0.25)
    # wooden tap at lantern_down, a tiny bounce, the paper shade rustles, the flame whuffs
    r = rng_for("lantern-down")
    for dt, a in ((0.0, 1.0), (0.075, 0.28)):
        tap = modal([r.uniform(340, 420), r.uniform(780, 900), r.uniform(1350, 1550), r.uniform(2400, 2800)],
                    [1.0, 0.6, 0.35, 0.2], [0.07, 0.045, 0.03, 0.018], 0.2, r, 0.0005)
        nb = int(0.004 * SR)
        tap[:nb] += filt(sos_bp(1500.0, 5000.0), r.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0015) * 0.3
        tap += np.sin(TAU * 140.0 * taxis(len(tap))) * np.exp(-taxis(len(tap)) / 0.015) * 0.4
        mx.add("foley", edge_fade(tap, 0.0005, 0.01) * a, ld + dt, gain=0.75, pan=p_set, space="hall", send=0.45,
               cue="lantern_down" if dt == 0.0 else None)
    mx.add("foley", crinkle(r, 0.14, 1800.0), ld + 0.01, gain=0.14, pan=p_set, space="hall", send=0.3)
    wf = filt(sos_bp(120.0, 600.0), r.standard_normal(int(0.4 * SR))) * env_ar(int(0.4 * SR), 0.02, 0.12)
    mx.add("foley", wf, ld + 0.02, gain=0.08, pan=p_set, space="hall", send=0.25)


# ================================================================ 14.0 to 18.0: the close-up
def closeup(mx):
    """Ears, sniffs and the blink at the animation's frames, panned from the head anchor."""
    rng = rng_for("closeup")
    a3 = TL.ANCHORS["3"]
    hx, hh = a3["head_centre_screen"][0], a3["head_height_screen"]

    def ear(t, pan, cue):
        L = int(0.16 * SR)
        tt = taxis(L)
        sw = filt(sos_bp(900.0, 4200.0), rng.standard_normal(L)) * env_bell(L, 0.35, 0.18)
        flp = filt(sos_bp(150.0, 600.0), rng.standard_normal(L)) * np.exp(-tt / 0.02) * np.clip(tt / 0.003, 0, 1) * 0.6
        mx.add("foley", edge_fade(sw * 0.8 + flp, 0.001, 0.01), t, gain=0.8, pan=pan, space="close", send=0.35, cue=cue)

    ear(C("ear_left"), screen_pan(hx - 0.28 * hh), "ear_left")
    ear(C("ear_right"), screen_pan(hx + 0.28 * hh), "ear_right")

    def sniff(t, cue):
        for k, (dt, a) in enumerate(((0.0, 1.0), (0.105, 0.8))):
            L = int(0.085 * SR)
            src = rng.standard_normal(L)
            y = sum(filt(sos_reson(f, q), src) * g for f, q, g in ((1900.0, 4.0, 1.0), (3300.0, 5.0, 0.6), (5200.0, 6.0, 0.22)))
            y *= env_bell(L, 0.4, 0.22)
            mx.add("foley", edge_fade(y, 0.002, 0.01) * a, t + dt, gain=0.75, pan=screen_pan(hx + 0.1 * hh), space="close",
                   send=0.3, cue=cue if k == 0 else None)

    sniff(C("sniff_1"), "sniff_1")
    sniff(C("sniff_2"), "sniff_2")
    # the blink: the softest moist tick as the lids meet, a breath of fur as they open
    nb = int(0.012 * SR)
    tk = filt(sos_bp(1800.0, 4500.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0012)
    tk += filt(sos_bp(300.0, 900.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.003) * 0.5
    mx.add("foley", edge_fade(tk, 0.0003, 0.004), C("blink_close"), gain=0.05, pan=screen_pan(hx), space="close",
           send=0.25, cue="blink_close")
    mx.add("foley", fur_rustle(rng, 0.07, 1.0, 1200.0, 4000.0), C("blink_open"), gain=0.025, pan=screen_pan(hx),
           space="close", send=0.25, cue="blink_open")


# ================================================================ 18.0 to 21.0: the hop
def hop(mx):
    rng = rng_for("hop")
    tk, tlnd = C("hop_takeoff"), C("hop_land")
    p0 = plate_pan("4a", tk, TL.ANCHORS["4a"]["fox_start"][0])
    p1 = plate_pan("4a", tlnd, TL.ANCHORS["4a"]["fox_land"][0])
    if EV("hop_crouch") is not None:
        tc = C("hop_crouch")      # the crouch: claws grip the stone, the fur bunches up
        for j in range(2):
            nb = int(0.004 * SR)
            click = filt(sos_bp(2200.0, 6500.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0006)
            mx.add("foley", click, tc + 0.03 * j, gain=0.09, pan=p0, space="hall", send=0.4, cue="hop_crouch" if j == 0 else None)
        mx.add("foley", fur_rustle(rng, 0.18, 1.0, 400.0, 2000.0), tc + 0.01, gain=0.06, pan=p0, space="hall", send=0.35)
    mx.add("foley", paw_step(rng, 0.9, 1.0), tk - 0.02, gain=0.30, pan=p0, space="hall", send=0.4, cue="hop_takeoff")
    for j in range(3):
        nb = int(0.004 * SR)
        click = filt(sos_bp(2200.0, 6500.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0006)
        mx.add("foley", click, tk + 0.012 * j, gain=0.12 * rng.uniform(0.6, 1.0), pan=p0, space="hall", send=0.4)
    wo = whoosh(tlnd - tk + 0.05, rng, 400.0, 1800.0, 0.55, 0.25, p0, p1)
    mx.add("foley", wo, tk, gain=0.10, space="hall", send=0.3)
    # landing on the open book: lectern wood, a paper fwap, crinkle, the old binding creaks
    thud = modal([rng.uniform(85, 110), rng.uniform(170, 210), rng.uniform(320, 380), rng.uniform(600, 700)],
                 [1.0, 0.55, 0.3, 0.12], [0.12, 0.08, 0.05, 0.03], 0.3, rng, 0.001)
    L = int(0.12 * SR)
    fwap = filt(sos_bp(350.0, 3000.0), rng.standard_normal(L)) * env_ar(L, 0.002, 0.03)
    mx.add("foley", edge_fade(thud, 0.001, 0.02), tlnd, gain=0.6, pan=p1, space="hall", send=0.45, cue="hop_land")
    mx.add("foley", fwap, tlnd, gain=0.35, pan=p1, space="hall", send=0.4)
    mx.add("foley", crinkle(rng, 0.14, 2500.0), tlnd + 0.01, gain=0.2, pan=p1, space="hall", send=0.3)
    d = 0.32
    cr = stick_slip(d, rng, np.linspace(30.0, 14.0, int(d * SR)),
                    [(330.0, 12.0, 1.0), (620.0, 14.0, 0.6), (1150.0, 16.0, 0.3)], amp_env=env_bell(int(d * SR), 0.4, 0.25))
    mx.add("foley", cr, tlnd + 0.06, gain=0.05, pan=p1 + 0.1, space="hall", send=0.45)


# ======================================================== 21.0 to 25.0: paw, ink, ink bird
def page_and_ink(mx):
    rng = rng_for("ink")
    tp = C("paw_down")
    a4 = TL.ANCHORS["4b"]
    pp, pb, px = (screen_pan(a4["paw_press_screen"][0]), screen_pan(a4["ink_bird_centre_screen"][0]),
                  screen_pan(a4["bird_exit_screen"][0]))
    mx.add("foley", crinkle(rng, 0.16, 1500.0, 1200.0, 5500.0), tp, gain=0.2, pan=pp, space="close", send=0.3, cue="paw_down")
    mx.add("foley", paw_step(rng, 0.6, 0.0), tp, gain=0.2, pan=pp, space="close", send=0.3)
    # the line draws itself: one confident stroke, rising in pitch as the bird forms
    t0, t1 = C("ink_start"), C("ink_end")
    n = seg_n(t0, t1)
    x = np.linspace(0.0, 1.0, n)
    speed = (smoothstep(x / 0.12) * (0.62 + 0.25 * np.sin(TAU * (2.2 * x ** 1.2) + 0.3) + 0.13 * np.sin(TAU * 7.0 * x))
             * (1.0 - smoothstep((x - 0.9) / 0.1) * 0.85))
    speed = np.clip(speed, 0.0, 1.0)
    y = brush_stroke(speed, rng, rise=(600.0, 2400.0, 0.22))
    flick = np.exp(-np.maximum(x - 0.97, 0) / 0.01) * smoothstep((x - 0.93) / 0.03)
    y += band_noise(n, 2000.0, 6000.0, rng) * flick * 0.15
    mx.add("foley", edge_fade(place_stereo(y, pp + (pb - pp) * x), 0.002, 0.02), t0, gain=0.10, space="close", send=0.35, cue="ink_start")
    # the bird shivers, peels off the page, and flaps away up to the right
    ts = C("bird_shiver")
    L = int(0.45 * SR)
    sh = filt(sos_bp(1200.0, 6000.0), rng.standard_normal(L)) * crackle_env(L, 1500.0, rng, tau=0.001)
    sh *= (0.5 + 0.5 * np.cos(TAU * 28.0 * taxis(L))) ** 2 * smoothstep(taxis(L) / 0.012) * np.exp(-taxis(L) / 0.25)
    sh = np.tanh(2.0 * sh / (np.abs(sh).max() + 1e-12)) / np.tanh(2.0)
    mx.add("foley", edge_fade(sh, 0.002, 0.02), ts, gain=0.12, pan=pb, space="close", send=0.3, cue="bird_shiver")
    tpl = C("bird_peel")
    L = int(0.24 * SR)
    xx = np.linspace(0, 1, L)
    pe = filt(sos_bp(900.0, 5500.0), rng.standard_normal(L)) * crackle_env(L, 1500.0 + 4000.0 * xx ** 1.5, rng, tau=0.0012)
    pe *= smoothstep(xx / 0.03) * (1.0 - smoothstep((xx - 0.85) / 0.15))
    nb0 = int(0.005 * SR)
    pe = np.tanh(2.5 * pe / (np.abs(pe).max() + 1e-12)) / np.tanh(2.5)
    pe[:nb0] += filt(sos_bp(1500.0, 5000.0), rng.standard_normal(nb0)) * np.exp(-taxis(nb0) / 0.001) * 0.3
    mx.add("foley", edge_fade(pe, 0.002, 0.01), tpl, gain=0.16, pan=pb, space="close", send=0.3, cue="bird_peel")
    nb = int(0.006 * SR)
    tk = filt(sos_bp(1500.0, 5000.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.001)
    mx.add("foley", tk, tpl + 0.22, gain=0.05, pan=pb, space="close", send=0.3)
    # the flaps: a downstroke on every drawn wingbeat (events.json), a lighter upstroke between
    tf = C("bird_flap_start")
    beats = [float(v) for v in (EV("bird_wingbeats") or [tf + 0.278 * k for k in range(4)])]
    period = (beats[-1] - beats[0]) / max(1, len(beats) - 1) if len(beats) > 1 else 0.28
    t_exit = C("cut_shelf")
    for k, t in enumerate(beats):
        for up in (False, True):
            tt_ = t + (0.5 * period if up else 0.0)
            if tt_ >= t_exit - 0.05:
                continue
            x = float(np.clip((tt_ - tf) / max(t_exit - tf, 0.2), 0.0, 1.0))
            L = int(0.09 * SR)
            fw = filt(sos_bp(600.0, 3500.0), rng.standard_normal(L)) * env_ar(L, 0.006, 0.035 if not up else 0.025)
            fw += filt(sos_bp(1500.0, 6000.0), rng.standard_normal(L)) * crackle_env(L, 600.0, rng, tau=0.001) * env_ar(L, 0.004, 0.02) * 0.4
            fw = filt(sos_lp(7000.0 - 4500.0 * x), fw)
            mx.add("foley", edge_fade(fw, 0.001, 0.01), tt_, gain=0.07 * (1.0 - 0.7 * x) * (0.4 if up else 1.0),
                   pan=pb + (px - pb) * x, space="hall", send=0.3 + 0.3 * x,
                   cue="bird_flap_start" if (k == 0 and not up) else None)


# ============================================================ 25.0 to 31.0: the library stirs
def rattle(dur, rng, inten, f_book=None, rate=(12.0, 20.0)):
    """A book trembling against its shelf: rapid irregular knocks plus a leather creak."""
    n = int(dur * SR)
    y = np.zeros(n + int(0.06 * SR))
    fb = f_book or rng.uniform(170.0, 280.0)
    t = rng.uniform(0.0, 0.03)
    while t < dur:
        a = float(inten(t / dur)) * rng.uniform(0.5, 1.0)
        kn = modal([fb * rng.uniform(0.95, 1.05), fb * 2.6 * rng.uniform(0.95, 1.05), fb * 5.3 * rng.uniform(0.9, 1.1)],
                   [1.0, 0.5, 0.25], [0.02, 0.012, 0.008], 0.05, rng, 0.0004)
        nb = int(0.002 * SR)
        kn[:nb] += filt(sos_bp(1500.0, 6000.0), rng.standard_normal(nb)) * 0.25
        i = int(t * SR)
        y[i:i + len(kn)] += kn * a
        t += 1.0 / rng.uniform(*rate)
    y = y[:n]
    y += filt(sos_bp(400.0, 2500.0), rng.standard_normal(n)) * crackle_env(n, 40.0, rng, tau=0.006) * 0.15 \
        * inten(np.linspace(0, 1, n))
    return edge_fade(y, 0.002, 0.012)


def tremor(mx):
    rng = rng_for("tremor")
    t1, tr, tst = C("tremble_one"), C("tremble_row"), C("stillness_start")
    # one book on a high shelf
    d = C("cut_row") - t1
    y = rattle(d, rng, lambda x: (0.4 + 0.6 * smoothstep(np.asarray(x) / 0.12)) * (0.7 + 0.3 * np.asarray(x)), f_book=210.0)
    mx.add("paper", filt(sos_lp(5000.0), y), t1, gain=1.5, pan=0.15, space="hall", send=0.5, cue="tremble_one")
    # the whole row, building until the stillness
    d = tst - tr
    acc = np.zeros((2, int(d * SR)))
    for b in range(16):
        r = rng_for(f"row-book{b}")
        st = 0.0 if b % 5 == 0 else r.uniform(0.0, 0.5)
        dd = d - st
        if dd < 0.2:
            continue
        lv = r.uniform(0.5, 1.0)
        yb = rattle(dd, r, lambda x, s=st: (0.35 + 0.65 * np.asarray(x) ** 0.8) * (0.5 + 0.5 * smoothstep(np.asarray(x) / 0.04)),
                    rate=(10.0, 24.0)) * lv
        yb = filt(sos_lp(r.uniform(3500.0, 6500.0)), yb)
        i = int(st * SR)
        acc[:, i:i + len(yb)] += place_stereo(yb, -0.85 + 1.7 * b / 15 + r.normal(0, 0.05))[:, : acc.shape[1] - i]
    n = acc.shape[1]
    x = np.linspace(0, 1, n)
    rum = filt(sos_bp(60.0, 250.0), rng.standard_normal((2, n))) * (0.6 + 0.4 * np.sin(TAU * 9.0 * x * d)) * (0.3 + 0.7 * x)
    acc += rum * 0.5
    acc *= curve(tl(tr, n), [(tr, 1.0), (C("cut_lantern"), 1.0), (C("cut_lantern") + 0.01, 0.5), (tst, 0.7)])
    acc[:, -int(0.008 * SR):] *= np.linspace(1, 0, int(0.008 * SR))
    mx.add("paper", acc, tr, gain=1.5, space="hall", send=0.45, cue="tremble_row")
    # dust sifts down in thin streams
    for j, (dt, pan) in enumerate(((0.0, -0.4), (0.25, 0.25), (0.6, -0.1), (1.1, 0.5))):
        r = rng_for(f"dust{j}")
        t0 = C("dust_fall") + dt
        n = seg_n(t0, tst)
        e = crackle_env(n, r.uniform(2500.0, 6000.0), r, tau=0.0004, sigma=0.9)
        dsig = filt(sos_bp(2500.0, 7000.0), r.standard_normal(n)) * e
        dsig *= smoothstep(np.arange(n) / ((0.04 if j == 0 else 0.3) * SR)) * np.clip(0.7 + 0.3 * smooth_rand(n, 1.0, r), 0, 1.5)
        dsig = np.tanh(3.0 * dsig / (np.abs(dsig).max() + 1e-12)) / np.tanh(3.0)
        dsig[-int(0.008 * SR):] *= np.linspace(1, 0, int(0.008 * SR))
        mx.add("paper", dsig, t0, gain=0.022, pan=pan, space="hall", send=0.4, cue="dust_fall" if j == 0 else None)
    # a deep low creak travels up through the tower
    tc = C("creak")
    d = min(2.4, tst - tc - 0.02)
    n = int(d * SR)
    x = np.linspace(0.0, 1.0, n)
    rate = 15.0 + 9.0 * smooth_rand(n, 1.3, rng) + 12.0 * x
    up = 1.0 + 2.2 * x ** 1.2
    res = [(55.0 * up, 7.0, 1.0), (110.0 * up, 9.0, 0.8), (190.0 * up, 11.0, 0.6), (320.0 * up, 13.0, 0.45),
           (540.0 * up, 15.0, 0.3), (900.0 * up, 17.0, 0.15)]
    env = smoothstep(x / 0.03) * (1.0 - 0.45 * x) * np.clip((1.0 - x) / 0.12, 0.0, 1.0)
    cr = stick_slip(d, rng, rate, res, amp_env=env)
    sub = np.sin(TAU * np.cumsum(36.0 * up ** 0.5) / SR) * onepole_lp(np.abs(cr), 12.0) * 1.5
    cr = cr + sub
    pan = -0.25 + 0.65 * x
    cr /= np.abs(cr).max() + 1e-12
    near = place_stereo(cr * (1.0 - 0.7 * x), pan)
    far = place_stereo(cr * (0.3 + 0.7 * x), pan)
    mx.add("fx", near, tc, gain=2.8, space="hall", send=0.3)
    mx.add("fx", filt(sos_lp(2500.0), far), tc, gain=1.9, space="hall", send=0.9)
    k0 = modal([62.0, 131.0, 240.0, 410.0], [1.0, 0.6, 0.35, 0.2], [0.12, 0.08, 0.05, 0.03], 0.15, rng, 0.002)
    mx.add("fx", k0, tc, gain=0.6, pan=-0.25, space="hall", send=0.4, cue="creak")
    # the flame gutters as it leans
    L = int(0.12 * SR)
    puff = filt(sos_bp(100.0, 600.0), rng.standard_normal(L)) * env_ar(L, 0.004, 0.035)
    mx.add("foley", puff, C("flame_lean"), gain=1.0, pan=screen_pan(TL.ANCHORS["5c"]["lantern_screen"][0]), space="hall", send=0.3, cue="flame_lean")
    # ears flatten
    mx.add("foley", fur_rustle(rng, 0.18, 1.0, 800.0, 3500.0), C("ears_flat"), gain=0.12, pan=screen_pan(TL.ANCHORS["5c"]["fox_head_screen"][0]), space="close", send=0.3, cue="ears_flat")


# ============================================================ 32.0 to 39.0: the pages burst
def burst(mx):
    rng = rng_for("burst")
    tb, tsw, tvp, tdm = C("burst"), C("cut_stairwell"), C("vortex_peak"), C("cut_dome")
    tg = C("gold_light")
    # every book flies open at once: a wall of cover slaps, the first ones exactly on the cue
    n = int(0.6 * SR)
    acc = np.zeros((2, n))
    for k in range(150):
        dt = rng.uniform(0.0, 0.012) if k < 10 else rng.uniform(0.0, 0.28) ** 1.3 * 1.2
        L = int(0.12 * SR)
        fb = rng.uniform(90.0, 230.0)
        sl = modal([fb, fb * 2.3, fb * 4.1], [1.0, 0.5, 0.25], [rng.uniform(0.03, 0.08), 0.03, 0.015], 0.12, rng, 0.0006)
        sl += filt(sos_bp(300.0, 3000.0), rng.standard_normal(L)) * env_ar(L, 0.001, 0.025) * 0.8
        sl = edge_fade(sl, 0.0005, 0.01) * rng.lognormal(0.0, 0.4) * (1.0 if k < 10 else 0.6)
        i = int(dt * SR)
        m = min(L, n - i)
        acc[:, i:i + m] += place_stereo(sl[:m], rng.uniform(-0.95, 0.95))
    acc[:, 0] = 0.0
    mx.add("paper", acc, tb, gain=0.10, space="hall", send=0.4, cue="burst")
    boom = np.sin(TAU * np.cumsum(70.0 * (1 + 0.5 * np.exp(-taxis(int(0.8 * SR)) / 0.05))) / SR)
    boom *= env_ar(len(boom), 0.004, 0.18)
    mx.add("fx", boom, tb, gain=0.16, space="hall", send=0.3)
    # the roar of thousands of pages tearing free
    t1 = tsw + 0.3
    n = seg_n(tb, t1)
    tt = tl(tb, n)
    roar = flutter_mass(n, rng_for("roar"), 56, 350.0, 4200.0, 8.0, 26.0)
    env = smoothstep((tt - tb) / 0.012) * curve(tt, [(tb, 1.0), (tb + 0.5, 1.0), (tsw, 0.7), (t1, 0.0)])
    mx.add("paper", roar * env, tb, gain=6.3, space="hall", send=0.35)
    hiss = band_noise(n, 400.0, 5000.0, rng, channels=2) * crackle_env(n, 9000.0, rng, tau=0.0008) * 0.18
    mx.add("paper", hiss * env, tb, gain=0.07, space="hall", send=0.3)
    for k in range(14):           # tears
        r = rng_for(f"tear{k}")
        d = r.uniform(0.15, 0.3)
        L = int(d * SR)
        xx = np.linspace(0, 1, L)
        e = crackle_env(L, r.uniform(4000.0, 9000.0), r, tau=0.0008) * env_bell(L, 0.4, 0.25)
        tear = svf(r.standard_normal(L) * e, 1500.0 * (3500.0 / 1500.0) ** xx, 2.0, "bp")
        mx.add("paper", tear, tb + 0.05 + r.uniform(0.0, 1.6), gain=0.10 * r.uniform(0.5, 1.0), pan=r.uniform(-0.8, 0.8),
               space="hall", send=0.35)
    for k in range(40):           # pages folding into birds on sharp drawn frames
        r = rng_for(f"snap{k}")
        t = tb + 0.25 + (tvp - tb - 0.3) * (k / 40.0) ** 0.9 + r.uniform(-0.04, 0.04)
        nb = int(0.025 * SR)
        sn = filt(sos_bp(1500.0, 5500.0), r.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0007)
        sn += filt(sos_bp(800.0, 3000.0), r.standard_normal(nb)) * env_ar(nb, 0.002, 0.008) * 0.5
        mx.add("paper", edge_fade(sn, 0.0003, 0.004), t, gain=0.06 * r.uniform(0.5, 1.0), pan=r.uniform(-0.9, 0.9),
               space="hall", send=0.35)
    # the vortex up the stairwell: rotating, rising, brightening, peaking, then far up in the dome
    t0, t2 = tsw, tg + 1.0
    n = seg_n(t0, t2)
    tt = tl(t0, n)
    x = np.clip((tt - t0) / (tvp - t0), 0.0, 1.2)
    nv = 44
    r = rng_for("vortex")
    rot = np.cumsum(0.35 + 0.85 * np.clip(x, 0, 1) ** 1.3) / SR
    pans = [0.9 * np.sin(TAU * rot + r.uniform(0, TAU)) for _ in range(nv)]
    starts = list(r.uniform(-0.3, 1.6, nv))
    vx = flutter_mass(n, r, nv, 300.0, 3800.0, 9.0, 24.0, pans=pans, starts=starts)
    lo = filt(sos_lp(1400.0, 2), vx)
    bright = np.clip(x, 0, 1) ** 1.2
    vx = lo * (1 - bright) + vx * bright
    venv = curve(tt, [(t0, 0.45), (tvp, 1.0), (tdm - 0.01, 0.95), (tdm, 0.55), (tg, 0.5), (tg + 0.4, 0.25), (t2, 0.0)])
    far = (tt >= tdm)
    vx = vx * (1 - far) + filt(sos_lp(2500.0, 2), vx) * far
    mx.add("paper", vx * venv, t0, gain=3.0, space="hall", send=0.45, cue="cut_stairwell")
    air = swept_noise(n, 220.0 * 4.0 ** np.clip(x, 0, 1), r, 0.9) * (0.3 + 0.7 * np.clip(x, 0, 1) ** 1.5)
    air *= curve(tt, [(t0, 1.0), (tdm - 0.01, 1.0), (tdm, 0.5), (tg, 0.4), (t2, 0.0)])
    mx.add("fx", air, t0, gain=0.10, space="hall", send=0.4)
    for k, tw in enumerate((0.4, 1.0, 1.5, 1.9, 2.22, 2.5)):
        w = whoosh(0.55 - 0.03 * k, rng_for(f"vwhoosh{k}"), 300.0, 2600.0, 0.5, 0.2, -0.9 if k % 2 else 0.9, 0.9 if k % 2 else -0.9)
        mx.add("paper", w, t0 + tw * (tvp - t0) / 2.5, gain=0.12, space="hall", send=0.35)


def shutters(mx):
    """The dome's copper shutters crack open, then air and the first light rush down the shaft."""
    rng = rng_for("shutters")
    ts, tg = C("shutters_break"), C("gold_light")
    d = 0.26
    n = int(d * SR)
    x = np.linspace(0, 1, n)
    gr = stick_slip(d, rng, 25.0 + 120.0 * x ** 2,
                    [(190.0, 30.0, 1.0), (420.0, 34.0, 0.7), (770.0, 38.0, 0.5), (1250.0, 40.0, 0.3)],
                    amp_env=smoothstep(x / 0.4) * (0.3 + 0.7 * x))
    mx.add("fx", gr, ts - d, gain=0.12, pan=0.0, space="hall", send=0.6)
    nb = int(0.04 * SR)
    crack = filt(sos_bp(600.0, 7000.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.004)
    crack += np.sin(TAU * 62.0 * taxis(nb)) * np.exp(-taxis(nb) / 0.02) * 0.8
    mx.add("fx", edge_fade(crack, 0.0003, 0.01), ts, gain=1.1, space="hall", send=0.5, cue="shutters_break")

    def clang(f0, r):
        ratios = [1.0, 1.004, 2.31, 3.87, 3.89, 5.52, 7.13, 9.6]
        amps = [1.0, 0.6, 0.6, 0.45, 0.3, 0.3, 0.2, 0.1]
        t60 = [2.2, 2.1, 1.6, 1.2, 1.1, 0.9, 0.6, 0.4]
        y = modal([f0 * q for q in ratios], amps, t60, 2.4, r, 0.0008)
        return edge_fade(filt(sos_lp(4500.0), y), 0.0008, 0.05)

    for dt, f0, pan, g in ((0.0, 118.0, 0.0, 1.0), (0.035, 161.0, -0.5, 0.7), (0.09, 97.0, 0.45, 0.8), (0.16, 203.0, -0.3, 0.5)):
        mx.add("fx", clang(f0, rng_for(f"clang{f0}")), ts + dt, gain=0.45 * g, pan=pan, space="hall", send=0.6)
    d = 0.38
    n = int(d * SR)
    x = np.linspace(0, 1, n)
    sq = stick_slip(d, rng, 220.0 + 180.0 * x, [(680.0, 45.0, 1.0), (1360.0, 50.0, 0.5)],
                    amp_env=env_bell(n, 0.3, 0.25))
    mx.add("fx", sq, ts + 0.04, gain=0.04, pan=0.2, space="hall", send=0.6)
    sc = filt(sos_bp(1000.0, 4000.0), rng.standard_normal(n)) * crackle_env(n, 3000.0, rng, tau=0.001) * np.exp(-x * 3.0)
    mx.add("fx", sc, ts + 0.02, gain=0.12, pan=-0.1, space="hall", send=0.5)
    for k in range(14):
        r = rng_for(f"debris{k}")
        f = r.uniform(1200.0, 3500.0)
        y = modal([f, f * 2.7, f * 5.1], [1.0, 0.4, 0.15], [r.uniform(0.05, 0.15), 0.05, 0.03], 0.2, r, 0.0003)
        mx.add("fx", filt(sos_lp(5000.0), y), ts + 0.1 + 1.0 * (k / 14.0) ** 0.8 + r.uniform(-0.03, 0.03),
               gain=0.04 * (1.0 - 0.6 * k / 14.0), pan=r.uniform(-0.7, 0.7), space="hall", send=0.6)
    # air rushes in through the open dome, the light blooms
    t0, t1 = ts + 0.02, C("cut_sunrise")
    n = seg_n(t0, t1)
    tt = tl(t0, n)
    fc = curve(tt, [(t0, 300.0), (tg, 1100.0), (t1, 500.0)])
    rush = swept_noise(n, fc, rng, 1.0) * curve(tt, [(t0, 0.0), (tg, 1.0), (t1 - 0.05, 0.6), (t1, 0.0)])
    mx.add("fx", rush, t0, gain=0.15, space="hall", send=0.5)
    bl = np.sin(TAU * 52.0 * taxis(int(1.2 * SR))) * env_ar(int(1.2 * SR), 0.08, 0.35)
    mx.add("fx", bl, tg, gain=0.08, space="hall", send=0.3, cue="gold_light")


# ============================================================== 39.0 to 44.0: sunrise outside
def sunrise(mx):
    rng = rng_for("sunrise-wind")
    t0, tb, t1 = C("cut_sunrise"), C("cut_great_bird"), C("cut_last_page")
    n = seg_n(t0, t1 + 1.0)
    tt = tl(t0, n)
    w, gust = wind_noise(n, rng, fc=650.0, gust_rate=0.35, depth=0.55, bright=0.45, whistle=(420.0, 900.0, 0.5))
    lev = curve(tt, [(t0, 1.0), (tb - 0.01, 1.0), (tb, 0.5), (t1, 0.5), (t1 + 1.0, 0.0)])
    mx.add("amb", edge_fade(w * lev, 0.004, 0.02), t0, gain=0.045, space="out", send=0.12, cue="cut_sunrise")
    gr = np.vstack([rustle(n, gust * 1.4, rng_for("grass-s1"), 2.0), rustle(n, np.roll(gust, int(0.25 * SR)) * 1.4, rng_for("grass-s2"), 2.0)])
    glev = curve(tt, [(t0, 1.0), (tb - 0.01, 1.0), (tb, 0.2), (t1, 0.2), (t1 + 1.0, 0.0)])
    mx.add("amb", edge_fade(np.vstack([gr[0], gr[1] * 0.8]) * glev, 0.004, 0.02), t0, gain=0.045, space="out", send=0.1)
    # fur whipping in the wind (the fox on the ledge)
    m = seg_n(t0, tb)
    fl = filt(sos_bp(250.0, 1800.0), rng.standard_normal(m))
    fl *= (0.5 + 0.5 * np.cos(TAU * np.cumsum(20.0 + 6.0 * smooth_rand(m, 2.0, rng)) / SR)) ** 2
    fl *= np.clip(gust[:m] - 0.5, 0, None) ** 1.5
    pfox = plate_pan("7", C("fox_mouth_open"), TL.ANCHORS["7"]["fox_paws"][0])
    mx.add("foley", edge_fade(fl, 0.004, 0.02), t0, gain=0.04, pan=pfox, space="out", send=0.1)
    # mouth slightly open: a tiny breath
    r = rng_for("breath")
    L = int(0.22 * SR)
    src = r.standard_normal(L)
    br = sum(filt(sos_reson(f, q), src) * g for f, q, g in ((900.0, 4.0, 1.0), (1400.0, 5.0, 0.6), (2600.0, 6.0, 0.25)))
    mx.add("foley", edge_fade(br * env_bell(L, 0.35, 0.22), 0.002, 0.01), C("fox_mouth_open"), gain=0.05, pan=pfox,
           space="out", send=0.15, cue="fox_mouth_open")
    # the great ring of wind rolls outward across the clouds
    tw = C("wind_ring")
    r = rng_for("wind-ring")
    d = 0.35
    nn_ = int(d * SR)
    xx = np.linspace(0, 1, nn_)
    pre = swept_noise(nn_, 200.0 * 4.5 ** xx, r, 0.9) * xx ** 2.5
    mx.add("fx", pre, tw - d, gain=0.5, space="out", send=0.2)
    L = int(1.6 * SR)
    t = taxis(L)
    f = 30.0 + 18.0 * np.exp(-t / 0.25)
    boom = np.sin(TAU * np.cumsum(f) / SR) * env_ar(L, 0.015, 0.45)
    boom += 0.35 * np.sin(TAU * np.cumsum(2.0 * f) / SR) * env_ar(L, 0.015, 0.25)
    mx.add("fx", boom, tw, gain=0.30, space="out", send=0.1, cue="wind_ring")
    L = int(3.2 * SR)
    t = taxis(L)
    fc = 200.0 + 1400.0 * np.exp(-t / 0.7)
    bl = swept_noise(L, fc, r, 1.0) * env_ar(L, 0.02, 0.9)
    mid = 0.5 * (bl[0] + bl[1])
    side = 0.5 * (bl[0] - bl[1]) * smoothstep(t / 0.9)
    blast = np.vstack([mid * (1 - 0.5 * smoothstep(t / 0.9)) + side * 1.6, mid * (1 - 0.5 * smoothstep(t / 0.9)) - side * 1.6])
    mx.add("fx", edge_fade(blast, 0.005, 0.05), tw, gain=3.2, space="out", send=0.3)
    L = int(min(2.6, tb - tw) * SR)
    tail = filt(sos_lp(160.0, 2), brown(L, r, 30.0, channels=2)) * env_bell(L, 0.25, 0.3)
    mx.add("fx", edge_fade(tail, 0.05, 0.05), tw + 0.1, gain=0.03, space="out", send=0.2)


# ============================================================== 44.0 to 52.0: the great bird
def great_bird(mx):
    tb, tf, tw, ta0, ta1 = C("cut_great_bird"), C("bird_formed"), C("wingbeat"), C("arc_start"), C("arc_end")
    # the flock pulls together: rates synchronise, the stereo image collapses to the centre
    r = rng_for("flock")
    t1 = tw + 0.05
    n = seg_n(tb, t1)
    tt = tl(tb, n)
    x = np.clip((tt - tb) / (tf - tb), 0.0, 1.0)
    nv = 52
    pans = [r.uniform(-0.95, 0.95) * (1.0 - 0.8 * x ** 1.5) for _ in range(nv)]
    fm = flutter_mass(n, r, nv, 300.0, 3500.0, 8.0, 24.0, pans=pans, rate_target=13.0, conv=x ** 2)
    fm = filt(sos_lp(5000.0, 2), fm) * (0.35 + 0.65 * x ** 1.4) * (1.0 - 0.4 * smoothstep((tt - (tf - 0.5)) / 0.6))
    fm[:, -int(0.04 * SR):] *= np.linspace(1, 0.3, int(0.04 * SR))
    mx.add("paper", fm, tb, gain=6.0, space="out", send=0.35, cue="cut_great_bird")
    # the upstroke draws air in, then the one enormous downstroke
    d = tw - (tf - 0.3)
    m = int(d * SR)
    xx = np.linspace(0, 1, m)
    up = swept_noise(m, 250.0 * 5.6 ** xx, r, 0.9) * xx ** 2.0
    up += np.vstack([1, 1]) * band_noise(m, 1500.0, 6000.0, r) * crackle_env(m, 2000.0 + 8000.0 * xx, r, tau=0.001) * xx ** 2 * 0.4
    mx.add("fx", up, tf - 0.3, gain=0.22, space="out", send=0.25)
    L = int(2.0 * SR)
    t = taxis(L)
    f = 36.0 + 26.0 * np.exp(-t / 0.18)
    sub = 0.6 * np.sin(TAU * np.cumsum(f) / SR) * env_ar(L, 0.01, 0.5)
    sub += 0.6 * np.sin(TAU * np.cumsum(2.0 * f) / SR) * env_ar(L, 0.01, 0.32)
    sub += 0.35 * np.sin(TAU * np.cumsum(3.0 * f) / SR) * env_ar(L, 0.01, 0.2)
    mx.add("fx", sub, tw, gain=0.40, space="out", send=0.08, cue="wingbeat")
    nz = r.standard_normal((2, L))
    fc = 140.0 + 1700.0 * np.exp(-t / 0.18)
    whoomph = svf(nz, fc, 0.9, "lp") * env_ar(L, 0.012, 0.42)
    mx.add("fx", whoomph, tw, gain=2.4, space="out", send=0.25)
    slam = band_noise(L, 400.0, 5000.0, r, channels=2) * crackle_env(L, 3000.0 + 17000.0 * np.exp(-t / 0.15), r, tau=0.001)
    slam *= env_ar(L, 0.006, 0.22)
    slam /= np.abs(slam).max() + 1e-12
    mx.add("paper", slam, tw, gain=1.3, space="out", send=0.3)
    m = int(1.0 * SR)
    tail = flutter_mass(m, rng_for("wing-tail"), 30, 350.0, 3500.0, 10.0, 18.0) * np.exp(-taxis(m) / 0.3)
    mx.add("paper", tail, tw + 0.02, gain=2.0, space="out", send=0.35)
    rw = rng_for("wash")
    m = int(2.0 * SR)
    wash, _ = wind_noise(m, rw, fc=900.0, gust_rate=1.0, depth=0.2, bright=0.4)
    mx.add("amb", wash * env_bell(m, 0.35, 0.25), tw + 0.2, gain=0.12, space="out", send=0.2)
    # the sweeping arc: one long pass following the bird across the sky, a trail of loose paper
    n = seg_n(ta0, ta1 + 0.6)
    tt = tl(ta0, n)
    knots_t = [ta0 + (ta1 - ta0) * k / 4.0 for k in range(5)]
    pan = np.interp(tt, knots_t, [0.0, 0.47, 0.81, 0.17, -0.79])
    prox = np.interp(tt, knots_t + [ta1 + 0.6], [0.35, 0.5, 0.42, 0.85, 0.45, 0.0])
    prox = onepole_lp(prox - prox[0], 2.0) + prox[0]
    dp = np.gradient(prox) * SR
    fc = (300.0 + 2000.0 * prox) * (1.0 + 0.35 * np.clip(dp, -1.5, 1.5))
    rp = rng_for("arc")
    ps = swept_noise(n, fc, rp, 1.0, channels=1)[0] * prox ** 1.6
    mx.add("fx", place_stereo(filt(sos_lp(7000.0), ps), pan), ta0, gain=2.0, space="out", send=0.3, cue="arc_start")
    trail = flutter_mass(n, rng_for("trail"), 20, 400.0, 3500.0, 6.0, 14.0, pans=[np.roll(pan, int(0.25 * SR)) * 0.9] * 20)
    trail *= np.clip(prox, 0.2, 1.0) * curve(tt, [(ta0, 0.6), (ta1, 0.8), (ta1 + 0.6, 0.0)])
    mx.add("paper", trail, ta0, gain=1.4, space="out", send=0.4)
    sp = band_noise(n, 3000.0, 7000.0, rp, channels=2) * crackle_env(n, 300.0, rp, tau=0.003) * prox
    mx.add("fx", sp, ta0, gain=0.03, space="out", send=0.5)


# ============================================================== 52.0 to 60.0: the last page
def last_page(mx):
    rng = rng_for("last-page")
    tq, tland, tend = C("cut_last_page"), C("page_land"), C("end")
    n = seg_n(tq - 0.5, tend)
    tt = tl(tq - 0.5, n)
    w, gust = wind_noise(n, rng, fc=420.0, gust_rate=0.15, depth=0.35, bright=0.12)
    lev = curve(tt, [(tq - 0.5, 0.0), (tq + 0.5, 1.0), (tend, 1.0)])
    mx.add("amb", w * lev, tq - 0.5, gain=0.03, space="out", send=0.15)
    gr = rustle(n, gust * 0.6, rng_for("grass-end"), 0.6)
    mx.add("amb", gr * lev, tq - 0.5, gain=0.025, pan=0.4, space="out", send=0.1)

    def page_fall(r, t0, t_end, pan0, gain, land=False):
        t = t0
        k = 0
        while t < t_end - 0.15:
            d = r.uniform(0.28, 0.5)
            L = int(d * SR)
            sw = filt(sos_bp(350.0, 2200.0), r.standard_normal(L)) * env_bell(L, r.uniform(0.35, 0.6), 0.2)
            sw += filt(sos_bp(1500.0, 5000.0), r.standard_normal(L)) * crackle_env(L, 150.0, r, tau=0.002) * env_bell(L, 0.5, 0.25) * 0.4
            pan = float(np.clip(pan0 + (0.3 if k % 2 else -0.3), -0.95, 0.95))
            mx.add("paper", edge_fade(sw, 0.005, 0.01), t, gain=4.0 * gain * r.uniform(0.7, 1.0), pan=pan, space="out", send=0.25)
            t += d * r.uniform(0.75, 1.0)
            k += 1

    for j, (t0, d, pan0, g) in enumerate(((0.10, 2.2, -0.5, 0.05), (0.35, 2.6, 0.45, 0.04), (0.6, 1.8, -0.1, 0.06),
                                          (0.9, 2.4, 0.7, 0.03), (1.3, 2.0, -0.75, 0.035), (1.7, 2.5, 0.2, 0.03))):
        page_fall(rng_for(f"page{j}"), tq + t0, tq + t0 + d, pan0, g)
    page_fall(rng_for("page-last"), tq + 0.6, tland, 0.23, 0.07)
    L = int(0.2 * SR)
    pf = filt(sos_bp(250.0, 2500.0), rng.standard_normal(L)) * env_ar(L, 0.003, 0.025)
    nb = int(0.004 * SR)
    pf[:nb] += filt(sos_bp(1500.0, 5000.0), rng.standard_normal(nb)) * 0.3
    mx.add("foley", edge_fade(pf, 0.0005, 0.01), tland, gain=0.9, pan=screen_pan(TL.ANCHORS["9a"]["page_land_screen"][0]), space="out", send=0.2, cue="page_land")
    sl = filt(sos_bp(800.0, 4000.0), rng.standard_normal(int(0.14 * SR))) * crackle_env(int(0.14 * SR), 800.0, rng) * env_bell(int(0.14 * SR), 0.3, 0.3)
    mx.add("foley", sl, tland + 0.02, gain=0.06, pan=screen_pan(TL.ANCHORS["9a"]["page_land_screen"][0]), space="out", send=0.2)
    mx.add("foley", fur_rustle(rng, 0.2, 1.0, 700.0, 3200.0), C("fox_looks_page"), gain=0.025, pan=screen_pan(TL.ANCHORS["9a"]["fox_seat_screen"][0]), space="close", send=0.2, cue="fox_looks_page")
    mx.add("foley", fur_rustle(rng, 0.2, 1.0, 700.0, 3200.0), C("fox_looks_us"), gain=0.022, pan=screen_pan(TL.ANCHORS["9a"]["fox_seat_screen"][0]), space="close", send=0.2, cue="fox_looks_us")
    # the paw goes down on the blank page
    tp = C("final_paw")
    p_paw = screen_pan(TL.ANCHORS["9b"]["page_quad_screen"][0][0] + 100.0)
    mx.add("foley", crinkle(rng, 0.08, 1200.0, 1200.0, 5000.0), tp, gain=0.06, pan=p_paw, space="close", send=0.25, cue="final_paw")
    mx.add("foley", paw_step(rng, 0.55, 0.0), tp, gain=0.10, pan=p_paw, space="close", send=0.25)
    # ink blooms and writes the title in one hand-brushed stroke
    # one brush contact per drawn stroke (events.json title_strokes): heavier at the start of a stroke,
    # tapering as it lifts, a soft touch at every landing; a dot is a single tap
    t0, t1 = C("title_ink_start"), C("title_ink_end")
    strokes = [(float(u), float(v)) for u, v in (EV("title_strokes") or [(t0, t1)])]
    n = seg_n(t0, t1 + 0.05)
    tt = tl(t0, n)
    speed = np.zeros(n)
    for u, v in strokes:
        d = max(v - u, 0.004)
        x_ = (tt - u) / d
        lobe = smoothstep((tt - u) / 0.01) * (1.0 - smoothstep((tt - (v - min(0.015, d / 2))) / min(0.015, d / 2)))
        speed = np.maximum(speed, lobe * (0.95 - 0.35 * np.clip(x_, 0, 1)))
    y = brush_stroke(np.clip(speed, 0, 1), rng, rise=(900.0, 1300.0, 0.06))
    for u, v in strokes:
        i = int((u - t0) * SR)
        nb = int(0.006 * SR)
        y[i:i + nb] += filt(sos_bp(1500.0, 5000.0), rng.standard_normal(nb)) * np.exp(-taxis(nb) / 0.0012) * (0.5 if v - u > 0.02 else 0.8)
    q = TL.ANCHORS["9b"]["page_quad_screen"]
    xl, xr = q[0][0] + 0.15 * (q[1][0] - q[0][0]), q[0][0] + 0.85 * (q[1][0] - q[0][0])
    mids = np.array([0.5 * (u + v) for u, v in strokes])
    px_ = np.interp(tt, mids, np.linspace(xl, xr, len(mids))) if len(mids) > 1 else np.full(n, 0.5 * (xl + xr))
    pan = np.clip((px_ - 960.0) / 960.0, -1, 1)
    mx.add("foley", edge_fade(place_stereo(y, pan), 0.0005, 0.02), t0, gain=0.045, space="close", send=0.3, cue="title_ink_start")
    # the hanko presses in, then lifts with a sticky little tick
    th = C("hanko")
    hk = modal([rng.uniform(90, 105), rng.uniform(170, 190), rng.uniform(320, 360), 920.0, 1600.0],
               [1.0, 0.5, 0.3, 0.1, 0.05], [0.06, 0.04, 0.025, 0.02, 0.012], 0.15, rng, 0.0015)
    L = int(0.08 * SR)
    hk[:L] += filt(sos_bp(300.0, 2000.0), rng.standard_normal(L)) * env_ar(L, 0.002, 0.02) * 0.5
    hk[:L] += filt(sos_bp(150.0, 900.0), rng.standard_normal(L)) * env_ar(L, 0.01, 0.03) * 0.3
    mx.add("foley", edge_fade(hk, 0.001, 0.02), th, gain=0.28, pan=0.35, space="close", send=0.25, cue="hanko")
    nb = int(0.03 * SR)
    lift = filt(sos_bp(1200.0, 5000.0), rng.standard_normal(nb)) * crackle_env(nb, 2500.0, rng, tau=0.0008)
    mx.add("foley", lift, th + 0.32, gain=0.04, pan=0.35, space="close", send=0.25)
    # the smaller line of lettering: quicker, lighter strokes
    t0, t1 = C("line_ink_start"), C("line_ink_end")
    n = seg_n(t0, t1)
    x = np.linspace(0, 1, n)
    speed = np.clip(smoothstep(x / 0.02) * (0.25 + 0.75 * np.abs(np.sin(np.pi * 12.0 * x + 0.5)) ** 0.7) * (1.0 - smoothstep((x - 0.94) / 0.06)), 0, 1)
    y = brush_stroke(speed, rng, wet=0.6)
    mx.add("foley", edge_fade(place_stereo(y, 0.0 + 0.4 * x), 0.003, 0.02), t0, gain=0.025, space="close", send=0.3, cue="line_ink_start")
    # a last tiny tail flick
    tf = C("tail_flick")
    mx.add("foley", fur_rustle(rng, 0.2, 1.0, 600.0, 4000.0), tf, gain=0.06, pan=screen_pan(TL.ANCHORS["9b"]["fox_seat_screen"][0]), space="close", send=0.25, cue="tail_flick")
    p9 = screen_pan(TL.ANCHORS["9b"]["fox_seat_screen"][0])
    wo = whoosh(0.22, rng, 300.0, 1200.0, 0.4, 0.25, p9 - 0.05, p9 + 0.1)
    mx.add("foley", wo, tf, gain=0.025, space="close", send=0.2)


def render_sfx(mx):
    before_dawn(mx)
    room_tone(mx)
    steps = fox_walk(mx)
    lantern(mx)
    closeup(mx)
    hop(mx)
    page_and_ink(mx)
    tremor(mx)
    burst(mx)
    shutters(mx)
    sunrise(mx)
    great_bird(mx)
    last_page(mx)
    return {"paw_steps": steps}
