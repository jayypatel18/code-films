"""The score of TINY ROCKET: one motif, arranged for felt piano, music box,
celesta, strings, horns, timpani and cymbal, placed on the timeline from
cues.json.

The motif (D major, 7 notes, hummable):

    D4  A4  B4 A4  F#4  E4  D4
    1   1   .5 .5  1    1   2      (beats at 72 bpm, quarter = 0.8333 s)

A leap up a fifth (the reach for the sky), one step higher, then it falls
back home. It is first heard alone on the felt piano when the panda admires
the rocket (5.6 s), in slow motion on the music box under the stars, and at
full orchestral scale when the trail ignites into a shooting star (46.0 s).
A single felt piano D5 is the brush stroke at 58.9 s.
"""

import numpy as np

from dsp import nn
from instruments import (bass_drum, brass, celesta, cymbal, felt_piano, glock,
                         music_box, pad, string_run, strings, timpani,
                         timpani_roll)

BEAT = 60.0 / 72.0

# (note, beats)
MOTIF = [("D", 1.0), ("A", 1.0), ("B", 0.5), ("A", 0.5), ("F#", 1.0), ("E", 1.0), ("D", 2.0)]


def motif_times(start, beat=BEAT):
    """Absolute start times of the motif's notes."""
    out = []
    t = start
    for _, b in MOTIF:
        out.append(t)
        t += b * beat
    return out


def motif_notes(octave):
    return [f"{name}{octave}" for name, _ in MOTIF]


def arrange(put):
    """Place every musical event. put(bus, signal, at, gain_db, pan, send_db)
    is provided by score.py. Buses: 'music'. send_db is the hall send."""

    # ---- shot 2: the lone felt piano as the panda admires the rocket (5.6 s)
    times = motif_times(5.6)
    for t, note, (_, beats) in zip(times, motif_notes(4), MOTIF):
        put("music", felt_piano(note, length=beats * BEAT + 2.2, velocity=0.42), t, -14.0, 0.05, -8.0)
    # left hand, gentle
    for t, notes in [(5.6, ["D3", "A3"]), (7.27, ["G3", "D4"]), (8.93, ["A2", "E3"]), (9.77, ["D3", "F#3"])]:
        for i, nt in enumerate(notes):
            put("music", felt_piano(nt, length=3.0, velocity=0.3), t + 0.015 * i, -19.0, -0.15, -8.0)

    # ---- shot 3: the night. a low sine pad, a music box arpeggio, the motif
    # in augmentation on the celesta with the shooting star at 15.6 s
    put("music", pad("D2", length=9.5, velocity=0.5, attack=2.5, release=2.0), 12.0, -30.0, 0.0, -10.0)
    put("music", pad("A3", length=8.0, velocity=0.3, attack=3.0, release=2.0), 12.5, -30.0, 0.2, -10.0)
    arp = ["D5", "F#5", "A5", "B5", "A5", "F#5"]
    t = 12.4
    k = 0
    while t < 19.3:
        note = arp[k % len(arp)]
        vel = 0.55 * (1.0 if t < 17.5 else max(0.2, 1 - (t - 17.5) / 2.0))
        put("music", music_box(note, length=1.6, velocity=vel, seed=k), t, -21.0, -0.35 + 0.15 * (k % 3), -12.0)
        t += 0.4
        k += 1
    slow = [(13.6, "D6"), (14.4, "A6"), (15.2, "B6"), (16.4, "A6"), (17.2, "F#6"), (18.0, "E6"), (18.8, "D6")]
    for t, note in slow:
        put("music", celesta(note, length=2.4, velocity=0.5), t, -20.0, 0.3, -10.0)
    # the shooting star gets a glockenspiel sparkle
    put("music", glock("D7", length=1.6, velocity=0.4), 15.62, -28.0, -0.4, -10.0)
    put("music", glock("A7", length=1.6, velocity=0.3), 15.78, -31.0, 0.4, -10.0)

    # ---- shot 4: first light. the pad turns to B minor, uneasy
    put("music", pad("B2", length=5.5, velocity=0.45, attack=2.0, release=1.5), 20.4, -26.0, 0.0, -10.0)
    put("music", pad("F#3", length=5.0, velocity=0.3, attack=2.5, release=1.5), 20.6, -31.0, -0.2, -10.0)
    put("music", pad("D4", length=4.5, velocity=0.25, attack=3.0, release=1.5), 21.0, -34.0, 0.2, -10.0)
    # a high string note creeps in at the cockpit and holds its breath
    put("music", strings("A5", length=7.0, velocity=0.35, attack=2.5, release=0.1, voices=3, brightness=0.8),
        22.5, -30.0, 0.1, -9.0)

    # ---- shot 5: the fuse. low drone swelling, high tremolo rising a half
    # step at the eye squeeze. everything is cut at 29.4 by the stillness gate
    put("music", strings("D2", length=4.6, velocity=0.6, attack=3.5, release=0.1, voices=4, brightness=0.5, tremolo=0.6),
        24.9, -22.0, 0.0, -10.0)
    put("music", strings("A2", length=3.8, velocity=0.5, attack=3.0, release=0.1, voices=3, brightness=0.5, tremolo=0.6),
        25.7, -26.0, 0.0, -10.0)
    put("music", strings("A5", length=3.8, velocity=0.55, attack=1.5, release=0.05, voices=4, brightness=1.0, tremolo=0.8),
        25.7, -24.0, -0.1, -9.0)
    put("music", strings("Bb5", length=0.95, velocity=0.65, attack=0.08, release=0.05, voices=4, brightness=1.1, tremolo=0.9),
        28.58, -21.0, 0.1, -9.0)
    put("music", strings("E6", length=0.95, velocity=0.5, attack=0.1, release=0.05, voices=3, brightness=1.1, tremolo=0.9),
        28.6, -27.0, -0.2, -9.0)

    # ---- shot 6: ignition and lift. drums, low horns, strings climbing
    put("music", bass_drum(velocity=0.9), 31.0, -8.0, 0.0, -12.0)
    put("music", timpani("D2", velocity=0.8), 31.0, -10.0, -0.1, -8.0)
    put("music", timpani_roll("D2", 1.2, np.linspace(0.2, 1.0, 12), rate=15.0), 31.0, -13.0, 0.0, -8.0)
    put("music", timpani("D2", velocity=0.9, seed=2), 32.2, -9.0, 0.0, -8.0)
    put("music", bass_drum(velocity=0.8, seed=2), 32.2, -9.0, 0.0, -12.0)
    put("music", brass("D2", length=3.4, velocity=0.6, attack=0.9, release=0.6), 31.0, -17.0, -0.2, -8.0)
    put("music", brass("A2", length=3.0, velocity=0.5, attack=1.1, release=0.6), 31.4, -20.0, 0.2, -8.0)
    scale1 = ["D3", "E3", "F#3", "G3", "A3", "B3", "C#4", "D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"]
    put("music", string_run(scale1, 0.36, 0.2, velocity=0.6), 33.0, -16.0, 0.0, -7.0)
    for note, p in [("D3", -0.3), ("F#3", 0.3), ("A3", 0.0)]:
        put("music", brass(note, length=4.2, velocity=0.6, attack=0.4, release=0.6), 33.0, -19.0, p, -8.0)
    put("music", timpani("D2", velocity=0.6, seed=3), 34.0, -14.0, 0.0, -8.0)
    put("music", strings("D4", length=4.5, velocity=0.5, attack=1.0, release=0.8, voices=6), 33.4, -20.0, 0.0, -7.0)

    # ---- shot 7: through the clouds. chords step IV, V and a second run up
    for note, p in [("G3", -0.3), ("B3", 0.3), ("D4", 0.0)]:
        put("music", brass(note, length=2.8, velocity=0.65, attack=0.3, release=0.5), 37.0, -18.0, p, -7.0)
    for note, p in [("G4", -0.4), ("D5", 0.4)]:
        put("music", strings(note, length=2.8, velocity=0.55, attack=0.5, release=0.5), 37.0, -20.0, p, -7.0)
    put("music", timpani("G2", velocity=0.7, seed=4), 37.0, -12.0, 0.0, -8.0)
    for note, p in [("A3", -0.3), ("C#4", 0.3), ("E4", 0.0)]:
        put("music", brass(note, length=3.2, velocity=0.7, attack=0.3, release=0.5), 39.6, -17.0, p, -7.0)
    for note, p in [("A4", -0.4), ("E5", 0.4)]:
        put("music", strings(note, length=3.2, velocity=0.6, attack=0.5, release=0.5), 39.6, -19.0, p, -7.0)
    put("music", timpani("A2", velocity=0.7, seed=5), 39.6, -12.0, 0.0, -8.0)
    scale2 = ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5", "E5", "F#5", "G5", "A5", "B5", "C#6", "D6"]
    put("music", string_run(scale2, 0.4, 0.22, velocity=0.65, seed=20), 38.0, -15.0, 0.0, -7.0)
    put("music", timpani_roll("A2", 1.6, np.linspace(0.3, 1.0, 12), rate=16.0, seed=6), 41.0, -13.0, 0.0, -8.0)
    put("music", cymbal(length=4.0, swell=1.5, velocity=0.5), 41.1, -20.0, 0.3, -6.0)

    # ---- 42.6 break into open sky: the big D major, held into the wonder
    put("music", bass_drum(velocity=1.0, seed=3), 42.6, -7.0, 0.0, -12.0)
    put("music", timpani("D2", velocity=1.0, seed=7), 42.6, -9.0, 0.0, -7.0)
    for note, p, g in [("D3", -0.3, -15.0), ("A3", 0.3, -16.0), ("D4", -0.1, -16.0), ("F#4", 0.2, -17.0)]:
        put("music", brass(note, length=2.6, velocity=0.85, attack=0.15, release=1.2), 42.6, g, p, -6.0)
    for note, p, g in [("D4", -0.5, -17.0), ("A4", 0.5, -17.0), ("F#5", -0.2, -18.0), ("D6", 0.2, -20.0)]:
        put("music", strings(note, length=3.0, velocity=0.8, attack=0.12, release=1.4, voices=6), 42.6, g, p, -6.0)
    put("music", glock("D6", length=2.0, velocity=0.5), 42.6, -22.0, -0.3, -8.0)
    put("music", glock("A6", length=2.0, velocity=0.4), 42.62, -25.0, 0.3, -8.0)

    # ---- shot 8 first half: the speck. pp high strings swelling into 46.0
    put("music", strings("D6", length=2.3, velocity=0.5, attack=1.6, release=0.1, voices=4, brightness=0.9), 44.0, -26.0, -0.2, -8.0)
    put("music", strings("A5", length=2.3, velocity=0.5, attack=1.6, release=0.1, voices=4, brightness=0.9), 44.0, -27.0, 0.2, -8.0)
    put("music", strings("F#4", length=2.1, velocity=0.4, attack=1.5, release=0.1, voices=4, brightness=0.8), 44.2, -28.0, 0.0, -8.0)
    put("music", cymbal(length=3.0, swell=1.25, velocity=0.5, seed=2), 44.75, -20.0, -0.3, -6.0)
    put("music", timpani_roll("D2", 1.3, np.linspace(0.1, 1.0, 12), rate=16.0, seed=8), 44.7, -15.0, 0.0, -8.0)

    # ---- 46.0 star ignites: the motif at full scale
    times = motif_times(46.0)
    for i, (t, (_, beats)) in enumerate(zip(times, MOTIF)):
        ln = beats * BEAT
        name = MOTIF[i][0]
        last = i == len(MOTIF) - 1
        hold = ln + (3.5 if last else 0.25)
        put("music", brass(f"{name}4", length=hold, velocity=0.85, attack=0.06, release=1.0 if last else 0.25), t, -14.0, -0.15, -6.0)
        put("music", brass(f"{name}3", length=hold, velocity=0.75, attack=0.08, release=1.0 if last else 0.25, seed=1), t, -17.0, 0.15, -6.0)
        put("music", strings(f"{name}5", length=hold, velocity=0.85, attack=0.07, release=1.2 if last else 0.3, voices=6, seed=i), t, -15.0, 0.25, -6.0)
        put("music", strings(f"{name}6", length=hold, velocity=0.7, attack=0.07, release=1.2 if last else 0.3, voices=5, seed=i + 7), t, -19.0, -0.25, -6.0)
        put("music", felt_piano(f"{name}5", length=hold + 1.0, velocity=0.9, seed=i), t, -17.0, 0.1, -6.0)
        put("music", felt_piano(f"{name}4", length=hold + 1.0, velocity=0.85, seed=i + 3), t, -19.0, -0.1, -6.0)
        if i in (0, 1, 2, 6):
            put("music", glock(f"{name}6", length=2.2, velocity=0.55, seed=i), t, -20.0, 0.35 if i % 2 else -0.35, -6.0)
    chords = [(46.0, ["D3", "F#3", "A3"], "D2"), (47.67, ["G3", "B3", "D4"], "G2"),
              (48.5, ["B2", "D3", "F#3"], None), (49.33, ["A2", "C#3", "E3"], "A2"),
              (50.17, ["D3", "F#3", "A3"], "D2")]
    for idx, (t, notes, tp) in enumerate(chords):
        nxt = chords[idx + 1][0] if idx + 1 < len(chords) else 53.0
        ln = nxt - t + (0.3 if idx + 1 < len(chords) else 0.0)
        last = idx == len(chords) - 1
        for j, nt in enumerate(notes):
            put("music", strings(nt, length=ln, velocity=0.7, attack=0.1, release=1.8 if last else 0.35, voices=5, seed=idx * 3 + j),
                t, -20.0, -0.3 + 0.3 * j, -6.0)
            put("music", brass(nt, length=ln, velocity=0.6, attack=0.1, release=1.5 if last else 0.3, seed=idx * 3 + j),
                t, -22.0, 0.3 - 0.3 * j, -6.0)
        if tp:
            put("music", timpani(tp, velocity=0.9 if idx in (0, 4) else 0.7, seed=10 + idx), t, -10.0, 0.0, -7.0)
    put("music", bass_drum(velocity=1.0, seed=4), 46.0, -7.0, 0.0, -12.0)
    put("music", bass_drum(velocity=0.9, seed=5), 50.17, -8.0, 0.0, -12.0)
    put("music", timpani_roll("A2", 0.84, np.linspace(0.4, 1.0, 8), rate=16.0, seed=9), 49.33, -14.0, 0.0, -8.0)
    put("music", cymbal(length=3.5, swell=0.05, velocity=0.5, seed=3), 46.0, -22.0, 0.3, -6.0)
    put("music", cymbal(length=3.5, swell=0.05, velocity=0.45, seed=4), 50.17, -24.0, -0.3, -6.0)

    # ---- shot 9: home. a quiet afterglow and the motif's last three notes
    for note, p in [("D5", -0.3), ("F#5", 0.3), ("A5", 0.0)]:
        put("music", strings(note, length=3.6, velocity=0.35, attack=0.4, release=2.2, voices=4, brightness=0.7, seed=30), 52.0, -30.0, p, -8.0)
    put("music", pad("D3", length=6.0, velocity=0.4, attack=1.5, release=2.5), 52.5, -30.0, 0.0, -10.0)
    put("music", felt_piano("A3", length=3.5, velocity=0.3), 53.2, -24.0, -0.2, -8.0)
    for t, nt in [(53.2, "F#4"), (54.03, "E4"), (54.87, "D4")]:
        put("music", felt_piano(nt, length=3.5, velocity=0.32), t, -20.0, 0.1, -8.0)

    # ---- 58.9 the brush stroke: one soft felt piano note
    put("music", felt_piano("D5", length=4.0, velocity=0.4), 58.9, -18.0, 0.05, -7.0)

    # ---- 60.0 the title: a rolled felt piano chord that rings into the paper
    for i, (nt, g) in enumerate([("D3", -24.0), ("A3", -25.0), ("F#4", -23.0), ("A4", -24.0), ("D5", -22.0)]):
        put("music", felt_piano(nt, length=4.4, velocity=0.38, seed=50 + i), 60.0 + 0.03 * i, g, -0.2 + 0.1 * i, -7.0)
    put("music", pad("D3", length=4.2, velocity=0.3, attack=1.2, release=1.8), 60.2, -34.0, 0.0, -10.0)
    put("music", celesta("A5", length=2.0, velocity=0.3), 62.3, -32.0, 0.2, -8.0)
