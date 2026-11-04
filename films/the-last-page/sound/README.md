# Sound of THE LAST PAGE

Original score and sound design, all synthesized in Python (numpy, scipy, numba). No samples,
no impulse response files, no borrowed melodies, nothing downloaded. Every time comes from
`films/the-last-page/build/events.json` (the animation's exact event frames, written by the lead
with `render.py --events`) where it has an entry, and from `films/the-last-page/timeline.py` `CUES`
otherwise. Re-timing to a new picture pass is one command: re-run `score.py`; it re-reads both and
prints every event that overrides a planned cue.

Render (heavy job, use the slot wrapper and run it in the background):

    python3 tools/heavy.py -- \
        python3 films/the-last-page/sound/score.py [--stems]

Outputs (48 kHz, stereo, 24 bit PCM):
- `films/the-last-page/build/audio/the-last-page.wav`: the film, exactly 60.000 s.
- `films/the-last-page/build/audio/the-last-page-coldopen.wav`: the X cut's cold open, exactly
  87,216 samples (1.817 s) from frame 362 (15.083 s) of the close-up.

The script prints loudness (own BS.1770 meter and ffmpeg `ebur128` and `loudnorm`), per-section
levels, the stillness and climax checks, a per-event cue check, and the cold open's numbers.

## Files

| file | what it holds |
|---|---|
| `score.py` | entry script: mixer, reverb sends, stillness gate, mastering, report, cue check |
| `music.py` | the motif, tempo maps pinned to cues, the composition, music rendering |
| `instruments.py` | felt piano, strings, horns, choir, timpani, bass drum, cymbal, harp, celesta |
| `sfx.py` | every sound effect, placed at events and cues, panned from the timeline anchors and camera |
| `coldopen.py` | the X cut's cold open (close-up window, piano hint, fades) |
| `dsp.py` | filters, envelopes, noise textures, buses, synthetic reverb, loudness, limiter, 24 bit WAV writer |

## The music

**Motif** (7 notes, original): `6 1' 5' | 4' 3' 1' 2'`, rhythm "da-da DAAA, da-da da DAAA".
In D flat major: `Bb4 Db5 Ab5 | Gb5 F5 Db5 Eb5`. It rises a minor third and a fifth, falls by
step and stops on the second degree, an open question. The question is only answered by the
last note of the film.

| time | music |
|---|---|
| 0.0 to 7.6 | no music: wind and near silence |
| 7.6 (`piano_in`) | solo felt piano, D flat major, the motif as the fox walks in; its Db lands as the lantern is set down (`lantern_down`) and the open Eb as the fox looks up (`look_up`) |
| 14.0 to 19.4 | the motif developed: the leap reaches a sixth (Bb5) over the close-up; a Bbm(add9) chord lands with `hop_land` |
| 21.3 (`paw_down`) | the pedal and dampers drop together: silence for the ink |
| 24.0 (`bird_flap_start`) | the motif head, high and light, as the ink bird flies |
| 25.0 to 31.0 | B flat minor: muted low string tremolo, the head in low piano, violas and celli with a raised seventh that bends down a semitone at `flame_lean`, dark horn cluster at `creak`, violin tremolo a minor second apart, F7(b9) crescendo cut dead at `stillness_start` |
| 32.0 to 37.75 | tutti hit on `burst`, string ostinato, a rising spiral of sixteenths up the stairwell, the motif head in horns at `cut_stairwell`, F7 roll, a caesura so the shutters crack in the open |
| 38.0 (`gold_light`) | the F7 resolves deceptively to G flat major: harp glissando, strings, choir; horns and violins carry the motif over the sunrise |
| 44.0 to 46.5 | the motif's last note Eb is held and becomes D sharp over B major: the pivot into E major, timpani roll and cymbal swell |
| 46.5 (`wingbeat`) | climax in E major (a chromatic third above the opening key): the motif at full scale, felt piano opened up and in octaves, violins in octaves, horns, low brass, choir, timpani, bass drum; harp glissandi on the arc; it ends on the open F sharp over B and fades as quiet returns |
| 52.0 to 55.0 | piano alone echoes the second half of the motif; the open F sharp rings until the pedal lifts at `cut_title` |
| 55.4 (`final_paw`) | one soft E5, the answer, pedal down so the other strings ring along, into the fade |

**Tempo maps.** Each metric section has a `TempoMap` whose pins are cue names (for example
`piano_in`, `cut_closeup`, `hop_land`, `paw_down`); beats between pins follow a monotone cubic
curve so tempo changes are smooth. Moving a cue stretches the music between its pins. Free-time
passages (the tremor, the bird, the ending) are written as offsets from cues.

## Instruments

- **Felt piano**: additive synthesis per note. Up to 90 inharmonic partials (string stiffness
  rising with pitch), two-stage decay per partial (prompt sound and aftersound), one to three
  detuned strings per key spread in stereo, a strike-point comb, and a felt strip model: a low,
  velocity-dependent lowpass with faster decay of the high partials. A woolly hammer thump, key
  bottoming and damper noises (close miked), sustain pedal noises, and **sympathetic resonance**:
  while the pedal is down the piano is convolved with a synthesized bank of 88 free strings (their
  first partials, piano-like decay times), which dies when the dampers fall. At the climax the
  felt is lifted (`felt=0.3`) so the piano opens up like a grand.
- **Strings** (modelled player by player; 10, 9, 7, 6 and 5 players): each player's source is a
  bow-position pulse, a band-limited sawtooth minus its copy shifted by the bow position beta,
  which carves the harmonic notches a real bow makes; beta drifts as the bow moves. Each player
  has an intonation offset, slow pitch drift, a pitch that settles on the attack, vibrato with its
  own rate wobble, depth and delayed onset, and bow pressure flutter. Rosin noise is pulsed at each
  slip of the string (once per period), a bow-bite scratch marks attacks (and every stroke of a
  tremolo). Brightness opens with the attack and closes on the release. The two desks of each
  section go through their own synthesized instrument body (modal: air mode, wood modes, dense upper
  modes with a bridge hill; scaled for viola, cello, bass). Articulations: legato, marcato,
  spiccato, tremolo, con sordino, glides. Loudness is matched to the v1 sections so the mix holds.
- **Horns and low brass**: per player, a lip scoop up into the pitch, brightness that follows the
  player's level (brighter when louder, a blip on tongued attacks), slow drift and almost no
  vibrato, tongue and breath noise, then a brassy edge only at fortissimo, the horn's formants and a
  closed, round bell. A darker stopped colour for the creak.
- **Choir**: per singer, a vocal tract of its own (formants scaled a few percent), vibrato with
  its own rate and onset, jitter and shimmer, a small scoop, breath shaped by the same formants;
  vowels morph from "oo" to "ah".
- **Percussion**: modal timpani (rolls of single strokes), a pitched-sweep bass drum, a suspended
  cymbal (soft mallet crash and roll swell, kept below 9 kHz).
- **Harp and celesta**: modal plucked strings with a pluck-point comb; celesta bars.

## Sound design (`sfx.py`)

| sound | how it is made |
|---|---|
| wind (before dawn, sunrise, morning) | two cascaded lowpasses of noise, gust curve, brighter air on gusts, a narrow wandering whistle on strong gusts |
| pampas grass, fur | crackle grains (Poisson spikes with exponential decay) on high band noise plus a swish, following the wind's gust curve |
| room tone of the library | low stone rumble, faint air, wind muffled by thick walls; its own bus so the stillness keeps it |
| paw steps | felt-like pad (band noise burst) and a soft sine thump, sometimes one or two tiny claw ticks; timing follows a walking gait that slows into `fox_stop` |
| lantern | a breathy flame flutter with rare pops, level and pan per shot; bamboo frame creaks with the stride; a wooden tap with a bounce at `lantern_down` |
| ears, sniffs | short band-passed fur swishes with a low flap; nasal inhales through three formant resonances, in pairs |
| hop and landing | claw scrabble, a body whoosh, lectern wood modes, a paper fwap and crinkle, the old binding creaks (stick-slip) |
| ink drawing itself | brush model: bristle scratch grains, wet drag, tiny wet pops, and a resonance rising two octaves as the bird forms |
| ink bird | paper shiver (crinkle at 28 Hz), a peel (accelerating crackle), tiny flaps receding up to the right |
| books trembling | rattles: irregular knocks of modal book-on-shelf resonances, sixteen books spread across the row, a woody rumble |
| dust | very dense, very fine crackle grains, thin streams |
| tower creak | stick-slip friction impulses through six resonances that rise as it travels up, a sub groan, near and far versions crossfaded |
| stillness | all buses gated to zero for one second (30 ms fade); only the room tone and one grain of dust |
| paper roar and vortex | many flapping voices (band noise gated by flapping waves of their own rates), book cover slaps, tears, crisp fold snaps; the vortex rotates in stereo, brightens and peaks at `vortex_peak`, then sounds far up in the dome |
| copper shutters | a metal groan, a crack, four inharmonic dull copper clangs, a squeal, a scrape, falling debris, air rushing in |
| wind ring | an inhale, a falling sub boom, a blast whose stereo width grows as the ring rolls outward, a long low tail |
| the great bird | the flock's flapping rates synchronise and its stereo image collapses to the centre; an upstroke inhale; the wingbeat is a 58 to 31 Hz sub with harmonics, a falling lowpass "whoomph", a slam of pages and a flutter tail, then an air wash; the arc is one long pass following the bird's path with a trail of paper |
| pages falling, page lands | slow swishes swinging left and right, a soft paper slap |
| title ink, hanko, tail flick | brush strokes (one long stroke, then small quick ones), a dense stamp thud with a wet squish and a sticky lift, a fur swish |

## Measured (Stage 2 render, events.json of 20:06 IST)

| check | result |
|---|---|
| format | film: 2,880,000 samples (60.000 s); cold open: 87,216 samples (1.817 s); both 48 kHz, stereo, pcm_s24le |
| integrated loudness | -14.0 LUFS (ffmpeg ebur128), -14.05 (loudnorm), -14.01 (own meter) |
| true peak | -1.3 dBTP, 0 clipped samples; limiter above 3 dB for 0.21% of the time (max 5.9 dB on the burst hit) |
| loudness range | 14.4 LU (ebur128), 14.9 LU (loudnorm); quiet parts lifted 4 to 6 dB for phones |
| loudest moment | climax: momentary -8.4 LUFS at 47.5 s, short-term -9.2 LUFS centred 47.2 s |
| stillness (31.0 to 32.0) | room tone and a faint breath of air, -37.8 LUFS momentary at most |
| quiet parts, short-term | wind -24.0, felt piano -20.5, ink -22.7, quiet return -18.5, last note -28 (LUFS) |
| above 8 kHz | -29.9 dB of total energy (climax -34.7 dB) |
| stereo | L/R correlation 0.73, mono fold-down -0.55 LU |
| cues | all 43 tagged effects within one frame of their event (most within 3 ms), 22 of 22 paw steps and 4 of 4 ink bird wingbeats within 2 ms |
| cold open | mean -22.2 LUFS, momentary max -19.0 (the film's mix in the same window: -19.5), ends in digital silence |

`--stems` prints per-bus loudness per section, per-bus loudness in the 0.4 s after each key cue
(to check that the effects are not masked by the music), and per-bus peaks around the big hits.

**Re-timing to a new picture pass.** When the lead rewrites `build/events.json`, re-run the one
command above. Every effect with an event (paw steps, lantern, look up, ears, sniffs, blink, hop
crouch, takeoff and land, paw down, ink, bird shiver, peel and wingbeats, ears flat, flame lean,
mouth open, looks, final paw, title strokes, line, hanko, tail flick) moves to it, the piano's
tempo map follows `lantern_down`, `look_up`, `hop_land` and `paw_down`, pans follow the timeline
anchors and camera, and the report lists every override and checks every event.

## Space, mix, master

Reverbs are synthesized, not loaded: band-split decaying noise (highs decay faster) with early
reflections. Four spaces: a scoring stage for the music (2.6 s), the stone library (4.6 s),
open air (1.3 s, sparse late reflections), and a small close space for the intimate sounds.
Inside and outside reverbs are cut with the picture. Before the master, the strings, brass, paper and
percussion buses have transparent peak control (rare transient peaks only), the quiet first half
and the ending are lifted 4 to 6 dB (loudness range for phones), and the room tone comes up about
11 dB inside the stillness. Master: 25 Hz highpass, a gentle high shelf at 9 kHz and a 15.5 kHz
lowpass (nothing harsh up top), a low band (below 110 Hz) peak compressor, a 2:1 compressor, a
look-ahead true peak limiter at -1.3 dBTP, loudness set to -14 LUFS integrated, TPDF dither to
24 bit. The cold open goes through the same gains and compressor curves as the film.
