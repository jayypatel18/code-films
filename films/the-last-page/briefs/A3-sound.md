# A3: sound for Film A, "THE LAST PAGE"

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

You write the original score and the sound design, all synthesized in code. Read first:
1. `films/the-last-page/briefs/COMMON.md` (rules, binding; the sound rules are the same: no samples, no impulse response files, no borrowed melodies, nothing downloaded).
2. `films/the-last-page/PROMPT.md`, especially the SOUND section.
3. `films/the-last-page/timeline.py`: `CUES` (every beat in seconds) and `CUTS`.
4. For technique only: `soundtrack.py of our previous film (github.com/jayypatel18/last-train)` is the score of our previous film (numpy synthesis: felt piano, strings, choir, reverb, a tempo map pinned to cues, mastering). You may reuse its techniques and code, since the same model wrote it, but write a new score for this film with a new melody.

The whole team is Opus 5.5 (you included).

## What you own (write only here)
- `films/the-last-page/sound/` (the synthesis code; one entry script, `score.py`, plus modules if you want).
- Output: `films/the-last-page/build/audio/the-last-page.wav`, 48 kHz, stereo, 24 bit PCM, exactly 60.000 s (2,880,000 samples).
- Append to `films/the-last-page/NOTES.md` with entries starting `A3 HH:MM IST:`.

## The sound of the film
From the prompt: start with wind and near silence, then a lone felt piano motif when the fox appears (`piano_in`). Low strings and creaking wood during the tremor (cut 5). A sudden roar of fluttering paper at `burst`. A full orchestral swell at sunrise (from `gold_light` through cut 7). At the climax (`wingbeat`) the piano motif returns at full scale with the orchestra. Quiet returns at `cut_last_page`. End with a single soft piano note as the ink touches the final page (`final_paw`), and let it ring into the fade.

Sound effects: paw steps on stone (soft pads, tiny claw ticks), the lantern (a soft flame flutter while it is near, a wooden tap at `lantern_down`), the ears and sniffs in the close-up (tiny, intimate), the hop and landing on the book, the paw on paper, the ink drawing itself (wet brush on paper, rising pitch as the bird forms), the paper peel and tiny flaps of the ink bird, books trembling and rattling, dust sifting, a deep low creak travelling up the tower, the total stillness beat (`stillness_start` to `stillness_end`: almost nothing, a held breath of room tone), the paper roar and vortex, the copper shutters cracking open, the wind ring, the one huge wingbeat with real weight, pages fluttering down, the hanko press, a last tiny tail flick.

Music craft: an original motif of 5 to 9 notes that is simple enough to hum, first on solo felt piano (close, soft hammers, felt noise, sympathetic resonance, pedal), developed under the fox's discovery, broken into fragments under the tremor, then carried by full strings, horns and timpani at sunrise and the climax. Real dynamics: the quiet parts must be truly quiet, the climax must feel huge. Pin the music to the cues with a tempo map so moving a cue does not break the music. Synthesize space with a reverb you build in code (an algorithmic reverb or a convolution with a decaying noise tail you generate).

Mix and master: stereo image with depth, nothing harsh above 8 kHz, no clipping. Integrated loudness about -14 LUFS, true peak at most -1 dBTP (measure with ffmpeg's `ebur128` or `loudnorm` filter and print the numbers). Check that every cue lands within one frame (42 ms).

## Stages
**Stage 1: full score and sound design v1** against `timeline.CUES`. Render the WAV, measure loudness and peaks, and listen through the numbers: print per-section levels, check that the stillness beat is near silence and the climax is the loudest moment. Write a short `sound/README.md` (what every layer is and how it is made). Report back with the WAV path, the loudness numbers and anything you are unsure about. Aim for about 2 hours.

**Stage 2: sync to picture.** When the lead writes exact event times to `films/the-last-page/build/events.json`, re-time paw steps, blinks, hops and ink to them and render again. Also render a short cold-open WAV for the X cut once the lead picks the cold-open segment. The director session will message you when this is ready.

Begin by writing your start time in NOTES.md.
