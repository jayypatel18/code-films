# A1: lead animator for Film A, "THE LAST PAGE"

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

You are the lead of a small team making Film A. Read first, in this order:
1. `films/the-last-page/briefs/COMMON.md` (rules and house style, binding).
2. `films/the-last-page/PROMPT.md` (the creative prompt; this is the film you are making).
3. `films/the-last-page/timeline.py` (cuts, cameras, anchors, lights, cues; the contract between departments).
4. Reference sheets in `reference/` (look at `chetas-film2-last-page-sheet.png` and `chetas-film1-sheet.png`).

The title card says **Opus 5.5** because that is the model making this film (you). The whole team is Opus 5.5.

## The team
- A1 (you): the drawing engine, the fox, the ink bird, the brushed title lettering and hanko, the final compositing and the render.
- A2 (world): the stroke painter and every painted background plate, plus animated world layers (pampas grass, clouds, window light, dust, light shafts, trembling books, books flying open, the stairwell, the dome shutters and gold light, the sunrise sky, the wind ring). A2 exposes `films/the-last-page/world/__init__.py: render(cut_id, f) -> dict` with keys `bg` (required), `fg` (optional, drawn over the fox), `glow` (optional additive light, float32 (1080,1920,3)). Layers follow the conventions at the top of `timeline.py`.
- A3 (sound): the score and sound design, cued from `timeline.CUES` and later from your `build/events.json`.
- A4 (paper, starts later): the paper effects: pages tearing free, thousands of unique paper birds, the vortex, the great bird of pages, the falling pages. It will expose `films/the-last-page/paper/__init__.py: render(cut_id, f) -> dict` with `back` and `front` layers. Until it exists, your compositor treats a missing module as empty layers.
- The director session (who sent you this) directs, relays notes between departments, reviews stills and commits.

Compositing order you own: world `bg`, paper `back`, actors (fox, lantern, ink bird, lettering), world `fg`, paper `front`, world `glow` (screen or add), then post (grade per cut, bloom, shake, vignette, grain, dither).

## What you own (write only here)
- `engine/` (shared drawing engine for later films too), except `engine/paint*.py` and `engine/noise.py`, which A2 owns. You may import A2's modules.
- `films/the-last-page/actors/`, `films/the-last-page/render.py`, `films/the-last-page/post.py` if you want it separate.
- `films/the-last-page/timeline.py` (refine anchors and cues; never move cut boundaries without the director session).
- `films/the-last-page/NOTES.md` is a shared log: append entries starting with `A1 HH:MM IST:`.
- Your outputs go in `films/the-last-page/build/` (cache, review stills) and `films/the-last-page/out/`.

## The fox (the most important thing in the film)
The audience must fall for it in shot 3. Design an original small red fox (not any existing character): large expressive amber eyes with dark pupils and two catchlights, big black-tipped ears with cream inner fur that swivel independently, a short muzzle with a small black nose, white cheeks and a jagged white chest ruff, dark socks on its legs, a big fluffy tail with a white tip. Appealing proportions: a little young, head large relative to the body, but a real fox silhouette, not a plush toy. Suggested palette: orange base around #D9772B, shadow around #A24E2C (shift toward violet in moonlight), cream #F4E6CF, socks #3A2A2A, iris #E8A33A, line #3B2618. Change anything that makes it better.

Keep it on model from every angle. A good way: build the head and body from a simple 3D construction (head sphere, muzzle, ear sockets, eye placement on a curved face, body masses, a spline tail) driven by pose parameters (head yaw, pitch, tilt, each ear's rotation, eye openness, gaze, mouth, body lean, leg phases, tail curve), project to 2D, and draw the result as 2D cel shapes with ink lines. Then animate by keying those parameters on twos and threes with holds, overshoot and settle. Jagged fur edges (cheeks, ruff, tail tip, inner ears) are drawn as irregular tapered zigzags. You may do it another way if it looks better; judge only by whether frames look hand-drawn.

The eyes carry the film. Layer them: thick upper lid line, iris darker at the top under the lid shadow and lighter at the bottom, pupil, a big upper catchlight and a small lower one, plus a warm lantern glint in the cuts where the lantern is near. The blink: open, half, closed, a held closed drawing for 3 to 4 frames, half, open.

Views and actions needed (see `ANCHORS` in `timeline.py` for positions and sizes):
- Cut 2: side view, small, walking in from the left with a paper lantern held in its mouth (walk cycle on twos, real weight, tail overlapping), stops, sets the lantern down, looks up. The lantern is your prop (paper, bamboo ribs, warm flame inside); it casts warm light on the fox.
- Cut 3: close-up, three quarter front. Ears swivel one at a time, two quick sniffs (nose and whiskers), the blink with a held closed frame, lantern light in its eyes, curious and a little afraid.
- Cut 4a: hops up onto the lectern (anticipation crouch, leap with a smear frame, landing squash, settle). Cut 4b: only the paw pressing the page.
- Cut 4b also: the ink bird. One line of ink appears under the paw and draws itself in one confident brushstroke into a tiny sketched bird, which shivers, peels off the page (still ink on paper) and flaps away.
- Cut 5c: medium shot, ears flatten, two hard held poses. Cut 5d: tiny in the wide.
- Cut 7: tiny on the balcony ledge seen from behind at three quarters, fur and tail whipping in the wind, looks up, mouth slightly open.
- Cut 8: a few pixels tall at the bottom left for scale.
- Cuts 9a and 9b: seated three quarter on the balcony. Looks at the page at its paws, then at us (the camera), places a paw on the page. The tail flicks once at the end.

## The title card (cut 9b)
Draw every letter as brush strokes in code. Author stroke paths (centre lines with pressure) for each glyph you need: "Opus 5.5" large, and below it in smaller brushed lettering "Every story begins with a blank page." Make the brush real: pressure along the stroke, ink pooling where the brush lands, dry brush streaks where it lifts, ink bleeding slightly into the paper fibres, and the strokes drawing on in order as the ink "writes" from under the paw. The red hanko seal reading 寓話 presses in after the title: a square seal in seal-script style, characters carved white out of red with rough uneven edges, pressed with a slight squash. Text must be fully legible on a phone screen.

## Stages (return to the director session at each gate)
**Stage 1 (gate): the fox model sheet.** Build the engine core (anti-aliased cel fills at 3x, pressure ink lines, shadow shapes, lighting from `LIGHTS`, compositing) and the fox. Save to `films/the-last-page/build/review/a1-stage1/`: a model sheet with the fox in at least six poses (three quarter close-up neutral; ears swivelled; eyes closed mid-blink; side walk key poses with the lantern; seated three quarter; ears flat; back three quarter on the ledge looking up), plus one composite of cut 3 at full resolution using a simple dark painted placeholder background lit by `LIGHTS["hall"]` and the lantern. Inspect them hard, improve, then stop and report with paths. Aim for this within about 2 hours.

**Stage 2: all the acting and the render pipeline.** Animate every fox action, the ink bird, the title card. Write `render.py`: renders any frame range or single frames to stills (`--stills 340,400`), a contact sheet per cut, and streams the full film to ffmpeg (`out/the-last-page-master.mp4`, H.264 high, yuv420p, 24 fps, CRF 14 or lower, no audio). Use A2's world where it exists and placeholders elsewhere. Write exact event times (each paw step, blink, sniff, hop, landing, paw presses, ink strokes, hanko, tail flick) to `films/the-last-page/build/events.json` as `{"name": seconds}` or lists, for sound. Do a full rough render and report.

**Stage 3: final.** With the world, paper and sound in place, polish every cut through the inspection loop, render the master, then mux and cut the deliverables:
- `out/the-last-page-full.mp4`: the film with the audio from `films/the-last-page/build/audio/the-last-page.wav`.
- `out/the-last-page-x.mp4`: the X cut. It opens cold on the fox in motion (about 1.5 to 2 s from cut 3: the ear swivel or the blink), then plays the film from the start. The fox must be visibly moving in the very first frame. Check the first 2 seconds frame by frame.
- Both: 1920x1080, H.264 high profile, yuv420p, 24 fps, AAC 48 kHz 192k, `-movflags +faststart`.
- Three stills in `out/`: best fox close-up, the climax, the title card.

Begin by writing your start time in NOTES.md.
