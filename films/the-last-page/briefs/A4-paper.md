# A4: paper effects for Film A, "THE LAST PAGE"

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo, and a remark comparing our aims with the reference film was trimmed. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

You make everything made of paper that flies: pages tearing free, thousands of paper birds, the vortex, the great bird of pages, and the pages that fall at the end. Read first, in this order:
1. `films/the-last-page/briefs/COMMON.md` (rules and house style, binding).
2. `films/the-last-page/PROMPT.md` (shots 6, 7, 8, 9 and the paper part of the quality standard).
3. `films/the-last-page/timeline.py` (cuts, cameras, anchors, cues; the contract).
4. `films/the-last-page/NOTES.md` (what the team has done so far).
5. The engine A1 has built in `engine/` (anti-aliased fills, ink lines, cel shading). Use it so your paper matches the fox's line work; if it lacks something you need for speed (batch drawing of thousands of small shapes), write that in your own folder.
6. Chetas's version of the same shots: `reference/chetas-film2-last-page-sheet.png`.

The whole team is Opus 5.5 (you included). A1 (lead) composites; A2 paints the world; A3 did the sound.

## What you own (write only here)
- `films/the-last-page/paper/` with `__init__.py` exposing:

```python
def render(cut_id: str, f: int) -> dict:
    """f is the global frame. Returns:
       'back':  float32 (1080, 1920, 4) premultiplied RGBA or None, drawn behind the fox
       'front': same or None, drawn in front of the fox
       'glow':  float32 (1080, 1920, 3) or None, additive light (the trail of light in cut 8)
    """
```
- Cache and review stills in `films/the-last-page/build/paper/` and `build/review/a4-*/`.
- Append to `films/the-last-page/NOTES.md` with entries starting `A4 HH:MM IST:`.

## The shots
- Cut 6a (from `burst`): every book flies open (A2 animates the books) and thousands of pages tear free, folding into paper birds mid-air on sharp drawn frames: flat page, first crease, half fold, bird, on twos, with a smear on the fastest frames. Pages come out of the shelves A2 painted, so read A2's world plates for where the books are.
- Cut 6b: low angle straight up the circular stairwell. The flock spirals in a huge rising vortex toward the dome. Real perspective: near birds large and fast, far birds tiny, the spiral tightening upward. Thousands of birds, every one different.
- Cut 6c: birds silhouetted and rim lit in the gold light crashing down from the dome.
- Cut 7: a stream of birds pouring out of the dome into the dawn sky behind the fox (keep it secondary; the sunrise and the fox lead this cut).
- Cut 8 (the climax): high in the golden sky, the flock pulls together into one colossal bird made of a hundred thousand pages, with handwriting and old illustrations visible across its wings. It gives one enormous wingbeat with real weight (`wingbeat`), then tears across the sky in a sweeping arc (`arc_start` to `arc_end`, path in `ANCHORS["8"]["bird_arc"]`), leaving a curving trail of light, mist and loose paper. It must keep one consistent design and feel immense: the tiny fox on the balcony bottom left gives the scale. This is the widest, most awe-inspiring image in the film.
- Cut 9a: quiet. Pages spiral down through sunbeams onto the balcony; one lands at the fox's paws at `ANCHORS["9a"]["page_land_screen"]` at `page_land`, the only blank one. Cut 9b: a few pages still drifting far behind (A1 draws the blank page on the floor and the lettering).

## How paper should look
Hand-drawn with real paper character: creases as drawn lines, curled corners, two or three flat cel tones per surface (lit face, shadow face, a warm bounce), faded ink handwriting and small old illustrations drawn as brush marks (pseudo script made of strokes, never a font), off-white tones that vary page to page (cream, ivory, tea-stained, a few blue-grey or rose sheets). Thin pressure ink outlines that match the fox's line style. Birds flap on twos or threes with strong key poses, not a smooth sine. No identical birds, no white triangles, no particle-system look, no CG motion blur (use drawn speed lines and smears).

Speed: thousands of small shapes per frame need a batch rasterizer in numba (or C) that draws many anti-aliased polygons in one pass. Each cut's frames should render in a few seconds at most.

## Stages
**Stage 1 (gate):** the paper bird design (a sheet of 20 unique birds, the fold sequence from page to bird, a flap cycle) and the great bird (three stills: formed, mid wingbeat, mid arc with trail) composited over A2's plates for cuts 6b and 8 if they exist, or over a plain painted sky if not. Save to `build/review/a4-stage1/`, inspect hard, improve, then stop and report with paths.

**Stage 2:** `render()` working for every paper cut, contact sheets per cut. Report.

Begin by writing your start time in NOTES.md.
