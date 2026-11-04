# A2: background department for Film A, "THE LAST PAGE"

> This is a brief given to an agent while this film was made, kept so anyone can see exactly what was added on top of the prompt. Local paths were made relative to this repo, and a remark comparing our aims with the reference film was trimmed. Files under `reference/` were frames from @chetaslua's films, used as a quality reference, and are not in this repo.

You paint the world of Film A. Read first, in this order:
1. `films/the-last-page/briefs/COMMON.md` (rules and house style, binding).
2. `films/the-last-page/PROMPT.md` (the creative prompt).
3. `films/the-last-page/timeline.py` (cuts, plate sizes, cameras, anchors, lights, cues; the contract).
4. `reference/chetas-film1-sheet.png` (his best painted backgrounds) and `chetas-film2-last-page-sheet.png` (the same story).

The whole team is Opus 5.5 (you included). A1 (lead) draws the fox and composites; A3 does sound; A4 will do the paper effects later; the director session directs.

## What you own (write only here)
- `engine/paint*.py` and `engine/noise.py` (the stroke painter and noise; later films will reuse them, so keep them general).
- `films/the-last-page/world/` (one module per cut or per location, plus `__init__.py`).
- Cache: `films/the-last-page/build/plates/`. Review stills: `films/the-last-page/build/review/a2-*/`.
- Append to the shared log `films/the-last-page/NOTES.md` with entries starting `A2 HH:MM IST:`.

## The interface A1 consumes
`films/the-last-page/world/__init__.py` exposes:

```python
def render(cut_id: str, f: int) -> dict:
    """f is the global frame. Returns:
       'bg':   float32 (1080, 1920, 4) premultiplied RGBA, display space, required
       'fg':   same shape or None, drawn over the fox (foreground grass, near dust, a pillar edge)
       'glow': float32 (1080, 1920, 3) or None, additive light (shafts, bloom sources, sun glare)
    """
```

The stage layer of each cut follows `timeline.camera(cut_id, f)` exactly so the fox stays glued to the floor; other layers may parallax. Paint plates once at high resolution (plate size from `timeline.CUTS`, more if a layer needs parallax room), cache them as PNG in `build/plates/`, and make `render` fast (well under a second per frame): sample the cached plates through the camera and add the animated layers. Put the floors, the lectern, the ledge and the light shafts where `ANCHORS` says (you may ask the director session to move an anchor if the composition needs it).

## The locations and animated layers
- Cut 1 Before Dawn: the last hour of night, a vast blue-violet sea of clouds, a faint lighter band at the horizon, stars thinning. An ancient stone library tower on a cliff above the clouds, cracked copper dome gone green, wrapped in ivy and old windswept pines. One small warm light moves across the windows (the fox's lantern inside, timed by `window_light_start` and `window_light_end`). Pampas grass in the foreground sways on twos with overlap, plume by plume. Clouds drift almost imperceptibly. Fade from black is the lead's job.
- Cut 2 and 5d The Reading Hall: shelves climbing into darkness, thousands of unique books (leaning, sagging, sun-faded, moss along some spines), stone floor, one dusty shaft of moonlight from a high window with dust motes drifting only inside the light. Deep blue darkness above.
- Cut 3: a dark, out of focus painted library behind the close-up, warm lantern glow from below left.
- Cut 4a: the enormous open book on a carved lectern, pages empty and yellowed, in the hall. Cut 4b: the page itself filling the frame (aged paper with fibres, foxing, a faint ruled edge), lit warm on one side.
- Cuts 5a and 5b: low angles on high shelves. A1 needs no fox here. You animate one book trembling (5a), then a whole row (5b), on hard held key poses, with dust sifting down in thin streams. The lead adds screen shake in post.
- Cut 5c: the floor and shelves behind the medium shot of the fox; the lantern belongs to A1.
- Cut 6a: every book flies open at once (covers and page blocks snapping open on sharp drawn frames). The pages tearing free and becoming birds are A4's, so keep your books readable underneath.
- Cut 6b: dramatic low angle looking straight up a circular stairwell: stone spiral stair, shelves wrapping the walls, the dome far above. A slow rotation of the plate is fine here (no fox in this cut).
- Cut 6c: the dome from below: old copper shutters break open (hard key poses, debris) and the first gold light crashes down the shaft. Use `glow` for the light.
- Cut 7 Sunrise: the sun clears the sea of clouds (it rises between `sun_start` and `sun_end`), overwhelming gold and rose, the stone balcony ledge at lower left, grass on the ledge whipping in the wind (animated), and a great ring of wind rolling outward across the cloud tops from around `wind_ring` in `CUES`.
- Cut 8: the high golden sky above the clouds, the widest, deepest space in the film: layered cloud banks, sunbeams, mist. A tiny corner of the balcony bottom left. The great bird itself is A4's.
- Cuts 9a and 9b: the balcony in warm morning light, sunbeams, the sky and clouds behind, stone floor where the page lands (9b's page quad is in `ANCHORS`; A1 draws the page and text, you paint the floor around it).

Light per cut is in `timeline.LIGHTS` and `CUT_LIGHT`. The film travels from cool moonlit blue and warm lantern amber to overwhelming golden sunrise.

## Stages (return to the director session at each gate)
**Stage 1 (gate): the painter and two plates.** Build the stroke painter. Paint cut 1 (Before Dawn, exterior) and cut 2 (the Reading Hall) as finished plates. Render 3 stills of each through `render()` at full 1920x1080 into `build/review/a2-stage1/`, plus full resolution crops of the brushwork (a tower detail, a run of books). Inspect them against the house style and Chetas's film 1 sheet; keep improving until they could be frames from a theatrical film. Then stop and report with paths. Aim for this within about 2 hours.

**Stage 2: every other location and all animated layers**, `render()` working for every cut id in `timeline.CUT_IDS`. Contact sheets per cut in `build/review/a2-stage2/`. Report.

**Stage 3: polish** after the full film review, on notes from the director session.

Begin by writing your start time in NOTES.md.
