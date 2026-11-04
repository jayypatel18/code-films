# TINY ROCKET: production notes (Fable 5.1)

Start: Wed Nov 4 16:33 IST 2026

## Plan
Everything is written from scratch in `tr/`:
- `tr/raster.py`: numba rasterizer. Anti-aliased nonzero-winding polygon coverage, pressure ink strokes (polyline with per-vertex width, wobble, taper), textured gouache dabs, value noise, box downsample.
- `tr/geom.py`: Catmull-Rom splines, hand wobble, stroke outlines, ellipses.
- `tr/panda.py`: the red panda as a parametric 2D drawing (pose dict in, cel layer out). No sprites, no rotated parts: every drawing is rebuilt from shapes.
- `tr/bg.py`: background painter. Compose a reference image, then paint over it with oriented dabs, big to small, then detail strokes.
- `tr/post.py`: grade, bloom, vignette, grain, dither, downsample.
- `look_test.py`: checkpoint 1 (model sheet, shot 1, shot 2 stills).

Canvas: work at 2x (3840x2160), box-downsample to 1920x1080.

## Log
- 16:33 read brief, prompt, common rules and the reference sheets. Start coding the rasterizer.
- 16:45 rasterizer, geometry, cel layer, panda, background painter, rocket and scenes written. First head test: too round, bear-like; tufts read as stubble. Rewrote the outline as a polar curve with fur bumps, one shadow crescent over the whole head after the features, bigger eyes, shorter legs.
- 16:50 model sheet v1: appealing. Sleep pose had the blanket over the head (fixed order and shape). Cheek tufts moved up from the chin to the cheeks.
- 16:52 shot 2 v1: painted look works (sky, clouds, tree, grass blades). Valley and village nearly invisible, rocket stubby, trunk cut flat at the base, ridges scribbly, river invisible. Lowered the hill, deepened the valley, widened the river, 150 houses, slimmer taller rocket, trunk into the ground under the grass.
- 16:53 shot 1 v1: head works. Knot floated beside the bamboo, paws and chest one dark blob, flat paper slab top left, brow line cut the eyebrow spot. Restaged: bamboo passes under the knot, far paw behind it, near paw grips past the knot, string to the teeth; russet chest with dark belly; sharper near grass; brow only when strong.
- 17:00 shot 2 v2: valley, river and village lights now read; slim rocket with legs. Shot 1 v2 restaged; v3 adds a fat overhand knot, four loops, toe lines, fur tufts on the shoulder.

## Checkpoint 1 reached: about 17:05 IST (about 0.5 h so far)
Stills: `build/review/model-sheet.png`, `build/review/shot1-knot.png`, `build/review/shot2-hill.png`.
Working: the panda design (broad head, fluffy cheeks, big anime eyes, tear marks, white ear rims, one hard shadow crescent, pressure ink with a lifted gap on the lit edge); on-model from front to profile through the sphere feature mapping; expressions (squeezed shut, one eye open, wonder, determined, goggles); gouache sky, clouds with lit undersides, camphor tree, grass blades, village lights, river catching the dusk; close-up backlight rim and bokeh.
Not yet: shot 1 paws and knot still smaller and stiffer than the head deserves (toes barely show, near paw covers the knot); shoulder is a large flat area; shot 2 mid-hill big dabs read as leaf blobs; village houses are a flat row of tents; rocket silhouette is a can with a dome, needs a taper and a visible fin; look-back pose on the sheet is a plain blob; whiskers very faint; no boil or animation tests yet.
Next: animation framework (shot timing, twos and threes, redrawn in-betweens), cues.json first version, then shots 3 to 9.

## Director notes after checkpoint 1 (about 17:06) and what changed
1. Sticker outline: rim light moved inside the ink line (erode the cel alpha past the ink, thin band on the lit side, broken by noise). No more halo on panda or bamboo.
2. Plush body: new `fluffy()` outline (sparse irregular fur tufts), neck ruff, chunky forearms wide at the shoulder with dark paws and three toe strokes, cast shadow of the head on the chest, belly shadow shape under the body. Tufts first came out as a uniform frill; retuned to sparse varied tufts.
3. Shot 1 restaged: knot and paws large at the left, four fat loops and an overhand twist round the bamboo, far paw behind the pole, near paw gripping the working end, string taut to the teeth with a short end past the mouth.
4. Shot 2: rocket slimmer and taller (640 px at 2x, about three pandas), kettle wire handle and spout, large patched sailcloth fin, bigger painted stars, thicker seam and tape lines. Clouds repainted as masses of overlapping soft puffs lit from below. Village redone with tile roofs (two slopes, tile courses) and thatch roofs, lit and shaded walls, eave shadows, warm windows.
- 17:12 animation framework: `tr/film.py` (Shot/Film, camera crop zoom, finish at 1x, timing helpers, ffmpeg streaming, contact sheets), `tr/shots.py` (shots 1 and 2 acted on twos with a 3 drawing boil cycle), `render.py` CLI.
- 17:14 to 17:23 shots 3 to 9 and the title written and test rendered one by one (contact montages in build/review/*-montage.png):
  s3 night: lighter night palette, 1800 stars plus a milky way band rotating round a pole, shooting star, fireflies, panda asleep against the rocket; medium framing.
  s4: wake, yawn, stand, walk, crouch, leap into the window (head clipped to the glass, drawn over the rocket); s4b porthole close-up with goggles snapping down, swallow, nod.
  s5: match strike on the tin band with flame; fuse spark on threes across grass; eyes squeezed shut with a 1.1 s hold on one drawing (no boil), one eye opens.
  s6: ignition with painted smoke burst, sparks, flattened grass ring, shudder, leap with ghost smear; low angle past the canopy with a smoke trail and birds (first version flew off the top; fixed to fly sideways).
  s7: three cloud layers streaming past the climbing rocket; face in the window with cheeks pushed back, fear to wonder, gold light.
  s8: the world from very high: curved horizon, sea of clouds, peaks, rivers, sun; the rocket speck, then the trail ignites into the arc. First sun was a blown white blob (reduced), arc left the frame (lowered).
  s9: parachute descent and settle, panda at the window, hop out, look up at the fading star, paint a new star on the map, look at us. Title brushed in stroke by stroke (every letter is 1 to 3 strokes), hanko 星 pressed with stamp breakup, "Fable 5.1", fade to paper then black.
  Known gaps for the final pass: no tail curl yet in the last beat; the soot on the face is faint; the cockpit close-ups reuse the same framing three times; village at predawn and morning reads as clutter at this size; the 6b trail is still a column.
- 17:23 full rough render started in two halves (frames 0 to 780 and 780 to 1560) through heavy.py.

Time note: the log times above were corrected against `date` at 17:46; earlier entries after checkpoint 1 had been estimated too high. Measured: start 16:33, part A of the rough render took 1364 s.

## Checkpoint 2: full rough render, 17:52 IST (1 h 20 min since start)
- `out/tiny-rocket-rough.mp4`: 1560 frames, 65 s, 1920x1080, H.264 high, yuv420p, CRF 14, no audio. Rendered as two halves (23 min each on the two slots), concatenated.
- Contact sheets per shot: `build/sheets/sheet-*.png` (from the rendered video, `sheets.py`).
- `cues.json` updated to the measured timings.
Inspection of the rough:
- s2: the panda never turned to the star map (turn keyed at 7.3 s local in a 7 s shot). Fixed: walk back 0 to 1.6, admire, turn at 4.0, step to the map, look down at 5.4.
- s9: the parachute hung above the frame through the whole descent. Fixed: chute 400 px above the kettle handle, camera framed higher during the descent, drop starts nearer.
- s3: stars still faint in the sheet; strength raised.
- s8 reads as the climax (arc over the sea of clouds); the head of the star is pure white, will warm it in the final pass.
- Still weak: cockpit close-ups share one framing (4b, 5c, 7b); village clutter at predawn and morning; the 6b trail; the lettering is readable but not a true brush yet (strokes are even, a real brush has dry breaks and a loaded start).
- 17:58 final render started (two halves) with the shot 2 turn, the shot 9 descent framed in (descent starts 720 px up, canopy 200 px over the kettle handle), brighter night stars, warmer star head, tail curl in the last beat, stronger soot. `finalize.py` concatenates the halves to `out/tiny-rocket-master.mp4`, renders the three stills and cuts the X version from `cold_open` in cues.json.
- 18:31 final master assembled: `out/tiny-rocket-master.mp4` (1560 frames, 65 s). Title shot re-rendered with the lettering lowered so the camera crop keeps it in frame (the first final had the top of the title cut). Part B trimmed to 660 frames with one re-encode at CRF 12 before the spliced title. Stills: `out/still-closeup.png` (frame 104), `out/still-climax.png` (frame 1176), `out/still-title.png` (frame 1518). X cut: `out/tiny-rocket-x.mp4` (cold open 0.85 to 2.6 s, then the film), 66.75 s. Rough master deleted to save disk.

- 19:07 title shot re-rendered once more with the model name moved left of the rocket (it collided with the kettle handle). Master, stills and X cut reassembled.

## End: 19:07 IST. Total 2 h 34 min since 16:33 (about 1 h 25 min of that was render time).

## Render slot rule (films session, from 18:40)
Every heavy job from now on: `python3 tools/heavy.py --slots 1 --threads 6 -- <command>` so only slot 0 is ever used and slot 1 stays free for Film A. The title re-render running at 19:00 already holds slot 0 only.

## Second pass (films session notes, 19:10 IST Wed): deadline about 14:00 IST Thu
Order: 1 climax repaint, 2 close-up body (fur cel shapes, arm, ruff, paws, knot), 3 brush lettering and title layout, 4 rocket silhouette, 5 village, 6 own list (cockpit framings, smoke wisps, soot, shot 3 lift, keyed leap). Then the inspection loop over every shot, a clean single pass master and X cut, frame check of the X cut's first 2 s.
- 19:10 resumed. Starting the climax repaint.
- 19:10 to 19:34 second pass work: climax repainted (tr/world.py: layered ridges painted from a lit reference with oriented dabs, snow on crests, cloud seas lit on top and toward the sun, rivers as subtle glinting ribbons, star streak with sparkling head, warm fringe, tapering trail with sparks and light on the clouds); paws with four toes and claw ticks; chest fur with tufted edge and arm cast shadow; ruff narrowed with only its hanging edge inked; knot loops hug the pole; brush_stroke() lettering (loaded start, swell, dry bristle tails, a pool at the touch) with fixed i and l and spacing; title drawn after the camera crop so it composes at frame scale; rocket tapered toward the nose, swept fin, legs splayed, a patch; village clustered into hamlets along the river with ridge lines and tile courses, morning mist in the valley.
- 19:34 to 19:46 gap: account usage limit, no work (12 min, not counted in the hours).
- 19:46 resumed. Checking shot 2 with the swept fin and the hamlets.
- 19:50 item 6 done: 5c reframed tighter and off centre with a flickering fuse light, 7b wider showing the ring, 6b trail as drifting wisps, shot 3 lifted, soot bolder, the leap keyed as three held poses (launch stretch, tuck, reach). Whole-film sample sheet (every 30 frames) inspected: nothing off model; 6a smoke still a big mass (left, it reads as a painted burst).
- 19:58 clean single pass master render started on slot 0 (6 threads).
- 20:44 single pass master done (2727 s on one slot). finalize.py: three stills and the X cut from the master. X cut first 54 frames checked frame by frame: cold open frames 0 to 41 are the knot tug (head dips, string snaps taut with a smear, ears), frame 42 cuts to the film start. Per shot sheets rebuilt from the master in build/sheets/.
- Inspection of the final sheets: s1 fur silhouette, toes and knot read; s2 hamlets and tapered rocket read; s3 lifted; s4 leap keyed in three held poses; s8 climax is now the widest, deepest frame; s9 parachute in frame through the descent; title card composed with brush strokes.
- Still short of the bar, honestly: s5a (match strike) is a big dark paw against a tin band and would benefit from a face in frame; s6a ignition smoke is a single large mass; the village at the title crop is thin; the hanko breakup is uniform; the ruff edge still reads as a collar in some drawings.

## End of second pass: 20:46 IST Wed. Hours: 16:33 to 19:07 (2 h 34 min) plus 19:10 to 19:34 (24 min) plus 19:46 to 20:46 (1 h 00 min) = 3 h 58 min, of which about 2 h 15 min was render time. The 12 min usage limit gap is excluded.

## Sound (Fable 5.1, separate agent)
SOUND 20:43 IST: start. Read brief, PROMPT SOUND section, COMMON rules, cues.json, picture notes. Starting the synthesis code in films/tiny-rocket/sound/ from an empty folder (numpy and scipy only, no samples).
SOUND 20:58 IST: dsp.py (filters, FDN reverb, limiter, WAV writer), instruments.py (felt piano, music box, celesta, strings, horns, timpani, cymbal, glock, pad), sfx.py (about 45 effects), music.py (the motif D4 A4 B4 A4 F#4 E4 D4 at 72 bpm and the arrangement), score.py (timeline, buses, stillness gate, master, loudness loop) and README.md written and unit checked (spectra, speed, no NaN). First full quick render running on slot 0.
SOUND 21:18 IST: first full render measured -16.9 LUFS, LRA 22.8 (quiet scenes -34 to -38 short term, too faint). Added a per scene trim curve, rebalanced night pad against the music box, rerendered: -14.2 LUFS, LRA 16.2, TP -1.3. Final tweak (solo piano scene -3 dB, night -2 dB) queued on slot 0 behind Film A pass 4.
SOUND 21:30 IST: final render done. tiny-rocket.wav: I -14.0 LUFS, LRA 16.1 LU, TP -1.3 dBTP, 3,120,000 samples. tiny-rocket-coldopen.wav: I -19.1 LUFS, TP -2.0 dBTP, 84,000 samples. Per shot short term: knot -22, hill -19, night -19, cockpit -26, fuse -22, stillness gate about -65 dBFS, ignition -13, face -11, star -10, home -18, title -22. End 21:30 IST. Sound hours: 20:43 to 21:30 = 47 min, of which 18 min was waiting for slot 0 behind Film A (about 29 min of work and own renders).
- 22:19 to 22:22 IST: portability pass only. Hard-coded machine paths removed from render.py, sheets.py, look_test.py, finalize.py and tr/film.py (every path now comes from the file's own location); the wrapper command in the slot rule written without the absolute path. py_compile clean on every file; one still rendered through render.py on one slot and byte-identical to out/still-closeup.png. No picture changes, no re-render. 3 min.
