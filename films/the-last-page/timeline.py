"""THE LAST PAGE: the timeline every department reads.

Picture (world, actors, paper) and sound all import this file, so a beat that
moves here moves everywhere. 24 fps, 1920x1080, 1440 frames (60.0 s).

Conventions
- Frame ranges are [start, end) in global frame numbers. Frame f shows time f / 24.
- Every layer any module returns is a float32 array of shape (1080, 1920, 4):
  premultiplied RGBA in display (sRGB) space, values 0..1.
- Each cut has a painted "stage" plate (the layer the fox stands in). Plate
  coordinates are pixels in that plate. The camera maps plate to screen:
      screen = (plate - (cx, cy)) * zoom + (960, 540)
  camera(cut_id, f) gives (zoom, cx, cy) for any frame. World and actors both
  use it, so the fox stays glued to the painted floor during camera moves.
  Parallax layers may move at other rates; only the stage layer must follow.
- Screen shake (the tremor, the burst, the shutters) is applied by the lead in
  post to the whole composed frame, never inside a single layer.
- ANCHORS are positions in plate coordinates unless the key ends in _screen.

Ownership: the lead (A1) owns this file. Cut boundaries are frozen; change them
only through the films session. Anchors and cues may be refined by the lead,
who notes every change in NOTES.md.
"""

FPS = 24
W, H = 1920, 1080
FRAMES = 1440  # 60.0 s


def sec(s):
    """Seconds to the nearest frame."""
    return int(round(s * FPS))


# id, start s, end s, plate (w, h), camera at start (zoom, cx, cy), camera at end, what is on screen
CUTS = [
    ("1", 0.0, 7.0, (2304, 1296), (0.8333, 1152, 648), (0.8889, 1240, 640),
     "Before Dawn. Black, then fade up on a blue-violet sea of clouds at the last hour of night. "
     "Stone library tower on a cliff, cracked green copper dome, ivy, old pines. One warm light "
     "moves across a window. Pampas grass sways in the foreground. Slow push toward the tower."),
    ("2", 7.0, 14.0, (2304, 1296), (0.8333, 1152, 648), (0.8600, 1100, 660),
     "The Reading Hall, wide. Shelves climb into darkness. One dusty shaft of moonlight. The fox "
     "pads in from the left carrying a paper lantern in its mouth, stops, sets it down, looks up."),
    ("3", 14.0, 18.0, (2112, 1188), (0.9091, 1056, 594), (0.9500, 1030, 590),
     "The Fox, close-up. Ears swivel one at a time, a sniff, a blink with a held closed frame, "
     "lantern light in its eyes. The audience falls in love here."),
    ("4a", 18.0, 21.0, (2304, 1296), (0.8333, 1152, 648), (0.8500, 1170, 640),
     "The Blank Book. An enormous open book on a lectern, pages empty and yellowed. The fox hops up."),
    ("4b", 21.0, 25.0, (2112, 1188), (0.9091, 1056, 594), (0.9400, 1056, 580),
     "On the page. The paw presses down. Silence. A line of ink draws itself in one brushstroke "
     "into a tiny sketched bird, which shivers, peels off the page and flaps away."),
    ("5a", 25.0, 27.0, (2112, 1188), (0.9091, 1056, 594), (0.9091, 1056, 594),
     "The Library Stirs. Low angle on a high shelf: one book trembles."),
    ("5b", 27.0, 29.0, (2112, 1188), (0.9091, 1056, 594), (0.9091, 1056, 594),
     "A whole row trembles. Dust sifts down in thin streams. A deep creak travels up the tower."),
    ("5c", 29.0, 31.0, (2112, 1188), (0.9091, 1056, 594), (0.9091, 1056, 594),
     "The lantern flame leans sideways with no wind. The fox flattens its ears. Hard held poses."),
    ("5d", 31.0, 32.0, (2304, 1296), (0.8600, 1100, 660), (0.8600, 1100, 660),
     "One beat of total stillness: the hall, wide, nothing moves."),
    ("6a", 32.0, 34.5, (2304, 1296), (0.8333, 1152, 648), (0.8700, 1152, 600),
     "The Pages Burst. Every book flies open at once. Thousands of pages tear free and fold into "
     "paper birds mid-air on sharp drawn frames."),
    ("6b", 34.5, 37.5, (2112, 1188), (0.9091, 1056, 594), (1.0000, 1056, 560),
     "Dramatic low angle straight up the circular stairwell. The flock spirals in a huge rising "
     "vortex toward the dome."),
    ("6c", 37.5, 39.0, (2112, 1188), (0.9091, 1056, 594), (0.9600, 1056, 594),
     "The dome's old copper shutters break open and the first gold light crashes down the shaft."),
    ("7", 39.0, 44.0, (2304, 1296), (0.8333, 1152, 700), (0.8333, 1152, 610),
     "Sunrise. The sun clears the sea of clouds. The fox on a stone balcony ledge, tiny against the "
     "sky, fur and grass whipping in the wind, looks up with its mouth slightly open. A great ring "
     "of wind rolls outward across the clouds. Slow tilt up."),
    ("8", 44.0, 52.0, (2560, 1440), (0.7500, 1280, 720), (0.7800, 1300, 690),
     "The Great Bird (climax). The flock pulls together into one colossal bird of a hundred thousand "
     "pages, gives one enormous wingbeat, then tears across the sky in a sweeping arc with a trail "
     "of light, mist and loose paper. The widest, most awe-filled image. Tiny fox on the balcony "
     "bottom left for scale."),
    ("9a", 52.0, 55.0, (2112, 1188), (0.9091, 1056, 594), (0.9300, 1056, 600),
     "The Last Page. Quiet. Pages spiral down through sunbeams onto the balcony. One lands at the "
     "fox's paws, the only blank one. The fox looks at it, then at us."),
    ("9b", 55.0, 60.0, (2112, 1188), (0.9091, 1056, 594), (0.9300, 1056, 594),
     "The paw goes down. Ink blooms and writes Opus 5.5 in one brushed stroke, the red 寓話 hanko "
     "presses in, then one smaller brushed line: Every story begins with a blank page. Hold. The "
     "tail flicks once. Fade to paper, then black."),
]

CUT_IDS = [c[0] for c in CUTS]


def cut_range(cut_id):
    """(start frame, end frame) of a cut, end exclusive."""
    for c in CUTS:
        if c[0] == cut_id:
            return sec(c[1]), sec(c[2])
    raise KeyError(cut_id)


def cut_at(f):
    """Cut id showing at global frame f."""
    for c in CUTS:
        if sec(c[1]) <= f < sec(c[2]):
            return c[0]
    return CUTS[-1][0]


def plate_size(cut_id):
    for c in CUTS:
        if c[0] == cut_id:
            return c[3]
    raise KeyError(cut_id)


def _ease(u):
    # Gentle ease in and out, like a cranked camera stand.
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def camera(cut_id, f):
    """(zoom, cx, cy) for the stage plate of a cut at global frame f."""
    for c in CUTS:
        if c[0] == cut_id:
            f0, f1 = sec(c[1]), sec(c[2])
            u = _ease((f - f0) / max(1, (f1 - 1 - f0)))
            a, b = c[4], c[5]
            return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))
    raise KeyError(cut_id)


def plate_to_screen(cut_id, f, x, y):
    zoom, cx, cy = camera(cut_id, f)
    return (x - cx) * zoom + W / 2, (y - cy) * zoom + H / 2


def screen_to_plate(cut_id, f, sx, sy):
    zoom, cx, cy = camera(cut_id, f)
    return (sx - W / 2) / zoom + cx, (sy - H / 2) / zoom + cy


# Where things are, so the painted world and the drawn fox agree.
ANCHORS = {
    "1": {
        "tower_base": (1560, 760),       # where the tower meets the cliff top
        "tower_top": (1560, 250),        # top of the dome
        "lit_window_path": [(1515, 520), (1600, 520)],  # the warm light crosses this window row
        "cloud_sea_top_y": 700,
    },
    "2": {
        "floor_y": 1090,                 # fox paws touch the floor here
        "fox_enter_x": 10,               # starts behind A2's foreground column at the left edge
        "fox_stop_x": 790,               # where it stops (body centre), a gentle walk away
        "lantern_set": (955, 1090),      # where the lantern is put down (under the fox's muzzle, in front of it)
        "moon_shaft_top": (1700, 0),     # shaft from a high window, upper right
        "moon_shaft_floor": (1040, 1090),
        "moon_shaft_width_at_floor": 280,
        "fox_height": 265,               # standing fox, ears included, in plate px
    },
    "3": {
        "head_centre_screen": (640, 660),  # left third, looking right toward the lantern
        "head_height_screen": 440,       # crown to chin, ears excluded (ears inside the frame)
        # lantern below right, off frame: the fox faces screen right (as it walked in)
        # and the lantern it set down is in front of it, so warm light comes from
        # below right and the glint sits low in the eyes on that side
        "lantern_glow_from_screen": (1450, 1200),
    },
    "4a": {
        "floor_y": 1150,
        "lantern": (830, 1150),
        "fox_start": (720, 1150),
        "lectern_centre_x": 1330,
        "book_top_y": 690,               # surface of the open pages at the gutter
        "book_span_x": (960, 1700),
        "fox_land": (1040, 700),         # the fox lands on the near page edge
        "fox_height": 260,
    },
    "4b": {
        "paw_press_screen": (420, 840),  # front of the toes; the paw enters from the bottom left corner
        "ink_bird_centre_screen": (1060, 520),
        "ink_bird_width_screen": 280,
        "bird_exit_screen": (1900, -100),
    },
    "5c": {
        "fox_head_screen": (1180, 460),
        "lantern_screen": (430, 830),
    },
    "5d": {
        "fox": (790, 1090),              # where it stopped in cut 2
        "lantern": (955, 1090),
    },
    "7": {
        "ledge_top_y": 1040,             # top of the balcony ledge, spans x 260 to 1120
        "ledge_x": (260, 1120),
        "fox_paws": (700, 1040),
        "fox_height": 330,
        "sun_start": (1560, 860),        # the sun clears the cloud sea during the cut
        "sun_end": (1560, 720),
        "cloud_sea_top_y": 820,
    },
    "8": {
        "bird_form_centre": (1280, 520),
        "bird_arc": [(1280, 520), (1900, 380), (2300, 600), (1500, 260), (300, 180)],
        "balcony_corner": (0, 1240),     # tiny balcony bottom left
        "fox_paws": (330, 1330),
        "fox_height": 70,
    },
    "9a": {
        "fox_seat_screen": (640, 830),    # where the seated fox's paws rest
        "fox_height_screen": 560,
        "page_land_screen": (930, 905),
    },
    "9b": {
        "fox_seat_screen": (400, 880),
        "fox_height_screen": 680,
        # the blank page lying on the balcony floor, corners clockwise from top left
        "page_quad_screen": [(420, 620), (1420, 575), (1510, 1020), (390, 1050)],
    },
}

# Light per cut. key = main light colour, key_dir = screen direction TOWARD the
# light, ambient = shadow fill, rim = edge light or None, warm = lantern colour.
LIGHTS = {
    "night_out": dict(key=(0.62, 0.70, 0.92), key_dir=(-0.6, -0.8), ambient=(0.17, 0.17, 0.32),
                      rim=None, warm=(1.0, 0.72, 0.40)),
    "hall": dict(key=(0.62, 0.71, 0.90), key_dir=(0.55, -0.83), ambient=(0.13, 0.12, 0.25),
                 rim=(0.70, 0.78, 1.0), warm=(1.0, 0.72, 0.40)),
    "gold_burst": dict(key=(1.0, 0.82, 0.50), key_dir=(0.0, -1.0), ambient=(0.35, 0.25, 0.30),
                       rim=(1.0, 0.90, 0.65), warm=(1.0, 0.72, 0.40)),
    "sunrise": dict(key=(1.0, 0.78, 0.48), key_dir=(0.85, -0.2), ambient=(0.62, 0.52, 0.66),
                    rim=(1.0, 0.86, 0.55), warm=None),
    "morning": dict(key=(1.0, 0.86, 0.66), key_dir=(0.8, -0.5), ambient=(0.70, 0.62, 0.70),
                    rim=(1.0, 0.92, 0.75), warm=None),
}
CUT_LIGHT = {
    "1": "night_out", "2": "hall", "3": "hall", "4a": "hall", "4b": "hall",
    "5a": "hall", "5b": "hall", "5c": "hall", "5d": "hall",
    "6a": "hall", "6b": "hall", "6c": "gold_burst",
    "7": "sunrise", "8": "sunrise", "9a": "morning", "9b": "morning",
}

# Story beats in seconds. Picture and sound both read these. The lead refines
# them as the animation is timed and writes exact per-frame events (paw steps,
# blinks, page hits) to build/events.json for the sound pass.
CUES = {
    # 1 Before Dawn
    "wind_in": 0.0,
    "fade_up_start": 1.0,
    "fade_up_end": 2.6,
    "window_light_start": 3.0,
    "window_light_end": 6.4,
    # 2 Reading Hall
    "cut_hall": 7.0,
    "fox_enter": 7.05,           # first paw step; the felt piano motif starts with the fox
    "piano_in": 7.6,
    "fox_stop": 11.6,
    "lantern_down": 12.15,       # soft wooden tap on stone
    "look_up": 12.8,
    # 3 The Fox
    "cut_closeup": 14.0,
    "ear_left": 14.35,
    "ear_right": 14.95,
    "sniff_1": 15.5,
    "sniff_2": 15.75,
    "blink_close": 16.5,
    "blink_open": 16.85,
    "eye_glint": 17.0,
    # 4 The Blank Book
    "cut_lectern": 18.0,
    "hop_takeoff": 18.9,
    "hop_land": 19.4,
    "cut_page": 21.0,
    "paw_down": 21.3,            # then silence: the music drops out
    "ink_start": 22.0,
    "ink_end": 23.2,
    "bird_shiver": 23.3,
    "bird_peel": 23.8,
    "bird_flap_start": 24.0,
    # 5 The Library Stirs (low strings and creaking wood come in)
    "cut_shelf": 25.0,
    "tremble_one": 25.4,
    "cut_row": 27.0,
    "tremble_row": 27.1,
    "dust_fall": 27.4,
    "creak": 27.8,
    "cut_lantern": 29.0,
    "flame_lean": 29.2,
    "ears_flat": 29.9,
    "stillness_start": 31.0,
    "stillness_end": 32.0,
    # 6 The Pages Burst
    "burst": 32.0,               # sudden roar of fluttering paper
    "cut_stairwell": 34.5,
    "vortex_peak": 37.0,
    "cut_dome": 37.5,
    "shutters_break": 37.8,
    "gold_light": 38.0,          # orchestral swell begins
    # 7 Sunrise
    "cut_sunrise": 39.0,
    "fox_mouth_open": 41.0,
    "wind_ring": 41.5,
    # 8 The Great Bird
    "cut_great_bird": 44.0,
    "bird_formed": 46.3,
    "wingbeat": 46.5,            # the huge wingbeat; the piano motif returns at full scale
    "arc_start": 48.0,
    "arc_end": 52.0,
    # 9 The Last Page
    "cut_last_page": 52.0,       # quiet returns
    "page_land": 54.0,
    "fox_looks_page": 54.2,
    "fox_looks_us": 54.9,
    "cut_title": 55.0,
    "final_paw": 55.4,           # a single soft piano note as the ink touches the page
    "title_ink_start": 55.5,
    "title_ink_end": 56.9,
    "hanko": 57.2,
    "line_ink_start": 57.5,
    "line_ink_end": 58.5,
    "tail_flick": 58.7,
    "fade_paper_start": 59.0,
    "fade_black_start": 59.5,
    "end": 60.0,
}
