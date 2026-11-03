"""Take 4's measured stand-ins: writes things/stand-ins.json for worker/stand_ins.py (desk corner, bed, laptop,
pillow, wardrobe, window and curtains, door, light switch, heater). Every number here was measured from take 4's frames
on Nov 2 (SAM 3.1 outlines on the fused surface, or projected onto the shell's walls with worker/wall_things.py); the
comments say from where. Take 4 is ghosted (frames COLMAP couldn't place land 0.3-0.7 m off), so some spots are a
judgement between the frames' copies, said in place.
    python worker/stand_ins_take4.py <take 4's request folder> --far
    python worker/stand_ins.py <take 4's request folder>
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from rebuild_things import _mask
from room_objects import _records

out = Path(sys.argv[1])
FAR = "--far" in sys.argv  # the desk corner where frames 0-6 put it instead (the close-ups' less the offset)
OFFSET = np.array([0.30, 0.0, 0.19])  # far to close, metres
shift = -OFFSET if FAR else np.zeros(3)
shell = json.loads((out / "room-shell.json").read_text())
answers = json.loads((out / "things" / "sam-answers.json").read_text())

# The right wall (the shell's +x face): a point on it at floor level, along it, and out of it into the room
yaw = math.radians(shell["yawDegrees"])
x_axis = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
z_axis = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
up = np.array([0.0, 1.0, 0.0])
centre = np.array(shell["centre"], float)
wall = centre + x_axis * shell["size"][0] / 2
wall[1] = 0.0
wall_line = wall.copy()  # the right wall itself, for the record
wall = wall + shift  # where the desk's frame sits
along, outward = z_axis, -x_axis  # along x up = outward: a right-handed box frame


def at(s, d, y):
    return wall + along * s + outward * d + up * y


def wall_box(s0, s1, d0, d1, y0, y1, colour, paint=False, **extra):
    return {"centre": at((s0 + s1) / 2, (d0 + d1) / 2, (y0 + y1) / 2).round(4).tolist(),
            "axes": np.array([along, up, outward]).round(5).tolist(),
            "size": [round(s1 - s0, 4), round(y1 - y0, 4), round(d1 - d0, 4)], "colour": colour, "paint": paint, **extra}


# The desk's wood: frame 30's SAM 'desk' outline, without the red mat and the dark keyboard
picture = cv2.imread(str(out / "splat-data" / "images" / "frame_0030.png"))
h, w = picture.shape[:2]
mask = np.zeros((h, w), bool)
for record in _records(answers["frame_0030.png|desk"]):
    mask |= _mask(record, h, w)
pixels = picture[mask][:, ::-1].astype(int)
wood = pixels[(pixels.sum(1) > 420) & (np.abs(pixels[:, 0] - pixels[:, 2]) < 90)]
wood_colour = [int(c) for c in np.median(wood, axis=0)]

TOP, TRAY = 0.53, 0.42  # metres up: the fused points' two levels in the desk corner (0.525-0.575 and 0.425-0.475:
# the mat and mouse on the top, the keyboard on the tray), less the mat's and keyboard's own thickness
desk = [
    wall_box(-0.80, 0.40, 0.01, 0.52, TOP - 0.025, TOP, wood_colour, paint=True, carries=True),  # top
    wall_box(-0.50, 0.10, 0.30, 0.68, TRAY - 0.02, TRAY, wood_colour, paint=True, carries=True),  # keyboard tray
    wall_box(-0.80, -0.78, 0.01, 0.50, 0.0, TOP - 0.025, wood_colour),  # side panels
    wall_box(0.38, 0.40, 0.01, 0.50, 0.0, TOP - 0.025, wood_colour),
    wall_box(-0.78, 0.38, 0.01, 0.03, 0.30, TOP - 0.025, wood_colour),  # back rail
    wall_box(-0.50, -0.485, 0.30, 0.48, TRAY - 0.02, TOP - 0.025, wood_colour),  # tray brackets
    wall_box(0.085, 0.10, 0.30, 0.48, TRAY - 0.02, TOP - 0.025, wood_colour),
]

# The monitors. B (the right one, YouTube on it) from frame 30, where SAM's outline isn't cut: centre, its plane's
# normal, 3-97% extents widened to the whole. A (the left one, a white page) stands beside it, angled in as frames
# 0-6 show (their plane), with its right edge at B's left edge (frames 29 and 30 see them meet); the same size and
# height as B (frames 0-6 measure it 10% smaller, as they do everything on the desk)
PLASTIC, SCREEN = [30, 30, 33], [6, 6, 8]


def monitor(middle, facing, width=0.48, height=0.27):
    facing = np.array([facing[0], 0.0, facing[2]])
    facing /= np.linalg.norm(facing)
    across = np.array([facing[2], 0.0, -facing[0]])  # the viewer's right, facing the screen
    axes = np.array([across, up, facing]).round(5).tolist()
    middle = np.array(middle, float)

    def piece(offset, size, colour):
        return {"centre": (middle + np.asarray(offset)).round(4).tolist(), "axes": axes,
                "size": [round(s, 4) for s in size], "colour": colour}

    neck = middle[1] - TOP
    return [
        piece([0, 0, 0], [width, height, 0.02], PLASTIC),
        piece(facing * 0.0105, [width - 0.02, height - 0.02, 0.002], SCREEN),  # black screen (owner's rule 2)
        piece(-facing * 0.035 + up * (-neck / 2), [0.04, neck, 0.025], PLASTIC),  # neck, from the top up
        piece(-facing * 0.06 + up * (TOP + 0.006 - middle[1]), [0.22, 0.012, 0.16], PLASTIC),  # base
    ], across


b_middle = (np.array([0.941, 0.743, -1.100]) + shift).tolist()
b_boxes, b_across = monitor(b_middle, [-0.98, 0.0, 0.16])
b_left = np.array(b_middle) - b_across * 0.24
a_facing = np.array([-0.914, 0.0, 0.406]) / np.linalg.norm([-0.914, 0.0, 0.406])
a_across = np.array([a_facing[2], 0.0, -a_facing[0]])
a_middle = b_left - b_across * 0.015 - a_across * 0.24
a_boxes, _ = monitor(a_middle.tolist(), a_facing.tolist())

# ---- Things built into the walls: measured by projecting every frame's SAM 3.1 outlines onto the shell's wall
# (scratchpad wall_things.py); columns are metres along the face's photo, left to right as seen from the room ----
from room_textures import FACES

PAD = 0.01
half = np.array(shell["size"], float) / 2
box_axes = np.array([x_axis, up, z_axis])


def face_frame(face):
    axis, side, (col_axis, col_sign), _ = FACES[face]
    right, inward = col_sign * box_axes[col_axis], -side * box_axes[axis]
    across = 2 * (half[col_axis] + PAD)

    def point(c, y, off):
        local = np.zeros(3)
        local[axis] = side * half[axis]
        local[col_axis] = col_sign * (c - across / 2)
        p = centre + local @ box_axes + inward * off
        p[1] = y
        return p

    return point, right, inward


def face_box(frame, c0, c1, y0, y1, off0, off1, colour, **extra):
    point, right, inward = frame
    return {"centre": point((c0 + c1) / 2, (y0 + y1) / 2, (off0 + off1) / 2).round(4).tolist(),
            "axes": np.array([right, up, inward]).round(5).tolist(),
            "size": [round(c1 - c0, 4), round(y1 - y0, 4), round(off1 - off0, 4)], "colour": colour, **extra}


def median_colour(frames, phrase):
    found = []
    for f in frames:
        picture = cv2.imread(str(out / "splat-data" / "images" / f"frame_{f:04d}.png"))
        hh, ww = picture.shape[:2]
        for record in _records(answers.get(f"frame_{f:04d}.png|{phrase}") or ""):
            found.append(picture[_mask(record, hh, ww)][:, ::-1])
    return [int(c) for c in np.median(np.vstack(found), axis=0)]


WHITE = [232, 230, 224]
front = face_frame("front")
inward_front = front[2].round(5).tolist()
curtain_colour = median_colour(range(0, 9), "curtain")
# The window: the glass shows between the curtains (columns 1.77-2.01) down to its sill (0.69 m up, frames 7-8 head
# on); the curtains hang to 0.66 m, the left one from 1.03 (frames 7-8 see its edge) to 1.83, the right one 2.00-2.31.
# No frame looked up far enough to see their tops: the window is assumed 1.0 x 1.05 m under the curtains' middle, the
# rod 1.86 m up (the owner's rule 1)
CW, WW, SILL, WTOP, ROD, HEM, RAIL = 1.67, 1.0, 0.70, 1.75, 1.86, 0.66, 0.08
window = [
    face_box(front, CW - WW / 2, CW + WW / 2, SILL, WTOP, 0.0, 0.01, [22, 26, 36], paint=True,
             paintFacing=inward_front),
    face_box(front, CW - WW / 2 - 0.05, CW - WW / 2, SILL, WTOP + 0.05, 0.0, 0.025, WHITE),
    face_box(front, CW + WW / 2, CW + WW / 2 + 0.05, SILL, WTOP + 0.05, 0.0, 0.025, WHITE),
    face_box(front, CW - WW / 2 - 0.05, CW + WW / 2 + 0.05, WTOP, WTOP + 0.05, 0.0, 0.025, WHITE),
    face_box(front, CW - WW / 2 - 0.08, CW + WW / 2 + 0.08, SILL - 0.025, SILL, 0.0, 0.07, WHITE),  # sill
    face_box(front, 0.98, 2.36, ROD - 0.01, ROD + 0.01, RAIL - 0.01, RAIL + 0.01, [205, 205, 208]),  # rod
]
curtains = [
    {"kind": "curtain", "left": front[0](c0, 0, RAIL).round(4).tolist(), "right": front[0](c1, 0, RAIL).round(4).tolist(),
     "top": ROD - 0.01, "bottom": HEM, "out": inward_front, "folds": 0.11, "depth": 0.05, "colour": curtain_colour,
     "paint": True, "paintFacing": inward_front, "paintMin": 0.3}
    for c0, c1 in ((1.03, 1.81), (1.99, 2.31))
]

back = face_frame("back")
inward_back = back[2].round(5).tolist()
door_colour = median_colour((14, 15), "door")
# The door: only filmed close up (frames 13-15) around its knob (0.965 m up, column 1.85) and the light switch just
# past its casing (column 2.0, 0.92 m up). The knob is on the latch side, so the door runs from it away from the
# switch: a standard door, 0.70 x 1.80 m in the room's units, its latch edge 6 cm past the knob (rule 1)
door = [
    # unseen parts a light door white: the frames saw it in the hall light's shadow (median door_colour)
    face_box(back, 1.21, 1.91, 0.0, 1.80, -0.0175, 0.0175, [206, 204, 198], paint=True, paintFacing=inward_back),
    face_box(back, 1.145, 1.21, 0.0, 1.865, 0.0, 0.02, WHITE),  # casing
    face_box(back, 1.91, 1.965, 0.0, 1.865, 0.0, 0.02, WHITE),
    face_box(back, 1.145, 1.965, 1.80, 1.865, 0.0, 0.02, WHITE),
    face_box(back, 1.835, 1.865, 0.95, 0.98, 0.0175, 0.04, [150, 150, 154]),  # knob: its neck
    face_box(back, 1.825, 1.875, 0.94, 0.99, 0.04, 0.065, [172, 172, 176]),  # and its round end, squared
]
switch = [
    face_box(back, 1.969, 2.031, 0.87, 0.97, 0.0, 0.006, [226, 216, 190]),  # a standard single plate, in the room's units
    face_box(back, 1.994, 2.006, 0.908, 0.932, 0.006, 0.016, [232, 224, 198]),  # toggle
]

# ---- The bed, in the shell's own axes (x across: the left wall at -half; z along: the window wall at -half) ----
def local_box(x0, x1, y0, y1, z0, z1, colour, **extra):
    middle = centre + np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2]) @ box_axes
    middle[1] = (y0 + y1) / 2
    return {"centre": middle.round(4).tolist(), "axes": box_axes.round(5).tolist(),
            "size": [round(x1 - x0, 4), round(y1 - y0, 4), round(z1 - z0, 4)], "colour": colour, **extra}


# The fused bedding (0.30-0.70 m up) runs from the left wall to x = 0.0 and z -1.15 to 0.45 (two copies of it, one
# per group of frames, but the chair beside its room-side edge agrees in frames 0-6 and 19-25): the bed fills that,
# against the wall, its mattress top at 0.46 (the bedding's level), on a wooden frame with legs
WOOD_BED = [146, 108, 74]  # the frame's honey wood (frames 19-25)
BX0, BX1, BZ0, BZ1, TOP_BED = -half[0] + 0.01, 0.0, -1.15, 0.45, 0.46
bedding_colour = median_colour(range(16, 26), "bed")
bed = [
    local_box(BX0 + 0.01, BX1 - 0.02, 0.20, TOP_BED, BZ0 + 0.01, BZ1 - 0.01, bedding_colour, paint=True,
              carries=True, fill="match"),  # mattress: the close-ups see a 0.75 m strip; the rest in its colour
    local_box(BX1 - 0.03, BX1, 0.12, 0.30, BZ0, BZ1, WOOD_BED),  # rails: room side, wall side, foot, head
    local_box(BX0, BX0 + 0.03, 0.12, 0.30, BZ0, BZ1, WOOD_BED),
    local_box(BX0, BX1, 0.12, 0.30, BZ0, BZ0 + 0.03, WOOD_BED),
    local_box(BX0, BX1, 0.12, 0.62, BZ1 - 0.03, BZ1, WOOD_BED),  # the head end a little higher
] + [local_box(x, x + 0.05, 0.0, 0.12, z, z + 0.05, WOOD_BED)
     for x in (BX0, BX1 - 0.05) for z in (BZ0, BZ1 - 0.05)]  # legs
pillow = [local_box(BX0 + 0.06, BX0 + 0.66, TOP_BED, TOP_BED + 0.12, BZ1 - 0.36, BZ1 - 0.05, [214, 205, 186],
                    paint=True)]

# The laptop on the bed: TRELLIS made a 21 cm cube of it; a closed laptop is a thin silver slab (30 x 2 x 21 cm in
# the room's units), where it was placed and turned, on the mattress
placed = {t["slug"]: t for t in json.loads((out / "things" / "placed-things.json").read_text())["things"]}
laptop_yaw = math.radians(placed["laptop"]["yawDegrees"])
laptop_axes = np.array([[math.cos(laptop_yaw), 0.0, math.sin(laptop_yaw)], up,
                        [-math.sin(laptop_yaw), 0.0, math.cos(laptop_yaw)]])
laptop = [{"centre": [placed["laptop"]["centre"][0], TOP_BED + 0.01, placed["laptop"]["centre"][2]],
           "axes": laptop_axes.round(5).tolist(), "size": [0.30, 0.02, 0.21], "colour": [178, 180, 186]}]

# The wardrobe: frames 32-36 put its doors at the back wall's own plane (z 1.25-1.35, the shell's wall at 1.236)
# and its body, the luggage on it and the bag beside it behind the wall (its mirror door and the close-ups'
# ghosting). It stands inside the room instead, against the back wall between the door's casing (x 0.31) and the
# right wall, 45 cm deep, its top at 1.65 where the luggage sits; its front painted from frames 33-36 where they see
# it (0.51 m further back). The things on it, beside it and in it move with it
WARDROBE_X, WARDROBE_FRONT, WARDROBE_TOP = (0.32, half[0] - 0.01), half[2] - 0.45, 1.65
SEEN_FRONT = 1.30  # where frames 33-36 put its doors
white = [234, 233, 229]
wardrobe = [
    local_box(WARDROBE_X[0], WARDROBE_X[1], 0.0, WARDROBE_TOP, WARDROBE_FRONT, half[2] - 0.01, white, paint=True,
              paintFacing=(-z_axis).round(5).tolist(), carries=True),
]
wardrobe_shift = ((SEEN_FRONT - WARDROBE_FRONT) * z_axis).round(4).tolist()


def onto(slug, z_local):
    """A move along the room's length that puts a placed thing's middle at z_local (shell axes)."""
    now = (np.array(placed[slug]["centre"], float) - centre) @ z_axis
    return ((z_local - now) * z_axis).round(4).tolist()


wardrobe_moves = {
    "suitcase-2": onto("suitcase-2", WARDROBE_FRONT + 0.22),  # on top
    "suitcase-3": onto("suitcase-3", WARDROBE_FRONT + 0.22),
    "shopping-bag": onto("shopping-bag", half[2] - 0.18),  # on the floor beside it, toward the door
    "trash-can": onto("trash-can", WARDROBE_FRONT - 0.15),  # on the floor in front of it
    "cardboard-box": onto("cardboard-box", WARDROBE_FRONT - 0.15),
}

# The baseboard heater under the window: SAM 3.1 outlines it in frames 18 and 21-25 (left of the suitcase, local x
# 0.26, 25 cm of it showing past the suitcase, 20 cm tall), but those steep downward views land it 25 cm under the
# floor and 14 cm behind the wall: a standard white baseboard heater at the window wall's base, at that spot, running
# on behind the suitcase
heater = [
    local_box(0.0, 0.55, 0.05, 0.23, -half[2] + 0.01, -half[2] + 0.08, [228, 226, 220]),
    local_box(0.01, 0.54, 0.23, 0.24, -half[2] + 0.02, -half[2] + 0.075, [168, 166, 160]),  # its grille
]

spec = {
    "why": ("Take 4's desk corner was filmed from across the room (frames 0-6) and close up (frames 29-31), and the "
            "two land about 30 cm apart (keyboard, mouse and mat each sit twice in the fused room). The close-up one "
            "is used: it is against the wall, and both monitors stand on it. The far-measured things on the desk are "
            "moved by that offset onto it."),
    "replaces": ["desk", "desk-mat", "computer-monitor", "computer-monitor-2", "computer-tower", "bed", "laptop",
                 "wardrobe", "puffer-jacket", "black-jacket"],  # the jackets hang inside it (its painted front shows them)
    "moves": {**wardrobe_moves, **({
        "keyboard": [-0.028, -0.0044, -0.0315],  # onto the tray, under frames 0-6's keyboard
        "computer-mouse": [-0.30, 0.0124, -0.19],  # measured close up: by the offset, its bottom onto the top
    } if FAR else {
        "keyboard": [0.272, -0.0044, 0.1585],  # to frames 29-31's keyboard (0.68, -1.32), on the tray
        "headphones": [0.30, 0.0, 0.19],  # the far-to-close offset (keyboard, mouse, mat: 0.27-0.33, 0.10-0.29)
        "computer-mouse": [0.0, 0.0124, 0.0],  # its bottom onto the top
    })},
    "parts": [
        {"slug": "desk", "name": "desk", "boxes": desk, "why": "measured stand-in: top, keyboard tray, sides",
         # placed where frames 0-6 put it, it's painted from them (COLMAP placed 0-8: the mat and keyboard land where
         # those frames see them; the close-ups' copy, shifted by OFFSET, landed beside them: round 6). Close up, the
         # close-ups paint their own copy
         "paintFrom": [f"frame_{f:04d}.png" for f in range(9)] if FAR
         else ["frame_0029.png", "frame_0030.png", "frame_0031.png"]},
        # The PC tower under the desk: TRELLIS built a dark blob from a far, dim view; a plain black tower of a
        # mid tower's size (45 x 20 x 45 cm, in the room's units) where it was measured, moved like the rest
        {"slug": "computer-tower", "name": "computer tower", "why": "measured stand-in: a black tower",
         "boxes": [wall_box(-0.145, 0.045, 0.07, 0.47, 0.0, 0.39, [24, 24, 27])]},
        # painted only from the close-ups COLMAP placed (17-25): they agree with each other, while the corner's frames
        # (9-12, 16, placed by MapAnything's cameras) land their copy of the duvet beside theirs
        {"slug": "bed", "name": "bed", "boxes": bed, "why": "measured stand-in: mattress on a wooden frame",
         "paintFrom": [f"frame_{f:04d}.png" for f in range(17, 26)]},
        {"slug": "laptop", "name": "laptop", "screen": True, "boxes": laptop, "why": "stand-in: a closed laptop"},
        {"slug": "wardrobe", "name": "wardrobe", "boxes": wardrobe, "why": "measured stand-in: a white wardrobe",
         "paintFrom": [f"frame_{f:04d}.png" for f in (33, 34, 35, 36)], "paintShift": wardrobe_shift},
        {"slug": "pillow", "name": "pillow", "boxes": pillow, "why": "measured stand-in: a pillow",
         "paintFrom": [f"frame_{f:04d}.png" for f in (9, 10, 11, 12)]},
        {"slug": "window", "name": "window", "fixed": True, "boxes": window,
         "why": "measured stand-in: glass, trim, sill and rod", "paintFrom": [f"frame_{f:04d}.png" for f in range(9)]},
        {"slug": "curtains", "name": "curtains", "fixed": True, "boxes": curtains,
         "why": "measured stand-in: two folded curtains", "paintFrom": [f"frame_{f:04d}.png" for f in range(9)]},
        {"slug": "door", "name": "door", "fixed": True, "boxes": door, "why": "measured stand-in: door, casing, knob",
         "paintFrom": ["frame_0013.png", "frame_0014.png", "frame_0015.png"]},
        {"slug": "heater", "name": "baseboard heater", "fixed": True, "boxes": heater,
         "why": "measured stand-in: a baseboard heater"},
        {"slug": "light-switch", "name": "light switch", "fixed": True, "boxes": switch,
         "why": "measured stand-in: a switch plate"},
        {"slug": "computer-monitor", "name": "computer monitor", "screen": True, "boxes": b_boxes,
         "why": "measured stand-in: panel with a black screen, on a stand"},
        {"slug": "computer-monitor-2", "name": "computer monitor", "screen": True, "boxes": a_boxes,
         "why": "measured stand-in: panel with a black screen, on a stand"},
    ],
    "measured": {"bedding": bedding_colour, "curtain": curtain_colour, "door": door_colour, "rightWall": wall_line.round(4).tolist(), "along": along.round(4).tolist(), "wood": wood_colour,
                 "monitorA": a_middle.round(4).tolist(), "monitorB": b_middle},
}
(out / "things" / "stand-ins.json").write_text(json.dumps(spec, indent=2))
print(json.dumps(spec["measured"]))
