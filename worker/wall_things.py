"""Measures things built into a wall: every frame's SAM 3.1 outlines of them (cached in things/sam-answers.json)
projected onto one face of the shell, drawn over its photo, with each outline's extent printed in the face's own metres
(columns left to right as seen from the room, rows down from the top). How take 4's window, curtains, door, knob and
switch were measured (worker/stand_ins_take4.py). ON_FACE (env, default 0.8): the share of an outline that must land
on the face.
    python worker/wall_things.py <request folder> <face> <out.jpg> <phrase,phrase> [frame numbers, comma-separated]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from rebuild_things import FLIP, _mask, _room
from room_objects import _cameras, _records
from room_textures import FACES

out, face, target = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
phrases = sys.argv[4].split(",")
only = [int(f) for f in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
shell = json.loads((out / "room-shell.json").read_text())
answers = json.loads((out / "things" / "sam-answers.json").read_text())
level, lift = _room(out)
cameras = _cameras(out / "splat-data")

axis, side, (col_axis, col_sign), (row_axis, row_sign) = FACES[face]
yaw = math.radians(shell["yawDegrees"])
axes = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
half = np.array(shell["size"], float) / 2
centre = np.array(shell["centre"], float)
normal = side * axes[axis]  # outward, upright axes
plane_point = centre + normal * half[axis]

picture = cv2.imread(str(out / f"room-shell-{face}.jpg"))
across, down = 2 * (half[col_axis] + 0.01), 2 * (half[row_axis] + 0.01)
S = 300  # pixels a metre
canvas = cv2.resize(picture, (int(across * S), int(down * S)))
colours = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200)]
report = []
for k, camera in enumerate(cameras):
    if only and k not in only:
        continue
    h, w = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
    eye = level @ FLIP @ (-camera["turn"].T @ camera["shift"]) - lift  # upright
    for p, phrase in enumerate(phrases):
        text = answers.get(f"{camera['name']}|{phrase}")
        if not text:
            continue
        for record in _records(text):
            mask = _mask(record, h, w)
            if mask.sum() < 200:
                continue
            edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            rows, cols = np.nonzero(edge)
            inside = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols, float)], -1) @ np.linalg.inv(camera["lens"]).T
            directions = (inside @ camera["turn"]) @ FLIP.T @ level.T
            t = ((plane_point - eye) @ normal) / (directions @ normal)
            ok = t > 0
            hits = eye + directions[ok] * t[ok, None]
            local = (hits - centre) @ axes.T
            c = (local[:, col_axis] * col_sign + across / 2) * S
            r = (local[:, row_axis] * row_sign + down / 2) * S
            on = (c >= 0) & (c < across * S) & (r >= 0) & (r < down * S)
            if not len(on) or on.mean() < float(__import__("os").environ.get("ON_FACE", "0.8")):
                continue  # mostly off this face: it's on another wall
            c, r = c[on], r[on]
            for x, y in zip(c.astype(int), r.astype(int)):
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    canvas[y, x] = colours[p % 4]
            cut = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
            report.append((camera["name"][6:10], phrase, cut, round(float(c.min() / S), 2), round(float(c.max() / S), 2),
                           round(float(r.min() / S), 2), round(float(r.max() / S), 2)))
for gx in range(0, canvas.shape[1], S // 10):
    cv2.line(canvas, (gx, 0), (gx, 6 if gx % S else 14), (0, 0, 0), 1)
for gy in range(0, canvas.shape[0], S // 10):
    cv2.line(canvas, (0, gy), (6 if gy % S else 14, gy), (0, 0, 0), 1)
cv2.imwrite(target, canvas)
print("face", face, "across", round(across, 3), "down", round(down, 3), "(columns run", "+" if col_sign > 0 else "-",
      "box axis", col_axis, ")")
for row in report:
    print(*row)
