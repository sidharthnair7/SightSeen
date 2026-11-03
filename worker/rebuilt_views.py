"""Four standard views of the Rebuilt look, for the owner to see how each loop round left the room: from the middle of
the room at eye height, looking at the front, right, back and left walls (the shell's own axes), drawn the way the
viewer's Rebuilt look draws it: the rebuilt things (room-things-all.ply) inside walls, floor and ceiling in their clean
materials (room-materials.json: each tile repeated at its real size, or a plain colour). Unlit, as the renderer is.
Writes checks/loop-views/<front|right|back|left>.jpg and a sheet of all four, checks/loop-views/sheet.jpg.
    python worker/rebuilt_views.py <request folder>
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from room_checks import FACES, PAD, WalkRoom

EYE = 1.15  # metres: take 4's camera height above its floor (room-frame.json "eyeHeight")
PIXELS_PER_METRE = 400


def _material_pictures(out, room):
    """Each face's picture as the viewer's material: its tile repeated across the face at the tile's real size."""
    materials = json.loads((out / "room-materials.json").read_text())["faces"]
    pictures = {}
    for face, (_, _, (col_axis, _), (row_axis, _)) in FACES.items():
        across, down = 2 * (room.half[col_axis] + PAD), 2 * (room.half[row_axis] + PAD)
        width, height = int(across * PIXELS_PER_METRE), int(down * PIXELS_PER_METRE)
        entry = materials[face]
        if entry.get("file") and entry.get("tileMetres"):
            tile = cv2.imread(str(out / entry["file"]))[:, :, ::-1].astype(np.float32) / 255
            side = max(1, int(entry["tileMetres"] * PIXELS_PER_METRE))
            tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_AREA)
            picture = np.tile(tile, (height // side + 1, width // side + 1, 1))[:height, :width]
        else:
            picture = np.ones((height, width, 3), np.float32) * (np.array(entry["colour"], np.float32) / 255)
        pictures[face] = np.ascontiguousarray(picture)
    return pictures


def views(out):
    out = Path(out)
    room = WalkRoom(out, furniture="room-things-all.ply")
    room.pictures = _material_pictures(out, room)
    room.seen = {face: None for face in FACES}
    folder = out / "checks" / "loop-views"
    folder.mkdir(parents=True, exist_ok=True)
    middle = np.array([room.centre[0], EYE, room.centre[2]])
    drawn = []
    for name, ahead in (("front", -room.axes[2]), ("right", room.axes[0]), ("back", room.axes[2]),
                        ("left", -room.axes[0])):
        ahead = ahead + np.array([0.0, -0.35, 0.0])  # a little down, as you'd look round a room
        colour, _ = room.view(middle, ahead, width=720, height=540, fov=80)
        picture = np.ascontiguousarray((np.clip(colour, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.putText(picture, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(folder / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
        drawn.append(picture)
    sheet = np.vstack([np.hstack(drawn[:2]), np.hstack(drawn[2:])])
    cv2.imwrite(str(folder / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"views": [p.name for p in sorted(folder.glob("*.jpg"))], "eye": EYE}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/rebuilt_views.py <request folder>")
    print(json.dumps(views(sys.argv[1]), indent=2))
