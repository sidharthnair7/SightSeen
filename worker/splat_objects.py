"""The room's splats split into its things (bed, desk, chair...), so each can be picked up and moved in the Walk look
and still look like the video.

SAM 3.1's outlines (cached in sam-outlines.json by room_objects.py and room_checks.py) say which pixels of a frame
are the bed. Every splat is projected into those frames; where nothing is in front of it (its distance from the
camera is within VISIBLE of the fused surface's there), it votes for the outline it lands in. A splat belongs to the
thing it was inside in at least SHARE of the frames that saw it, and at least TWICE; the rest stay with the room.

Writes room-splats-<thing>.ply per thing and room-splats-rest.ply (Brush's layout, the room's OpenCV space, like
room-splats.ply) and room-splat-objects.json (each thing's file, splat count, and its box in the upright room).
    python worker/splat_objects.py <request folder>
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from room_objects import _cameras, _records

THINGS = ["bed", "desk", "chair", "wardrobe", "computer monitor", "computer tower", "suitcase"]
VISIBLE = 0.04  # metres: a splat this close to (or in front of) the surface the camera sees there is visible
SHARE = 0.5
TWICE = 2
FLIP = np.diag([1.0, -1.0, -1.0])


def _turn(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mask(records, width, height):
    """All of one name's outlines in a frame as one frame-sized picture (each mask covers only its own box)."""
    from meta_sam_parser import decode_mask_to_raster

    picture = np.zeros((height, width), dtype=bool)
    for record in records:
        mask, bounds = record.mask, record.bounds
        raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(mask.height, mask.width)
        left, top = int(round(bounds.left)), int(round(bounds.top))
        right, bottom = int(round(bounds.right)), int(round(bounds.bottom))
        if right <= left or bottom <= top:
            continue
        box = cv2.resize(raster, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST) > 0
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
        picture[y0:y1, x0:x1] |= box[y0 - top:y1 - top, x0 - left:x1 - left]
    return picture


def split_splats(out):
    import open3d as o3d

    out = Path(out)
    started = time.perf_counter()
    data = (out / "room-splats.ply").read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    names = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    splats = np.frombuffer(data, dtype=np.float32, offset=end).reshape(-1, len(names))
    column = {name: k for k, name in enumerate(names)}
    centres = splats[:, [column["x"], column["y"], column["z"]]].astype(np.float64)  # OpenCV axes, room space

    saved = json.loads((out / "sam-outlines.json").read_text())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))))
    votes = np.zeros((len(splats), len(THINGS)), dtype=np.int32)
    seen = np.zeros((len(splats), len(THINGS)), dtype=np.int32)
    for camera in _cameras(out / "splat-data"):
        asked = [k for k, thing in enumerate(THINGS) if f"{camera['name']}|{thing}" in saved]
        if not asked:
            continue
        height, width = cv2.imread(str(out / "splat-data" / "images" / camera["name"])).shape[:2]
        # where each splat lands in this frame, and how far from the camera
        inside = centres @ camera["turn"].T + camera["shift"]
        ahead = inside[:, 2] > 0.05
        u = camera["lens"][0, 0] * inside[:, 0] / np.where(ahead, inside[:, 2], 1) + camera["lens"][0, 2]
        v = camera["lens"][1, 1] * inside[:, 1] / np.where(ahead, inside[:, 2], 1) + camera["lens"][1, 2]
        on = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        # the distance to the surface the camera sees at each pixel, from the fused surface (three.js axes)
        rows, cols = np.indices((height, width))
        rays_in_camera = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols)], axis=-1).reshape(-1, 3)
        rays_in_camera = rays_in_camera @ np.linalg.inv(camera["lens"]).T
        directions = (rays_in_camera @ camera["turn"]) @ FLIP.T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origin = FLIP @ (-camera["turn"].T @ camera["shift"])
        rays = np.hstack([np.repeat(origin[None], len(directions), axis=0), directions]).astype(np.float32)
        surface = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(height, width)
        ui, vi = u[on].astype(int), v[on].astype(int)
        visible = np.zeros(len(splats), dtype=bool)
        visible[np.nonzero(on)[0]] = np.linalg.norm(inside[on], axis=1) <= surface[vi, ui] + VISIBLE
        for k in asked:
            mask = _mask(_records(saved[f"{camera['name']}|{THINGS[k]}"]), width, height)
            seen[visible, k] += 1
            hit = np.zeros(len(splats), dtype=bool)
            hit[np.nonzero(on)[0]] = mask[vi, ui]
            votes[visible & hit, k] += 1

    share = votes / np.maximum(seen, 1)
    best = share.argmax(axis=1)
    owner = np.where((share[np.arange(len(splats)), best] >= SHARE) & (votes[np.arange(len(splats)), best] >= TWICE),
                     best, -1)

    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = _turn(*frame["quaternion"])
    upright = (centres @ FLIP.T) @ level.T - np.array([0.0, frame["floorHeight"], 0.0])
    yaw = math.radians(shell["yawDegrees"])
    across, along = np.array([math.cos(yaw), 0.0, math.sin(yaw)]), np.array([-math.sin(yaw), 0.0, math.cos(yaw)])

    def write(name, keep):
        file = f"room-splats-{name.replace(' ', '-')}.ply"
        text = header.replace(f"element vertex {len(splats)}\n", f"element vertex {int(keep.sum())}\n")
        (out / file).write_bytes(text.encode("ascii") + splats[keep].tobytes())
        return file

    things = []
    for k, name in enumerate(THINGS):
        keep = owner == k
        if keep.sum() < 200:
            continue
        part = upright[keep]
        x, z = part @ across, part @ along
        low = np.array([np.percentile(x, 2), np.percentile(part[:, 1], 2), np.percentile(z, 2)])
        high = np.array([np.percentile(x, 98), np.percentile(part[:, 1], 98), np.percentile(z, 98)])
        middle = (low + high) / 2
        things.append({"name": name, "file": write(name, keep), "splats": int(keep.sum()),
                       "centre": [round(float(c), 4) for c in across * middle[0] + along * middle[2] + [0, middle[1], 0]],
                       "size": [round(float(s), 3) for s in high - low], "yawDegrees": shell["yawDegrees"]})
    rest = ~np.isin(owner, [THINGS.index(thing["name"]) for thing in things])
    result = {"things": things, "rest": {"file": write("rest", rest), "splats": int(rest.sum())},
              "seconds": round(time.perf_counter() - started, 1)}
    (out / "room-splat-objects.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/splat_objects.py <request folder>")
    print(json.dumps(split_splats(sys.argv[1]), indent=2))
