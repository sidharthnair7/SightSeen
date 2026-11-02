"""The room's big things rebuilt the way a game level is: first a grey blockout of each thing's key parts, then the real
look laid over it. The fused surface of a thing nobody filmed all the way round comes out torn (take 3's wardrobe);
a blockout is clean and solid from every side, and the video's photos on it keep it looking like the real thing.

For each thing:
  1. Its measured box: the room's dots (room.ply, every frame's depth) inside the thing's SAM 3.1 box from
     room_objects.py, widened by REACH, in the walls' directions (2nd to 98th percentile), standing on the floor, and
     snapped to a wall it's within SNAP of (furniture stands against walls; the gap behind it is never filmed).
  2. Its key parts: Muse Spark (Meta Model API) looks at crops of the thing from the frames that see it most and
     answers as a game designer would: up to 8 boxes in the thing's own box (0 to 1 from left to right as you face its
     front, floor to top, front to back), with a one-line reason. Asked VOTES times; the answer with the most common
     part count wins, and its parts are used.
  3. The parts as boxes, each face a 1 cm grid, coloured from the frames that see it best (room_furniture.recolour,
     from the base-lit frames when base_lighting.py has made them), with the new parts in the scene so they hide what's
     behind them. Faces no frame saw keep the thing's median colour.
Then room-furniture.ply gets each thing's blockout in place of the old surface inside its box (built each time from
room-furniture-before-blockout.ply, kept the first time). room-blockout.json keeps the boxes, Muse Spark's answers and reasons.
    python worker/blockout.py <request folder> [thing ...]
"""

import base64
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from rebuild_objects import _owned_dots
from room_furniture import _read_ply, _write_ply, recolour
from room_objects import _cameras, _records
from splat_objects import _mask
from sam_check import API, load_env

THINGS = ["wardrobe", "bed", "desk"]
# Smaller things that sit on or by the big ones get a block of their own, measured from their own dots, so the big
# ones' photos never take them in (the desk was painted with the monitors and the keyboard)
ITEMS = ["computer monitor", "keyboard", "computer tower"]
# The SAM 3.1 names whose pixels may paint each thing (a bed's photo includes its blanket and pillows; the wardrobe's
# own doors come back as "door", and its suitcases sit on it)
PAINT_FROM = {"bed": ["bed", "blanket", "pillow", "plush toy", "body pillow"], "desk": ["desk", "desk mat"],
              "wardrobe": ["wardrobe", "door", "suitcase"]}
SPARK = "muse-spark-1.3"
VOTES = 3
CROPS = 4
FRONT_SHARE = 0.3  # a 2 cm slice needs this share of the busiest one to be the front face
HEAD_ON = 0.5  # cos 60 degrees: a frame paints a blockout face only if it sees it at least this head-on (no stretching)
REACH = 0.25  # metres: the measured box looks this far past the SAM box for the thing's dots
SNAP = 0.30  # metres: a side this close to a wall goes to the wall
REPLACE = 0.01  # metres around a part where the old surface gives way to it
GRID = 0.01  # metres between the grid points on a part's faces
FLIP = np.diag([1.0, -1.0, -1.0])

ASK = (
    "These {n} pictures show the same {name} in a room, cut from a phone video. Describe it the way a game designer "
    "blocks out a level: as 1 to 8 boxes for its key parts (a wardrobe: its body, maybe a top shelf; a desk: the top, "
    "its legs or side panels, a keyboard tray; a bed: the frame and the mattress). Use the {name}'s own bounding box: "
    "x from its left (0) to its right (1) as you face its front, y from the floor (0) to its top (1), z from its front "
    "(0) to its back (1). Answer with JSON only, no other text: {{\"parts\": [{{\"part\": \"short name\", "
    "\"x\": [from, to], \"y\": [from, to], \"z\": [from, to]}}], \"reason\": \"one sentence on what you see\"}}.")


def _turn(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _spark(pictures, prompt, key):
    content = [{"type": "input_text", "text": prompt}] + [
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", p)[1]).decode()}
        for p in pictures]
    body = {"model": SPARK, "input": [{"type": "message", "role": "user", "content": content}]}
    request = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read().decode())
    text = "".join(part.get("text", "") for item in reply.get("output", []) for part in item.get("content", []) or [])
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _own_pixels(out, names, images):
    """Frame name -> True where SAM 3.1 outlined any of `names` in that frame (only frames it was asked about), and
    the median colour of those pixels (the thing's own colour, for sides no frame saw)."""
    saved = json.loads((out / "sam-outlines.json").read_text())
    allowed, samples = {}, []
    for camera in _cameras(out / "splat-data"):
        asked = [name for name in names if f"{camera['name']}|{name}" in saved]
        if not asked:
            continue
        picture = cv2.imread(str(images / camera["name"]))
        height, width = picture.shape[:2]
        mask = np.zeros((height, width), bool)
        for name in asked:
            mask |= _mask(_records(saved[f"{camera['name']}|{name}"]), width, height)
        allowed[camera["name"]] = mask
        if mask.any():
            samples.append(picture[mask][:, ::-1] / 255.0)
    median = np.median(np.vstack(samples), axis=0) if samples else np.array([0.6, 0.6, 0.6])
    return allowed, median


def _box_grid(low, high, axes):
    """The six faces of a box (corners `low`, `high` in the walls' frame: across, up, along) as 1 cm grids of
    triangles in the upright room: vertices, outward normals, triangles."""
    vertices, normals, triangles = [], [], []
    for axis in range(3):
        for side, value in ((-1, low[axis]), (1, high[axis])):
            u_axis, v_axis = [k for k in range(3) if k != axis]
            us = np.linspace(low[u_axis], high[u_axis], max(2, int(round((high[u_axis] - low[u_axis]) / GRID)) + 1))
            vs = np.linspace(low[v_axis], high[v_axis], max(2, int(round((high[v_axis] - low[v_axis]) / GRID)) + 1))
            grid_u, grid_v = np.meshgrid(us, vs)
            local = np.zeros((grid_u.size, 3))
            local[:, axis], local[:, u_axis], local[:, v_axis] = value, grid_u.ravel(), grid_v.ravel()
            start = sum(len(v) for v in vertices)
            vertices.append(local @ axes)
            normals.append(np.repeat((side * axes[axis])[None], len(local), axis=0))
            rows, cols = grid_u.shape
            index = np.arange(rows * cols).reshape(rows, cols) + start
            a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
            face = np.stack([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)]).reshape(-1, 3)
            # wind every triangle so it faces out of the box (the checks read a triangle's back as torn)
            here = local @ axes
            first = here[face[0] - start]
            if np.cross(first[1] - first[0], first[2] - first[0]) @ (side * axes[axis]) < 0:
                face = face[:, [0, 2, 1]]
            triangles.append(face)
    return np.vstack(vertices), np.vstack(normals), np.vstack(triangles)


def blockout(out, things=THINGS):
    import open3d as o3d

    out = Path(out)
    started = time.perf_counter()
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    blocks = {thing["name"]: thing for thing in json.loads((out / "room-objects.json").read_text())["things"]}
    level = _turn(*frame["quaternion"])
    lift = np.array([0.0, frame["floorHeight"], 0.0])
    yaw = math.radians(shell["yawDegrees"])
    axes = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
    centre = np.array(shell["centre"])
    walls_low = centre @ axes.T - np.array(shell["size"]) / 2  # the room's box in the walls' frame
    walls_high = centre @ axes.T + np.array(shell["size"]) / 2
    dots = o3d.io.read_point_cloud(str(out / "room.ply"))
    local_dots = (np.asarray(dots.points) @ level.T - lift) @ axes.T  # upright, then the walls' frame
    dot_colours = np.asarray(dots.colors)
    cameras = _cameras(out / "splat-data")
    images = out / "frames-base-light" if (out / "frames-base-light").exists() else out / "splat-data" / "images"
    surface_three, _, surface_triangles = _read_ply(out / "room-surface.ply")

    previous = json.loads((out / "room-blockout.json").read_text()) if (out / "room-blockout.json").exists() else {}
    report, built = {}, []
    for name in things:
        if name not in blocks:
            report[name] = "no SAM 3.1 box for it"
            continue
        block = blocks[name]
        middle = np.array(block["centre"]) @ axes.T
        half = np.array(block["size"]) / 2
        near = np.all((local_dots > middle - half - REACH) & (local_dots < middle + half + REACH), axis=1)
        near &= local_dots[:, 1] > 0.03  # not the floor
        near &= np.all((local_dots > walls_low + 0.03) & (local_dots < walls_high - 0.03), axis=1)  # nor the walls
        mine = local_dots[near]
        low, high = np.percentile(mine, 2, axis=0), np.percentile(mine, 98, axis=0)
        low[1] = 0.0  # standing on the floor
        snapped = []
        for axis in (0, 2):
            if low[axis] - walls_low[axis] < SNAP:
                low[axis], _ = walls_low[axis], snapped.append(f"{'xyz'[axis]}-")
            if walls_high[axis] - high[axis] < SNAP:
                high[axis], _ = walls_high[axis], snapped.append(f"{'xyz'[axis]}+")
        # its front faces away from the wall it's closest to (into the room)
        gaps = {(0, -1): low[0] - walls_low[0], (0, 1): walls_high[0] - high[0],
                (2, -1): low[2] - walls_low[2], (2, 1): walls_high[2] - high[2]}
        # in a corner (touching two walls) its back is the wall along its long side: a wardrobe stands width-on
        back_axis, back_side = min(gaps, key=lambda g: (round(gaps[g] / 0.05), high[g[0]] - low[g[0]]))
        side_axis = 2 if back_axis == 0 else 0
        # the front is its biggest upright face nearest the room (a wardrobe's doors), found above the floor clutter
        # and open doors that stretch the dots' box: the outermost 2 cm slice holding FRONT_SHARE of the busiest one
        band = mine[(mine[:, 1] > 0.4 * high[1]) & (mine[:, 1] < 0.9 * high[1])
                    & (mine[:, side_axis] > low[side_axis]) & (mine[:, side_axis] < high[side_axis])]
        if len(band) > 100:
            counts, edges = np.histogram(band[:, back_axis], bins=np.arange(low[back_axis], high[back_axis] + 0.02, 0.02))
            busy = np.nonzero(counts >= FRONT_SHARE * counts.max())[0]
            if back_side > 0:
                low[back_axis] = edges[busy[0]]
            else:
                high[back_axis] = edges[busy[-1] + 1]

        # the frames that see it most, cropped round it, for Muse Spark
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]) @ axes
        corners_cv = (corners + lift) @ level @ FLIP.T
        views = []
        for camera in cameras:
            inside = corners_cv @ camera["turn"].T + camera["shift"]
            if (inside[:, 2] < 0.1).any():
                continue
            u = camera["lens"][0, 0] * inside[:, 0] / inside[:, 2] + camera["lens"][0, 2]
            v = camera["lens"][1, 1] * inside[:, 1] / inside[:, 2] + camera["lens"][1, 2]
            width, height = 2 * camera["lens"][0, 2], 2 * camera["lens"][1, 2]
            x0, x1, y0, y1 = max(u.min(), 0), min(u.max(), width), max(v.min(), 0), min(v.max(), height)
            if x1 - x0 > 20 and y1 - y0 > 20:
                views.append(((x1 - x0) * (y1 - y0), camera["name"], (int(x0), int(y0), int(x1), int(y1))))
        crops = []
        for _, picture_name, (x0, y0, x1, y1) in sorted(views, reverse=True)[:CROPS]:
            picture = cv2.imread(str(images / picture_name))
            crops.append(picture[y0:y1, x0:x1])
        answers = [a for a in previous.get(name, {}).get("votes", []) if a.get("parts")]  # asked before: reuse
        for _ in range(VOTES - len(answers)):
            try:
                answers.append(_spark(crops, ASK.format(n=len(crops), name=name), key))
            except Exception as error:  # noqa: BLE001 - a failed vote just doesn't count
                answers.append({"error": str(error)})
        counted = [a for a in answers if a.get("parts")]
        if not counted:
            parts = [{"part": "body", "x": [0, 1], "y": [0, 1], "z": [0, 1]}]
            reason = "Muse Spark gave no usable parts; one box"
        else:
            common = Counter(len(a["parts"]) for a in counted).most_common(1)[0][0]
            chosen = next(a for a in counted if len(a["parts"]) == common)
            parts, reason = chosen["parts"], chosen.get("reason", "")

        # parts from the thing's own frame (x as you face its front, y up, z front to back) to the walls' frame
        boxes = []
        for part in parts:
            frac = {k: sorted(float(np.clip(v, 0, 1)) for v in part[k]) for k in ("x", "y", "z")}
            part_low, part_high = low.copy(), high.copy()
            part_low[1], part_high[1] = low[1] + frac["y"][0] * (high[1] - low[1]), low[1] + frac["y"][1] * (high[1] - low[1])
            # depth: front (0) is away from the back wall
            span = high[back_axis] - low[back_axis]
            if back_side > 0:
                part_low[back_axis], part_high[back_axis] = low[back_axis] + frac["z"][0] * span, low[back_axis] + frac["z"][1] * span
            else:
                part_low[back_axis], part_high[back_axis] = high[back_axis] - frac["z"][1] * span, high[back_axis] - frac["z"][0] * span
            across = high[side_axis] - low[side_axis]
            part_low[side_axis] = low[side_axis] + frac["x"][0] * across
            part_high[side_axis] = low[side_axis] + frac["x"][1] * across
            if np.all(part_high - part_low > 0.005):
                boxes.append((part["part"], part_low, part_high))
        report[name] = {"measured": {"low": low.round(3).tolist(), "high": high.round(3).tolist(),
                                     "sizeCm": ((high - low) * 100).round(0).tolist(), "snappedToWalls": snapped},
                        "parts": [{"part": p, "low": lo.round(3).tolist(), "high": hi.round(3).tolist()} for p, lo, hi in boxes],
                        "reason": reason, "votes": answers}
        built.append((name, boxes, low, high))

    # the smaller things, one block per piece of each (two monitors are two blocks), from their own dots
    owned, _ = _owned_dots(out, np.asarray(dots.points) @ FLIP.T, ITEMS)
    for name in ITEMS:
        points = local_dots[owned[name]]
        if len(points) < 60:
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        groups = np.asarray(cloud.cluster_dbscan(eps=0.04, min_points=15))
        pieces = []
        for group in set(groups) - {-1}:
            piece = points[groups == group]
            if len(piece) < 60:
                continue
            low, high = np.percentile(piece, 3, axis=0), np.percentile(piece, 97, axis=0)
            high = np.maximum(high, low + 0.015)  # at least 1.5 cm thick
            pieces.append((name, low, high))
            built.append((name, [(name, low, high)], low, high))
        report[name] = {"pieces": [{"low": lo.round(3).tolist(), "high": hi.round(3).tolist(),
                                    "sizeCm": ((hi - lo) * 100).round(0).tolist()} for _, lo, hi in pieces]}

    if built:
        mesh_vertices, mesh_normals, mesh_triangles, owners, offset = [], [], [], [], 0
        for k, (name, boxes, low, high) in enumerate(built):
            for _, part_low, part_high in boxes:
                v, n, t = _box_grid(part_low, part_high, axes)
                mesh_vertices.append(v), mesh_normals.append(n), mesh_triangles.append(t + offset)
                owners.append(np.full(len(v), k))
                offset += len(v)
        vertices, normals = np.vstack(mesh_vertices), np.vstack(mesh_normals)
        triangles, owners = np.vstack(mesh_triangles), np.concatenate(owners)
        colours = np.zeros((len(vertices), 3))
        occluders = np.vstack([surface_three @ FLIP.T, ((vertices + lift) @ level) @ FLIP.T])
        all_triangles = np.vstack([surface_triangles, triangles + len(surface_three)])
        import room_furniture
        least, room_furniture.LEAST_FACING = room_furniture.LEAST_FACING, HEAD_ON
        try:
            for k, (name, _, _, _) in enumerate(built):  # each thing painted only from its own outlined pixels
                allowed, median = _own_pixels(out, PAINT_FROM.get(name, [name]), images)
                mine = owners == k
                colours[mine], _ = recolour(out, vertices[mine], normals[mine],
                                            np.repeat(median[None], mine.sum(), axis=0), occluders, all_triangles,
                                            level, lift, images, allowed=allowed,
                                            truth=(surface_three @ FLIP.T, surface_triangles))
        finally:
            room_furniture.LEAST_FACING = least

        furniture = out / "room-furniture.ply"
        base = out / "room-furniture-before-blockout.ply"  # every run starts from the furniture before any blockout
        if not base.exists():
            shutil.copyfile(furniture, base)
        old_vertices, old_colours, old_triangles = _read_ply(base)
        old_local = old_vertices @ axes.T
        inside = np.zeros(len(old_vertices), bool)
        for _, boxes, _, _ in built:  # only what's inside a part goes: things standing on a part stay
            for _, part_low, part_high in boxes:
                inside |= np.all((old_local > part_low - REPLACE) & (old_local < part_high + REPLACE), axis=1)
        keep = ~inside[old_triangles].any(axis=1)
        _write_ply(furniture,
                   np.vstack([old_vertices, vertices]).astype(np.float32),
                   (np.clip(np.vstack([old_colours, colours]), 0, 1) * 255 + 0.5).astype(np.uint8),
                   np.vstack([old_triangles[keep], triangles + len(old_vertices)]).astype(np.int32))
        report["_furniture"] = {"oldTrianglesReplaced": int((~keep).sum()), "megabytes": round(furniture.stat().st_size / 1e6, 1)}
    report["_seconds"] = round(time.perf_counter() - started, 1)
    (out / "room-blockout.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python worker/blockout.py <request folder> [thing ...]")
    result = blockout(sys.argv[1], sys.argv[2:] or THINGS)
    for name, item in result.items():
        if isinstance(item, dict) and "parts" in item:
            print(name, item["measured"]["sizeCm"], item["measured"]["snappedToWalls"], [p["part"] for p in item["parts"]], "|", item["reason"])
        else:
            print(name, item)
