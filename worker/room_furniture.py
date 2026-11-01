"""The room's furniture looking like the real things, for the Walk look: everything inside the shell that isn't the
walls, floor or ceiling, cut out of the fused surface, with each spot coloured from the one or two frames that saw it
best instead of the blurry average the fusion keeps.

The fused surface (room-surface.ply) has the real shapes, but its colours are an average of every frame that voted
into each 2 cm cube, so a desk's edge or a bag's print comes out as mush. Here the surface is moved upright, cut to
what stands inside the shell (the wall agent draws the walls, floor and ceiling), split finer than 2 cm, and every
vertex looks itself up in the frames: a frame counts if the vertex lands inside it, faces the camera, isn't hidden
behind something nearer (a depth map of the whole surface per frame) and isn't on an edge where colour bleeds across.
The best few frames (most head-on, closest, sharpest, COLMAP's own cameras first) are blended, each frame's exposure
matched to the fused colours first so there are no seams where one frame hands over to the next. Spots no frame sees
keep the fused colour.

Needs the request folder's splat-data/ (frames and cameras), room-surface.ply, room-frame.json, room-shell.json and,
for naming the pieces, room-objects.json. Writes room-furniture.ply (upright frame, metres, floor at y = 0, three.js
axes; coloured triangles) and room-furniture.json.
    python worker/room_furniture.py <request folder>
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from room_objects import FLIP, _cameras, _turn

WALL_GAP = 0.05  # metres: the surface this close to a wall plane is the wall's, not furniture
FLOOR_GAP = 0.03
CEILING_GAP = 0.06
WALL_SHEET = 0.15  # metres: within this of a shell face, a surface parallel to it in its colour is still that face
FLOOR_SHEET = 0.08
CEILING_SHEET = 0.12
SHEET_DEGREES = 30  # "parallel": the normal within this of the face's
LIKE_THE_SHELL = 0.04  # chromaticity (r, g over r + g + b) this close to the face's median colour is its paint
BRIGHTER = 1.7  # and its brightness within this factor of the median, either way
SMALLEST_PIECE = 200  # triangles: loose bits smaller than this are fusion noise, not things
MAX_TRIANGLES = 1_000_000  # split the surface as fine as this allows (about 20 MB), at most twice
MAX_SPLITS = 2
EDGE_PX = 4  # a vertex this close to a frame's border is left to frames that see it whole
HIDDEN = 0.03  # metres: a vertex further than this behind the frame's depth map is hidden in that frame
NEAR_EDGE_PX = 3  # look this many pixels around a vertex for an edge where colour bleeds across
BLEED_BEHIND = 0.15  # metres: something this much further away next to the vertex means it's on a silhouette
LEAST_FACING = 0.26  # cos 75 degrees: frames that see the surface more edge-on than this don't colour it
KEEP = 3  # blend the best 3 frames per vertex
SHARPER = 4  # weights go as (score / best score) ** SHARPER: a clearly better frame wins, near-ties blend
ADDED_BACK = 0.4  # frames COLMAP couldn't place (lined up by MapAnything instead) are rougher: their score x this
GAIN_LIMITS = (0.6, 1.6)


def _read_ply(path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)


def _write_ply(path, points, colours, triangles):
    """Binary PLY, float32 positions and uchar colours, triangles: what three.js's PLYLoader reads."""
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    faces = np.empty(len(triangles), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    faces["n"] = 3
    faces["a"], faces["b"], faces["c"] = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
        f.write(faces.tobytes())


def _shell_axes(shell):
    yaw = math.radians(shell["yawDegrees"])
    return np.array([math.cos(yaw), 0.0, math.sin(yaw)]), np.array([-math.sin(yaw), 0.0, math.cos(yaw)])


def _alike(colours, face):
    """Colours that are the face's paint: the same hue and saturation (light falls unevenly on a wall, so brightness
    may differ up to BRIGHTER either way)."""
    total = colours.sum(axis=1, keepdims=True)
    hue = colours / np.maximum(total, 1.0)
    same_hue = np.linalg.norm(hue - face / max(face.sum(), 1.0), axis=1) < LIKE_THE_SHELL
    ratio = total[:, 0] / max(face.sum(), 1.0)
    return same_hue & (ratio > 1 / BRIGHTER) & (ratio < BRIGHTER)


def cut_furniture(points, colours, triangles, shell):
    """The triangles inside the shell that aren't the shell itself, without loose bits. Upright points in."""
    import open3d as o3d

    across, along = _shell_axes(shell)
    middle = np.array(shell["centre"])
    width, height, length = shell["size"]
    rel = points - middle
    inside = ((np.abs(rel @ across) < width / 2 - WALL_GAP) & (np.abs(rel @ along) < length / 2 - WALL_GAP)
              & (points[:, 1] > FLOOR_GAP) & (points[:, 1] < height - CEILING_GAP))
    kept = triangles[inside[triangles].all(axis=1)]

    # Where the fused wall bulges or the plane sits a little off, sheets of wall survive the cut: drop the triangles
    # near a shell face that lie parallel to it and have its colour (a curtain or a poster there keeps its own colour)
    corners = points[kept]
    middles, colour = corners.mean(axis=1) - middle, colours[kept].mean(axis=1) * 255.0
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    flat = math.cos(math.radians(SHEET_DEGREES))
    like = {name: _alike(colour, np.array(rgb, float)) for name, rgb in shell["faces"].items()}
    sheet = np.zeros(len(kept), bool)
    walls_near = np.zeros(len(kept), int)
    painted = np.zeros(len(kept), bool)
    for axis, half, sides in ((across, width / 2, ("left", "right")), (along, length / 2, ("front", "back"))):
        offset, parallel = middles @ axis, np.abs(normals @ axis) > flat
        for sign, name in zip((-1, 1), sides):
            near = half - sign * offset < WALL_SHEET
            sheet |= near & parallel & like[name]
            walls_near += near
            painted |= near & like[name]
    sheet |= (walls_near >= 2) & painted  # the rounded crease the fusion leaves in a corner, whichever way it faces
    level_with = np.abs(normals[:, 1]) > flat
    sheet |= (middles[:, 1] + middle[1] < FLOOR_SHEET) & level_with & like["floor"]
    sheet |= (middles[:, 1] + middle[1] > height - CEILING_SHEET) & level_with & like["ceiling"]
    kept = kept[~sheet]

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(kept))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    mesh.remove_unreferenced_vertices()
    pieces, sizes, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(sizes)[np.asarray(pieces)] < SMALLEST_PIECE)
    mesh.remove_unreferenced_vertices()
    return mesh


def _placed_by_colmap(out, names):
    """The frames COLMAP itself placed (its model lists them); the rest were lined up by MapAnything and are rougher."""
    model = out / "colmap" / "model" / "0" / "images.bin"
    if not model.exists():
        return set(names)
    stems = {m.decode() for m in re.findall(rb"(frame_\d+)\.(?:png|jpg)", model.read_bytes())}
    placed = {name for name in names if Path(name).stem in stems}
    return placed or set(names)


def _sharpness(pictures):
    """How sharp each frame is next to its neighbours in time: its Laplacian's variance over the median of the 3 on
    each side. Neighbours film nearly the same things, so this measures blur, not how busy the view is."""
    import cv2

    raw = np.array([cv2.Laplacian(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for p in pictures])
    near = [np.median(raw[max(0, i - 3):i + 4]) for i in range(len(raw))]
    return np.clip(raw / np.maximum(near, 1e-6), 0.3, 1.3)


def _bilinear(picture, u, v):
    """Colours at pixel coordinates (pixel centres at i + 0.5), bilinear."""
    height, width = picture.shape[:2]
    x = np.clip(u - 0.5, 0, width - 1.001)
    y = np.clip(v - 0.5, 0, height - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((picture[y0, x0] * (1 - fx) + picture[y0, x0 + 1] * fx) * (1 - fy)
            + (picture[y0 + 1, x0] * (1 - fx) + picture[y0 + 1, x0 + 1] * fx) * fy)


def recolour(out, upright_points, normals, fused_colours, surface_cv, triangles_all, level, lift):
    """Each vertex's colour from the frames that see it best (see the module doc). Returns colours 0-1 and stats."""
    import cv2
    import open3d as o3d

    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    pictures = [cv2.imread(str(images / camera["name"]))[:, :, ::-1].astype(np.float32) / 255.0 for camera in cameras]
    sharp = _sharpness([(p * 255).astype(np.uint8) for p in pictures])
    placed = _placed_by_colmap(out, [camera["name"] for camera in cameras])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(surface_cv.astype(np.float32)), o3d.core.Tensor(triangles_all.astype(np.uint32)))

    # Upright to OpenCV room space: p_three = level^T (p + lift), p_cv = FLIP p_three
    to_cv = FLIP @ level.T
    points = (upright_points + lift) @ to_cv.T
    turned = normals @ to_cv.T
    count = len(points)
    best_score = np.zeros((count, KEEP), np.float32)
    best_colour = np.zeros((count, KEEP, 3), np.float32)
    best_frame = np.full((count, KEEP), -1, np.int32)
    gains = []
    for index, (camera, picture) in enumerate(zip(cameras, pictures)):
        height, width = picture.shape[:2]
        lens, turn, shift = camera["lens"], camera["turn"], camera["shift"]
        # The frame's depth map: one ray per pixel centre, directions with camera z = 1 so t_hit is the depth
        cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        in_camera = np.stack([(cols - lens[0, 2]) / lens[0, 0], (rows - lens[1, 2]) / lens[1, 1], np.ones_like(cols)], -1)
        centre = -turn.T @ shift
        directions = in_camera.reshape(-1, 3) @ turn  # turn^T applied to each row
        rays = np.hstack([np.repeat(centre[None], len(directions), 0), directions]).astype(np.float32)
        depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(height, width)
        depth = np.where(np.isfinite(depth), depth, 1e3).astype(np.float32)
        kernel = np.ones((2 * NEAR_EDGE_PX + 1, 2 * NEAR_EDGE_PX + 1), np.uint8)
        nearest, furthest = cv2.erode(depth, kernel), cv2.dilate(depth, kernel)

        seen = points @ turn.T + shift
        z = seen[:, 2]
        ahead = z > 0.05
        u = np.where(ahead, lens[0, 0] * seen[:, 0] / np.maximum(z, 1e-6) + lens[0, 2], -1)
        v = np.where(ahead, lens[1, 1] * seen[:, 1] / np.maximum(z, 1e-6) + lens[1, 2], -1)
        inside = ahead & (u >= EDGE_PX) & (u < width - EDGE_PX) & (v >= EDGE_PX) & (v < height - EDGE_PX)
        which = np.nonzero(inside)[0]
        px, py = u[which].astype(int), v[which].astype(int)
        visible = z[which] < depth[py, px] + HIDDEN
        which, px, py = which[visible], px[visible], py[visible]
        to_camera = centre - points[which]
        distance = np.linalg.norm(to_camera, axis=1)
        facing = np.abs(np.sum(turned[which] * to_camera, axis=1)) / np.maximum(distance, 1e-6)
        good = facing > LEAST_FACING
        which, px, py, distance, facing = which[good], px[good], py[good], distance[good], facing[good]
        if len(which) == 0:
            gains.append([1.0, 1.0, 1.0])
            continue
        # Next to an occluding edge (something nearer close by) or on a silhouette (the background close by), the
        # colour can come from the wrong side: such frames count for less there
        zs = z[which]
        edge = (nearest[py, px] < zs - HIDDEN) | (furthest[py, px] > zs + BLEED_BEHIND)
        colour = _bilinear(picture, u[which], v[which])

        # Match this frame's exposure to the fused colours (an average of all frames) on what it sees well
        steady = (facing > 0.5) & ~edge
        if steady.sum() > 500:
            gain = np.median(fused_colours[which[steady]] / np.maximum(colour[steady], 0.02), axis=0)
            gain = np.clip(gain, *GAIN_LIMITS)
        else:
            gain = np.ones(3)
        gains.append([round(float(g), 3) for g in gain])
        colour = np.clip(colour * gain, 0.0, 1.0)

        score = facing / np.maximum(distance, 0.3) * sharp[index] ** 2
        score = score * (1.0 if camera["name"] in placed else ADDED_BACK) * np.where(edge, 0.15, 1.0)

        # Keep the best KEEP frames per vertex
        all_score = np.concatenate([best_score[which], score[:, None].astype(np.float32)], axis=1)
        all_colour = np.concatenate([best_colour[which], colour[:, None, :].astype(np.float32)], axis=1)
        all_frame = np.concatenate([best_frame[which], np.full((len(which), 1), index, np.int32)], axis=1)
        order = np.argsort(-all_score, axis=1)[:, :KEEP]
        best_score[which] = np.take_along_axis(all_score, order, axis=1)
        best_colour[which] = np.take_along_axis(all_colour, order[:, :, None], axis=1)
        best_frame[which] = np.take_along_axis(all_frame, order, axis=1)

    found = best_score[:, 0] > 0
    weights = np.where(best_score > 0, (best_score / np.maximum(best_score[:, :1], 1e-9)) ** SHARPER, 0.0)
    blended = (best_colour * weights[:, :, None]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)[:, None]
    colours = np.where(found[:, None], blended, fused_colours)
    frames_used = np.bincount(best_frame[found, 0], minlength=len(cameras))
    stats = {
        "verticesFromFrames": int(found.sum()),
        "verticesKeptFused": int((~found).sum()),
        "framesPlacedByColmap": len(placed),
        "bestFrameCounts": {cameras[i]["name"]: int(n) for i, n in enumerate(frames_used) if n},
        "gains": {camera["name"]: gain for camera, gain in zip(cameras, gains)},
    }
    return colours, stats


def _name_pieces(out, points, triangles, across, along):
    """Which triangles are which thing, from room-objects.json's blocks (a block, 8 cm wider each way, that holds the
    triangle's middle; the smallest such block wins). Triangles come back sorted by thing, with each thing's run."""
    objects_file = out / "room-objects.json"
    things = json.loads(objects_file.read_text())["things"] if objects_file.exists() else []
    middles = points[triangles].mean(axis=1)
    label = np.full(len(triangles), len(things), np.int32)
    volume = np.full(len(triangles), np.inf)
    for number, thing in enumerate(things):
        rel = middles - np.array(thing["centre"])
        size = np.array(thing["size"]) / 2 + 0.08
        inside = (np.abs(rel @ across) < size[0]) & (np.abs(rel[:, 1]) < size[1]) & (np.abs(rel @ along) < size[2])
        smaller = inside & (np.prod(size) < volume)
        label[smaller], volume[smaller] = number, np.prod(size)
    order = np.argsort(label, kind="stable")
    pieces, start = [], 0
    for number in range(len(things) + 1):
        n = int((label == number).sum())
        if n:
            name = things[number]["name"] if number < len(things) else "other"
            pieces.append({"name": name, "firstTriangle": start, "triangles": n})
        start += n
    return triangles[order], pieces


def build_furniture(out):
    import open3d as o3d

    out = Path(out)
    started = time.perf_counter()
    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = _turn(*frame["quaternion"])
    lift = np.array([0.0, frame["floorHeight"], 0.0])
    surface, fused, triangles = _read_ply(out / "room-surface.ply")  # room space, three.js axes
    upright = surface @ level.T - lift

    mesh = cut_furniture(upright, fused, triangles, shell)
    cut_triangles = len(mesh.triangles)
    splits = 0
    while splits < MAX_SPLITS and cut_triangles * 4 ** (splits + 1) <= MAX_TRIANGLES:
        splits += 1
    if splits:
        mesh = mesh.subdivide_midpoint(number_of_iterations=splits)
    mesh.compute_vertex_normals()
    timings = {"cut": round(time.perf_counter() - started, 1)}

    points = np.asarray(mesh.vertices)
    colours, stats = recolour(out, points, np.asarray(mesh.vertex_normals), np.asarray(mesh.vertex_colors),
                              surface * np.array([1.0, -1.0, -1.0]), triangles, level, lift)
    timings["recolour"] = round(time.perf_counter() - started - timings["cut"], 1)

    across, along = _shell_axes(shell)
    faces, pieces = _name_pieces(out, points, np.asarray(mesh.triangles), across, along)
    _write_ply(out / "room-furniture.ply", points.astype(np.float32),
               (np.clip(colours, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), faces.astype(np.int32))
    timings["total"] = round(time.perf_counter() - started, 1)
    result = {
        "ply": "room-furniture.ply",
        "triangles": int(len(faces)),
        "vertices": int(len(points)),
        "trianglesBeforeSplitting": cut_triangles,
        "splits": splits,
        **{k: v for k, v in stats.items() if k != "gains"},
        "pieces": pieces,  # triangles are sorted by piece; names and runs from room-objects.json's blocks, rough
        "seconds": timings,
        "megabytes": round((out / "room-furniture.ply").stat().st_size / 1e6, 1),
        "frameGains": stats["gains"],
    }
    (out / "room-furniture.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_furniture.py <request folder>")
    summary = build_furniture(sys.argv[1])
    print(json.dumps({k: v for k, v in summary.items() if k not in ("frameGains", "bestFrameCounts")}, indent=2))
