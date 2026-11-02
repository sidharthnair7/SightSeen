"""Rebuilds the room's things whose fused surface lost pieces the video did see. On take 3, Test 1 found SAM 3.1's
outlines landing on no 3D piece for 10% of the suitcase, 17% of the chair, 31% of the ceiling fan and all of the
slipper: thin parts wash out of the 2 cm fused surface, and room_furniture.py's cleanup cuts everything within 3 cm of
the floor (the slipper) and 5 cm of the walls.

The dots (room.ply, every frame's depth at 1 cm) still hold those parts. For each thing:
  1. SAM 3.1's outlines (sam-outlines.json) pick out its own dots: a dot belongs to it when it's inside the outline in
     at least SHARE of the frames that see it. "Sees" uses the dots' own nearest depth at each pixel (not the fused
     surface, which has the holes), so the wall behind a lost piece can't pass for it.
  2. Open3D's Poisson surface reconstruction (open source, MIT) wraps one closed surface through those dots, which
     closes the holes; the thinnest-supported part of that surface is cut and it's kept inside the dots' box.
  3. Each vertex is coloured from the frames that see it best (room_furniture.recolour), from the base-lit frames
     when base_lighting.py has made them.
Parts no frame saw (the back) are Poisson's smooth guess: room-rebuilt.json says so per thing.

Then room-furniture.ply gets each rebuilt thing in place of the old surface around it (the old file is kept once as
room-furniture-fused.ply).
    python worker/rebuild_objects.py <request folder>
"""

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from room_furniture import _read_ply, _write_ply, recolour
from room_objects import _cameras, _records
from splat_objects import _mask

THINGS = ["suitcase", "chair", "ceiling fan", "slipper"]  # SAM 3.1's name for each, as in sam-outlines.json
VISIBLE = 0.04  # metres: a dot within this of the nearest dot at its pixel is visible there
NEAREST_PX = 5  # the dots' own depth map is filled over this many pixels square (dots are sparser than pixels)
SHARE = 0.5
POISSON_DEPTH = 8  # octree depth over the thing's own box: about 2 mm cells for a 50 cm suitcase
THIN = 0.08  # the least-supported 8% of Poisson's surface is cut (it balloons where there are no dots)
PAD = 0.02  # metres around the dots' box that the surface may keep
REPLACE = 0.025  # metres: old furniture surface this close to a rebuilt thing's dots gives way to it
FLIP = np.diag([1.0, -1.0, -1.0])


def _turn(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _owned_dots(out, dots_cv, things=THINGS):
    """For each thing, which dots are its own, by SAM 3.1's outlines over the frames."""
    saved = json.loads((out / "sam-outlines.json").read_text())
    votes = np.zeros((len(dots_cv), len(things)), np.int32)
    seen = np.zeros((len(dots_cv), len(things)), np.int32)
    found_in = np.zeros(len(things), np.int32)
    for camera in _cameras(out / "splat-data"):
        asked = [k for k, thing in enumerate(things) if f"{camera['name']}|{thing}" in saved]
        if not asked:
            continue
        height, width = cv2.imread(str(out / "splat-data" / "images" / camera["name"])).shape[:2]
        inside = dots_cv @ camera["turn"].T + camera["shift"]
        ahead = inside[:, 2] > 0.05
        depth = np.where(ahead, inside[:, 2], 1.0)
        u = camera["lens"][0, 0] * inside[:, 0] / depth + camera["lens"][0, 2]
        v = camera["lens"][1, 1] * inside[:, 1] / depth + camera["lens"][1, 2]
        on = np.nonzero(ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height))[0]
        ui, vi = u[on].astype(int), v[on].astype(int)
        distance = np.linalg.norm(inside[on], axis=1).astype(np.float32)
        nearest = np.full((height, width), np.inf, np.float32)
        np.minimum.at(nearest, (vi, ui), distance)
        nearest = cv2.erode(np.where(np.isfinite(nearest), nearest, 1e3).astype(np.float32),
                            np.ones((NEAREST_PX, NEAREST_PX), np.uint8))
        visible = on[distance <= nearest[vi, ui] + VISIBLE]
        visible_ui, visible_vi = u[visible].astype(int), v[visible].astype(int)
        for k in asked:
            records = _records(saved[f"{camera['name']}|{things[k]}"])
            mask = _mask(records, width, height)
            if mask.any():
                found_in[k] += 1
            seen[visible, k] += 1
            votes[visible[mask[visible_vi, visible_ui]], k] += 1
    share = votes / np.maximum(seen, 1)
    least = np.minimum(2, np.maximum(found_in, 1))  # things found in only one frame (the slipper) need one vote
    return {thing: (share[:, k] >= SHARE) & (votes[:, k] >= least[k]) for k, thing in enumerate(things)}, found_in


def _closed_surface(points, colours, towards):
    """One closed surface through a thing's dots (Open3D Poisson), trimmed to where the dots support it."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colours)
    # only the biggest group: outlines bleed a little onto the wall or floor next to the thing
    groups = np.asarray(cloud.cluster_dbscan(eps=0.05, min_points=10))
    if (groups >= 0).any():
        cloud = cloud.select_by_index(np.nonzero(groups == np.bincount(groups[groups >= 0]).argmax())[0])
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    cloud.orient_normals_towards_camera_location(towards)  # the side the cameras saw faces out
    mesh, support = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=POISSON_DEPTH,
                                                                              scale=1.1, linear_fit=True)
    support = np.asarray(support)
    mesh.remove_vertices_by_mask(support < np.quantile(support, THIN))
    box = cloud.get_axis_aligned_bounding_box()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(box.min_bound - PAD, box.max_bound + PAD))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, np.asarray(cloud.points)


def rebuild(out):
    import open3d as o3d

    out = Path(out)
    started = time.perf_counter()
    frame = json.loads((out / "room-frame.json").read_text())
    level = _turn(*frame["quaternion"])
    lift = np.array([0.0, frame["floorHeight"], 0.0])
    dots = o3d.io.read_point_cloud(str(out / "room.ply"))
    dots_three, dot_colours = np.asarray(dots.points), np.asarray(dots.colors)
    owned, found_in = _owned_dots(out, dots_three @ FLIP.T)
    cameras = _cameras(out / "splat-data")
    towards = np.mean([FLIP @ (-camera["turn"].T @ camera["shift"]) for camera in cameras], axis=0)

    surface_three, _, surface_triangles = _read_ply(out / "room-surface.ply")
    images = out / "frames-base-light" if (out / "frames-base-light").exists() else None
    rebuilt, report = [], {}
    for k, thing in enumerate(THINGS):
        mine = owned[thing]
        if mine.sum() < 50:
            report[thing] = {"dots": int(mine.sum()), "foundInFrames": int(found_in[k]), "rebuilt": False}
            continue
        mesh, used = _closed_surface(dots_three[mine], dot_colours[mine], towards)
        vertices = np.asarray(mesh.vertices) @ level.T - lift  # upright, like room-furniture.ply
        normals = np.asarray(mesh.vertex_normals) @ level.T
        # colour from the frames, with this thing in the scene so it can hide what's behind it
        occluders = np.vstack([surface_three, np.asarray(mesh.vertices)]) @ FLIP.T
        triangles = np.vstack([surface_triangles, np.asarray(mesh.triangles) + len(surface_three)])
        colours, _ = recolour(out, vertices, normals, np.asarray(mesh.vertex_colors), occluders, triangles,
                              level, lift, images)
        rebuilt.append((vertices, colours, np.asarray(mesh.triangles), used @ level.T - lift))
        report[thing] = {"dots": int(mine.sum()), "foundInFrames": int(found_in[k]), "rebuilt": True,
                         "triangles": int(len(mesh.triangles)),
                         "note": "the sides no frame saw are Poisson's smooth guess, not measured"}

    furniture = out / "room-furniture.ply"
    if rebuilt:
        original = out / "room-furniture-fused.ply"
        if not original.exists():
            shutil.copyfile(furniture, original)
        old_vertices, old_colours, old_triangles = _read_ply(original)
        near = np.zeros(len(old_vertices), bool)
        for _, _, _, used in rebuilt:
            gap = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(old_vertices))
                             .compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(used))))
            near |= gap < REPLACE
        keep = ~near[old_triangles].any(axis=1)
        parts_vertices, parts_colours, parts_triangles, offset = [old_vertices], [old_colours], [old_triangles[keep]], len(old_vertices)
        for vertices, colours, triangles, _ in rebuilt:
            parts_vertices.append(vertices)
            parts_colours.append(colours)
            parts_triangles.append(triangles + offset)
            offset += len(vertices)
        _write_ply(furniture, np.vstack(parts_vertices).astype(np.float32),
                   (np.clip(np.vstack(parts_colours), 0, 1) * 255 + 0.5).astype(np.uint8),
                   np.vstack(parts_triangles).astype(np.int32))
        report["_furniture"] = {"oldTrianglesReplaced": int((~keep).sum()), "megabytes": round(furniture.stat().st_size / 1e6, 1)}
    report["_seconds"] = round(time.perf_counter() - started, 1)
    (out / "room-rebuilt.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/rebuild_objects.py <request folder>")
    print(json.dumps(rebuild(sys.argv[1]), indent=2))
