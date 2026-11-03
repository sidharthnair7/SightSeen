"""Measured stand-ins for things a single photo can't build well (the owner's rule 1, Nov 1: "put a very basic/normal
thing that we assume" instead of a blur): a desk as its top, keyboard tray and side panels, a computer monitor as an
upright panel with a black screen (his rule 2) on a stand. Each is a few boxes, sized and placed by
things/stand-ins.json (measured from the frames; its "why" says how). The faces the video saw are painted from the
frames that saw them best (a desk top keeps its mat, a tray its keyboard); the rest get the part's plain colour with a
little light and shade baked in, so they read as solid next to the other rebuilt things, which are drawn unlit.

The spec can also move things that rest on a stand-in onto it ("moves", metres), and lists the built things the
stand-ins replace. Rewrites room-thing-*.ply for both, room-things.json and room-things-all.ply.
    python worker/stand_ins.py <request folder>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from rebuild_things import FLIP, _box, _rays, _room, _surface, _write_thing_ply
from room_furniture import _bilinear, _write_ply
from room_objects import _cameras

STEP = 0.006  # metres between painted vertices: about what the close-up frames show
LIGHT = np.array([0.35, 0.85, 0.4]) / np.linalg.norm([0.35, 0.85, 0.4])  # the baked shade's light, from above
FACING = 0.25  # a frame paints a face only when it sees it at least this head-on (cosine)
DEPTH = 0.03  # metres: and only where the frame's own surface is no nearer than this (nothing stands in front)
REACH = 0.35  # metres: a thing whose bottom is this far below a stand-in's top, over it, rests on it


def _cuboid(centre, axes, size, step=None):
    """A box as six flat faces (their own vertices, so each face shades flat), each a grid of quads `step` apart (one
    quad when None). Returns points, triangles (counter-clockwise from outside) and each vertex's outward normal."""
    centre, axes, half = np.asarray(centre, float), np.asarray(axes, float), np.asarray(size, float) / 2
    points, triangles, normals = [], [], []
    for axis in range(3):
        for side in (-1, 1):
            u_axis, v_axis = [(1, 2), (2, 0), (0, 1)][axis]
            if side < 0:
                u_axis, v_axis = v_axis, u_axis  # keeps the winding counter-clockwise seen from outside
            nu = max(1, int(np.ceil(2 * half[u_axis] / step))) if step else 1
            nv = max(1, int(np.ceil(2 * half[v_axis] / step))) if step else 1
            a, b = np.meshgrid(np.linspace(-1, 1, nu + 1), np.linspace(-1, 1, nv + 1), indexing="ij")
            local = np.zeros((a.size, 3))
            local[:, axis] = side * half[axis]
            local[:, u_axis] = a.ravel() * half[u_axis]
            local[:, v_axis] = b.ravel() * half[v_axis]
            start = sum(len(p) for p in points)
            points.append(centre + local @ axes)
            normals.append(np.repeat((side * axes[axis])[None], a.size, axis=0))
            index = np.arange(a.size).reshape(nu + 1, nv + 1) + start
            corner = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
            triangles.append(np.concatenate([np.stack([corner[0], corner[1], corner[2]], 1),
                                             np.stack([corner[0], corner[2], corner[3]], 1)]))
    return np.vstack(points), np.vstack(triangles), np.vstack(normals)


def _curtain(left, right, top, bottom, out, folds, depth, step=0.01):
    """A curtain hanging from `top` down to `bottom` (metres up) between `left` and `right` (on its rail, left to right
    as seen from the room), folded in and out of the wall (`out`, into the room) every `folds` metres, `depth` deep.
    Returns points, triangles (counter-clockwise from the room) and normals, like _cuboid."""
    left, right, out = np.asarray(left, float), np.asarray(right, float), np.asarray(out, float)
    across = right - left
    across[1] = 0.0
    width = np.linalg.norm(across)
    across /= width
    nu, nv = max(2, int(np.ceil(width / step))), max(2, int(np.ceil((top - bottom) / (2 * step))))
    a, height = np.meshgrid(np.linspace(0, width, nu + 1), np.linspace(bottom, top, nv + 1), indexing="ij")
    wave = 2 * np.pi * a.ravel() / folds
    points = left[None] + a.ravel()[:, None] * across + (depth / 2 * np.sin(wave))[:, None] * out
    points[:, 1] = height.ravel()
    normals = out[None] - (depth / 2 * 2 * np.pi / folds * np.cos(wave))[:, None] * across  # across the folds
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    index = np.arange(a.size).reshape(nu + 1, nv + 1)
    corner = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([corner[0], corner[1], corner[2]], 1),
                                np.stack([corner[0], corner[2], corner[3]], 1)])
    return points, triangles, normals


def _shape(box):
    """A spec entry's mesh: a box (painted ones finely divided), or a curtain."""
    if box.get("kind") == "curtain":
        return _curtain(box["left"], box["right"], box["top"], box["bottom"], box["out"], box["folds"], box["depth"])
    return _cuboid(box["centre"], box["axes"], box["size"], STEP if box.get("paint") else None)


def _paint(out, points, normals, frames, level, lift, scene):
    """Each vertex's colour (0-1) from the frame among `frames` that sees it most head-on and nearest, where nothing
    stands in front (`scene`: the fused room and the stand-ins themselves, so a tray under a desk top isn't painted
    with the top's mat); and which vertices were painted."""
    import cv2
    import open3d as o3d

    cameras = {camera["name"]: camera for camera in _cameras(out / "splat-data")}
    best = np.zeros(len(points))
    colours = np.zeros((len(points), 3))
    room_cv = ((points + lift) @ level) @ FLIP  # upright -> three.js -> OpenCV room
    normals_cv = (normals @ level) @ FLIP
    for name in frames:
        camera = cameras[name]
        photo = cv2.imread(str(out / "frames" / f"{Path(name).stem}.jpg"))
        if photo is None:
            continue
        height, width = camera["size"]
        x = room_cv @ camera["turn"].T + camera["shift"]
        ahead = x[:, 2] > 0.05
        z = np.where(ahead, x[:, 2], 1.0)
        u = camera["lens"][0, 0] * x[:, 0] / z + camera["lens"][0, 2]
        v = camera["lens"][1, 1] * x[:, 1] / z + camera["lens"][1, 2]
        inside = ahead & (u > 2) & (u < width - 2) & (v > 2) & (v < height - 2)
        eye = -camera["turn"].T @ camera["shift"]
        to_eye = eye - room_cv
        distance = np.linalg.norm(to_eye, axis=1)
        facing = np.sum(normals_cv * to_eye, axis=1) / np.maximum(distance, 1e-9)
        use = np.nonzero(inside & (facing > FACING))[0]
        if not len(use):
            continue
        # nothing nearer in front: the frame's own surface (its depth along the same pixel) agrees
        hit = scene.cast_rays(o3d.core.Tensor(_rays(camera, u[use], v[use])))["t_hit"].numpy()
        use = use[~(hit < z[use] - DEPTH)]
        score = facing[use] / distance[use]
        better = score > best[use]
        use, score = use[better], score[better]
        grow = photo.shape[0] / height
        colours[use] = _bilinear(photo[:, :, ::-1].astype(np.float32) / 255, u[use] * grow, v[use] * grow)
        best[use] = score
    return colours, best > 0


def _match_fill(colours, seen, normals, facing, grain=0.025, most=0.9):
    """A painted face's unseen part in its seen part's own colour (the median), with a faint grain, instead of the
    part's plain colour: a duvet the frames saw only a strip of reads as one duvet, not a strip on a grey slab.
    (Reflecting the seen strip across it mirrored the things lying on it, so they showed up three times.) Still marked
    unseen, so "Not filmed" shows it."""
    face = normals @ np.asarray(facing, float) > most  # (a folded curtain's faces lean further: paintMin)
    known, fill = face & seen, face & ~seen
    if known.sum() < 20 or not fill.any():
        return colours
    noise = np.random.default_rng(7).normal(0, grain, (int(fill.sum()), 1)) if grain else 0.0
    result = colours.copy()
    result[fill] = np.clip(np.median(colours[known], axis=0)[None] + noise, 0, 1)
    return result


def _carriers(spec):
    """The stand-ins' surfaces that things rest on (boxes marked "carries": a mattress, a desk top, a tray): each one's
    centre, axes, half size and top."""
    found = []
    for part in spec["parts"]:
        for box in part["boxes"]:
            if box.get("carries"):
                half = np.asarray(box["size"], float) / 2
                found.append((np.asarray(box["centre"], float), np.asarray(box["axes"], float), half,
                              box["centre"][1] + half[1]))
    return found


def _rest(middle, bottom, carriers):
    """How far up a thing must go to sit on the stand-in surface it rests on: its middle over the surface, its bottom
    at most REACH below the top (sunk into a mattress the stand-in made thicker) or just above it; the surface whose
    top is nearest its bottom (a keyboard on a tray under a desk top). 0 when it rests on none."""
    lifts = [top + 0.002 - bottom for centre, axes, half, top in carriers
             if np.all(np.abs(((middle - centre) @ axes.T)[[0, 2]]) <= half[[0, 2]]) and top - REACH <= bottom <= top + 0.03]
    return min(lifts, key=abs) if lifts else 0.0


def build(out):
    out = Path(out)
    spec = json.loads((out / "things" / "stand-ins.json").read_text())
    level, lift = _room(out)
    # From the things as place() and the loop left them (kept aside on the first run), so a second run doesn't move
    # anything twice
    aside = out / "things" / "placed-things.json"
    if not aside.exists():
        shutil.copy(out / "room-things.json", aside)
    summary = json.loads(aside.read_text())
    replaced = set(spec.get("replaces", []))
    kept = [thing for thing in summary["things"] if thing["slug"] not in replaced]

    # Things moved by the spec, and things that rest on a stand-in set on it
    carriers = _carriers(spec)
    for thing in kept:
        path, original = out / thing["file"], out / "things" / thing["slug"] / "placed.ply"
        data = (original if original.exists() else path).read_bytes()
        end = data.index(b"end_header\n") + len(b"end_header\n")
        header = data[:end].decode("ascii")
        names = [line.split()[2] for line in header.splitlines() if line.startswith("property") and "list" not in line]
        count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
        types = {"x": "<f4", "y": "<f4", "z": "<f4", "red": "u1", "green": "u1", "blue": "u1", "seen": "u1"}
        vertex = np.frombuffer(data, dtype=[(n, types[n]) for n in names], count=count, offset=end).copy()
        shift = np.array(spec.get("moves", {}).get(thing["slug"], (0.0, 0.0, 0.0)), float)
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(float) + shift
        shift[1] += _rest(points.mean(axis=0), points[:, 1].min(), carriers)
        if not shift.any():
            if original.exists():  # moved by an earlier run, not by this one: back as placed
                path.write_bytes(original.read_bytes())
            continue
        if not original.exists():
            shutil.copy(path, original)
        for axis, name in enumerate("xyz"):
            vertex[name] += shift[axis]
        path.write_bytes(data[:end] + vertex.tobytes() + data[end + vertex.nbytes:])
        thing["centre"] = [round(c + s, 4) for c, s in zip(thing["centre"], shift)]
        thing["movedBy"] = shift.round(4).tolist()
    # and the grey boxes of things that couldn't be built, the same way (none for things a stand-in replaces)
    summary["failed"] = [thing for thing in summary.get("failed", []) if thing["slug"] not in replaced]
    for thing in summary["failed"]:
        box = thing.get("box")
        if box and thing["slug"] not in replaced:
            middle = np.array(box["centre"], float)
            box["centre"][1] = round(box["centre"][1] + _rest(middle, middle[1] - abs(box["size"][1]) / 2, carriers), 4)

    # Every stand-in's boxes first, so each one's painting knows what the others hide (with the fused room)
    import open3d as o3d

    shapes = [[_shape(box) for box in part["boxes"]] for part in spec["parts"]]
    scenes = {}

    def hiding(shift):
        """The fused room and every stand-in moved by `shift`: what hides what, for frames that saw the room there
        (a part painted from frames whose copy of it sits `shift` away, "paintShift")."""
        if shift not in scenes:
            scenes[shift] = _surface(out)[0]
            for points, triangles, _ in (shape for boxes in shapes for shape in boxes):
                three = ((points + np.array(shift) + lift) @ level).astype(np.float32)  # upright -> three.js axes
                scenes[shift].add_triangles(o3d.core.Tensor(three), o3d.core.Tensor(triangles.astype(np.uint32)))
        return scenes[shift]

    made = []
    for part, boxes in zip(spec["parts"], shapes):
        pieces = []
        for box, (points, triangles, normals) in zip(part["boxes"], boxes):
            shade = 0.55 + 0.45 * (0.5 + 0.5 * normals @ LIGHT)
            colours = np.clip(np.array(box["colour"], float)[None] / 255 * shade[:, None], 0, 1)
            seen = np.zeros(len(points), bool)
            if box.get("paint") and part.get("paintFrom"):
                # only the faces looking the painted way (up, unless the box says): the frames see an edge at a
                # slant, and paint it with what's beyond
                shift = tuple(part.get("paintShift", (0.0, 0.0, 0.0)))
                painted, seen = _paint(out, points + np.array(shift), normals, part["paintFrom"], level, lift,
                                       hiding(shift))
                seen &= normals @ np.array(box.get("paintFacing", (0.0, 1.0, 0.0))) > box.get("paintMin", 0.9)
                colours[seen] = painted[seen]
                if box.get("fill") == "match":
                    colours = _match_fill(colours, seen, normals, box.get("paintFacing", (0.0, 1.0, 0.0)),
                                          box.get("fillGrain", 0.025), box.get("paintMin", 0.9))
            pieces.append((points, triangles, colours, seen))
        offsets = np.cumsum([0] + [len(p) for p, _, _, _ in pieces[:-1]])
        points = np.vstack([p for p, _, _, _ in pieces])
        triangles = np.vstack([t + o for (_, t, _, _), o in zip(pieces, offsets)])
        colours = np.clip(np.vstack([c for _, _, c, _ in pieces]) * 255 + 0.5, 0, 255).astype(np.uint8)
        seen = np.concatenate([s for _, _, _, s in pieces])
        name = f"room-thing-{part['slug']}.ply"
        _write_thing_ply(out / name, points, colours, triangles, seen)
        made.append({"name": part["name"], "slug": part["slug"], "file": name, **_box(points),
                     "seenShare": round(float(seen.mean()), 3), "screen": bool(part.get("screen")), "moving": False,
                     "fit": "measured stand-in", "made": part.get("why", "stand-in"),
                     **({"fixed": True} if part.get("fixed") else {}), **({"wall": True} if part.get("wall") else {})})

    summary["things"] = made + kept
    summary["standIns"] = {"replaced": sorted(replaced), "moved": sorted(spec.get("moves", {}))}
    (out / "room-things.json").write_text(json.dumps(summary, indent=2))
    merged = []
    for thing in summary["things"]:
        mesh = o3d.io.read_triangle_mesh(str(out / thing["file"]))
        merged.append((np.asarray(mesh.vertices), (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint8),
                       np.asarray(mesh.triangles)))
    offsets = np.cumsum([0] + [len(p) for p, _, _ in merged[:-1]])
    _write_ply(out / "room-things-all.ply", np.vstack([p for p, _, _ in merged]), np.vstack([c for _, c, _ in merged]),
               np.vstack([t + o for (_, _, t), o in zip(merged, offsets)]))
    return {"made": [m["slug"] for m in made], "replaced": sorted(replaced), "moved": sorted(spec.get("moves", {}))}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/stand_ins.py <request folder>")
    print(json.dumps(build(sys.argv[1]), indent=2))
