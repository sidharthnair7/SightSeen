"""The rebuild loop's placement round: every rebuilt thing nudged (turned, sized, moved across the floor) until its
outline from every frame SAM 3.1 outlined it in agrees best with SAM's, instead of only the one photo it was built
from. A nudge is kept only when the thing's mean overlap (IoU, over those frames) goes up.

    python worker/rebuild_refine.py outline <request folder>   SAM 3.1 outlines each placed thing in up to MOST_FRAMES
                                                                of the frames Muse Spark said it's in (cached)
    python worker/rebuild_refine.py refine <request folder>    the nudging; rewrites room-thing-*.ply,
                                                                room-things.json and room-things-all.ply
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from rebuild_things import FLIP, IN_FRONT, _box, _iou, _mask, _rays, _room, _silhouette, _surface, _write_ply
from room_objects import _cameras, _outline, _records
from sam_check import load_env

MOST_FRAMES = 6  # frames each thing is outlined in, beyond the ones it already has
SHRINK = 4  # outlines compared at a quarter of the cameras' size
TURNS = (-24, -12, -6, 6, 12, 24)  # degrees tried, then halved
SIZES = (0.85, 0.93, 1.07, 1.15)
SHIFTS = (-0.12, -0.06, 0.06, 0.12)  # metres, along each floor axis


def _placed(out):
    return json.loads((out / "room-things.json").read_text())


def outline(out):
    """SAM 3.1 outlines of each placed thing in more of its frames (the sharpest Muse Spark said it's in)."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    from room_furniture import _sharpness

    out = Path(out)
    load_env()
    key = os.environ["MODEL_API_KEY"]
    folder = out / "things"
    listed = json.loads((folder / "picks.json").read_text())
    seen_in = json.loads((folder / "seen-in.json").read_text())
    answers_file = folder / "sam-answers.json"
    answers = json.loads(answers_file.read_text())
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    sharp = _sharpness([cv2.imread(str(images / camera["name"]))[:, :, ::-1] for camera in cameras])
    wanted = set()
    for thing in _placed(out)["things"]:
        frames = set()
        for k, item in enumerate(listed):
            if str(item.get("name", "")).strip().lower() == thing["name"]:
                frames |= {f for f in seen_in.get(str(k), []) if isinstance(f, int) and 0 <= f < len(cameras)}
        for f in sorted(frames, key=lambda f: -sharp[f])[:MOST_FRAMES]:
            if f"{cameras[f]['name']}|{thing['name']}" not in answers:
                wanted.add((cameras[f]["name"], thing["name"]))

    def one(pair):
        name, phrase = pair
        return pair, _outline(images / name, phrase, key)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for (name, phrase), text in pool.map(one, sorted(wanted)):
            answers[f"{name}|{phrase}"] = text
    answers_file.write_text(json.dumps(answers))
    return {"newOutlines": len(wanted)}


def _views(out, thing, upright, answers, cameras, level, lift, scene):
    """The frames this thing is outlined in, each with its small outline (the one nearest where the thing projects),
    what hides it there (room surface clearly nearer), and the small camera."""
    import cv2
    import open3d as o3d

    views = []
    middle = upright.mean(axis=0)
    for camera in cameras:
        key = f"{camera['name']}|{thing['name']}"
        if key not in answers:
            continue
        height, width = camera["size"]
        small = (height // SHRINK, width // SHRINK)
        masks = [cv2.resize(_mask(r, height, width).astype(np.uint8), (small[1], small[0]),
                            interpolation=cv2.INTER_NEAREST) > 0 for r in _records(answers[key])]
        masks = [m for m in masks if m.sum() > 5]
        spot = (((middle + lift) @ level) @ FLIP) @ camera["turn"].T + camera["shift"]
        if not masks or spot[2] <= 0:
            continue
        px = camera["lens"][0, 0] * spot[0] / spot[2] + camera["lens"][0, 2]
        py = camera["lens"][1, 1] * spot[1] / spot[2] + camera["lens"][1, 2]
        centres = [np.argwhere(m).mean(axis=0) * SHRINK for m in masks]
        mask = masks[int(np.argmin([np.hypot(c[0] - py, c[1] - px) for c in centres]))]
        rows, cols = np.mgrid[0:small[0], 0:small[1]]
        real = scene.cast_rays(o3d.core.Tensor(_rays(camera, (cols.ravel() + 0.5) * SHRINK,
                                                     (rows.ravel() + 0.5) * SHRINK)))["t_hit"].numpy()
        hidden = (np.where(np.isfinite(real), real, 1e3) < spot[2] - IN_FRONT).reshape(small)
        lens = camera["lens"].copy()
        lens[:2] /= SHRINK
        views.append((camera, lens, small, mask, hidden))
    return views


def _score(upright, triangles, views, level, lift):
    if not views:
        return 0.0
    total = 0.0
    for camera, lens, small, mask, hidden in views:
        in_camera = (((upright + lift) @ level) @ FLIP) @ camera["turn"].T + camera["shift"]
        total += _iou(_silhouette(in_camera, triangles, lens, small), mask, hidden)
    return total / len(views)


def _nudge(upright, base, turn_degrees=0.0, size=1.0, shift=(0.0, 0.0)):
    """The thing turned about its own upright axis and sized about its base centre, then moved across the floor."""
    angle = math.radians(turn_degrees)
    spin = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    moved = upright - base
    moved[:, [0, 2]] = moved[:, [0, 2]] @ spin.T
    moved = moved * size + base
    moved[:, 0] += shift[0]
    moved[:, 2] += shift[1]
    return moved


def refine(out):
    import open3d as o3d

    out = Path(out)
    folder = out / "things"
    summary = _placed(out)
    answers = json.loads((folder / "sam-answers.json").read_text())
    cameras = _cameras(out / "splat-data")
    level, lift = _room(out)
    scene, _, _ = _surface(out)
    report, merged = [], []
    for thing in summary["things"]:
        path = out / thing["file"]
        mesh = o3d.io.read_triangle_mesh(str(path))
        upright = np.asarray(mesh.vertices).copy()
        triangles = np.asarray(mesh.triangles)
        views = _views(out, thing, upright, answers, cameras, level, lift, scene)
        before = _score(upright, triangles, views, level, lift)
        best, best_score = upright, before
        standing = upright[:, 1].min() < 0.02
        for scale in (1.0, 0.5):  # a coarse pass, then a finer one about the best so far
            base = np.array([best[:, 0].mean(), best[:, 1].min(), best[:, 2].mean()])
            for options, make in ((TURNS, lambda t: _nudge(best, base, turn_degrees=t * scale)),
                                  (SIZES, lambda s: _nudge(best, base, size=1 + (s - 1) * scale)),
                                  (SHIFTS, lambda d: _nudge(best, base, shift=(d * scale, 0.0))),
                                  (SHIFTS, lambda d: _nudge(best, base, shift=(0.0, d * scale)))):
                for option in options:
                    trial = make(option)
                    if standing:
                        trial[:, 1] -= trial[:, 1].min()
                    score = _score(trial, triangles, views, level, lift)
                    if score > best_score + 0.002:
                        best, best_score = trial, score
                        base = np.array([best[:, 0].mean(), best[:, 1].min(), best[:, 2].mean()])
        if best is not upright:
            data = path.read_bytes()
            end = data.index(b"end_header\n") + len(b"end_header\n")
            header = data[:end].decode("ascii")
            names = [line.split()[2] for line in header.splitlines() if line.startswith("property") and "list" not in line]
            count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
            types = {"x": "<f4", "y": "<f4", "z": "<f4", "red": "u1", "green": "u1", "blue": "u1", "seen": "u1"}
            vertex = np.frombuffer(data, dtype=[(n, types[n]) for n in names], count=count, offset=end).copy()
            vertex["x"], vertex["y"], vertex["z"] = best[:, 0], best[:, 1], best[:, 2]
            path.write_bytes(data[:end] + vertex.tobytes() + data[end + vertex.nbytes:])
            thing.update(_box(best))
        thing["framesFit"] = round(best_score, 3)
        report.append({"thing": thing["slug"], "frames": len(views), "before": round(before, 3), "after": round(best_score, 3)})
        colours = (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint8)
        merged.append((best, colours, triangles))
    (out / "room-things.json").write_text(json.dumps(summary, indent=2))
    offsets = np.cumsum([0] + [len(points) for points, _, _ in merged[:-1]])
    _write_ply(out / "room-things-all.ply", np.vstack([p for p, _, _ in merged]), np.vstack([c for _, c, _ in merged]),
               np.vstack([t + o for (_, _, t), o in zip(merged, offsets)]))
    return {"things": report}


if __name__ == "__main__":
    steps = {"outline": outline, "refine": refine}
    if len(sys.argv) != 3 or sys.argv[1] not in steps:
        sys.exit("usage: python worker/rebuild_refine.py outline|refine <request folder>")
    print(json.dumps(steps[sys.argv[1]](sys.argv[2]), indent=2))
