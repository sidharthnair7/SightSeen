"""Things built into the walls (the window and its curtains, the door, the heater, switches and outlets) as sharp
cut-outs on the Rebuilt look's clean walls. The clean paint (room_materials.py) leaves the walls bare, and these are
what make the room this room (the owner's ask, Nov 2: the 3D model should look like a 3D version of the room).

SAM 3.1 (Meta Model API) outlines each kind in every frame. Each wall of room-shell.json's box is laid out as TEXEL
squares on the real wall's surface (on the curtain where it hangs in front, as room_textures.py does); every frame that
sees a square head-on and unblocked gives it a colour and a vote (is this thing there?), weighted towards the sharpest,
nearest, most head-on view. Where most of the weight says it is, the square is part of the cut-out; each connected
patch is one picture with a soft see-through edge, set on its wall where it was filmed. A part no frame saw, inside a
patch (behind the monitor), is filled from around it.

Writes room-decal-<kind>-<n>.png and room-decals.json ({"decals": [{"name", "file", "centre", "right", "up",
"normal", "width", "height"}]}, upright room axes, metres; the viewer draws each as a picture on its wall).
    python worker/room_decals.py <request folder>
Costs one image call per phrase per frame not asked before: about 40 cents for take 4's 37 frames.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import room_textures as textures
from rebuild_things import _mask
from room_objects import _outline, _records
from sam_check import load_env

KINDS = {  # each kind and the SAM 3.1 phrases whose outlines make it, joined
    "window": ("window", "curtain"),
    "door": ("door",),
    "heater": ("baseboard heater",),
    "light switch": ("light switch",),
    "outlet": ("electrical outlet",),
}
WALLS = ("right", "left", "back", "front")
TEXEL = 0.005  # metres a square: about what a frame shows of a wall 2 m away, so the cut-outs stay sharp
SMALLEST = {"window": 0.05, "door": 0.2, "heater": 0.02, "light switch": 0.003, "outlet": 0.002}  # square metres
VOTE = 0.5  # share of a square's weight that must say "this thing"
MARGIN = 6  # squares of see-through border round each picture, for its soft edge


def make(out):
    import cv2
    import open3d as o3d

    out = Path(out)
    load_env()
    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = textures._turn(*frame["quaternion"])
    lift = np.array([0.0, frame["floorHeight"], 0.0])
    cameras = textures._cameras(out / "splat-data")
    solved = textures._colmap_solved(out / "colmap" / "model" / "0")
    images = out / "splat-data" / "images"

    # SAM 3.1's outlines of every kind in every frame, kept with the rebuild step's
    answers_file = out / "things" / "sam-answers.json"
    answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}
    phrases = sorted({phrase for group in KINDS.values() for phrase in group})
    wanted = [(c["name"], p) for c in cameras for p in phrases if f"{c['name']}|{p}" not in answers]
    if wanted:
        key = os.environ["MODEL_API_KEY"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for (name, phrase), text in pool.map(lambda pair: (pair, _outline(images / pair[0], pair[1], key)), wanted):
                answers[f"{name}|{phrase}"] = text
        answers_file.write_text(json.dumps(answers))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))))

    # Each wall's squares, on the real surface within BAND of the shell's plane (room_textures.py's way)
    textures.TEXEL, textures.LONGEST = TEXEL, 4096
    faces = {}
    for face in WALLS:
        upright, inward, grid = textures._face_points(shell, face)
        three, inward3 = (upright.reshape(-1, 3) + lift) @ level, inward @ level
        rays = np.hstack([three + inward3 * textures.BAND, np.repeat(-inward3[None], len(three), axis=0)])
        reach = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        found = np.isfinite(reach) & (reach < 2 * textures.BAND)
        offset = float(np.median(textures.BAND - reach[found])) if found.any() else 0.0
        three = np.where(found[:, None], three + inward3 * (textures.BAND - np.where(found, reach, 0))[:, None],
                         three + inward3 * offset)
        faces[face] = {"upright": upright, "inward": inward, "three": three, "inward3": inward3, "grid": grid,
                       "score": np.zeros((len(cameras), len(three)), np.float32),
                       "colour": np.zeros((len(cameras), len(three), 3), np.uint8),
                       "bits": np.zeros((len(cameras), len(three)), np.uint8)}

    sharpness, kernel = [], np.ones((textures.NEAR_PIXELS, textures.NEAR_PIXELS), np.uint8)
    for k, camera in enumerate(cameras):
        picture = cv2.imread(str(images / camera["name"]))
        height, width = picture.shape[:2]
        sharpness.append(float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()))
        bits = np.zeros((height, width), np.uint8)  # one bit per kind: SAM outlined it here
        for bit, group in enumerate(KINDS.values()):
            for phrase in group:
                for record in _records(answers.get(f"{camera['name']}|{phrase}") or ""):
                    bits |= _mask(record, height, width).astype(np.uint8) << bit
        lens, turn, shift = camera["lens"], camera["turn"], camera["shift"]
        origin = textures.FLIP * (-turn.T @ shift)  # the camera's centre, three.js axes

        rows, cols = np.indices((height, width))
        in_camera = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols)], axis=-1).reshape(-1, 3) @ np.linalg.inv(lens).T
        directions = (in_camera @ turn) * textures.FLIP
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rays = np.hstack([np.repeat(origin[None], len(directions), axis=0), directions]).astype(np.float32)
        reach = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(height, width)
        reach = cv2.erode(np.where(np.isfinite(reach), reach, 100.0).astype(np.float32), kernel)
        soft = cv2.GaussianBlur(picture, (0, 0), 0.7)  # a square covers a few pixels: no shimmer from one

        for data in faces.values():
            points = data["three"]
            x = (points * textures.FLIP) @ turn.T + shift
            ahead = x[:, 2] > 0.05
            z = np.where(ahead, x[:, 2], 1.0)
            u = lens[0, 0] * x[:, 0] / z + lens[0, 2]
            v = lens[1, 1] * x[:, 1] / z + lens[1, 2]
            inside = ahead & (u >= 0.5) & (u <= width - 0.5) & (v >= 0.5) & (v <= height - 0.5)
            to_camera = origin - points
            distance = np.linalg.norm(to_camera, axis=1)
            head_on = (to_camera @ data["inward3"]) / np.maximum(distance, 1e-6)
            iu, iv = np.clip(u.astype(int), 0, width - 1), np.clip(v.astype(int), 0, height - 1)
            seen = inside & (head_on > textures.GRAZING) & (
                reach[iv, iu] > distance - textures.SEEN - textures.SEEN_PER_METRE * distance)
            if not seen.any():
                continue
            edge = np.minimum.reduce([u, width - u, v, height - v]) / textures.EDGE
            data["score"][k] = np.where(seen, head_on / distance * np.clip(edge, 0, 1), 0)
            if solved is not None and Path(camera["name"]).stem not in solved:
                data["score"][k] *= textures.ROUGH
            grid = data["grid"]
            sample = cv2.remap(soft, (u - 0.5).astype(np.float32).reshape(grid["rows"], grid["cols"]),
                               (v - 0.5).astype(np.float32).reshape(grid["rows"], grid["cols"]),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            data["colour"][k] = sample.reshape(-1, 3)
            data["bits"][k] = np.where(seen, bits[iv, iu], 0)

    sharp = np.clip(np.sqrt(np.array(sharpness) / np.median(sharpness)), 0.4, 1.3).astype(np.float32)
    decals, counts = [], {}
    for face, data in faces.items():
        grid = data["grid"]
        shape = (grid["rows"], grid["cols"])
        score = data["score"] * sharp[:, None]
        best = score.max(axis=0)
        weight = np.where(score > 0, (score / np.maximum(best, 1e-9)) ** textures.SHARPEST, 0).astype(np.float32)
        weight[weight < 0.02] = 0
        total = weight.sum(axis=0)
        seen = (total > 0).reshape(shape)
        colour = np.einsum("kn,knc->nc", weight, data["colour"].astype(np.float32)) / np.maximum(total, 1e-9)[:, None]
        colour = np.clip(colour, 0, 255).astype(np.uint8).reshape(*shape, 3)
        texel_area = grid["texel"][0] * grid["texel"][1]
        for bit, kind in enumerate(KINDS):
            vote = (weight * ((data["bits"] >> bit) & 1)).sum(axis=0) / np.maximum(total, 1e-9)
            inside = ((vote > VOTE) & (total > 0)).reshape(shape).astype(np.uint8)
            inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(inside)
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] * texel_area < SMALLEST[kind]:
                    continue
                left, top, w, h = stats[label, :4]
                r0, r1 = max(0, top - MARGIN), min(shape[0], top + h + MARGIN)
                c0, c1 = max(0, left - MARGIN), min(shape[1], left + w + MARGIN)
                part = labels[r0:r1, c0:c1] == label
                rgb = colour[r0:r1, c0:c1].copy()
                unseen = part & ~seen[r0:r1, c0:c1]
                if unseen.any():  # inside the patch but behind something in every frame
                    rgb = cv2.inpaint(rgb, unseen.astype(np.uint8), 5, cv2.INPAINT_TELEA)
                alpha = cv2.GaussianBlur(part.astype(np.float32), (0, 0), 1.2)
                n = counts[kind] = counts.get(kind, 0) + 1
                name = f"room-decal-{re.sub(r'[^a-z0-9]+', '-', kind)}-{n}.png"
                cv2.imwrite(str(out / name), np.dstack([rgb, (alpha * 255).astype(np.uint8)]))
                corners = data["upright"][[r0, r0, r1 - 1, r1 - 1], [c0, c1 - 1, c0, c1 - 1]]  # on the shell's plane
                right = corners[1] - corners[0]
                up = corners[0] - corners[2]
                decals.append({
                    "name": kind, "file": name, "face": face,
                    "centre": [round(float(c), 4) for c in corners.mean(axis=0)],
                    "right": [round(float(c), 5) for c in right / np.linalg.norm(right)],
                    "up": [round(float(c), 5) for c in up / np.linalg.norm(up)],
                    "normal": [round(float(c), 5) for c in data["inward"]],
                    "width": round(float((c1 - c0) * grid["texel"][0]), 4),
                    "height": round(float((r1 - r0) * grid["texel"][1]), 4),
                    "seenShare": round(float(1 - unseen.sum() / max(1, part.sum())), 3),
                })
    (out / "room-decals.json").write_text(json.dumps({"texelMetres": TEXEL, "decals": decals}, indent=2))
    return decals


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_decals.py <request folder>")
    for decal in make(sys.argv[1]):
        print(decal["name"], decal["face"], decal["width"], "x", decal["height"], "seen", decal["seenShare"])
