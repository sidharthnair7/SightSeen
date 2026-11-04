"""The issues mode (the owner's ask, Nov 2): potential problems in the landlord's video, each boxed where it is and
saying why, flagging less and being right, with a plain-rules gate on top of the AI. Two kinds come from here:

  damage      Muse Spark (Meta) lists damage a renter should know about in the frames (mould, water stains, cracks,
              holes, peeling paint, broken fixtures, stained or torn carpet, pests); SAM 3.1 outlines each in the frames
              Muse named. THE GATE: a flag shows only when SAM 3.1 outlines it in at least GATE_FRAMES frames, from
              cameras at least GATE_APART metres apart or GATE_TURN degrees round, whose outlines land on the room's
              surface within GATE_SAME metres of each other, and at least SMALLEST across. One AI guess from one frame
              never shows; every candidate that fails says which rule it failed.
  not filmed  the parts of the walls and ceiling no frame saw (room_textures.py's seen maps) where nothing stands in
              front (looking from the middle of the room at eye height, no rebuilt thing is in the way: what's hidden
              behind a wardrobe isn't a red flag, a wall the camera skipped is), at least NOT_FILMED square metres,
              one flag a face. The floor is left out: most of it is under things.

The size check against the listing and the badge for how the video came in are worked out in the viewer (the
student types the ad's size; the server knows how the video arrived): plain arithmetic and facts, no AI.

Writes room-issues.json and issue-<n>-<k>.jpg (each damage flag's frames, its outline and box drawn in red).
    python worker/room_issues.py <request folder>
Costs one Muse Spark call, and a SAM 3.1 call per candidate per frame named (cached): a few cents.
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rebuild_things import THUMB, _mask, _rays, _room, _spark, _surface
from room_objects import _cameras, _outline, _records
from room_textures import FACES, PAD
from sam_check import load_env

GATE_FRAMES = 2  # frames that must outline it
GATE_SAME = 0.25  # metres: their outlines' spots on the room's surface agree this closely
GATE_APART = 0.20  # metres between cameras, or
GATE_TURN = 10.0  # degrees between the ways they look at it: really two views, not one view twice
SMALLEST = 0.02  # metres across: smaller is a speck, not damage worth a box
MOST_CANDIDATES = 8
MOST_FRAMES = 6  # frames asked per candidate (the sharpest Muse named)
NOT_FILMED = 0.5  # square metres of open, unseen wall or ceiling that make a flag
EYE = 1.5  # real metres: where the not-filmed check looks from, in the middle of the room

ISSUES = (
    "These {n} photos are frames of a landlord's video of a room for rent, each numbered in its top-left corner. A "
    "student will decide whether to rent the room from this video alone. List the damage or condition problems you can "
    "clearly see that a renter should know about: mould or mildew, water stains or leaks, cracks or holes in walls, "
    "ceiling or floor, peeling paint, broken or missing fixtures (blinds, handles, outlets, lights, smoke alarm), "
    "stained or torn carpet, signs of pests. Only list what is clearly visible; do not guess, and leave out ordinary "
    "mess (clothes, bags, bottles, an unmade bed). Answer with only a JSON list, empty if there is nothing: "
    '[{{"name": "<2 to 4 words that outline it, e.g. water stain>", "frames": [frame numbers that show it], '
    '"why": "<one sentence for the student: what it is and why it matters>"}}]'
)


def find(out):
    import cv2
    import open3d as o3d

    out = Path(out)
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    level, lift = _room(out)
    folder = out / "things"
    folder.mkdir(exist_ok=True)

    # Muse Spark's list (kept, so running again costs nothing)
    listed_file = folder / "issues-listed.json"
    if listed_file.exists():
        listed = json.loads(listed_file.read_text())
    else:
        pictures = []
        for k, camera in enumerate(cameras):
            picture = cv2.imread(str(images / camera["name"]))
            scale = THUMB / max(picture.shape[:2])
            picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.rectangle(picture, (0, 0), (58, 30), (0, 0, 0), -1)
            cv2.putText(picture, str(k), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            pictures.append(cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        answer = _spark(pictures, ISSUES.format(n=len(pictures)), key)
        try:
            listed = json.loads(answer[answer.index("["): answer.rindex("]") + 1])
        except ValueError as error:  # not an answer: the step fails, and the page says this check didn't finish
            raise RuntimeError("Muse Spark's list of damage couldn't be read") from error
        listed_file.write_text(json.dumps(listed, indent=2))
    listed = [item for item in listed if isinstance(item, dict) and str(item.get("name", "")).strip()][:MOST_CANDIDATES]

    # SAM 3.1's outlines of each candidate in the sharpest frames Muse named (cached with the rebuild step's)
    sharp = {}
    for k, camera in enumerate(cameras):
        gray = cv2.cvtColor(cv2.imread(str(images / camera["name"])), cv2.COLOR_BGR2GRAY)
        sharp[k] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    jobs = []
    for item in listed:
        frames = [f for f in item.get("frames", []) if isinstance(f, int) and 0 <= f < len(cameras)]
        item["frames"] = sorted(frames, key=lambda f: -sharp[f])[:MOST_FRAMES]
        jobs += [(cameras[f]["name"], str(item["name"]).strip().lower()) for f in item["frames"]]
    answers_file = folder / "sam-answers.json"
    answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}
    wanted = sorted({job for job in jobs if f"{job[0]}|{job[1]}" not in answers})
    if wanted:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for (name, phrase), text in pool.map(lambda job: (job, _outline(images / job[0], job[1], key)), wanted):
                answers[f"{name}|{phrase}"] = text
        answers_file.write_text(json.dumps(answers))

    # Each outline onto the room's surface: where it is and how big
    scene, _, _ = _surface(out)
    flags, dropped = [], []
    for item in listed:
        name = str(item["name"]).strip().lower()
        seen = []
        for f in item["frames"]:
            camera = cameras[f]
            height, width = camera["size"]
            for record in _records(answers.get(f"{camera['name']}|{name}") or ""):
                mask = _mask(record, height, width)
                if mask.sum() < 30:
                    continue
                rows, cols = np.nonzero(mask[::2, ::2])
                rays = _rays(camera, cols * 2 + 0.5, rows * 2 + 0.5)
                hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                ok = np.isfinite(hits)
                if ok.sum() < 10:
                    continue
                points = (rays[ok, :3] + rays[ok, 3:] * hits[ok, None]) @ level.T - lift  # upright
                middle = np.median(points, axis=0)
                low, high = np.percentile(points, [5, 95], axis=0)
                eye = level @ np.diag([1.0, -1.0, -1.0]) @ (-camera["turn"].T @ camera["shift"]) - lift
                seen.append({"frame": f, "mask": mask, "middle": middle, "size": high - low, "eye": eye})
        verdict, group = _gate(seen)
        if verdict:
            dropped.append({"name": name, "why": item.get("why"), "failed": verdict,
                            "framesOutlined": sorted({s["frame"] for s in seen})})
            continue
        n = len(flags) + 1
        pictures = []
        for s in sorted(group, key=lambda s: -s["mask"].sum())[:3]:
            picture = cv2.imread(str(images / cameras[s["frame"]]["name"]))
            contours, _ = cv2.findContours(s["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(picture, contours, -1, (0, 0, 255), 3)
            ys, xs = np.nonzero(s["mask"])
            cv2.rectangle(picture, (int(xs.min()) - 12, int(ys.min()) - 12), (int(xs.max()) + 12, int(ys.max()) + 12),
                          (0, 0, 255), 4)
            picture = cv2.resize(picture, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            file = f"issue-{n}-{s['frame']}.jpg"
            cv2.imwrite(str(out / file), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            pictures.append({"frame": s["frame"], "file": file})
        middles = np.array([s["middle"] for s in group])
        flags.append({
            "kind": "damage", "name": name, "why": item.get("why"),
            "position": [round(float(v), 3) for v in np.median(middles, axis=0)],
            "size": [round(float(v), 3) for v in np.median([s["size"] for s in group], axis=0)],
            "frames": pictures,
            "gate": {"frames": len({s["frame"] for s in group}), "sameSpotCm": round(100 * _spread(middles), 1),
                     "viewsApartCm": round(100 * _apart(group), 1), "viewsTurnDeg": round(_turn(group), 1)},
            "ask": f"Ask the landlord to film the {name} up close, live.",
        })

    # Not filmed, flagging less: the ceiling, and the biggest open wall patch; the other walls' patches are listed as
    # not shown (a handheld video rarely looks up, so every wall's top goes unfilmed)
    unfilmed = _not_filmed(out)
    ceiling = [f for f in unfilmed if f["face"] == "ceiling"]
    walls = sorted((f for f in unfilmed if f["face"] != "ceiling"), key=lambda f: -f["areaM2"])
    flags += ceiling + walls[:1]
    dropped += [{"name": f["name"], "why": f["why"], "failed": "a smaller unfilmed wall patch (the biggest one is shown)",
                 "face": f["face"], "areaM2": f["areaM2"]} for f in walls[1:]]
    report = {"gate": {"frames": GATE_FRAMES, "sameSpotCm": round(100 * GATE_SAME), "viewsApartCm": round(100 * GATE_APART),
                       "viewsTurnDeg": GATE_TURN, "smallestCm": round(100 * SMALLEST)},
              "framesChecked": len(cameras), "damageListed": len(listed), "flags": flags, "notShown": dropped,
              "askedSam": len(wanted)}
    (out / "room-issues.json").write_text(json.dumps(report, indent=2))
    return report


def _spread(middles):
    """How far apart the two furthest spots are (what "landing within X of each other" says)."""
    return float(max(np.linalg.norm(a - b) for a in middles for b in middles)) if len(middles) > 1 else 0.0


def _apart(group):
    eyes = np.array([s["eye"] for s in group])
    return float(max(np.linalg.norm(a - b) for a in eyes for b in eyes)) if len(eyes) > 1 else 0.0


def _turn(group):
    spot = np.median([s["middle"] for s in group], axis=0)
    ways = [(spot - s["eye"]) / max(np.linalg.norm(spot - s["eye"]), 1e-9) for s in group]
    return max(math.degrees(math.acos(float(np.clip(a @ b, -1, 1)))) for a in ways for b in ways) if len(ways) > 1 else 0.0


def _gate(seen):
    """The plain rules: None and the outlines that pass, or which rule failed. The biggest group of outlines (one per
    frame) whose spots are all within GATE_SAME of each other, every pair, is the candidate."""
    if not seen:
        return "SAM 3.1 didn't outline it in any frame Muse Spark named", []
    best = []
    for anchor in seen:
        group, used = [], set()
        for s in sorted(seen, key=lambda s: np.linalg.norm(s["middle"] - anchor["middle"])):
            if s["frame"] not in used and all(np.linalg.norm(s["middle"] - g["middle"]) <= GATE_SAME
                                              for g in group + [anchor]):
                group.append(s)
                used.add(s["frame"])
        if len(group) > len(best):
            best = group
    if len(best) < GATE_FRAMES:
        return f"outlined in only {len(best)} frame(s) at the same spot (needs {GATE_FRAMES})", best
    if _apart(best) < GATE_APART and _turn(best) < GATE_TURN:
        return "its frames are one view seen twice (cameras too close, too little turn)", best
    if float(np.median([s["size"].max() for s in best])) < SMALLEST:
        return f"smaller than {round(100 * SMALLEST)} cm across", best
    return None, best


def _not_filmed(out):
    """Unseen, open patches of the shell's walls and ceiling (room_textures.py's seen maps, less what the room's rebuilt
    things hide from its middle), each as a rectangle in the room."""
    import cv2
    import open3d as o3d

    shell = json.loads((out / "room-shell.json").read_text())
    textures = json.loads((out / "room-shell-textures.json").read_text())["faces"]
    frame = json.loads((out / "room-frame.json").read_text())
    scale = frame.get("scale") or 1.0
    yaw = math.radians(shell["yawDegrees"])
    axes = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
    half = np.array(shell["size"], float) / 2
    centre = np.array(shell["centre"], float)
    # The room's rebuilt things, to tell "behind a wardrobe" from "never pointed at"
    things = o3d.t.geometry.RaycastingScene()
    things_file = out / "room-things-all.ply"
    if things_file.exists():
        things.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(things_file))))
    eye = np.array([centre[0], EYE / scale, centre[2]])
    flags = []
    for face, (axis, side, (col_axis, col_sign), (row_axis, row_sign)) in FACES.items():
        if face == "floor":
            continue
        seen_file = out / textures[face]["seen"]
        seen = cv2.imread(str(seen_file), cv2.IMREAD_GRAYSCALE) > 127 if seen_file.exists() else None
        if seen is None:
            continue
        rows, cols = seen.shape
        across, down = 2 * (half[col_axis] + PAD), 2 * (half[row_axis] + PAD)
        texel = (across / cols) * (down / rows) * scale * scale  # real square metres

        def corner(c, r):
            local = np.zeros(3)
            local[axis] = side * half[axis]
            local[col_axis] = col_sign * (c / cols * across - across / 2)
            local[row_axis] = row_sign * (r / rows * down - down / 2)
            return centre + local @ axes

        # Open: a ray from the middle of the room to the texel meets no rebuilt thing on the way
        r_idx, c_idx = np.mgrid[0:rows, 0:cols]
        targets = np.array([corner(c + 0.5, r + 0.5) for r, c in zip(r_idx.ravel(), c_idx.ravel())])
        ways = targets - eye
        lengths = np.linalg.norm(ways, axis=1)
        rays = np.hstack([np.repeat(eye[None], len(ways), 0), ways / lengths[:, None]]).astype(np.float32)
        hits = things.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        open_view = (~(hits < lengths - 0.05)).reshape(rows, cols)
        unseen = (~seen) & open_view
        unseen = cv2.morphologyEx(unseen.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(unseen)
        patches = sorted(range(1, count), key=lambda label: -stats[label, cv2.CC_STAT_AREA])
        for label in patches[:1]:  # one flag a face: its biggest open unseen patch
            area = stats[label, cv2.CC_STAT_AREA] * texel
            if area < NOT_FILMED:
                continue
            left, top, w, h = stats[label, :4]

            corners = [corner(left, top), corner(left + w, top), corner(left + w, top + h), corner(left, top + h)]
            # Its own shape, not its bounding box: the edge as a polygon, and the inside as cells on a coarse grid
            patch = (labels == label).astype(np.uint8)
            edges, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge = cv2.approxPolyDP(max(edges, key=cv2.contourArea), 1.5, True)[:, 0, :]
            step = max(1, round(min(rows, cols) / 32))
            cells = [[r, c] for r in range(0, rows, step) for c in range(0, cols, step)
                     if patch[r:r + step, c:c + step].mean() >= 0.5]
            origin = corner(0, 0)
            share = stats[label, cv2.CC_STAT_AREA] / (rows * cols)  # of the whole face
            where = {"ceiling": "the ceiling", "floor": "the floor"}.get(face, f"the {face} wall")
            flags.append({
                "kind": "not filmed", "name": f"{where}, never filmed" if share > 0.9 else f"part of {where}, never filmed",
                "why": f"No frame of the video shows this {round(area, 1)} m² of {where}: a renter can't judge what "
                       f"isn't shown.",
                "face": face, "areaM2": round(float(area), 2), "share": round(float(share), 2),
                "corners": [[round(float(v), 3) for v in c] for c in corners],
                "outline": [[round(float(v), 3) for v in corner(float(c), float(r))] for c, r in edge],
                "fill": {"origin": [round(float(v), 4) for v in origin],
                         "across": [round(float(v), 5) for v in corner(step, 0) - origin],
                         "down": [round(float(v), 5) for v in corner(0, step) - origin], "cells": cells},
                "ask": f"Ask the landlord to film {where} too.",
            })
    return flags


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_issues.py <request folder>")
    result = find(sys.argv[1])
    print(json.dumps({"flags": [(f["kind"], f["name"]) for f in result["flags"]],
                      "notShown": [(d["name"], d["failed"]) for d in result["notShown"]],
                      "askedSam": result["askedSam"]}, indent=2))
