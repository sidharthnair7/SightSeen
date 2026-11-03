"""The Rebuilt look against the video, frame by frame and item by item: what the owner asked for (Nov 2, 3pm: "compare
it with what the items and how the room looks like from the video frame by frame and then make it better").

Every frame is drawn from its own camera the way the viewer's Rebuilt look draws it (the rebuilt things inside the
clean walls), next to the real frame. Every item Muse Spark listed (things.json's picks, the skipped and failed ones
too) and every wall thing is outlined by SAM 3.1 in the frames it shows in (cached in things/sam-answers.json; missing
answers are asked for, about a third of a cent each). For each item and frame: how much of its outline the rebuilt
item covers from that camera (coverage), the overlap both ways (IoU), and how far its colour is off where it's drawn.
Outlines are drawn on the real frame: green where the rebuilt item covers it, orange where it's off, red where the
room has nothing there.

Writes checks/compare/frame-NN.jpg (real | drawn), checks/compare/sheet.jpg (all frames, small) and
checks/compare/report.json (per item: frames, coverage, IoU, colour, status, worst frame; per frame: its items), and
prints the items worst first.
    python worker/rebuild_compare.py <request folder>
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from rebuild_things import _mask, _room
from rebuilt_views import _material_pictures
from room_checks import FACES, FLIP, WalkRoom
from room_objects import _cameras, _outline, _records
from sam_check import load_env

SHRINK = 3  # frames are compared at a third of the cameras' size
WALL_THINGS = {"window": ("window",), "curtains": ("curtain",), "door": ("door",),
               "baseboard heater": ("baseboard heater",), "light switch": ("light switch",)}
GOOD, OFF = 0.5, 0.2  # coverage: at least GOOD is matched, under OFF the room shows nothing of it there


def compare(out):
    import open3d as o3d

    out = Path(out)
    folder = out / "things"
    placed = json.loads((out / "room-things.json").read_text())
    things = placed["things"]
    picked = json.loads((folder / "things.json").read_text())
    listed = json.loads((folder / "picks.json").read_text())
    seen_in = json.loads((folder / "seen-in.json").read_text())
    answers_file = folder / "sam-answers.json"
    answers = json.loads(answers_file.read_text())
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    level, lift = _room(out)

    # Every item to look for: the rebuilt things (by slug), the picks that never got built (skipped or failed), and
    # the wall things. Each with the SAM phrase(s) that outline it and the frames it's in
    built = {t["slug"]: t for t in things}
    items = []
    for thing in things:
        phrases = WALL_THINGS.get(thing["name"], (thing["name"],))
        items.append({"key": thing["slug"], "name": thing["name"], "phrases": phrases, "status": "rebuilt"})
    for failed in placed.get("failed", []):
        items.append({"key": failed["slug"], "name": failed["name"], "phrases": (failed["name"],),
                      "status": "grey box" if failed.get("box") else "missing"})
    for skipped in picked.get("skipped", []):
        if skipped["name"] in {i["name"] for i in items}:
            continue
        items.append({"key": skipped["name"].replace(" ", "-"), "name": skipped["name"],
                      "phrases": (skipped["name"],), "status": "not built"})
    frames_of = {}
    for k, item in enumerate(listed):
        name = str(item.get("name", "")).strip().lower()
        frames_of.setdefault(name, set()).update(f for f in seen_in.get(str(k), []) if isinstance(f, int))
    for item in items:
        if item["name"] in WALL_THINGS or item["name"] not in frames_of:
            item["frames"] = list(range(len(cameras)))  # wall things: SAM looks everywhere (cached already)
        else:
            item["frames"] = sorted(f for f in frames_of[item["name"]] if 0 <= f < len(cameras))

    # SAM 3.1's outlines for every item in every frame it's in
    wanted = sorted({(cameras[f]["name"], p) for item in items for f in item["frames"] for p in item["phrases"]
                     if f"{cameras[f]['name']}|{p}" not in answers})
    if wanted:
        load_env()
        key = os.environ["MODEL_API_KEY"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for (name, phrase), text in pool.map(lambda job: (job, _outline(images / job[0], job[1], key)), wanted):
                answers[f"{name}|{phrase}"] = text
        answers_file.write_text(json.dumps(answers))

    # The room drawn like the Rebuilt look: things inside clean walls; and each thing as its own geometry
    room = WalkRoom(out, furniture="room-things-all.ply")
    room.pictures = _material_pictures(out, room)
    room.seen = {face: None for face in FACES}
    scene = o3d.t.geometry.RaycastingScene()
    ids, middles = {}, {}
    for thing in things:
        mesh = o3d.io.read_triangle_mesh(str(out / thing["file"]))
        ids[thing["slug"]] = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        middles[thing["slug"]] = np.asarray(mesh.vertices).mean(axis=0)
    boxes = {f["slug"]: f["box"] for f in placed.get("failed", []) if f.get("box")}

    target = out / "checks" / "compare"
    target.mkdir(parents=True, exist_ok=True)
    records, per_frame, sheet = {item["key"]: [] for item in items}, {}, []
    for k, camera in enumerate(cameras):
        height, width = camera["size"]
        small = (height // SHRINK, width // SHRINK)
        rows, cols = np.mgrid[0:small[0], 0:small[1]]
        u, v = (cols.ravel() + 0.5) * SHRINK, (rows.ravel() + 0.5) * SHRINK
        origins, directions = room.camera_rays(camera, u, v)
        colour, _ = room.cast(origins, directions)
        drawn = (np.clip(colour, 0, 1).reshape(small[0], small[1], 3)[:, :, ::-1] * 255).astype(np.uint8)
        real = cv2.resize(cv2.imread(str(images / camera["name"])), (small[1], small[0]), interpolation=cv2.INTER_AREA)
        rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        shown = np.where(np.isfinite(hit["t_hit"].numpy()), hit["geometry_ids"].numpy().astype(np.int64), -1)
        shown = shown.reshape(small)
        marked = real.copy()
        found = []
        for item in items:
            if k not in item["frames"]:
                continue
            masks = []
            for phrase in item["phrases"]:
                for record in _records(answers.get(f"{camera['name']}|{phrase}") or ""):
                    mask = cv2.resize(_mask(record, height, width).astype(np.uint8), (small[1], small[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
                    if mask.sum() > 8:
                        masks.append(mask)
            if not masks:
                continue
            if item["key"] in ids:
                # the outline nearest where the rebuilt thing projects is its own (three suitcases share a name)
                spot = ((middles[item["key"]] + lift) @ level) @ FLIP @ camera["turn"].T + camera["shift"]
                if spot[2] > 0:
                    px = camera["lens"][0, 0] * spot[0] / spot[2] + camera["lens"][0, 2]
                    py = camera["lens"][1, 1] * spot[1] / spot[2] + camera["lens"][1, 2]
                    centres = [np.argwhere(m).mean(axis=0) * SHRINK for m in masks]
                    masks = [masks[int(np.argmin([np.hypot(c[0] - py, c[1] - px) for c in centres]))]]
                mask = np.any(masks, axis=0) if item["name"] in WALL_THINGS else masks[0]
                mine = shown == ids[item["key"]]
            else:
                mask = masks[0]
                mine = np.zeros(small, bool)
            coverage = float((mine & mask).sum() / mask.sum())
            union = (mine | mask).sum()
            iou = float((mine & mask).sum() / union) if union else 0.0
            both = mine & mask
            colour_off = (float(np.abs(real[both].astype(float).mean(0) - drawn[both].astype(float).mean(0)).mean())
                          if both.sum() > 20 else None)
            record = {"frame": k, "coverage": round(coverage, 3), "iou": round(iou, 3),
                      "colourOff": None if colour_off is None else round(colour_off, 1), "pixels": int(mask.sum())}
            records[item["key"]].append(record)
            found.append({"item": item["key"], **record})
            tint = (0, 200, 0) if coverage >= GOOD else (0, 140, 255) if coverage >= OFF else (0, 0, 255)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked, contours, -1, tint, 2)
            y, x = np.argwhere(mask).mean(axis=0).astype(int)
            cv2.putText(marked, item["key"][:16], (max(0, x - 40), y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, tint, 1)
        per_frame[k] = found
        pair = np.hstack([marked, drawn])
        cv2.putText(pair, f"frame {k}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(str(target / f"frame-{k:02d}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheet.append(cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2), interpolation=cv2.INTER_AREA))

    summary = []
    for item in items:
        found = records[item["key"]]
        if not found:
            summary.append({"item": item["key"], "name": item["name"], "status": item["status"], "frames": 0})
            continue
        weights = np.array([r["pixels"] for r in found], float)
        coverage = float(np.average([r["coverage"] for r in found], weights=weights))
        colours = [r["colourOff"] for r in found if r["colourOff"] is not None]
        worst = min(found, key=lambda r: r["coverage"])
        summary.append({"item": item["key"], "name": item["name"], "status": item["status"], "frames": len(found),
                        "coverage": round(coverage, 3),
                        "iou": round(float(np.average([r["iou"] for r in found], weights=weights)), 3),
                        "colourOff": round(float(np.mean(colours)), 1) if colours else None,
                        "worstFrame": worst["frame"], "worstCoverage": worst["coverage"],
                        "box": boxes.get(item["key"])})
    summary.sort(key=lambda s: (s.get("coverage", -1), s["item"]))
    columns = 6
    while len(sheet) % columns:
        sheet.append(np.zeros_like(sheet[0]))
    grid = np.vstack([np.hstack(sheet[i:i + columns]) for i in range(0, len(sheet), columns)])
    cv2.imwrite(str(target / "sheet.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    report = {"items": summary, "frames": per_frame, "askedSam": len(wanted)}
    (target / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/rebuild_compare.py <request folder>")
    result = compare(sys.argv[1])
    print("asked SAM", result["askedSam"], "times")
    for s in result["items"]:
        print(f"{s['item']:22s} {s['status']:9s} frames {s['frames']:2d}  coverage {s.get('coverage', '-')!s:6s} "
              f"iou {s.get('iou', '-')!s:6s} colour off {s.get('colourOff', '-')!s:5s} worst frame {s.get('worstFrame', '-')}")
