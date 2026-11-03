"""The room's real size, checked against things that come in standard sizes (the design in the README: outlets and
switches of known size check the room's scale). The 3D step's metric depth can be off by a constant factor (take 4
came out about 1.3x too small: a 74 cm desk top at 54 cm), and pictures can't show it: the same room twice as big,
filmed from twice as far, looks the same. A wall plate can: a standard North American single switch or outlet plate
is 4.5 x 2.75 inches (11.4 x 7.0 cm).

SAM 3.1 (Meta Model API) outlines every light switch and outlet in every frame; rays through each outline hit the fused
surface; the plate's height and width are measured along the wall (up, and across) from the hits, edge to edge (the
outer 1% each side trimmed as noise, then the 98% that's left widened back to the whole). The scale is the median of
standard / measured over every plate seen, in both directions. Plates more than 40% off the median are left out (a
double plate, a bad outline).

Writes "scale" (and the evidence) into room-frame.json, which the viewer applies to the whole room.
    python worker/room_scale.py <request folder>
Costs two image calls per frame: about 20 cents for 37 frames.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rebuild_things import _mask, _rays, _room, _surface
from room_objects import _cameras, _outline, _records
from sam_check import load_env

PLATE = (0.114, 0.070)  # metres: a standard single-gang wall plate, tall x wide
PHRASES = ("light switch", "electrical outlet")
TRIM = 1  # percent trimmed off each side of a plate's hits as edge noise (then widened back)
AGREE = 0.4  # plates more than this share off the median ratio are left out


def measure(out):
    import open3d as o3d

    out = Path(out)
    load_env()
    key = os.environ["MODEL_API_KEY"]
    folder = out / "things"
    folder.mkdir(exist_ok=True)
    answers_file = folder / "sam-answers.json"
    answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    level, lift = _room(out)
    scene, _, _ = _surface(out)

    wanted = [(c["name"], p) for c in cameras for p in PHRASES if f"{c['name']}|{p}" not in answers]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (name, phrase), text in pool.map(lambda pair: (pair, _outline(images / pair[0], pair[1], key)), wanted):
            answers[f"{name}|{phrase}"] = text
    answers_file.write_text(json.dumps(answers))

    plates = []
    for camera in cameras:
        height, width = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
        for phrase in PHRASES:
            for record in _records(answers[f"{camera['name']}|{phrase}"]):
                mask = _mask(record, height, width)
                if mask.sum() < 400 or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                    continue  # too small to measure, or cut by the picture's edge
                rows, cols = np.nonzero(mask)
                rays = _rays(camera, cols + 0.5, rows + 0.5)
                hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
                ok = np.isfinite(hits)
                points = (rays[ok, :3] + rays[ok, 3:] * hits[ok, None]) @ level.T - lift
                if len(points) < 300:
                    continue
                # Along the wall: up, and across (the horizontal direction in the plate's own plane)
                centred = points - points.mean(axis=0)
                _, _, axes = np.linalg.svd(centred, full_matrices=False)
                normal = axes[2]
                if abs(normal[1]) > 0.5:
                    continue  # not on a wall
                across = np.cross([0.0, 1.0, 0.0], normal)
                across /= np.linalg.norm(across)
                spread = 1 - 2 * TRIM / 100

                def extent(values):
                    low, high = np.percentile(values, [TRIM, 100 - TRIM])
                    return (high - low) / spread

                tall, wide = extent(centred[:, 1]), extent(centred @ across)
                plates.append({"frame": camera["name"], "phrase": phrase, "tall": round(float(tall), 4),
                               "wide": round(float(wide), 4), "height": round(float(points[:, 1].mean()), 3),
                               "pixels": int(mask.sum())})
    ratios = [PLATE[0] / p["tall"] for p in plates] + [PLATE[1] / p["wide"] for p in plates]
    if not ratios:
        return {"scale": None, "plates": [], "why": "no whole wall plate was found"}
    middle = float(np.median(ratios))
    kept = [r for r in ratios if abs(r / middle - 1) <= AGREE]
    scale = round(float(np.median(kept)), 4)
    frame_file = out / "room-frame.json"
    frame = json.loads(frame_file.read_text())
    frame["scale"] = scale
    frame["scaleEvidence"] = {"standard": "wall plate 11.4 x 7.0 cm", "plates": len(plates), "ratiosKept": len(kept),
                              "spread": [round(float(np.percentile(kept, 25)), 3), round(float(np.percentile(kept, 75)), 3)]}
    frame_file.write_text(json.dumps(frame, indent=2))
    return {"scale": scale, "plates": plates, "ratios": [round(r, 3) for r in ratios]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_scale.py <request folder>")
    result = measure(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k != "plates"}, indent=2))
    for plate in result.get("plates", []):
        print(plate)
