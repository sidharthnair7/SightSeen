"""The room's real size, checked against things that come in standard sizes (the design in the README: outlets and
switches of known size check the room's scale). The 3D step's metric depth can be off by a constant factor (take 4's
desk top came out 54 cm high, where desks are usually 72 to 76 cm), and pictures can't show it: the same room twice as big,
filmed from twice as far, looks the same. A wall plate can: a standard North American single switch or outlet plate
is 4.5 x 2.75 inches (11.4 x 7.0 cm).

SAM 3.1 (Meta Model API) outlines every light switch and outlet in every frame; rays through each outline hit the fused
surface; the plate's height and width are measured along the wall (up, and across) from the hits, edge to edge (the
outer 1% each side trimmed as noise, then the 98% that's left widened back to the whole). Outlines landing within
MERGE of each other are one plate (seen in several frames, or outlined twice in one), measured by their medians; a
plate measuring under half or over twice its standard size is a bad outline (or not one plate) and is left out. The
scale is the median of standard / measured over the plates left, in both directions; how sure it is counts plates,
not outlines (a plate's height and width share its errors).

Writes "scale" (and the evidence) into room-frame.json, which the viewer applies to the whole room. The owner's tape
measure beats the plates: `set` takes one real length (the room's width or length, the desk's or the bed's height)
and makes everything the scale sizes again (the shell's assumed ceiling, its wall photos and materials, the splats).
    python worker/room_scale.py <request folder>
    python worker/room_scale.py set <request folder> desk-height 0.74
    python worker/room_scale.py rebuild <request folder>
Costs two image calls per frame: about 20 cents for 37 frames.
"""
import json
import os
import re
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
MERGE = 0.3  # room units: outlines whose middles are closer are the same plate (plates sit further apart than this)
BELIEVABLE = (0.5, 2.0)  # standard / measured for a real plate: MapAnything's own scale is never off by 2x
SENSIBLE = {"room-width": (1.5, 15.0), "room-length": (1.5, 15.0), "desk-height": (0.4, 1.2), "bed-height": (0.15, 1.0)}
TYPES = {"x": "<f4", "y": "<f4", "z": "<f4", "red": "u1", "green": "u1", "blue": "u1", "seen": "u1"}


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

    sightings = []
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
                sightings.append({"frame": camera["name"], "phrase": phrase, "tall": float(tall), "wide": float(wide),
                                  "middle": points.mean(axis=0), "pixels": int(mask.sum())})

    # One plate however often it was seen or outlined: the sightings grouped by where they are, the biggest first
    plates = []
    for sighting in sorted(sightings, key=lambda s: -s["pixels"]):
        home = next((p for p in plates if np.linalg.norm(p["middle"] - sighting["middle"]) < MERGE), None)
        if home:
            home["seen"].append(sighting)
        else:
            plates.append({"middle": sighting["middle"], "seen": [sighting]})
    readings, report = [], []
    for plate in plates:
        tall = float(np.median([s["tall"] for s in plate["seen"]]))
        wide = float(np.median([s["wide"] for s in plate["seen"]]))
        ratios = [PLATE[0] / tall, PLATE[1] / wide]
        kept = all(BELIEVABLE[0] <= r <= BELIEVABLE[1] for r in ratios)
        if kept:
            readings += ratios
        report.append({"phrases": sorted({s["phrase"] for s in plate["seen"]}),
                       "frames": sorted({s["frame"] for s in plate["seen"]}), "outlines": len(plate["seen"]),
                       "tall": round(tall, 4), "wide": round(wide, 4), "ratios": [round(r, 3) for r in ratios],
                       "height": round(float(plate["middle"][1]), 3), "kept": kept})
    if not readings:
        return {"scale": None, "plates": report, "why": "no wall plate of a believable size was found"}
    scale = round(float(np.median(readings)), 4)
    count = sum(p["kept"] for p in report)
    evidence = {"standard": "wall plate 11.4 x 7.0 cm", "plates": count, "outlines": len(sightings),
                "ratiosKept": count,  # independent readings: one a plate
                "spread": [round(float(np.percentile(readings, 25)), 3), round(float(np.percentile(readings, 75)), 3)],
                "readings": [round(r, 3) for r in readings]}
    frame_file = out / "room-frame.json"
    frame = json.loads(frame_file.read_text())
    before = frame.get("scale")
    if (frame.get("scaleEvidence") or {}).get("source") == "tape measure":
        frame["plateCheck"] = {"scale": scale, **evidence}  # the tape measure beats the plates: kept beside it
    else:
        frame["scale"] = scale
        frame["scaleEvidence"] = evidence
    frame_file.write_text(json.dumps(frame, indent=2))
    if frame["scale"] != before:
        _rescale_items(out, frame["scale"])
    return {"scale": frame["scale"], "plateScale": scale, "before": before, "plates": report}


def _rescale_items(out, scale):
    """The student's things were built in the room's units at the scale then: a new scale sizes them again, so each
    stays its real size (their PLYs' points scaled in place, item.json's room units and scale updated)."""
    import items

    for meta_file in sorted((out / "items").glob("*/item.json")) if (out / "items").exists() else []:
        meta = json.loads(meta_file.read_text())
        ply = meta_file.parent / "item.ply"
        was = meta.get("roomScale")
        if meta.get("status") != "done" or not was or not ply.exists() or abs(was - scale) < 1e-9:
            continue
        factor = was / scale
        data = bytearray(ply.read_bytes())
        end = data.index(b"end_header\n") + len(b"end_header\n")
        header = data[:end].decode("ascii").splitlines()
        names = [line.split()[2] for line in header if line.startswith("property") and "list" not in line]
        count = int(next(line.split()[2] for line in header if line.startswith("element vertex")))
        kind = np.dtype([(n, TYPES[n]) for n in names])
        vertex = np.frombuffer(bytes(data[end:end + count * kind.itemsize]), dtype=kind).copy()
        for axis in "xyz":
            vertex[axis] *= factor
        data[end:end + count * kind.itemsize] = vertex.tobytes()
        ply.write_bytes(bytes(data))
        items._status(meta_file.parent, roomScale=scale,
                      sizeUnits=[round(v * factor, 4) for v in meta.get("sizeUnits", [])])


def _metres(text):
    """A length as typed: '0.74' or '0.74m' (metres), '74cm' or '74 cm' (centimetres)."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(cm|m)?\s*", str(text).lower())
    if not match:
        raise SystemExit(f"'{text}' isn't a length: give metres (0.74) or centimetres (74cm)")
    value = float(match.group(1))
    return value / 100 if match.group(2) == "cm" else value


KNOWN = ("room-width", "room-length", "desk-height", "bed-height")  # what a tape measure can set the scale from


def _known_length(out, what):
    """How long `what` is in the room's own units: the shell's width or length, or the top of a stand-in's surface (the
    desk's, the bed's) from things/stand-ins.json."""
    if what in ("room-width", "room-length"):
        shell = json.loads((out / "room-shell.json").read_text())
        return float(shell["size"][0 if what == "room-width" else 2])
    slug = what.split("-")[0]
    spec = json.loads((out / "things" / "stand-ins.json").read_text())
    part = next(p for p in spec["parts"] if p["slug"] == slug)
    boxes = [b for b in part["boxes"] if b.get("carries")] or part["boxes"]
    return max(b["centre"][1] + b["size"][1] / 2 for b in boxes if "centre" in b)


def set_scale(out, what, metres):
    """The owner's tape measure: one real length sets the room's scale (it beats the wall plates), then everything
    sized by it is made again."""
    out = Path(out)
    if what not in KNOWN:
        raise SystemExit(f"what's measured must be one of: {', '.join(KNOWN)}")
    low, high = SENSIBLE[what]
    if not low <= metres <= high:
        raise SystemExit(f"{metres:g} m isn't a believable {what.replace('-', ' ')} ({low:g} to {high:g} m). "
                         f"In centimetres? Write it as {metres:g}cm.")
    units = _known_length(out, what)
    frame_file = out / "room-frame.json"
    frame = json.loads(frame_file.read_text())
    before = frame.get("scale")
    frame["scale"] = round(metres / units, 4)
    frame["scaleEvidence"] = {"source": "tape measure", "what": what, "metres": metres, "roomUnits": round(units, 4),
                              "before": before, "plates": (frame.get("scaleEvidence") or {}).get("plates")}
    frame_file.write_text(json.dumps(frame, indent=2))
    rebuild(out)
    _rescale_items(out, frame["scale"])
    return {"scale": frame["scale"], "before": before, "what": what, "metres": metres, "roomUnits": round(units, 4)}


def rebuild(out):
    """Everything the scale sizes, made again: the shell (its assumed ceiling is 2.44 real metres), its wall photos and
    clean materials, and the splats trimmed to it."""
    import reconstruct
    import room_materials
    import room_textures

    out = Path(out)
    shell = reconstruct.write_room_shell(out)
    room_textures.make_textures(out)
    room_materials.make(out)
    if (out / "room-splats-full.ply").exists() or (out / "room-splats.ply").exists():
        reconstruct.trim_splats(out)
    return {"shell": shell["size"], "ceilingAssumed": shell["ceilingAssumed"]}


USAGE = """usage: python worker/room_scale.py <request folder>                     (the scale from the wall plates)
       python worker/room_scale.py set <request folder> <what> <length>   (a tape measure: room-width, room-length,
                                                                          desk-height or bed-height; 0.74 or 74cm)
       python worker/room_scale.py rebuild <request folder>               (shell, wall photos, materials, splats)"""


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "set":
        print(json.dumps(set_scale(sys.argv[2], sys.argv[3], _metres(sys.argv[4])), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "rebuild":
        print(json.dumps(rebuild(sys.argv[2]), indent=2))
    elif len(sys.argv) == 2:
        result = measure(sys.argv[1])
        print(json.dumps({k: v for k, v in result.items() if k != "plates"}, indent=2))
        for plate in result.get("plates", []):
            print(plate)
        print("then, when the scale changed: python worker/room_scale.py rebuild <request folder>")
    else:
        sys.exit(USAGE)
