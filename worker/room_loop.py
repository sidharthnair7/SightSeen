"""The owner's loop for the Walk room: build it with a recipe, without the held-back frames, draw it from those frames'
own cameras, score it against the real frames (the video itself, under the base lighting), and keep what scores
better. Scores are for the whole picture (Phase 2) and inside each thing's SAM 3.1 outline (Phase 1), so a step that
helps the wardrobe and hurts the desk shows as exactly that.

Recipes add steps to the painted room (photo walls and the fused furniture, both from the base-lit frames):
  "rebuild"  - rebuild_objects.py: things whose surface lost pieces, closed with Open3D Poisson
  "blockout" - blockout.py: Muse Spark's key parts as blocks, painted from each thing's own pixels
Muse Spark's answers are reused from room-blockout.json (no new calls). The geometry (fused surface and dots) still
comes from every frame; only the paint is held back, so the scores lean a little kind, equally for every recipe.
    python worker/room_loop.py <request folder> [recipe ...]
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import blockout as blockout_step
import rebuild_objects
from room_checks import HOLD_OUT, WalkRoom, _ssim
from room_furniture import build_furniture
from room_objects import _cameras, _records
from room_textures import _colmap_solved, make_textures
from splat_objects import _mask

RECIPES = {"painted": [], "rebuilt": ["rebuild"], "rebuilt+blockout": ["rebuild", "blockout"]}
PER_THING = ["bed", "desk", "chair", "wardrobe", "door", "window"]  # names SAM 3.1 outlined on every 3rd frame


def _held_out_copy(out, held, recipe):
    check = out / "checks" / f"loop-{recipe}"
    if check.exists():
        shutil.rmtree(check)
    (check / "splat-data" / "sparse" / "0").mkdir(parents=True)
    (check / "splat-data" / "images").mkdir()
    (check / "colmap" / "model" / "0").mkdir(parents=True)
    for name in ("room-frame.json", "room-shell.json", "room-objects.json", "room-surface.ply", "room.ply",
                 "room-blockout.json"):
        if (out / name).exists():
            shutil.copyfile(out / name, check / name)
    shutil.copyfile(out / "colmap" / "model" / "0" / "images.bin", check / "colmap" / "model" / "0" / "images.bin")
    sparse = out / "splat-data" / "sparse" / "0"
    for name in ("cameras.txt", "points3D.txt"):
        shutil.copyfile(sparse / name, check / "splat-data" / "sparse" / "0" / name)
    kept = []
    for line in (sparse / "images.txt").read_text().splitlines():
        parts = line.split()
        if line.startswith("#"):
            kept.append(line)
        elif len(parts) == 10 and parts[9] not in held:
            kept += [line, ""]
            shutil.copyfile(out / "frames-base-light" / parts[9], check / "splat-data" / "images" / parts[9])
    (check / "splat-data" / "sparse" / "0" / "images.txt").write_text("\n".join(kept) + "\n")
    outlines = json.loads((out / "sam-outlines.json").read_text())  # only the kept frames' outlines
    (check / "sam-outlines.json").write_text(json.dumps({k: v for k, v in outlines.items() if k.split("|")[0] not in held}))
    return check


def build(check, steps):
    make_textures(check)
    build_furniture(check)
    if "rebuild" in steps:
        rebuild_objects.rebuild(check)
    if "blockout" in steps:
        blockout_step.blockout(check)


def score(out, check, held, label):
    """Phase 2 (whole frames) and Phase 1 (inside each thing's outline) on the held-back frames, COLMAP-solved ones."""
    walk = WalkRoom(check)
    solved = _colmap_solved(out / "colmap" / "model" / "0") or set()
    outlines = json.loads((out / "sam-outlines.json").read_text())
    frames, things, sheet = [], {name: [0.0, 0, []] for name in PER_THING}, []
    for camera in _cameras(out / "splat-data"):
        if camera["name"] not in held or Path(camera["name"]).stem not in solved:
            continue
        real = cv2.imread(str(out / "frames-base-light" / camera["name"]))[:, :, ::-1].astype(np.float32) / 255
        height, width = real.shape[:2]
        v, u = np.indices((height, width)) + 0.5
        colour, _ = walk.cast(*walk.camera_rays(camera, u.reshape(-1), v.reshape(-1)))
        drawn = colour.reshape(height, width, 3)
        error = ((drawn - real) ** 2).mean(axis=2)
        grey = lambda x: cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255  # noqa: E731
        frames.append((10 * math.log10(1 / max(float(error.mean()), 1e-10)), _ssim(grey(drawn), grey(real))))
        sheet.append(np.hstack([real, drawn]))
        for name in PER_THING:
            key = f"{camera['name']}|{name}"
            if key not in outlines:
                continue
            mask = _mask(_records(outlines[key]), width, height)
            if mask.sum() < 200:
                continue
            things[name][0] += float(error[mask].sum())
            things[name][1] += int(mask.sum())
    cv2.imwrite(str(out / "checks" / f"loop-{label}.jpg"), (np.vstack(sheet[:6]) * 255).astype(np.uint8)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "psnr": round(float(np.mean([f[0] for f in frames])), 2), "ssim": round(float(np.mean([f[1] for f in frames])), 3),
        "frames": len(frames),
        "things": {name: round(10 * math.log10(n / e), 2) for name, (e, n, _) in things.items() if n},
    }


def loop(out, recipes=None):
    out = Path(out)
    cameras = _cameras(out / "splat-data")
    held = {camera["name"] for k, camera in enumerate(cameras) if k % HOLD_OUT == 3}
    results = {}
    for label in recipes or RECIPES:
        started = time.perf_counter()
        check = _held_out_copy(out, held, label)
        build(check, RECIPES[label])
        results[label] = score(out, check, held, label)
        results[label]["seconds"] = round(time.perf_counter() - started, 1)
        print(label, json.dumps(results[label]), flush=True)
    previous = json.loads((out / "room-loop.json").read_text()) if (out / "room-loop.json").exists() else {}
    previous.update(results)
    (out / "room-loop.json").write_text(json.dumps(previous, indent=2))
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python worker/room_loop.py <request folder> [recipe ...]")
    loop(sys.argv[1], sys.argv[2:] or None)
