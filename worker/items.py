"""The student's own things, built from a photo and set in the rebuilt room (the owner's ask, Nov 2: take a picture of
your current setup, and its 3D version is an item you can place in the room). In: a photo, what the thing is, and its
longest side across the floor. Out: a 3D model of it at exactly that size, standing on the floor, ready to move.

  1. SAM 3.1 (Meta Model API) outlines the thing by its name in the photo (the biggest outline when there are
     several; its last word alone when the whole name finds nothing: "gaming desk" -> "desk"; then whatever Muse Spark,
     looking at the photo, calls it).
  2. SAM 3D Objects (Meta) builds it whole from the photo and the outline, in its own axes, on this PC's card
     (worker/sam3d_local.py in WSL: about a minute, and two more while the model loads); TRELLIS
     (worker/trellis_local.py) when SAM 3D can't.
  3. It's stood up (TRELLIS's z up -> the room's y up), its bottom on the floor, its middle at x = z = 0, and sized
     so its longest side across the floor is the length the student typed: one real number, not the model's guess,
     since a fit check is only as good as its sizes. Real metres become the room's own units by room-frame.json's
     "scale" (worker/room_scale.py sizes it again when the scale changes). The viewer stands it on free floor.

Writes items/<id>/item.ply (the room's units, like room-thing-*.ply) and keeps items/<id>/item.json's "status" current
(queued, outlining, building, done, or failed with "why"), which the viewer reads through the server.
    python worker/items.py <request folder> <item id>
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from rebuild_things import _mask, _spark, _wsl, _write_thing_ply
from room_objects import _outline, _records
from sam_check import load_env

LONGEST = 1024  # pixels: the photo's long side for SAM 3.1 and TRELLIS (TRELLIS works at 518 itself)
MIN_SHARE = 0.01  # an outline smaller than this share of the photo is too small to build from
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # (x, y, z) -> (x, z, -y)
BUILD_LIMIT = 15 * 60  # seconds one model gets in WSL before it's stopped (it takes about 3 minutes)
on_status = None  # told each new item.json, when something else needs to know straight away (the hosted app)


def _status(item_dir, **fields):
    """item.json with `fields` changed, written whole (a new file swapped in), so the server never reads half of one."""
    meta = json.loads((item_dir / "item.json").read_text())
    meta.update(fields)
    fresh = item_dir / "item.json.new"
    fresh.write_text(json.dumps(meta, indent=2))
    os.replace(fresh, item_dir / "item.json")
    if on_status:
        on_status(meta)
    return meta


def _plain(text):
    """An error for the student: file paths cut down to their names (they mean nothing to them, and show the PC's)."""
    text = re.sub(r"[A-Za-z]:[\\/][^'\"\s]*", lambda m: Path(m.group(0)).name, text)
    return re.sub(r"/mnt/[a-z]/[^'\"\s]*|/home/[^'\"\s]*", lambda m: m.group(0).rsplit("/", 1)[-1], text)


def _names_from_muse(photo, name, key):
    """When SAM 3.1 finds nothing by the typed name: Muse Spark looks at the photo and says what to ask it for instead."""
    import cv2

    picture = cv2.imencode(".jpg", cv2.imread(str(photo)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    answer = _spark([picture], f"Someone photographed their {name}. Give up to 3 plain names an object detector could "
                               f"use to find it in this photo, one per line, 1 to 3 words each, nothing else. If it "
                               f"isn't clear, name the biggest thing in the photo.", key)
    names = [" ".join(re.sub(r"[^a-z0-9 ]", " ", line.lower()).split()) for line in answer.splitlines()]
    return [n for n in names if n][:3]


def _outline_thing(photo, name, key):
    """The biggest SAM 3.1 outline of `name` in the photo, as a picture-sized True/False mask, and the words that found
    it: the whole name, then its last word, then what Muse Spark calls it. (None, None) when nothing does."""
    from PIL import Image

    width, height = Image.open(photo).size

    def biggest(phrases):
        for phrase in phrases:
            masks = [_mask(record, height, width) for record in _records(_outline(photo, phrase, key))]
            masks = [m for m in masks if m.mean() >= MIN_SHARE]
            if masks:
                return max(masks, key=lambda m: m.sum()), phrase
        return None, None

    first = list(dict.fromkeys([name, name.split()[-1]]))
    found = biggest(first)
    return found if found[0] is not None else biggest([n for n in _names_from_muse(photo, name, key) if n not in first])


def _build_deployed(photo, outline):
    """SAM 3D on the deployed sightseen-sam3d app (the hosted pipeline): vertices in its own axes (z up), faces, colours."""
    import modal

    result = modal.Cls.from_name("sightseen-sam3d", "Sam3d")().build.remote(
        Path(photo).read_bytes(), [Path(outline).read_bytes()], None, 42)[0]
    if "error" in result:
        raise RuntimeError(f"SAM 3D couldn't build it: {result['error']}")
    return (np.frombuffer(result["canonical"], np.float32).reshape(-1, 3).astype(np.float64),
            np.frombuffer(result["faces"], np.int32).reshape(-1, 3), np.frombuffer(result["colours"], np.uint8).reshape(-1, 3))


def _build(item_dir, photo, outline, runner):
    """SAM 3D ("sam3d_local.py") or TRELLIS ("trellis_local.py") on the photo and outline, in WSL. Returns vertices in
    the model's own axes (z up), faces and colours, or raises with the model's own error."""
    work = item_dir / f"build-{Path(runner).stem}"
    work.mkdir(exist_ok=True)
    (work / "jobs.json").write_text(json.dumps({"jobs": [{"photo": _wsl(photo), "outlines": [_wsl(outline)],
                                                          "out": _wsl(work / "out")}], "seed": 42}))
    command = ('export MAMBA_ROOT_PREFIX=~/sam3d/mamba && eval "$(~/sam3d/bin/micromamba shell hook -s bash)" && '
               f"micromamba activate sam3d && timeout {BUILD_LIMIT} python {_wsl(Path(__file__).parent / runner)} "
               f"{_wsl(work / 'jobs.json')}")
    try:  # one hung build mustn't hold the GPU queue (the room builds wait in it too)
        subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], check=False, capture_output=True,
                       timeout=BUILD_LIMIT + 60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{runner} took more than {BUILD_LIMIT // 60} minutes and was stopped") from None
    result_file = work / "out" / "0.json"
    if not result_file.exists():
        raise RuntimeError(f"{runner} stopped before it finished (is WSL and its environment set up?)")
    result = json.loads(result_file.read_text())
    if "error" in result:
        raise RuntimeError(f"{runner} couldn't build it: {result['error']}")
    arrays = np.load(work / "out" / "0.npz")
    vertices = arrays["canonical"] if "canonical" in arrays else arrays["vertices"]  # SAM 3D's own axes, as TRELLIS's
    return vertices.astype(np.float64), arrays["faces"], arrays["colours"]


def build_item(request_dir, item_id):
    """Builds one item (its photo and item.json are already in items/<id>/). Returns the final item.json."""
    from PIL import Image

    request_dir = Path(request_dir)
    item_dir = request_dir / "items" / item_id
    meta = json.loads((item_dir / "item.json").read_text())
    try:
        load_env()
        key = os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
        try:
            picture = Image.open(item_dir / meta["photo"]).convert("RGB")
        except Exception:  # noqa: BLE001 - named .jpg, but not a picture
            raise RuntimeError("That file isn't a photo SightSeen can open. Use a JPG, PNG or WebP.") from None
        picture.thumbnail((LONGEST, LONGEST))
        photo = item_dir / "photo.png"
        picture.save(photo)

        _status(item_dir, status="outlining", why=None)
        outline, phrase = _outline_thing(photo, meta["name"], key)
        if outline is None:
            raise RuntimeError(f"Couldn't find the {meta['name']} in the picture. Try one where it fills more of the "
                               f"frame, with nothing in front of it.")
        Image.fromarray(outline.astype(np.uint8) * 255).save(item_dir / "outline.png")

        _status(item_dir, status="building", outlinedAs=phrase)
        built_by = "SAM 3D Objects"
        if os.environ.get("SIGHTSEEN_SAM3D_DEPLOYED") == "1":
            vertices, faces, colours = _build_deployed(photo, item_dir / "outline.png")
        else:
            try:
                vertices, faces, colours = _build(item_dir, photo, item_dir / "outline.png", "sam3d_local.py")
            except RuntimeError as error:
                _status(item_dir, status="building", sam3dError=str(error)[:200])
                vertices, faces, colours = _build(item_dir, photo, item_dir / "outline.png", "trellis_local.py")
                built_by = "TRELLIS"

        # Stood up, on the floor, at its typed width in the room's own units
        points = vertices @ Z_UP_TO_Y_UP.T
        frame_file = request_dir / "room-frame.json"
        scale = (json.loads(frame_file.read_text()).get("scale") or 1.0) if frame_file.exists() else 1.0
        scale = meta.get("scale") or scale  # a hosted visitor's thing says which room's scale it's for
        across = np.ptp(points[:, [0, 2]], axis=0)
        factor = (meta["widthCm"] / 100 / scale) / max(float(across.max()), 1e-6)
        points = points * factor
        points[:, 1] -= points[:, 1].min()
        points[:, [0, 2]] -= (points[:, [0, 2]].max(axis=0) + points[:, [0, 2]].min(axis=0)) / 2
        _write_thing_ply(item_dir / "item.ply", points.astype(np.float32), np.asarray(colours, np.uint8),
                         np.asarray(faces), np.ones(len(points), bool))
        size_units = np.ptp(points, axis=0)
        return _status(item_dir, status="done", file="item.ply", builtBy=built_by,
                       sizeCm=[round(float(v) * scale * 100, 1) for v in size_units],
                       sizeUnits=[round(float(v), 4) for v in size_units], roomScale=scale)
    except Exception as error:  # noqa: BLE001 - the student sees why, in a sentence
        return _status(item_dir, status="failed", why=_plain(str(error))[:300])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python worker/items.py <request folder> <item id>")
    print(json.dumps(build_item(sys.argv[1], sys.argv[2]), indent=2))
