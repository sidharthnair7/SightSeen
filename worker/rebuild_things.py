"""The room's things rebuilt whole with SAM 3D Objects (Meta's model that builds a whole object from one photo), so every
side of them looks like a real thing, even the sides the video never saw. The owner's rules for what wasn't filmed
(Nov 1): (1) it gets a normal version of itself instead of blur; (2) a screen is black, and something that moves, like
a fan, is built from the one frame where it's sharpest; (3) a grey box stands in only where the build fails (the same
box the viewer grabs a thing by when it's moved).

For each thing standing in the room:
  1. pick: Muse Spark (Meta Model API) looks at every frame and lists the things, each with the one frame that shows it
     most whole and a rough box round it there. SAM 3.1 outlines it in that frame (the outline whose box best matches
     Muse Spark's). Screens in that frame are outlined too and painted black. Rays through the outlined pixels hit the
     room's fused surface (room-surface.ply): those are the thing's measured points, in metres.
  2. build: SAM 3D Objects (run by fal.ai, since it needs a 32 GB graphics card) builds every outlined thing of a frame
     from that frame, one call per frame.
  3. place: SAM 3D's object goes into the frame's camera (its own pose, in whichever axes best cover the outline), is
     scaled about the camera until it sits on the measured points (scaling about the camera keeps it on the outline),
     and is then fitted to them (ICP). Things standing on the floor go down onto it.
  4. The sides the photo saw are painted from the full-size photo itself, where the real surface agrees; SAM 3D's own
     colours stay only where the photo didn't see, matched to the photo's exposure, and each vertex records which.
Things built into the walls (windows, doors, curtains, outlets, heaters) stay part of the walls' photos.

Needs the request folder's frames/ (the video's full-size frames), splat-data/ (frames and cameras), room-surface.ply,
room-frame.json and room-shell.json; MODEL_API_KEY for pick and FAL_KEY for build (both in .env).
    python worker/rebuild_things.py pick <request folder>    Muse Spark + SAM 3.1: a few cents
    python worker/rebuild_things.py build <request folder>   SAM 3D on fal.ai: $0.02 a call, one call per frame
    python worker/rebuild_things.py place <request folder>
Writes things/ (picks.json, and per thing its outline, measured points and SAM 3D's files), room-thing-<slug>.ply per
thing and room-things.json for the viewer.
"""

import base64
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from room_furniture import _bilinear, _write_ply
from room_objects import _cameras, _outline, _records, _turn
from sam_check import API, load_env

SPARK = "muse-spark-1.3"
THUMB = 448  # pixels: the long side of the frames Muse Spark looks at, so all of them fit one request
MOST_FRAMES = 50  # Muse Spark takes at most 50 images
SCREEN = "computer screen"
MOST_TRIES = 4  # frames tried per thing: Muse Spark's pick, then the sharpest others it shows in
CUT = 0.05  # a mask whose outermost pixel ring runs along the picture's edge for more than this share is cut off
DUPLICATE = 0.10  # metres: a thing whose measured middle is this close to one already picked (its size alike) is it
RAY_STEP = 2  # rays through every 2nd outlined pixel each way
SAME_THING = 0.06  # metres: measured points closer than this are one piece (DBSCAN); the biggest piece is the thing
FAL_APP = "fal-ai/sam-3/3d-objects"
SEED = 42
FIT_SIZE = 480  # pixels: the long side of the picture the outline fit is checked on
FAILED_FIT = 0.35  # an object covering its outline less well than this (IoU) didn't build: the grey box stands in
ON_FLOOR = 0.15  # metres: a thing whose bottom ends up this close to the floor stands on it
SEEN_HIDDEN = 0.01  # metres: a vertex further than this behind the object's own surface is hidden from the photo
SEEN_FACING = 0.26  # cos 75 degrees: surface turned further from the photo than this isn't painted from it
TRUTH = 0.05  # metres: the photo paints a vertex only where the real surface it sees there is this close
TRUTH_PER_METRE = 0.02  # plus this per metre of depth
OUTLINE_INSET = 2  # pixels: the outline shrinks this much before painting, so its edge doesn't bleed the background in
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes, as everywhere in the worker
# Axes SAM 3D's camera might use, to OpenCV's (x right, y down, z ahead): the fit to the outline picks one. The
# usual ones first (PyTorch3D's x left, y up; OpenGL's and glTF's y up, z back), then every other quarter turn,
# in case its files are z-up; a fit this good stops the search
AXES = {"opencv": np.diag([1.0, 1.0, 1.0]), "pytorch3d": np.diag([-1.0, -1.0, 1.0]), "opengl": np.diag([1.0, -1.0, -1.0]),
        "x-flipped": np.diag([-1.0, 1.0, -1.0])}
for _order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
    for _signs in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
        _swap = np.zeros((3, 3))
        _swap[np.arange(3), _order] = _signs
        if np.linalg.det(_swap) > 0 and not any(np.array_equal(_swap, a) for a in AXES.values()):
            AXES[f"turn {len(AXES)}"] = _swap
GOOD_FIT = 0.85

PICK = (
    "These {n} pictures are frames from one phone video walking through a single room, in order; each has its number "
    "in the top left corner. List every distinct physical thing standing, lying or hanging in the room: furniture, "
    "appliances, electronics, bags, boxes, fans, lamps, and smaller items like bottles, shoes or clothes piles. Leave "
    "out the walls, floor and ceiling, and things built into them (windows, doors, curtains, outlets, switches, "
    "heaters, vents). Each thing once, even when several frames show it; two of the same (two monitors) are two "
    "things. Answer with JSON only, no other text: an array of objects with \"name\" (a short plain noun phrase an "
    "object-finding model can search for, like \"desk chair\"), \"frame\" (the number of the one frame that shows the "
    "whole thing most completely and sharply: nothing cut off by the picture's edge, least hidden behind other things, "
    "as close as possible), \"box\" ([left, top, right, bottom] round the thing in that frame, as fractions 0 to 1 of "
    "the picture's width and height), \"screen\" (true when it has a display, like a monitor, TV or laptop), "
    "\"moving\" (true when it moves on its own, like a fan's blades) and \"restsOn\" (\"floor\", \"furniture\", "
    "\"wall\" or \"ceiling\").")


SEEN_IN = (
    "These are the same {n} numbered frames from one phone video walking through a room. For each thing in this list, "
    "give every frame number where at least part of it can be seen: {things}. Answer with JSON only, no other text: an "
    "object whose keys are the ids (as strings) and whose values are arrays of frame numbers.")


def _cut(mask):
    """How much of a mask's box edge runs along the picture's edge: 0 when it's whole in the picture."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 1.0
    edge = mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()
    perimeter = 2 * ((cols.max() - cols.min() + 1) + (rows.max() - rows.min() + 1))
    return float(edge / perimeter)


def _slug(name, taken):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "thing"
    candidate, n = slug, 2
    while candidate in taken:
        candidate, n = f"{slug}-{n}", n + 1
    taken.add(candidate)
    return candidate


def _room(out):
    frame = json.loads((out / "room-frame.json").read_text())
    return _turn(*frame["quaternion"]), np.array([0.0, frame["floorHeight"], 0.0])


def _surface(out):
    """room-surface.ply as a ray-casting scene, in three.js axes (FLIP times OpenCV room space), with its arrays."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene, np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _rays(camera, u, v):
    """Rays in three.js axes from a frame's camera through pixels (u, v); camera z = 1, so a hit's t is its depth."""
    inside = np.stack([(u - camera["lens"][0, 2]) / camera["lens"][0, 0],
                       (v - camera["lens"][1, 2]) / camera["lens"][1, 1], np.ones_like(u, dtype=float)], -1)
    directions = (inside @ camera["turn"]) @ FLIP.T
    origin = FLIP @ (-camera["turn"].T @ camera["shift"])
    return np.hstack([np.repeat(origin[None], len(directions), 0), directions]).astype(np.float32)


def _spark(pictures, prompt, key):
    """One Muse Spark call with JPEG pictures (bytes) and a question; returns its answer text."""
    content = [{"type": "input_text", "text": prompt}] + [
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(p).decode()} for p in pictures]
    body = {"model": SPARK, "input": [{"type": "message", "role": "user", "content": content}]}
    request = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.loads(response.read().decode())
    return "".join(part.get("text", "") for item in reply.get("output", []) for part in item.get("content", []) or [])


def _mask(record, height, width):
    """One SAM 3.1 mask record as a frame-sized True/False picture."""
    import cv2
    from meta_sam_parser import decode_mask_to_raster

    mask, bounds = record.mask, record.bounds
    raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(mask.height, mask.width) > 0
    left, top = max(0, int(round(bounds.left))), max(0, int(round(bounds.top)))
    right, bottom = min(width, int(round(bounds.right))), min(height, int(round(bounds.bottom)))
    full = np.zeros((height, width), bool)
    if right > left and bottom > top:
        full[top:bottom, left:right] = cv2.resize(raster.astype(np.uint8), (right - left, bottom - top),
                                                  interpolation=cv2.INTER_NEAREST) > 0
    return full


def _box_overlap(mask, box):
    """IoU of a mask's own box with a box in pixels."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    a = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    inter = max(0, min(a[2], box[2]) - max(a[0], box[0])) * max(0, min(a[3], box[3]) - max(a[1], box[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return inter / union if union > 0 else 0.0


def pick(out):
    """Step 1: the things, each one's frame and outline, its measured points, and the photos with screens black."""
    import cv2
    from sklearn.cluster import DBSCAN

    out = Path(out)
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
    folder = out / "things"
    folder.mkdir(exist_ok=True)
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    started = time.perf_counter()

    pictures = None

    def thumbnails():
        nonlocal pictures
        if pictures is None:
            pictures = []
            for k in range(0, len(cameras), max(1, math.ceil(len(cameras) / MOST_FRAMES))):
                picture = cv2.imread(str(images / cameras[k]["name"]))
                scale = THUMB / max(picture.shape[:2])
                picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.rectangle(picture, (0, 0), (58, 30), (0, 0, 0), -1)
                cv2.putText(picture, str(k), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                pictures.append(cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        return pictures

    # Muse Spark's answers are kept, so picking again costs nothing and picks the same things: its list, then every
    # frame each thing shows in (so a blurry, cut-off or unoutlined first pick has others to fall back on)
    picks_file, seen_in_file = folder / "picks.json", folder / "seen-in.json"
    if picks_file.exists():
        listed = json.loads(picks_file.read_text())
    else:
        answer = _spark(thumbnails(), PICK.format(n=len(thumbnails())), key)
        listed = json.loads(answer[answer.index("["): answer.rindex("]") + 1])
        picks_file.write_text(json.dumps(listed, indent=2))
    if seen_in_file.exists():
        seen_in = json.loads(seen_in_file.read_text())
    else:
        names = json.dumps([{"id": k, "name": item.get("name")} for k, item in enumerate(listed)])
        answer = _spark(thumbnails(), SEEN_IN.format(n=len(thumbnails()), things=names), key)
        seen_in = json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
        seen_in_file.write_text(json.dumps(seen_in, indent=2))

    # SAM 3.1's answers are kept too: frame|phrase -> answer text
    answers_file = folder / "sam-answers.json"
    answers = json.loads(answers_file.read_text()) if answers_file.exists() else {}

    def outline(name, phrase):
        if f"{name}|{phrase}" not in answers:
            answers[f"{name}|{phrase}"] = _outline(images / name, phrase, key)
            answers_file.write_text(json.dumps(answers))
        return _records(answers[f"{name}|{phrase}"])

    import open3d as o3d
    from room_furniture import _sharpness

    level, lift = _room(out)
    scene, _, _ = _surface(out)
    sharp = _sharpness([cv2.imread(str(images / camera["name"]))[:, :, ::-1] for camera in cameras])

    def measure(camera, mask):
        """A mask's measured points: rays through its pixels onto the fused surface, the biggest piece of the hits."""
        rows, cols = np.nonzero(mask[::RAY_STEP, ::RAY_STEP])
        rays = _rays(camera, cols * RAY_STEP + 0.5, rows * RAY_STEP + 0.5)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ok = np.isfinite(hits)
        points = rays[ok, :3] + rays[ok, 3:] * hits[ok, None]
        if len(points) < 50:
            return None
        labels = DBSCAN(eps=SAME_THING, min_samples=10).fit_predict(points)
        if (labels >= 0).any():
            points = points[labels == np.bincount(labels[labels >= 0]).argmax()]
        return points @ level.T - lift

    def duplicate(upright):
        """The same physical thing as one already picked (under another name, or the same name twice)."""
        middle, extent = upright.mean(axis=0), np.ptp(upright, axis=0).max()
        for other in things:
            if (np.linalg.norm(np.array(other["middle"]) - middle) < DUPLICATE
                    and max(extent, other["extent"]) < 2 * min(extent, other["extent"])):
                return other["slug"]
        return None

    taken, things, skipped = set(), [], []
    screens = {}  # frame name -> its screens' mask
    for index, item in enumerate(listed):
        name = str(item.get("name", "")).strip().lower()
        first = item.get("frame")
        if not name or not isinstance(first, int) or not 0 <= first < len(cameras):
            skipped.append({"name": name, "why": "no usable frame"})
            continue
        if item.get("restsOn") == "wall":
            skipped.append({"name": name, "why": "on the wall: part of the walls' photos"})
            continue
        # Muse Spark's frame first (for something moving, the sharpest frame first: the owner's rule 2), then the
        # other frames it shows in, sharpest first
        others = [k for k in seen_in.get(str(index), []) if isinstance(k, int) and 0 <= k < len(cameras) and k != first]
        others.sort(key=lambda k: -sharp[k])
        tries = ([first] + others)[:MOST_TRIES]
        if item.get("moving"):
            tries.sort(key=lambda k: -sharp[k])
        candidates, why = [], "SAM 3.1 didn't outline it in any of its frames"
        for k in tries:
            camera = cameras[k]
            height, width = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
            masks = [_mask(record, height, width) for record in outline(camera["name"], name)]
            masks = [m for m in masks if m.sum() > 200]
            box = item.get("box") if k == first else None
            if isinstance(box, list) and len(box) == 4 and all(isinstance(b, (int, float)) for b in box):
                if max(box) > 1.5:  # the thumbnail's pixels, not fractions
                    shrink = THUMB / max(height, width)
                    box = [box[0] / (width * shrink), box[1] / (height * shrink), box[2] / (width * shrink),
                           box[3] / (height * shrink)]
                pixels = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
                masks.sort(key=lambda m: (-_box_overlap(m, pixels), -m.sum()))
            else:
                masks.sort(key=lambda m: -m.sum())
            for mask in masks:
                upright = measure(camera, mask)
                if upright is None:
                    why = "its outline hits too little of the surface"
                    continue
                same = duplicate(upright)
                if same:
                    why = f"the same thing as {same}"
                    continue
                candidates.append((camera, mask, upright, box, _cut(mask), k))
                break
            # Muse Spark's own frame, whole and sharp, is taken as it is; otherwise the others get a look too
            if candidates and k == first and candidates[-1][4] <= CUT and sharp[k] >= 0.9 and not item.get("moving"):
                break
        if not candidates:
            skipped.append({"name": name, "tried": [cameras[k]["name"] for k in tries], "why": why})
            continue
        # The frame that shows the most of it (its measured footprint), biggest and sharpest; cut-off ones count less.
        # SAM 3D fills in what's hidden, so how much of the thing it sees matters most
        footprint = [float(np.prod(np.sort(np.ptp(c[2], axis=0))[1:])) for c in candidates]
        area = [float(c[1].sum()) for c in candidates]

        def score(i):
            camera, mask, upright, box, cut, k = candidates[i]
            blur = min(1.0, sharp[k]) ** (2 if item.get("moving") else 1)
            return ((footprint[i] / max(footprint)) * (0.5 + 0.5 * area[i] / max(area)) * blur
                    * (0.7 if cut > CUT else 1.0) * (1.1 if k == first else 1.0))

        chosen = candidates[max(range(len(candidates)), key=score)]
        camera, mask, upright, box, cut, k = chosen
        if item.get("screen") and camera["name"] not in screens:
            height, width = mask.shape
            found = [_mask(record, height, width) for record in outline(camera["name"], SCREEN)]
            screens[camera["name"]] = np.any(found, axis=0) if found else np.zeros((height, width), bool)
        slug = _slug(name, taken)
        own = folder / slug
        own.mkdir(exist_ok=True)
        cv2.imwrite(str(own / "outline.png"), mask.astype(np.uint8) * 255)
        np.save(own / "measured.npy", upright.astype(np.float32))
        things.append({"name": name, "slug": slug, "frame": camera["name"], "box": box,
                       "screen": bool(item.get("screen")), "moving": bool(item.get("moving")),
                       "restsOn": item.get("restsOn", "floor"), "outlinePixels": int(mask.sum()),
                       "cutByEdge": round(float(cut), 3), "sharpness": round(float(sharp[k]), 2),
                       "measuredPoints": len(upright),
                       "measuredSize": [round(float(v), 3) for v in np.ptp(upright, axis=0)],
                       "middle": [round(float(v), 4) for v in upright.mean(axis=0)],
                       "extent": round(float(np.ptp(upright, axis=0).max()), 4)})

    # Each used frame's photos with its screens painted black (the owner's rule 2): the half-size one SAM 3D builds
    # from (the cameras' own size) and the full-size one the seen sides are painted from
    photos = folder / "photos"
    photos.mkdir(exist_ok=True)
    for name in sorted({thing["frame"] for thing in things}):
        stem = Path(name).stem
        half = cv2.imread(str(images / name))
        full = cv2.imread(str(out / "frames" / f"{stem}.jpg"))
        if full is None:
            full = half
        black = screens.get(name)
        if black is not None and black.any():
            black = cv2.dilate(black.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            half[black] = (12, 12, 12)
            big = cv2.resize(black.astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            full[big] = (12, 12, 12)
        cv2.imwrite(str(photos / f"{stem}.png"), half)
        cv2.imwrite(str(photos / f"{stem}-full.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 95])

    summary = {"things": things, "skipped": skipped, "framesUsed": len({t["frame"] for t in things}),
               "seconds": round(time.perf_counter() - started, 1)}
    (folder / "things.json").write_text(json.dumps(summary, indent=2))
    return summary


def _pointmap(out, camera):
    """The room's own depth for every pixel of a frame, as SAM 3D takes a pointmap: H x W x 3 in PyTorch3D's camera
    axes (x left, y up, z ahead), metres, NaN where the ray meets nothing. Given this, SAM 3D lays its objects out in
    this camera at the room's real scale instead of guessing depth."""
    import open3d as o3d

    scene, _, _ = _surface(out)
    height, width = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
    rows, cols = np.mgrid[0:height, 0:width]
    u, v = cols.ravel() + 0.5, rows.ravel() + 0.5
    depth = scene.cast_rays(o3d.core.Tensor(_rays(camera, u, v)))["t_hit"].numpy()
    depth = np.where(np.isfinite(depth), depth, np.nan)
    x = (u - camera["lens"][0, 2]) / camera["lens"][0, 0] * depth
    y = (v - camera["lens"][1, 2]) / camera["lens"][1, 1] * depth
    return np.stack([-x, -y, depth], -1).reshape(height, width, 3).astype(np.float16)


def build(out, only=None, runner="local"):
    """Step 2: SAM 3D Objects, one call per frame with every outline picked in it. `runner` "local" runs Meta's model
    on this PC's own card (worker/sam3d_local.py in WSL: free, one model part on the card at a time), "modal" on a
    Modal GPU (worker/sam3d_modal.py: credits, but Modal wants a card on file); both get HF_TOKEN for the gated weights
    and the room's own depth, so each object comes back in the frame's camera in metres. "fal" uses fal.ai's hosted
    copy (paid, FAL_KEY). Files are kept, so building again only does frames that have none yet. `only`: some
    things' slugs, to build just their frames (a first check)."""
    out = Path(out)
    load_env()
    folder = out / "things"
    things = json.loads((folder / "things.json").read_text())["things"]
    frames = {}
    for thing in things:
        frames.setdefault(thing["frame"], []).append(thing)
    if only:
        frames = {name: group for name, group in frames.items() if any(t["slug"] in only for t in group)}
    started = time.perf_counter()
    if runner == "local":
        done = _build_locally(out, folder, frames)
    elif runner == "modal":
        done = _build_on_modal(out, folder, frames)
    else:
        done = _build_on_fal(folder, frames)
    return {"calls": sum(1 for d in done if d.get("called")), "frames": done,
            "seconds": round(time.perf_counter() - started, 1)}


def _wsl(path):
    """A Windows path as WSL sees it (C:\\x -> /mnt/c/x)."""
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def _save_object(own, vertices, faces, colours, layout):
    """SAM 3D's object, already in the frame's OpenCV camera, as sam3d.glb; its own layout kept for the record only,
    under a name place() doesn't apply again."""
    import trimesh

    trimesh.Trimesh(vertices, faces, vertex_colors=colours, process=False).export(own / "sam3d.glb")
    (own / "sam3d-pose.json").write_text(json.dumps({"inCamera": "opencv", "samLayout": layout}, indent=2))


def _build_locally(out, folder, frames):
    """Every frame's things through sam3d_local.py in WSL, one run for all of them (the model loads once)."""
    import subprocess

    cameras = {camera["name"]: camera for camera in _cameras(out / "splat-data")}
    work = folder / "local"
    (work / "pointmaps").mkdir(parents=True, exist_ok=True)
    jobs, todo = [], {}
    for name, group in frames.items():
        stem = Path(name).stem
        if (folder / f"sam3d-{stem}.json").exists():
            continue
        pointmap = work / "pointmaps" / f"{stem}.npz"
        if not pointmap.exists():
            np.savez_compressed(pointmap, pointmap=_pointmap(out, cameras[name]))
        jobs.append({"photo": _wsl(folder / "photos" / f"{stem}.png"),
                     "outlines": [_wsl(folder / t["slug"] / "outline.png") for t in group],
                     "pointmap": _wsl(pointmap), "out": _wsl(work / stem)})
        todo[name] = group
    if jobs:
        (work / "jobs.json").write_text(json.dumps({"jobs": jobs, "seed": SEED}, indent=2))
        command = ('export MAMBA_ROOT_PREFIX=~/sam3d/mamba && eval "$(~/sam3d/bin/micromamba shell hook -s bash)" && '
                   f"micromamba activate sam3d && python {_wsl(Path(__file__).parent / 'sam3d_local.py')} "
                   f"{_wsl(work / 'jobs.json')}")
        subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], check=False)
    done = []
    for name, group in todo.items():
        stem = Path(name).stem
        summary = []
        for n, thing in enumerate(group):
            result_file = work / stem / f"{n}.json"
            if not result_file.exists():
                summary.append({"thing": thing["slug"], "error": "no result (the run stopped before it)"})
                continue
            result = json.loads(result_file.read_text())
            if "error" in result:
                summary.append({"thing": thing["slug"], **result})
                continue
            arrays = np.load(work / stem / f"{n}.npz")
            _save_object(folder / thing["slug"], arrays["vertices"], arrays["faces"], arrays["colours"], result["layout"])
            summary.append({"thing": thing["slug"], "vertices": result["vertices"], "seconds": result["seconds"],
                            "peakGb": result["peakGb"]})
        if all("error" not in item for item in summary):
            (folder / f"sam3d-{stem}.json").write_text(json.dumps({"runner": "local", "things": summary}, indent=2))
        done.append({"frame": name, "things": summary, "called": True})
    return done


def _build_on_modal(out, folder, frames):
    import io

    import modal

    from sam3d_modal import Sam3d, app

    cameras = {camera["name"]: camera for camera in _cameras(out / "splat-data")}
    done = []
    with modal.enable_output(), app.run():
        sam = Sam3d()
        for name, group in frames.items():
            stem = Path(name).stem
            result_file = folder / f"sam3d-{stem}.json"
            if result_file.exists():
                done.append({"frame": name, "things": [t["slug"] for t in group], "called": False})
                continue
            buffer = io.BytesIO()
            np.savez_compressed(buffer, pointmap=_pointmap(out, cameras[name]))
            results = sam.build.remote((folder / "photos" / f"{stem}.png").read_bytes(),
                                       [(folder / t["slug"] / "outline.png").read_bytes() for t in group],
                                       buffer.getvalue(), SEED)
            summary = []
            for thing, result in zip(group, results):
                own = folder / thing["slug"]
                if "error" in result:
                    summary.append({"thing": thing["slug"], "error": result["error"], "seconds": result["seconds"]})
                    continue
                _save_object(own, np.frombuffer(result["vertices"], np.float32).reshape(-1, 3),
                             np.frombuffer(result["faces"], np.int32).reshape(-1, 3),
                             np.frombuffer(result["colours"], np.uint8).reshape(-1, 3),
                             {k: result[k] for k in ("rotation", "translation", "scale")})
                summary.append({"thing": thing["slug"], "vertices": len(vertices), "seconds": result["seconds"]})
            result_file.write_text(json.dumps({"runner": "modal", "things": summary}, indent=2))
            done.append({"frame": name, "things": summary, "called": True})
    return done


def _build_on_fal(folder, frames):
    from concurrent.futures import ThreadPoolExecutor

    import fal_client

    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY isn't set: make a key at fal.ai (Dashboard, Keys) and put FAL_KEY=... in .env")

    def one(name, group):
        stem = Path(name).stem
        result_file = folder / f"sam3d-{stem}.json"
        fresh = not result_file.exists()  # a new call's files replace whatever is there
        try:
            if fresh:
                image_url = fal_client.upload_file(folder / "photos" / f"{stem}.png")
                mask_urls = [fal_client.upload_file(folder / thing["slug"] / "outline.png") for thing in group]
                result = fal_client.subscribe(FAL_APP, arguments={
                    "image_url": image_url, "mask_urls": mask_urls, "seed": SEED, "enable_safety_checker": False})
                result_file.write_text(json.dumps({"things": [t["slug"] for t in group], "result": result}, indent=2))
            result = json.loads(result_file.read_text())["result"]
            glbs = result.get("individual_glbs") or ([result["model_glb"]] if result.get("model_glb") else [])
            splats = result.get("individual_splats") or [result["gaussian_splat"]]
            for k, thing in enumerate(group):
                own = folder / thing["slug"]
                for kind, files in (("glb", glbs), ("ply", splats)):
                    target = own / f"sam3d.{kind}"
                    if k < len(files) and (fresh or not target.exists()):
                        urllib.request.urlretrieve(files[k]["url"], target)
                metadata = [m for m in result.get("metadata", []) if m.get("object_index") == k]
                (own / "sam3d-pose.json").write_text(json.dumps(metadata[0] if metadata else {}, indent=2))
            return {"frame": name, "things": [t["slug"] for t in group], "called": fresh}
        except Exception as error:  # one frame failing shouldn't stop the others
            return {"frame": name, "things": [t["slug"] for t in group], "error": f"{type(error).__name__}: {error}"[:500]}

    # Two calls at a time: a new fal account's limit
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(lambda item: one(*item), frames.items()))


def _load_object(own):
    """SAM 3D's object: vertices, triangles and colours 0-1, as its GLB has them (vertex colours, or the texture's
    colour at each vertex)."""
    import trimesh

    mesh = trimesh.load(own / "sam3d.glb", force="mesh", process=False)
    visual = mesh.visual
    if hasattr(visual, "to_color") and getattr(visual, "kind", None) == "texture":
        visual = visual.to_color()
    colours = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64), colours


def _layouts(vertices, pose):
    """Ways SAM 3D's object might sit in its camera: as its file has it, or moved by its layout (fal's docs say the
    quaternion is x y z w; w x y z is tried too), with or without the inverse of its camera pose. The outline fit
    picks the one that's right."""
    layouts = {"as built": vertices}
    try:
        q = np.ravel(pose["rotation"])[:4].astype(float)
        shift = np.ravel(pose["translation"])[:3].astype(float)
        stretch = np.ravel(pose["scale"])[:3].astype(float)
    except (KeyError, TypeError, ValueError, AttributeError):
        return layouts
    camera_pose = np.asarray(pose.get("camera_pose") or [], float)
    for order, turn in (("xyzw", _turn(q[3], q[0], q[1], q[2])), ("wxyz", _turn(q[0], q[1], q[2], q[3]))):
        moved = (vertices * stretch) @ turn.T + shift
        layouts[f"its pose ({order})"] = moved
        if camera_pose.shape == (4, 4):
            back = np.linalg.inv(camera_pose)
            layouts[f"its pose ({order}) from its camera"] = moved @ back[:3, :3].T + back[:3, 3]
    return layouts


def _silhouette(vertices_cam, triangles, lens, size):
    """The object's outline as the camera sees it (vertices in OpenCV camera axes), at `size` (height, width): a ray
    per pixel from the camera, True where it hits. (OpenCV's fillPoly can't do this: where triangles overlap, as the
    front and back of any closed object do, it leaves holes.)"""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_cam.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.uint32)))
    rows, cols = np.mgrid[0:size[0], 0:size[1]]
    directions = np.stack([(cols.ravel() + 0.5 - lens[0, 2]) / lens[0, 0], (rows.ravel() + 0.5 - lens[1, 2]) / lens[1, 1],
                           np.ones(rows.size)], -1)
    rays = np.hstack([np.zeros_like(directions), directions]).astype(np.float32)
    return np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()).reshape(size)


def _iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def _fit(vertices, triangles, pose, camera, outline):
    """SAM 3D's object in the frame's OpenCV camera axes, in the layout and axes whose outline best covers SAM 3.1's,
    up to its size (the camera can't tell a big far thing from a small near one). Returns vertices, IoU, the choice."""
    import cv2

    height, width = outline.shape
    scale = FIT_SIZE / max(height, width)
    size = (int(round(height * scale)), int(round(width * scale)))
    small = cv2.resize(outline.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST) > 0
    lens = camera["lens"].copy()
    lens[:2] *= scale
    best = (None, -1.0, None)
    for layout, placed in _layouts(vertices, pose or {}).items():
        for axes, swap in AXES.items():
            candidate = placed @ swap.T
            if np.median(candidate[:, 2]) <= 0:
                continue
            score = _iou(_silhouette(candidate, triangles, lens, size), small)
            if score > best[1]:
                best = (candidate, score, f"{layout}, {axes} axes")
            if best[1] >= GOOD_FIT:
                return best
    return best


def _visible(vertices_cam, triangles, lens, size):
    """Which vertices the camera sees (in front, not behind the object's own surface), by a depth map of the object."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_cam.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.uint32)))
    z = vertices_cam[:, 2]
    u = lens[0, 0] * vertices_cam[:, 0] / np.maximum(z, 1e-6) + lens[0, 2]
    v = lens[1, 1] * vertices_cam[:, 1] / np.maximum(z, 1e-6) + lens[1, 2]
    inside = (z > 0.01) & (u >= 0) & (u < size[1]) & (v >= 0) & (v < size[0])
    which = np.nonzero(inside)[0]
    directions = np.stack([(u[which] - lens[0, 2]) / lens[0, 0], (v[which] - lens[1, 2]) / lens[1, 1],
                           np.ones(len(which))], -1)
    rays = np.hstack([np.zeros_like(directions), directions]).astype(np.float32)
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    seen = np.zeros(len(vertices_cam), bool)
    seen[which] = z[which] <= depth + SEEN_HIDDEN
    return seen, u, v


def _to_measured(vertices_cam, triangles, camera, measured_cam):
    """Scales the object about the camera (which keeps its outline) so its seen surface is as far away as the measured
    points, then fits it to them (ICP, rigid). Returns the moved vertices and the fit's mean gap in metres."""
    import open3d as o3d

    height, width = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
    seen, _, _ = _visible(vertices_cam, triangles, camera["lens"], (height, width))
    if seen.sum() < 20:
        seen = np.ones(len(vertices_cam), bool)
    ratio = np.median(measured_cam[:, 2]) / max(np.median(vertices_cam[seen, 2]), 1e-6)
    scaled = vertices_cam * ratio
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scaled[seen]))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(measured_cam))
    reach = max(0.05, 0.25 * float(np.linalg.norm(measured_cam.max(axis=0) - measured_cam.min(axis=0))))
    fit = o3d.pipelines.registration.registration_icp(
        source, target, reach, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    moved = scaled @ fit.transformation[:3, :3].T + fit.transformation[:3, 3]
    gap = np.asarray(target.compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moved[seen]))))
    return moved, float(np.mean(gap)) if len(gap) else float("nan"), float(ratio)


def _paint(vertices_cam, triangles, colours, camera, outline, photo, real_depth):
    """The sides the photo saw get the photo's own colours (the full-size photo); the rest keep SAM 3D's, matched to
    the photo's exposure so there's no seam. Returns colours 0-1 and which vertices came from the photo."""
    import cv2

    height, width = outline.shape
    seen, u, v = _visible(vertices_cam, triangles, camera["lens"], (height, width))
    inset = cv2.erode(outline.astype(np.uint8), np.ones((2 * OUTLINE_INSET + 1,) * 2, np.uint8)) > 0
    which = np.nonzero(seen)[0]
    px, py = u[which].astype(int), v[which].astype(int)
    keep = inset[py, px]
    z = vertices_cam[which, 2]
    keep &= np.abs(z - real_depth[py, px]) < TRUTH + TRUTH_PER_METRE * z
    # Facing the camera: the vertex normals from the triangles around each vertex
    normals = np.zeros_like(vertices_cam)
    corners = vertices_cam[triangles]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for k in range(3):
        np.add.at(normals, triangles[:, k], faces)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    to_camera = -vertices_cam[which]
    facing = np.abs(np.sum(normals[which] * to_camera, axis=1)) / np.maximum(np.linalg.norm(to_camera, axis=1), 1e-9)
    keep &= facing > SEEN_FACING
    which = which[keep]
    painted = np.zeros(len(vertices_cam), bool)
    painted[which] = True
    result = colours.copy()
    if len(which):
        grow = photo.shape[0] / height  # the photo is the full-size frame: the same camera, scaled
        picture = photo[:, :, ::-1].astype(np.float32) / 255.0
        from_photo = _bilinear(picture, u[which] * grow, v[which] * grow)
        gain = np.median(np.maximum(from_photo, 0.02) / np.maximum(colours[which], 0.02), axis=0)
        result = np.clip(colours * np.clip(gain, 0.5, 2.0), 0.0, 1.0)
        result[which] = from_photo
    return result, painted


def _box(upright):
    """A thing's box in the upright room, along its own longest way across the floor: what the viewer grabs it by."""
    flat = upright[:, [0, 2]] - upright[:, [0, 2]].mean(axis=0)
    _, _, axes = np.linalg.svd(flat, full_matrices=False)
    yaw = math.atan2(axes[0][1], axes[0][0])
    along = np.array([math.cos(yaw), math.sin(yaw)])
    across = np.array([-math.sin(yaw), math.cos(yaw)])
    a, b = flat @ along, flat @ across
    middle = upright[:, [0, 2]].mean(axis=0) + along * (a.max() + a.min()) / 2 + across * (b.max() + b.min()) / 2
    low, high = float(upright[:, 1].min()), float(upright[:, 1].max())
    return {"centre": [round(float(middle[0]), 4), round((low + high) / 2, 4), round(float(middle[1]), 4)],
            "size": [round(float(a.max() - a.min()), 4), round(high - low, 4), round(float(b.max() - b.min()), 4)],
            "yawDegrees": round(math.degrees(yaw), 2)}


def _grey_box(own, rests_on):
    """The owner's rule 3: where SAM 3D fails, a grey box of the thing's measured size stands in. The measured points
    are only its seen face, so the box is at least 20 cm deep, and reaches the floor when the thing stands on it."""
    measured = own / "measured.npy"
    if not measured.exists():
        return None
    box = _box(np.load(measured).astype(np.float64))
    box["size"][2] = max(box["size"][2], 0.2)
    if rests_on == "floor":
        top = box["centre"][1] + box["size"][1] / 2
        box["centre"][1], box["size"][1] = round(top / 2, 4), round(top, 4)
    return box


def _write_thing_ply(path, points, colours, triangles, seen):
    """Binary PLY like room_furniture._write_ply, plus a `seen` byte per vertex: 1 where the colour came from the photo,
    0 where it's SAM 3D's (what the video never saw), so the viewer can show which is which."""
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("seen", "u1")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    data["seen"] = seen.astype(np.uint8)
    faces = np.empty(len(triangles), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    faces["n"] = 3
    faces["a"], faces["b"], faces["c"] = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar seen\n"
              f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
        f.write(faces.tobytes())


def place(out):
    """Steps 3 and 4: every built thing in the room at its measured spot and size, its seen sides painted from the
    photo. Writes room-thing-<slug>.ply and room-things.json."""
    import cv2
    import open3d as o3d

    out = Path(out)
    folder = out / "things"
    things = json.loads((folder / "things.json").read_text())["things"]
    cameras = {camera["name"]: camera for camera in _cameras(out / "splat-data")}
    level, lift = _room(out)
    scene, _, _ = _surface(out)
    placed, failed, merged = [], [], []
    for thing in things:
        own = folder / thing["slug"]
        if not (own / "sam3d.glb").exists():
            failed.append({"name": thing["name"], "slug": thing["slug"], "why": "not built yet"})
            continue
        camera = cameras[thing["frame"]]
        outline = cv2.imread(str(own / "outline.png"), cv2.IMREAD_GRAYSCALE) > 127
        pose_file = own / "sam3d-pose.json"
        pose = json.loads(pose_file.read_text()) if pose_file.exists() else None
        vertices, triangles, colours = _load_object(own)
        in_camera, coverage, choice = _fit(vertices, triangles, pose, camera, outline)
        if in_camera is None or coverage < FAILED_FIT:
            failed.append({"name": thing["name"], "slug": thing["slug"], "why": f"outline fit {coverage:.2f}",
                           "fit": choice, "box": _grey_box(own, thing.get("restsOn"))})
            continue
        # The measured points in the camera's OpenCV axes
        measured = np.load(own / "measured.npy").astype(np.float64)
        measured_cv = (measured + lift) @ level @ FLIP  # upright -> three.js axes -> OpenCV (FLIP is its own inverse)
        measured_cam = measured_cv @ camera["turn"].T + camera["shift"]
        moved, gap, ratio = _to_measured(in_camera, triangles, camera, measured_cam)

        # Paint from the full-size photo, where the real surface (its depth, from the same camera) agrees
        height, width = outline.shape
        rows, cols = np.mgrid[0:height, 0:width]
        real = scene.cast_rays(o3d.core.Tensor(_rays(camera, cols.ravel() + 0.5, rows.ravel() + 0.5)))["t_hit"].numpy()
        real_depth = np.where(np.isfinite(real), real, 1e3).reshape(height, width)
        photo = cv2.imread(str(folder / "photos" / f"{Path(thing['frame']).stem}-full.jpg"))
        painted_colours, painted = _paint(moved, triangles, colours, camera, outline, photo, real_depth)

        # Camera -> OpenCV room -> three.js axes -> upright room
        room_cv = (moved - camera["shift"]) @ camera["turn"]
        upright = (room_cv @ FLIP.T) @ level.T - lift
        if thing.get("restsOn") == "floor" or upright[:, 1].min() < ON_FLOOR:
            upright[:, 1] -= upright[:, 1].min()
        colours_255 = np.clip(painted_colours * 255 + 0.5, 0, 255).astype(np.uint8)
        _write_thing_ply(out / f"room-thing-{thing['slug']}.ply", upright, colours_255, triangles, painted)
        merged.append((upright, colours_255, triangles))

        placed.append({
            "name": thing["name"], "slug": thing["slug"], "file": f"room-thing-{thing['slug']}.ply", **_box(upright),
            "seenShare": round(float(painted.mean()), 3), "screen": thing["screen"], "moving": thing["moving"],
            "outlineFit": round(coverage, 3), "fit": choice, "scaledBy": round(ratio, 4), "measuredGapCm": round(100 * gap, 1),
        })
    summary = {"things": placed, "failed": failed}
    (out / "room-things.json").write_text(json.dumps(summary, indent=2))
    # All of them in one mesh, so the room checks can draw the Rebuilt look (room_checks.WalkRoom's `furniture`)
    if merged:
        offsets = np.cumsum([0] + [len(points) for points, _, _ in merged[:-1]])
        _write_ply(out / "room-things-all.ply", np.vstack([points for points, _, _ in merged]),
                   np.vstack([colours for _, colours, _ in merged]),
                   np.vstack([triangles + offset for (_, _, triangles), offset in zip(merged, offsets)]))
    return summary


if __name__ == "__main__":
    steps = {"pick": pick, "build": build, "place": place}
    if len(sys.argv) < 3 or sys.argv[1] not in steps:
        sys.exit("usage: python worker/rebuild_things.py pick|build|place <request folder> "
                 "[build: thing slugs] [--runner local|modal|fal]")
    if sys.argv[1] == "build":
        extra = sys.argv[3:]
        runner = "local"
        if "--runner" in extra:
            at = extra.index("--runner")
            runner = extra[at + 1]
            extra = extra[:at] + extra[at + 2:]
        print(json.dumps(build(sys.argv[2], set(extra) or None, runner), indent=2))
    else:
        print(json.dumps(steps[sys.argv[1]](sys.argv[2]), indent=2))
