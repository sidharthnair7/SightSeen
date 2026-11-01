"""The room's furniture as clean blocks, for the Walk look: SAM 3.1 (Meta Model API) outlines each thing in the frames
by name, and the outlines are lifted onto the room's fused surface through each frame's camera.

SAM 3.1 works in 2D, one frame at a time. Every frame here has a camera from the 3D step (COLMAP's, or MapAnything's
lined up with its neighbours), so a ray from the camera through an outlined pixel hits the surface where that thing
is. The hits from all frames, grouped (DBSCAN) so separate things stay apart, give each thing a block in the
upright room: its real size and position, along the walls' directions, in the colour SAM outlined.

Needs the request folder's splat-data/ (frames and cameras), room-surface.ply, room-frame.json and room-shell.json,
and MODEL_API_KEY (in .env or the environment). Writes room-objects.json.
    python worker/room_objects.py <request folder>
Costs one image call per frame per name: $2.50 per 1,000, so 25 frames x 6 names is about 38 cents.
"""

import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from sam_check import API, load_env

NAMES = ["bed", "desk", "chair", "wardrobe", "door", "window"]
EVERY = 3  # every 3rd placed frame: the turns between them are small enough that each thing is seen several times
PIXEL_STEP = 2  # rays through every 2nd outlined pixel each way: plenty of hits, a quarter of the work
GROUP = 0.10  # metres: hits closer than this belong to the same thing (DBSCAN's eps)
MIN_HITS = 150  # a group needs this many hits to be a thing and not a stray outline
SHARE_OF_BIGGEST = 0.2  # and at least this share of the biggest group with the same name
MOST = {"chair": 2, "door": 2, "window": 2}  # how many of each a room usually has; everything else, one
STANDS_ON_FLOOR = 0.6  # metres: furniture whose seen part starts below this goes down to the floor
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes, as everywhere in the worker


def _turn(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _cameras(splat_data):
    """Each frame's lens and world-to-camera turn and shift, from splat-data's COLMAP text files."""
    sparse = splat_data / "sparse" / "0"
    lenses = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            number, _, width, height, fx, fy, cx, cy = line.split()
            lenses[number] = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]])
    frames = []
    for line in (sparse / "images.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 10 and not line.startswith("#"):
            qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
            frames.append({"name": parts[9], "lens": lenses[parts[8]], "turn": _turn(qw, qx, qy, qz),
                           "shift": np.array([tx, ty, tz])})
    return sorted(frames, key=lambda frame: frame["name"])


def _records(text):
    """SAM 3.1's answer text as its mask records, through Meta's own parser."""
    from meta_sam_parser import image_segmentation_format

    snapshots = image_segmentation_format().create_parser().push(text + "\n")
    return [record for record in (snapshots[-1].records if snapshots else ()) if record.kind == "mask"]


def _outline(image, name, key):
    """One SAM 3.1 call: every `name` in the image, as SAM's answer text (one line, a box and a mask per thing)."""
    body = {
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": name},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()},
        ]}],
    }
    request = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                reply = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise RuntimeError(f"SAM 3.1 said {error.code}: {error.read().decode()[:300]}") from error
            time.sleep(2)
    return "".join(part.get("text", "") for item in reply.get("output", []) for part in item.get("content", []) or [])


def _pixels(record):
    """The outlined pixels of one mask, in the frame's own pixel coordinates (the mask covers only its box)."""
    from meta_sam_parser import decode_mask_to_raster

    mask, bounds = record.mask, record.bounds
    raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(mask.height, mask.width)
    rows, cols = np.nonzero(raster[::PIXEL_STEP, ::PIXEL_STEP])
    rows, cols = rows * PIXEL_STEP, cols * PIXEL_STEP
    u = bounds.left + (cols + 0.5) * (bounds.right - bounds.left) / mask.width
    v = bounds.top + (rows + 0.5) * (bounds.bottom - bounds.top) / mask.height
    return u, v


def find_objects(out):
    import cv2
    import open3d as o3d

    out = Path(out)
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = _turn(*frame["quaternion"])
    cameras = _cameras(out / "splat-data")[::EVERY]
    images = out / "splat-data" / "images"

    # SAM's answers are kept in sam-outlines.json, so building the blocks again doesn't pay for the same calls twice
    saved_file = out / "sam-outlines.json"
    saved = json.loads(saved_file.read_text()) if saved_file.exists() else {}
    jobs = [(camera, name) for camera in cameras for name in NAMES]
    missing = [job for job in jobs if f"{job[0]['name']}|{job[1]}" not in saved]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (camera, name), text in zip(missing, pool.map(lambda job: _outline(images / job[0]["name"], job[1], key), missing)):
            saved[f"{camera['name']}|{name}"] = text
    calls = time.perf_counter() - started
    saved_file.write_text(json.dumps(saved))
    found = [_records(saved[f"{camera['name']}|{name}"]) for camera, name in jobs]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))))
    hits = {name: [] for name in NAMES}
    colours = {name: [] for name in NAMES}
    sightings = {name: 0 for name in NAMES}
    for (camera, name), records in zip(jobs, found):
        picture = cv2.imread(str(images / camera["name"]))[:, :, ::-1]
        for record in records:
            u, v = _pixels(record)
            if len(u) == 0:
                continue
            sightings[name] += 1
            rays_in_camera = np.stack([u, v, np.ones_like(u)]) .T @ np.linalg.inv(camera["lens"]).T
            directions = FLIP @ (camera["turn"].T @ rays_in_camera.T)  # into the room, three.js axes
            directions = (directions / np.linalg.norm(directions, axis=0)).T
            origin = FLIP @ (-camera["turn"].T @ camera["shift"])
            rays = np.hstack([np.repeat(origin[None], len(directions), axis=0), directions]).astype(np.float32)
            distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            hit = np.isfinite(distance)
            points = origin + directions[hit] * distance[hit, None]
            hits[name].append(points @ level.T - np.array([0.0, frame["floorHeight"], 0.0]))
            ys = np.clip(v[hit].astype(int), 0, picture.shape[0] - 1)
            xs = np.clip(u[hit].astype(int), 0, picture.shape[1] - 1)
            colours[name].append(picture[ys, xs])

    yaw = math.radians(shell["yawDegrees"])
    across, along = np.array([math.cos(yaw), 0.0, math.sin(yaw)]), np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
    things = []
    for name in NAMES:
        if not hits[name]:
            continue
        points, colour = np.concatenate(hits[name]), np.concatenate(colours[name])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        groups = np.asarray(cloud.cluster_dbscan(eps=GROUP, min_points=10))
        ranked = sorted(set(groups) - {-1}, key=lambda g: -(groups == g).sum())
        biggest = (groups == ranked[0]).sum() if ranked else 0
        for group in ranked[:MOST.get(name, 1)]:
            members = groups == group
            if members.sum() < max(MIN_HITS, SHARE_OF_BIGGEST * biggest):
                break
            part = points[members]
            x, z, y = part @ across, part @ along, part[:, 1]
            low = np.array([np.percentile(x, 3), np.percentile(y, 3), np.percentile(z, 3)])
            high = np.array([np.percentile(x, 97), np.percentile(y, 97), np.percentile(z, 97)])
            if name not in ("door", "window") and low[1] < STANDS_ON_FLOOR:
                low[1] = 0.0  # furniture stands on the floor; its sides and legs are usually hidden from the camera
            middle = (low + high) / 2
            things.append({
                "name": name,
                "centre": [round(float(v), 4) for v in across * middle[0] + along * middle[2] + [0, middle[1], 0]],
                "size": [round(float(v), 3) for v in high - low],  # along the walls' x, up, along their z
                "yawDegrees": shell["yawDegrees"],
                "colour": [int(c) for c in np.median(colour[members], axis=0)],
                "hits": int(members.sum()),
                "_box": (low, high),
            })
    # A wardrobe's own doors come back as "door" too; a door inside a wardrobe's footprint is the wardrobe's
    wardrobes = [thing["_box"] for thing in things if thing["name"] == "wardrobe"]
    things = [thing for thing in things if thing["name"] != "door" or not any(
        np.all((thing["_box"][0] + thing["_box"][1])[[0, 2]] / 2 > low[[0, 2]] - 0.1)
        and np.all((thing["_box"][0] + thing["_box"][1])[[0, 2]] / 2 < high[[0, 2]] + 0.1) for low, high in wardrobes)]
    for thing in things:
        del thing["_box"]
    result = {"things": things, "frames": len(cameras), "names": NAMES, "calls": len(jobs),
              "sightings": sightings, "seconds": round(calls, 1)}
    (out / "room-objects.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_objects.py <request folder>")
    print(json.dumps(find_objects(sys.argv[1]), indent=2))
