"""SightSeen's room checks: the owner's two tests, measured on a built room instead of judged by eye.

Test 1, "all the stuff from the video is in the 3D room". Muse Spark (Meta Model API) looks at a spread of the video's
frames and lists every thing it sees. SAM 3.1 outlines each one in the frames. A ray from each frame's camera through
each outlined pixel goes into the Walk room exactly as the viewer draws it (the photo walls plus the furniture mesh).
A thing is in the room when most of its outlined pixels land on something real: the furniture mesh for things
standing in the room, a filmed patch of wall for things on the walls. It's missing when the rays go through it to the
wall behind, or to wall that was never filmed.

Test 2, "the room looks like the room in the video". The Walk room is rebuilt without every HOLD_OUT-th frame, drawn
from those held-back frames' own cameras, and scored against the real frames (PSNR and SSIM, the usual picture
scores; frames whose camera COLMAP solved, so a bad camera can't be blamed on the room). Then, from the middle of the
room, where the phone never stood, the share of the view that's torn (the inside of the furniture mesh showing
through a gap) or never filmed (wall fill).

Writes room-checks.json and pictures in checks/ in the request folder.
    python worker/room_checks.py <request folder>
Costs: one Muse Spark call, and SAM 3.1 calls for names it hasn't been asked before (every 6th frame, $2.50 per 1,000).
"""

import base64
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from room_furniture import build_furniture
from room_objects import _cameras, _outline, _pixels, _records
from room_textures import FACES, PAD, _colmap_solved, make_textures
from sam_check import API, load_env

BRUSH = Path(__file__).parent / "vendor" / "brush" / "brush_app.exe"
SPLAT_STEPS, SPLAT_MAX, SPLAT_SH = 15_000, 500_000, 1  # the same as reconstruct.py's, for a fair held-out build
WSL_PYTHON = "~/sightseen-gs/.venv/bin/python"  # the gsplat venv in WSL that render_splats.py runs in

HOLD_OUT = 6  # Test 2 holds back every 6th frame (starting with the 4th), about 13 of take 3's 76
SPARK = "muse-spark-1.3"
INVENTORY_FRAMES = 12  # frames Muse Spark looks at, spread over the video
SAM_EVERY = 6  # frames SAM 3.1 outlines each name in (every 3rd were outlined before, so these are often cached)
MOST_NAMES = 25
PRESENT = 0.7  # a thing is in the room when this share of its outlined pixels land on something real
ROOM_PARTS = {"wall", "walls", "floor", "ceiling", "room", "carpet floor", "flooring"}
EYE = 1.4  # metres: eye height for the views from the middle of the room
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes, as everywhere in the worker


def _turn(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class WalkRoom:
    """The Walk room as the viewer draws it, for casting rays: the shell box 1 cm outside the walls with its photo
    faces, and the furniture mesh with its photo colours, in the upright room (metres, floor at 0)."""

    def __init__(self, folder):
        import open3d as o3d

        folder = Path(folder)
        self.frame = json.loads((folder / "room-frame.json").read_text())
        self.shell = json.loads((folder / "room-shell.json").read_text())
        self.level = _turn(*self.frame["quaternion"])
        self.lift = np.array([0.0, self.frame["floorHeight"], 0.0])
        yaw = math.radians(self.shell["yawDegrees"])
        self.axes = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
        self.centre = np.array(self.shell["centre"])
        self.half = np.array(self.shell["size"], dtype=float) / 2
        textures = json.loads((folder / "room-shell-textures.json").read_text())["faces"]
        self.pictures, self.seen = {}, {}
        for face, info in textures.items():
            self.pictures[face] = cv2.imread(str(folder / info["file"]))[:, :, ::-1].astype(np.float32) / 255
            seen = folder / info.get("seen", f"room-shell-{face}-seen.png")
            self.seen[face] = cv2.imread(str(seen), cv2.IMREAD_GRAYSCALE) > 127 if seen.exists() else None

        mesh = o3d.io.read_triangle_mesh(str(folder / "room-furniture.ply"))
        self.colours = np.asarray(mesh.vertex_colors)
        self.triangles = np.asarray(mesh.triangles)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        box = self.centre + (corners * (self.half + PAD)) @ self.axes
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        box_triangles = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
        self.scene = o3d.t.geometry.RaycastingScene()
        self.box_id = self.scene.add_triangles(o3d.core.Tensor(box.astype(np.float32)),
                                               o3d.core.Tensor(box_triangles.astype(np.uint32)))
        self.furniture_id = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def camera_rays(self, camera, u, v):
        """Rays in the upright room from one frame's COLMAP camera through pixels (u, v)."""
        inside = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(camera["lens"]).T
        directions = (inside @ camera["turn"]) @ FLIP.T @ self.level.T  # turn.T @ d, per row
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origin = self.level @ FLIP @ (-camera["turn"].T @ camera["shift"]) - self.lift
        return np.broadcast_to(origin, directions.shape), directions

    def cast(self, origins, directions):
        """What each ray sees: its colour, and whether it's furniture, torn (a furniture triangle's back), filmed
        wall, or never-filmed wall."""
        import open3d as o3d

        rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6).astype(np.float32)
        hit = self.scene.cast_rays(o3d.core.Tensor(rays))
        distance = hit["t_hit"].numpy()
        geometry = hit["geometry_ids"].numpy()
        primitive = hit["primitive_ids"].numpy()
        weights = hit["primitive_uvs"].numpy()
        normals = hit["primitive_normals"].numpy()
        flat_dirs = directions.reshape(-1, 3)
        count = len(rays)
        colour = np.zeros((count, 3), dtype=np.float32)
        kind = np.full(count, "nothing", dtype=object)

        furniture = (geometry == self.furniture_id) & np.isfinite(distance)
        if furniture.any():
            corners = self.triangles[primitive[furniture]]
            u, v = weights[furniture, 0:1], weights[furniture, 1:2]
            colour[furniture] = ((1 - u - v) * self.colours[corners[:, 0]] + u * self.colours[corners[:, 1]]
                                 + v * self.colours[corners[:, 2]])
            back = (normals[furniture] * flat_dirs[furniture]).sum(axis=1) > 0
            kind[np.nonzero(furniture)[0]] = np.where(back, "torn", "furniture")

        box = (geometry == self.box_id) & np.isfinite(distance)
        if box.any():
            points = rays[box, :3] + flat_dirs[box] * distance[box, None]
            local = (points - self.centre) @ self.axes.T
            ratio = local / (self.half + PAD)
            axis = np.abs(ratio).argmax(axis=1)
            side = np.sign(ratio[np.arange(len(ratio)), axis])
            indices = np.nonzero(box)[0]
            for face, (face_axis, face_side, (col_axis, col_sign), (row_axis, row_sign)) in FACES.items():
                on = (axis == face_axis) & (side == face_side)
                if not on.any():
                    continue
                picture = self.pictures[face]
                rows, cols = picture.shape[:2]
                across, down = 2 * (self.half[col_axis] + PAD), 2 * (self.half[row_axis] + PAD)
                c = (local[on, col_axis] * col_sign + across / 2) * cols / across - 0.5
                r = (local[on, row_axis] * row_sign + down / 2) * rows / down - 0.5
                colour[indices[on]] = _sample(picture, c, r)
                seen = self.seen[face]
                if seen is None:
                    kind[indices[on]] = "wall"
                else:
                    ri = np.clip(np.round(r).astype(int), 0, rows - 1)
                    ci = np.clip(np.round(c).astype(int), 0, cols - 1)
                    kind[indices[on]] = np.where(seen[ri, ci], "wall", "unfilmed")
        return colour, kind

    def view(self, position, ahead, width=480, height=360, fov=70):
        """A plain camera at `position` looking `ahead` (level), like the viewer's Walk look."""
        ahead = np.asarray(ahead, dtype=float) / np.linalg.norm(ahead)
        right = np.cross(ahead, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, ahead)
        tan_y = math.tan(math.radians(fov / 2))
        tan_x = tan_y * width / height
        ys, xs = np.indices((height, width))
        x = ((xs + 0.5) / width * 2 - 1) * tan_x
        y = (1 - (ys + 0.5) / height * 2) * tan_y
        directions = ahead + x[..., None] * right + y[..., None] * up
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        colour, kind = self.cast(np.broadcast_to(np.asarray(position, float), directions.shape), directions)
        return colour.reshape(height, width, 3), kind.reshape(height, width)


def _sample(picture, c, r):
    """Bilinear colours at fractional pixel spots; OpenCV's remap takes at most 32,767 per row, so in rows of 1024."""
    count = len(c)
    rows = -(-count // 1024)
    grid = lambda x: np.pad(x.astype(np.float32), (0, rows * 1024 - count)).reshape(rows, 1024)  # noqa: E731
    out = cv2.remap(picture, grid(c), grid(r), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, 3)[:count]


def _ssim(a, b):
    """Mean SSIM of two grey pictures (0 to 1), Gaussian window, Wang et al. 2004."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean())


def _wsl_path(path):
    """C:\\Users\\... as WSL sees it, /mnt/c/Users/..."""
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def draw_splats(splats, splat_data, out_dir):
    """The splats drawn from every camera in `splat_data`, into `out_dir` (render_splats.py in WSL's gsplat)."""
    script = _wsl_path(Path(__file__).parent / "render_splats.py")
    command = (f"{WSL_PYTHON} {shlex.quote(script)} {shlex.quote(_wsl_path(splats))} "
               f"{shlex.quote(_wsl_path(splat_data))} {shlex.quote(_wsl_path(out_dir))}")
    subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], check=True, capture_output=True, text=True)


def paint_from_splats(out):
    """The Walk room's walls and furniture painted from the splats' pictures instead of the video's frames: the
    splats were fitted to every frame at once, so they share one exposure and drop most of the blur."""
    out = Path(out)
    splats = out / "room-splats-full.ply" if (out / "room-splats-full.ply").exists() else out / "room-splats.ply"
    renders = out / "splat-renders"
    draw_splats(splats, out / "splat-data", renders)
    return {"textures": make_textures(out, images=renders), "furniture": build_furniture(out, images=renders)}


def _held_out_build(out, held, paint="video"):
    """A copy of the room's inputs without the held-back frames, with the walls and furniture rebuilt from the rest:
    painted from the kept video frames, or ("splats") from splats trained on the kept frames only."""
    check = out / "checks" / f"held-out-{paint}"
    if check.exists():
        shutil.rmtree(check)
    (check / "splat-data" / "sparse" / "0").mkdir(parents=True)
    (check / "splat-data" / "images").mkdir()
    (check / "colmap" / "model" / "0").mkdir(parents=True)
    for name in ("room-frame.json", "room-shell.json", "room-objects.json", "room-surface.ply"):
        shutil.copyfile(out / name, check / name)
    shutil.copyfile(out / "colmap" / "model" / "0" / "images.bin", check / "colmap" / "model" / "0" / "images.bin")
    sparse = out / "splat-data" / "sparse" / "0"
    shutil.copyfile(sparse / "cameras.txt", check / "splat-data" / "sparse" / "0" / "cameras.txt")
    shutil.copyfile(sparse / "points3D.txt", check / "splat-data" / "sparse" / "0" / "points3D.txt")
    kept = []
    for line in (sparse / "images.txt").read_text().splitlines():
        parts = line.split()
        if line.startswith("#"):
            kept.append(line)
        elif len(parts) == 10 and parts[9] not in held:
            kept += [line, ""]
            shutil.copyfile(out / "splat-data" / "images" / parts[9], check / "splat-data" / "images" / parts[9])
    (check / "splat-data" / "sparse" / "0" / "images.txt").write_text("\n".join(kept) + "\n")
    images = None
    if paint == "splats":
        subprocess.run([str(BRUSH), str(check / "splat-data"), "--total-steps", str(SPLAT_STEPS),
                        "--max-splats", str(SPLAT_MAX), "--sh-degree", str(SPLAT_SH), "--export-every", str(SPLAT_STEPS),
                        "--export-path", str(check), "--export-name", "splats-held-out.ply"],
                       check=True, capture_output=True)
        images = check / "splat-renders"
        draw_splats(check / "splats-held-out.ply", check / "splat-data", images)
    make_textures(check, images=images)
    build_furniture(check, images=images)
    return check


def test_looks_like_the_video(out, paint="video"):
    """Test 2: the held-back frames against the Walk room drawn from their cameras, and views from the middle."""
    cameras = _cameras(out / "splat-data")
    held = {camera["name"] for k, camera in enumerate(cameras) if k % HOLD_OUT == 3}
    solved = _colmap_solved(out / "colmap" / "model" / "0") or set()
    started = time.perf_counter()
    walk = WalkRoom(_held_out_build(out, held, paint))
    pictures = out / "checks"
    scores = []
    sheet = []
    for camera in cameras:
        if camera["name"] not in held or Path(camera["name"]).stem not in solved:
            continue
        real = cv2.imread(str(out / "splat-data" / "images" / camera["name"]))[:, :, ::-1].astype(np.float32) / 255
        height, width = real.shape[:2]
        v, u = np.indices((height, width)) + 0.5
        origins, directions = walk.camera_rays(camera, u.reshape(-1), v.reshape(-1))
        colour, kind = walk.cast(origins, directions)
        drawn = colour.reshape(height, width, 3)
        error = float(((drawn - real) ** 2).mean())
        grey = lambda x: cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255  # noqa: E731
        scores.append({"frame": camera["name"], "psnr": round(10 * math.log10(1 / max(error, 1e-10)), 2),
                       "ssim": round(_ssim(grey(drawn), grey(real)), 3),
                       "torn": round(float((kind == "torn").mean()), 3),
                       "unfilmed": round(float((kind == "unfilmed").mean()), 3)})
        sheet.append(np.hstack([real, drawn]))
    cv2.imwrite(str(pictures / f"test2-held-back-{paint}.jpg"),
                (np.vstack(sheet[:6]) * 255).astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    full = WalkRoom(out)  # the room as built, from where the phone never stood: the middle, eye height, 4 ways
    middle = np.array([full.centre[0], EYE, full.centre[2]])
    views = []
    for name, ahead in (("front", -full.axes[2]), ("right", full.axes[0]), ("back", full.axes[2]), ("left", -full.axes[0])):
        colour, kind = full.view(middle, ahead)
        views.append({"looking": name, "torn": round(float((kind == "torn").mean()), 3),
                      "unfilmed": round(float((kind == "unfilmed").mean()), 3)})
        cv2.imwrite(str(pictures / f"test2-middle-{name}.jpg"), (colour * 255).astype(np.uint8)[:, :, ::-1])
    return {
        "heldBack": len(held), "scored": len(scores),
        "psnr": round(float(np.mean([s["psnr"] for s in scores])), 2) if scores else None,
        "ssim": round(float(np.mean([s["ssim"] for s in scores])), 3) if scores else None,
        "frames": scores, "fromTheMiddle": views,
        "tornFromTheMiddle": round(float(np.mean([v["torn"] for v in views])), 3),
        "unfilmedFromTheMiddle": round(float(np.mean([v["unfilmed"] for v in views])), 3),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _spark(images, prompt, key):
    """One Muse Spark call with several frames and a question; returns its answer text."""
    content = [{"type": "input_text", "text": prompt}] + [
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}
        for path in images]
    body = {"model": SPARK, "input": [{"type": "message", "role": "user", "content": content}]}
    request = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read().decode())
    return "".join(part.get("text", "") for item in reply.get("output", []) for part in item.get("content", []) or [])


INVENTORY = (
    "These {n} pictures are frames from one phone video walking through a single room. List every distinct physical "
    "thing in the room that appears in any of the frames: furniture, appliances, electronics, decor, and smaller items "
    "like bags, boxes, bottles or clothes piles. Don't list the walls, floor, ceiling or the room itself, and don't "
    "list the same thing twice. Answer with JSON only, no other text: an array of objects with \"name\" (a short plain "
    "noun phrase of one to three words that an object-finding model can search for, like \"desk chair\" or \"ceiling "
    "fan\") and \"onWall\" (true when it hangs on or is built into a wall or the ceiling, like a window, curtain, door, "
    "light switch, outlet or ceiling fan; false when it stands in the room).")


def test_everything_is_there(out, key):
    """Test 1: every thing Muse Spark sees in the video has a real 3D piece where the video shows it."""
    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    spread = [cameras[round(k * (len(cameras) - 1) / (INVENTORY_FRAMES - 1))] for k in range(INVENTORY_FRAMES)]
    started = time.perf_counter()
    # Muse Spark's list is kept, so checking the same room again checks the same things (and costs nothing)
    inventory = out / "checks" / "inventory.json"
    if inventory.exists():
        listed = json.loads(inventory.read_text())
    else:
        answer = _spark([images / camera["name"] for camera in spread], INVENTORY.format(n=len(spread)), key)
        listed = json.loads(answer[answer.index("["): answer.rindex("]") + 1])
        inventory.write_text(json.dumps(listed, indent=2))
    things, names = [], set()
    for thing in listed:
        name = str(thing.get("name", "")).strip().lower()
        if name and name not in names and name not in ROOM_PARTS:
            names.add(name)
            things.append({"name": name, "onWall": bool(thing.get("onWall"))})
    things = things[:MOST_NAMES]

    saved_file = out / "sam-outlines.json"
    saved = json.loads(saved_file.read_text()) if saved_file.exists() else {}
    frames = cameras[::SAM_EVERY]
    jobs = [(camera, thing["name"]) for camera in frames for thing in things]
    missing = [job for job in jobs if f"{job[0]['name']}|{job[1]}" not in saved]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (camera, name), text in zip(missing, pool.map(lambda job: _outline(images / job[0]["name"], job[1], key), missing)):
            saved[f"{camera['name']}|{name}"] = text
    saved_file.write_text(json.dumps(saved))

    walk = WalkRoom(out)
    for thing in things:
        counts = {"furniture": 0, "torn": 0, "wall": 0, "unfilmed": 0, "nothing": 0}
        sightings = 0
        for camera in frames:
            for record in _records(saved[f"{camera['name']}|{thing['name']}"]):
                u, v = _pixels(record)
                if len(u) == 0:
                    continue
                sightings += 1
                _, kind = walk.cast(*walk.camera_rays(camera, u, v))
                for k in counts:
                    counts[k] += int((kind == k).sum())
        total = sum(counts.values())
        real = counts["furniture"] + (counts["wall"] if thing["onWall"] else 0)
        thing.update({"sightings": sightings, "pixels": total, "landsOnSomethingReal": round(real / total, 3) if total else None,
                      "hits": counts})
        thing["result"] = ("not found by SAM 3.1" if not total else "in the room" if real / total >= PRESENT else "missing")
    found = [thing for thing in things if thing["pixels"]]
    return {
        "listed": len(things), "foundBySam": len(found),
        "inTheRoom": sum(thing["result"] == "in the room" for thing in found),
        "things": things, "samCalls": len(missing), "framesForSam": len(frames),
        "seconds": round(time.perf_counter() - started, 1),
    }


def run_checks(out):
    out = Path(out)
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
    (out / "checks").mkdir(exist_ok=True)
    report = {"test2": test_looks_like_the_video(out), "test1": test_everything_is_there(out, key)}
    (out / "room-checks.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_checks.py <request folder>")
    result = run_checks(sys.argv[1])
    t1, t2 = result["test1"], result["test2"]
    print(json.dumps({"test2": {k: v for k, v in t2.items() if k != "frames"},
                      "test1": {k: v for k, v in t1.items() if k != "things"}}, indent=2))
    for thing in t1["things"]:
        print(f"{thing['name']:22s} {thing['result']:22s} real {thing['landsOnSomethingReal']}  seen in {thing['sightings']}")
