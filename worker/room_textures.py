"""The shell's walls, floor and ceiling as photos, for the Walk look: each face of room-shell.json's box gets a picture
made from the frames, so the clean box looks like the real room from anywhere inside it.

Every texel (about 1 cm) of a face is a point on that face's plane in the upright room. It goes back to room space,
into each frame through that frame's camera, and the frame counts only where it really sees the point: each frame's
rays are cast against room-surface.ply first, and when the surface its ray meets lies more than SEEN in front of the
wall point, something (the desk, the bed) stands between, and that frame would paint it onto the wall. The frames
that see a point are blended, the most head-on, nearest and sharpest by far the most, COLMAP-solved cameras before
the rougher ones, after a gain per frame and colour so their exposures agree. What no frame sees fades from the
nearby colours into the face's median colour.

Needs the request folder's splat-data/ (frames and cameras), room-surface.ply, room-frame.json and room-shell.json.
Writes room-shell-{right,left,ceiling,floor,back,front}.jpg and room-shell-textures.json.
    python worker/room_textures.py <request folder>
"""

import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

TEXEL = 0.01  # metres per texel
LONGEST = 1024  # texels on a face's long side at most
PAD = 0.01  # the viewer's box is this much bigger than the shell on every side (BoxGeometry(w + 0.02, ...))
BAND = 0.15  # metres either side of the shell's plane where a face looks for the real wall's surface
SEEN = 0.04  # metres: a frame sees a wall point when its ray meets no surface more than this in front of it
SEEN_PER_METRE = 0.02  # plus this much per metre away, for walls the camera sees at a slant
NEAR_PIXELS = 3  # the surface test takes the nearest surface in this many pixels square, so edges don't bleed
GRAZING = 0.2  # a frame that sees a face at less than this cosine (about 78 degrees off head-on) doesn't count
EDGE = 12  # pixels: a frame fades out over this many pixels at its borders, so its edge leaves no seam
TRIM = 5  # texels square: each frame's seen part of a face shrinks by 2 texels at its edges
SPECK = 60  # texels: a seen patch smaller than this (60 cm2) on its own is dropped and filled
ROUGH = 0.35  # score for frames whose camera MapAnything gave (lined up to neighbours), next to COLMAP's 1
SHARPEST = 3  # blend weight = (score / best score here) ** SHARPEST: the best frame leads, the next ones soften seams
# (8 to 3 with the furniture at 6 frames and 1: Test 2 on take 3's held-back frames, 19.10 to 19.41 dB)
GAIN_LIMITS = (0.7, 1.4)
FADE = 0.12  # metres: unseen texels go from the nearby colours to the face's median colour over this distance
QUALITY = 90
FLIP = np.array([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes, as everywhere in the worker

# Each face in the box's own axes (x along the walls' axis a, y up, z along axis b; 0, 1, 2): the axis the face sits
# on and which end, then which way the picture's columns (left to right) and rows (top to bottom) run. Every picture
# reads the right way round from inside the room: a wall as you face it, upright; the ceiling as you look up from
# facing the front wall, the floor as you look down facing it. Laid out as a cross they meet at their edges: left,
# front, right, back in a row, the ceiling above the front wall and the floor below it.
FACES = {
    "right": (0, 1, (2, 1), (1, -1)),
    "left": (0, -1, (2, -1), (1, -1)),
    "ceiling": (1, 1, (0, 1), (2, -1)),
    "floor": (1, -1, (0, 1), (2, 1)),
    "back": (2, 1, (0, -1), (1, -1)),
    "front": (2, -1, (0, 1), (1, -1)),
}


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


def _colmap_solved(model):
    """The frames (by name without extension) COLMAP solved itself, from its images.bin; None when there's no model."""
    path = model / "images.bin"
    if not path.exists():
        return None
    data = path.read_bytes()
    count, at = struct.unpack_from("<Q", data, 0)[0], 8
    names = set()
    for _ in range(count):
        at += 4 + 8 * 7 + 4  # image id, quaternion, shift, camera id
        end = data.index(b"\0", at)
        names.add(Path(data[at:end].decode()).stem)
        points = struct.unpack_from("<Q", data, end + 1)[0]
        at = end + 1 + 8 + points * 24
    return names


def _face_points(shell, face):
    """The texel centres of one face as upright-room points (rows x columns x 3), the face's inward normal, and the
    texel grid's numbers."""
    axis, side, (col_axis, col_sign), (row_axis, row_sign) = FACES[face]
    yaw = math.radians(shell["yawDegrees"])
    axes = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]])
    half = np.array(shell["size"], dtype=float) / 2
    across, down = 2 * (half[col_axis] + PAD), 2 * (half[row_axis] + PAD)
    texel = max(TEXEL, max(across, down) / LONGEST)
    cols, rows = max(1, round(across / texel)), max(1, round(down / texel))
    s = (np.arange(cols) + 0.5) * across / cols - across / 2  # along the columns' direction
    t = (np.arange(rows) + 0.5) * down / rows - down / 2  # along the rows' direction
    local = np.zeros((rows, cols, 3))
    local[:, :, axis] = side * half[axis]  # on the shell's own plane, where the real wall is
    local[:, :, col_axis] = col_sign * s[None, :]
    local[:, :, row_axis] = row_sign * t[:, None]
    points = np.array(shell["centre"]) + local @ axes
    return points, -side * axes[axis], {"cols": cols, "rows": rows, "texel": (across / cols, down / rows)}


def make_textures(out, images=None):
    """`images`: a folder of pictures to paint from instead of the video's frames (same names), e.g. the splats'
    renders from render_splats.py."""
    import cv2
    import open3d as o3d

    started = time.perf_counter()
    out = Path(out)
    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = _turn(*frame["quaternion"])
    lift = np.array([0.0, frame["floorHeight"], 0.0])
    cameras = _cameras(out / "splat-data")
    solved = _colmap_solved(out / "colmap" / "model" / "0")
    images = Path(images) if images else out / "splat-data" / "images"

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))))

    faces = {}
    for face in FACES:
        upright, inward, grid = _face_points(shell, face)
        three, inward = (upright.reshape(-1, 3) + lift) @ level, inward @ level  # upright = level . three - lift
        # The real wall (and what hangs on it: the curtain, a poster) sits a few cm off the shell's plane, so each
        # texel takes the surface it finds within BAND of the plane, straight out from the room: every frame then
        # projects the same real point, and the surface test checks that point
        rays = np.hstack([three + inward * BAND, np.repeat(-inward[None], len(three), axis=0)]).astype(np.float32)
        reach = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        found = np.isfinite(reach) & (reach < 2 * BAND)
        offset = float(np.median(BAND - reach[found])) if found.any() else 0.0  # into the room, metres
        three = np.where(found[:, None], three + inward * (BAND - np.where(found, reach, 0))[:, None],
                         three + inward * offset)
        faces[face] = {"three": three, "inward": inward, "grid": grid, "onSurface": float(found.mean()),
                       "offset": offset, "score": np.zeros((len(cameras), len(three)), np.float32),
                       "colour": np.zeros((len(cameras), len(three), 3), np.float32)}

    sharpness, kernel = [], np.ones((NEAR_PIXELS, NEAR_PIXELS), np.uint8)
    for k, camera in enumerate(cameras):
        picture = cv2.imread(str(images / camera["name"]))[:, :, ::-1].astype(np.float32) / 255
        height, width = picture.shape[:2]
        gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        lens, turn, shift = camera["lens"], camera["turn"], camera["shift"]
        origin = FLIP * (-turn.T @ shift)  # the camera's centre, three.js axes

        # how far each pixel's ray goes before it meets the surface (the nearest within NEAR_PIXELS, for edges)
        rows, cols = np.indices((height, width))
        in_camera = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols)], axis=-1).reshape(-1, 3) @ np.linalg.inv(lens).T
        directions = (in_camera @ turn) * FLIP  # turn.T @ d for each row, then to three.js axes
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rays = np.hstack([np.repeat(origin[None], len(directions), axis=0), directions]).astype(np.float32)
        reach = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(height, width)
        reach = cv2.erode(np.where(np.isfinite(reach), reach, 100.0).astype(np.float32), kernel)

        for data in faces.values():
            points = data["three"]
            x = (points * FLIP) @ turn.T + shift  # into the camera, OpenCV axes
            ahead = x[:, 2] > 0.05
            z = np.where(ahead, x[:, 2], 1.0)
            u = lens[0, 0] * x[:, 0] / z + lens[0, 2]
            v = lens[1, 1] * x[:, 1] / z + lens[1, 2]
            inside = ahead & (u >= 0.5) & (u <= width - 0.5) & (v >= 0.5) & (v <= height - 0.5)
            to_camera = origin - points
            distance = np.linalg.norm(to_camera, axis=1)
            head_on = (to_camera @ data["inward"]) / np.maximum(distance, 1e-6)
            iu = np.clip(u.astype(int), 0, width - 1)
            iv = np.clip(v.astype(int), 0, height - 1)
            seen = inside & (head_on > GRAZING) & (reach[iv, iu] > distance - SEEN - SEEN_PER_METRE * distance)
            if not seen.any():
                continue
            grid = data["grid"]
            # a frame's view stops short of where something stands in front, so that thing's edge stays off the wall
            seen = cv2.erode(seen.reshape(grid["rows"], grid["cols"]).astype(np.uint8), np.ones((TRIM, TRIM), np.uint8))
            seen = seen.reshape(-1).astype(bool)
            edge = np.minimum.reduce([u, width - u, v, height - v]) / EDGE
            data["score"][k] = np.where(seen, head_on / distance * np.clip(edge, 0, 1), 0)
            if solved is not None and Path(camera["name"]).stem not in solved:
                data["score"][k] *= ROUGH
            sample = cv2.remap(picture, (u - 0.5).astype(np.float32).reshape(grid["rows"], grid["cols"]),
                               (v - 0.5).astype(np.float32).reshape(grid["rows"], grid["cols"]),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            data["colour"][k] = sample.reshape(-1, 3)

    sharp = np.clip(np.sqrt(np.array(sharpness) / np.median(sharpness)), 0.4, 1.3).astype(np.float32)
    for data in faces.values():
        data["score"] *= sharp[:, None]
        best = data["score"].max(axis=0)
        data["weight"] = np.where(data["score"] > 0, (data["score"] / np.maximum(best, 1e-9)) ** SHARPEST, 0)
        data["weight"][data["weight"] < 0.02] = 0

    # a gain per frame and colour so exposures agree: each frame against the blend of all of them, twice
    gains = np.ones((len(cameras), 3), np.float32)
    for _ in range(2):
        ratios = [[] for _ in cameras]
        for data in faces.values():
            blend = _blend(data, gains)
            for k in range(len(cameras)):
                use = (data["weight"][k] > 0.05) & (data["colour"][k].min(axis=1) > 0.04)
                if use.sum() > 50:
                    ratios[k].append(blend[use] / data["colour"][k][use])
        for k, found in enumerate(ratios):
            if found:
                gains[k] = np.clip(np.median(np.concatenate(found), axis=0), *GAIN_LIMITS)
        gains /= np.median(gains, axis=0)

    report = {"texelMetres": None, "faces": {}, "frames": len(cameras),
              "colmapSolved": None if solved is None else sum(Path(c["name"]).stem in solved for c in cameras),
              "axes": "box's own axes: x along the walls' axis a (cos yaw, 0, sin yaw), y up, z along axis b "
                      "(-sin yaw, 0, cos yaw); the viewer's BoxGeometry axes, turned by rotation.y = -yaw",
              "layout": "each picture covers its whole face of the viewer's box (the shell plus PAD each side); "
                        "'columns' is the box axis the picture runs along left to right, 'rows' top to bottom; "
                        "every picture reads the right way round from inside the room"}
    for face, data in faces.items():
        grid, axis = data["grid"], FACES[face]
        seen = (data["weight"].sum(axis=0) > 0).reshape(grid["rows"], grid["cols"])
        # specks seen on their own are mostly glimpses through holes in the surface: let the fill have them
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seen.astype(np.uint8), connectivity=8)
        speck = stats[:, cv2.CC_STAT_AREA] < SPECK
        speck[0] = False
        seen &= ~speck[labels]
        picture = _blend(data, gains).reshape(grid["rows"], grid["cols"], 3)
        picture = _fill(picture, seen, np.array(shell["faces"][face]) / 255, grid["texel"][0])
        # which texels a frame really saw (white) and which are fill (black), for the room checks and the viewer
        cv2.imwrite(str(out / f"room-shell-{face}-seen.png"), seen.astype(np.uint8) * 255)
        seen = seen.reshape(-1)
        name = f"room-shell-{face}.jpg"
        cv2.imwrite(str(out / name), (np.clip(picture, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        label = lambda a, sign: ("+" if sign > 0 else "-") + "xyz"[a]  # noqa: E731
        used = data["weight"].sum(axis=1) > 0
        report["faces"][face] = {
            "file": name, "seen": f"room-shell-{face}-seen.png", "pixels": [grid["cols"], grid["rows"]],
            "texelMetres": [round(grid["texel"][0], 5), round(grid["texel"][1], 5)],
            "coverage": round(float(seen.mean()), 3),
            "columns": label(*axis[2]), "rows": label(*axis[3]),
            "framesUsed": int(used.sum()),
            "onSurface": round(data["onSurface"], 3),  # share of texels that found the real surface within BAND
            "surfaceOffset": round(data["offset"], 3),  # metres into the room from the shell's plane, median

        }
    report["texelMetres"] = TEXEL
    report["gains"] = [round(float(g), 3) for g in np.percentile(gains, [5, 95])]
    report["seconds"] = round(time.perf_counter() - started, 1)
    (out / "room-shell-textures.json").write_text(json.dumps(report, indent=2))
    return report


def _blend(data, gains):
    weight = data["weight"]
    total = weight.sum(axis=0)
    blend = np.einsum("kn,knc->nc", weight, data["colour"] * gains[:, None, :])
    return blend / np.maximum(total, 1e-9)[:, None]


def _fill(picture, seen, median, texel):
    """Texels no frame saw: the nearby seen colours spread in (a coarse-to-fine push-pull), fading into the face's
    median colour over FADE."""
    import cv2

    if seen.all():
        return picture
    if not seen.any():
        return np.broadcast_to(median, picture.shape).astype(np.float32)
    levels = [(picture * seen[:, :, None], seen.astype(np.float32))]
    while min(levels[-1][1].shape) > 2:
        colour, weight = levels[-1]
        levels.append((cv2.resize(colour, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                       cv2.resize(weight, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)))
    colour, weight = levels[-1]
    filled = np.where(weight[:, :, None] > 1e-6, colour / np.maximum(weight, 1e-6)[:, :, None], median)
    for colour, weight in reversed(levels[:-1]):
        up = cv2.resize(filled, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR)
        mine = colour / np.maximum(weight, 1e-6)[:, :, None]
        share = np.clip(weight, 0, 1)[:, :, None]
        filled = mine * share + up * (1 - share)
    away = cv2.distanceTransform((~seen).astype(np.uint8), cv2.DIST_L2, 5) * texel
    fade = np.clip(away / FADE, 0, 1)[:, :, None]
    filled = filled * (1 - fade) + median * fade
    return np.where(seen[:, :, None], picture, filled).astype(np.float32)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("python worker/room_textures.py <request folder>")
    print(json.dumps(make_textures(sys.argv[1]), indent=2))
