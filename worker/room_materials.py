"""The Rebuilt look's walls, floor and ceiling as clean real materials instead of photo patches: the owner's ask
(Nov 2: "create everything in the 3d model way", for VR). Each face of room-shell.json's box gets one sharp tile of its
own surface, cut from the video and repeated across the face at its real size, so the carpet looks like the carpet and
the paint like the paint, with no blur and no furniture painted on the walls.

For each face: on its photo (room_textures.py's, about 1 cm a texel) the cleanest TILE-square window the video saw
(every texel seen, the fewest edges, brightness near the face's own); its corners go into every frame that sees them
head-on and unblocked, and the sharpest, biggest view of it is cut from the full-size frame and straightened onto the
face (TILE_PX texels a side, 2 mm each); mirrored to a 2 x 2 block, it repeats without seams. A face the video never
saw (take 4's ceiling) gets a plain colour instead: the lightest wall's.

The photo walls stay (room-shell-*.jpg): the viewer's "Photo walls" switch shows them, so damage the landlord filmed
stays visible. Writes room-material-<face>.jpg and room-materials.json.
    python worker/room_materials.py <request folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

from room_objects import FLIP, _cameras, _turn
from room_textures import FACES, _face_points

TILES = (0.40, 0.25, 0.15)  # metres of real surface in one tile: the biggest that has a clean view
TILE_PX = 200  # texels a tile side: 2 mm each
STRIDE = 8  # texels between the windows tried
SEEN = 0.9  # share of a window's texels the video must have seen
HEAD_ON = 0.35  # cos 70 degrees: frames seeing the face more edge-on than this don't cut its tile (the floor is
# only ever seen at a slant from standing height)
BLOCKED = 0.05  # metres: a point whose ray meets the room's surface this much before the face is hidden there
LIKE = 30  # colour levels: a cut tile's mean may differ this much from the face photo's there, each channel
BUSIER = 0.04  # and its share of edge pixels may be at most this much higher
DETAIL = 0.5  # how much of a cut tile's own marks it keeps (the rest goes to its mean colour)
EVEN = 22  # brightness levels: an 8 x 8 grid of a tile's block averages may span at most this (a chair leg on the
# carpet, a shadow line on a wall: a tile repeated across a face would repeat them everywhere)


def make(out):
    import cv2
    import open3d as o3d

    out = Path(out)
    shell = json.loads((out / "room-shell.json").read_text())
    frame = json.loads((out / "room-frame.json").read_text())
    textures = json.loads((out / "room-shell-textures.json").read_text())["faces"]
    level, lift = _turn(*frame["quaternion"]), np.array([0.0, frame["floorHeight"], 0.0])
    cameras = _cameras(out / "splat-data")
    mesh = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))  # three.js axes

    # How much bigger each full-size frame is than its camera (take 4: 4K frames, 1080 x 1920 cameras)
    grow = {}
    for camera in cameras:
        size = cv2.imread(str(out / "frames" / f"{Path(camera['name']).stem}.jpg"), cv2.IMREAD_REDUCED_COLOR_8)
        grow[camera["name"]] = 8 * size.shape[0] / (2 * camera["lens"][1, 2]) if size is not None else 1.0

    report, own_colour = {}, {}
    for face in FACES:
        points, normal, grid = _face_points(shell, face)
        picture = cv2.imread(str(out / textures[face]["file"]))
        seen = cv2.imread(str(out / textures[face]["seen"]), cv2.IMREAD_GRAYSCALE) > 127
        if picture is not None and seen.any():
            own_colour[face] = [int(c) for c in np.median(picture[seen], axis=0)[::-1]]
        report[face] = None
        for tile_metres in TILES:
            report[face] = _tile(out, face, points, normal, grid, picture, seen, tile_metres, cameras, grow, level, lift,
                                 scene)
            if report[face] is not None:
                break

    # One paint for every wall: the wall tile nearest the walls' typical colour (a wall in shadow is grey, not its
    # paint), so the walls don't each repeat their own lighting; the viewer's lights shade them apart, as a game
    # engine would
    import cv2

    walls = [f for f in ("left", "right", "front", "back") if report.get(f)]
    if walls:
        typical = np.median([report[f]["colour"] for f in walls], axis=0)
        def evenness(entry):
            gray = cv2.cvtColor(cv2.imread(str(out / entry["file"])), cv2.COLOR_BGR2GRAY)
            blocks = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            return int(blocks.max()) - int(blocks.min())

        # near the typical colour first (within 10 levels counts as the same), then the most even (no stripes)
        paint = min(walls, key=lambda f: (round(np.abs(np.array(report[f]["colour"]) - typical).sum() / 10),
                                          evenness(report[f]), -report[f]["tileMetres"]))
        for face in ("left", "right", "front", "back"):
            report[face] = {**report[paint], "sharedFrom": paint}
    plain = report[paint]["colour"] if walls else [230, 230, 230]
    # A floor with no clean view (take 4's carpet is mostly under things): a fine carpet-like grain in its own filmed
    # colour, marked as made, not filmed
    if report.get("floor") is None and "floor" in own_colour:
        grain = np.random.default_rng(7).normal(0, 9, (TILE_PX, TILE_PX, 1))
        grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), 0.8)[..., None]
        floor = np.clip(np.array(own_colour["floor"][::-1], np.float32) + grain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "room-material-floor.jpg"), floor, [cv2.IMWRITE_JPEG_QUALITY, 92])
        report["floor"] = {"file": "room-material-floor.jpg", "tileMetres": 0.2, "frame": None,
                           "colour": own_colour["floor"], "made": "grain in the floor's filmed colour"}
    for face in FACES:
        if report.get(face) is None:  # its own filmed colour when it has one, else the walls' paint
            report[face] = {"file": None, "tileMetres": None, "frame": None, "colour": own_colour.get(face, plain)}
    (out / "room-materials.json").write_text(json.dumps({"faces": report}, indent=2))
    return report



def _tile(out, face, points, normal, grid, picture, seen, tile_metres, cameras, grow, level, lift, scene):
    """One face's tile at one size, or None when no clean, unblocked, head-on view of it was filmed."""
    import cv2
    import open3d as o3d

    if True:
        size = int(round(tile_metres / grid["texel"][0]))
        if picture is None or seen.mean() < 0.02 or min(seen.shape) < size:
            return None
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 90) > 0
        median = np.median(gray[seen])
        windows = []
        for top in range(0, seen.shape[0] - size + 1, STRIDE):
            for left in range(0, seen.shape[1] - size + 1, STRIDE):
                if seen[top:top + size, left:left + size].mean() < SEEN:
                    continue
                busy = edges[top:top + size, left:left + size].mean()
                shade = abs(float(gray[top:top + size, left:left + size].mean()) - median) / 255
                windows.append((busy + shade, top, left))
        if not windows:
            return None

        # The cleanest windows first; for each, the frames that see it head-on and unblocked (a 5 x 5 grid of its
        # points, not just its corners: take 4's fan stood in the middle of a floor patch), biggest view first; the
        # first cut that still looks like the face's own photo there (colour, no busier) is the tile
        tile, source = None, None
        grid_rows, grid_cols = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="ij")
        for _, top, left in sorted(windows)[:80]:
            window_colour = picture[top:top + size, left:left + size].reshape(-1, 3).mean(axis=0)
            corners = np.array([points[top, left], points[top, left + size - 1],
                                points[top + size - 1, left + size - 1], points[top + size - 1, left]])
            inside = points[top + (grid_rows * (size - 1)).astype(int), left + (grid_cols * (size - 1)).astype(int)]
            inside = inside.reshape(-1, 3)
            views = []
            for camera in cameras:
                full = grow[camera["name"]]  # the full-size frames against the cameras' size
                in_cv = ((corners + lift) @ level) @ FLIP
                in_camera = in_cv @ camera["turn"].T + camera["shift"]
                if (in_camera[:, 2] < 0.1).any():
                    continue
                u = (camera["lens"][0, 0] * in_camera[:, 0] / in_camera[:, 2] + camera["lens"][0, 2]) * full
                v = (camera["lens"][1, 1] * in_camera[:, 1] / in_camera[:, 2] + camera["lens"][1, 2]) * full
                width, height = 2 * camera["lens"][0, 2] * full, 2 * camera["lens"][1, 2] * full
                if (u < 0).any() or (u >= width).any() or (v < 0).any() or (v >= height).any():
                    continue
                eye = -camera["turn"].T @ camera["shift"]  # OpenCV room space
                facing = float(np.mean((eye - in_cv) @ (FLIP @ (level.T @ normal)))
                               / np.linalg.norm(eye - in_cv, axis=1).mean())
                if facing < HEAD_ON:
                    continue
                origin = FLIP @ eye
                targets = (inside + lift) @ level  # upright -> three.js axes
                rays = np.hstack([np.repeat(origin[None], len(targets), 0), targets - origin]).astype(np.float32)
                hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()  # 1 = the face point itself
                lengths = np.linalg.norm(targets - origin, axis=1)
                if ((1 - hit) * lengths > BLOCKED).any():
                    continue
                area = 0.5 * abs(np.dot(u, np.roll(v, 1)) - np.dot(v, np.roll(u, 1)))
                views.append((area * facing, camera, np.stack([u, v], -1)))
            for _, camera, quad in sorted(views, key=lambda view: -view[0]):
                photo = cv2.imread(str(out / "frames" / f"{Path(camera['name']).stem}.jpg"))
                if photo is None:
                    continue
                square = np.array([[0, 0], [TILE_PX - 1, 0], [TILE_PX - 1, TILE_PX - 1], [0, TILE_PX - 1]], np.float32)
                cut = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(quad.astype(np.float32), square),
                                          (TILE_PX, TILE_PX), flags=cv2.INTER_AREA)
                small = cv2.resize(cut, (size, size), interpolation=cv2.INTER_AREA)
                busy_cut = (cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 30, 90) > 0).mean()
                busy_window = edges[top:top + size, left:left + size].mean()
                blocks = cv2.resize(cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
                if (np.abs(small.reshape(-1, 3).mean(axis=0) - window_colour).max() < LIKE
                        and busy_cut <= busy_window + BUSIER and int(blocks.max()) - int(blocks.min()) <= EVEN):
                    tile, source = cut, camera["name"]
                    break
            if tile is not None:
                break
        if tile is None:
            return None
        # Its light evened out: the frame's own shading across the tile (take 4's: 12 levels darker in its middle)
        # would otherwise repeat as a band every tile across the face
        # and its marks softened by half, since a mark on a tile repeats in a row across the whole face
        tile = tile.astype(np.float32)
        mean = tile.reshape(-1, 3).mean(axis=0)
        shading = cv2.GaussianBlur(tile, (0, 0), TILE_PX / 4)
        tile = mean + DETAIL * (tile / np.maximum(shading, 1.0) * mean - mean)
        tile = np.clip(tile, 0, 255).astype(np.uint8)
        seamless = np.vstack([np.hstack([tile, tile[:, ::-1]]), np.hstack([tile[::-1], tile[::-1, ::-1]])])
        name = f"room-material-{face}.jpg"
        cv2.imwrite(str(out / name), seamless, [cv2.IMWRITE_JPEG_QUALITY, 92])
        colour = seamless.reshape(-1, 3).mean(axis=0)[::-1]
        return {"file": name, "tileMetres": 2 * tile_metres, "frame": source, "colour": [int(round(c)) for c in colour]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/room_materials.py <request folder>")
    print(json.dumps(make(sys.argv[1]), indent=2))
