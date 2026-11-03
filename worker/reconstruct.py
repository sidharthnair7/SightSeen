"""Video in, 3D room out: frames from the video, MapAnything, then the room as a solid surface and as coloured points.

A GPU with 8 GB can only take about 24 frames in one MapAnything run. A longer video gets one run of 24 frames spread
over the whole video first: that run sets the room's shape and size (the one-run version measured right). The video is
then split into equal parts. Each part is given the whole-video run's camera positions for the frames they share
(MapAnything takes known camera positions as input), so it's built to agree with that run, and is then lined up with
it on those frames (the fit is VGGT-Long's, arXiv 2507.16443, Section 3.1). Lining every part up with the same run means errors can't stack up from
part to part. Every join reports how well it fit.

MapAnything's cameras are then swapped for COLMAP's wherever COLMAP can place the frames (Schönberger and Frahm,
"Structure-from-Motion Revisited", CVPR 2016): it finds the same small features in many frames and solves every camera
together, so the views line up to under a pixel. MapAnything's lens and directions drift from frame to frame, and on
take 3 COLMAP's cameras took the splats from 21.4 to 31.2 dB on frames held back from training. COLMAP has no sense of
size, so MapAnything's depth still sets the metres; when COLMAP can't place enough frames, MapAnything's cameras stay.

All the frames are then fused into one surface (TSDF fusion: each frame's depth votes into a grid of small cubes, the
grid keeps the average, and marching cubes pulls out triangles), so the walls are solid instead of dots.

Last, Brush trains Gaussian splats on the frames and those cameras, so the room looks like the video.

Everything is written into the request's folder, next to the video:
  frames/            the frames MapAnything saw
  colmap/model/      COLMAP's cameras and points for those frames, in its own units (when it could place them)
  splat-data/        what Brush trains on (COLMAP format, OpenCV axes)
  room-splats.ply    the room as Gaussian splats, in OpenCV axes (x right, y down, z away from the first camera)
  room-surface.ply   the room as a coloured triangle mesh, in metres, in three.js axes (x right, y up, z towards the viewer)
  room.ply           the same room as coloured points, in three.js axes
  result.json        the run's numbers (the same dict run() returns)
"""

import json
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")  # Open3D's notebook viewer; SightSeen doesn't use it

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CHECKPOINTS = {
    "apache": "facebook/map-anything-apache",  # Apache 2.0, 6 training datasets: SightSeen's default
    "default": "facebook/map-anything",  # CC-BY-NC 4.0, 13 datasets: for comparison only
}
MAX_POINTS = 2_000_000  # about 30 MB for the browser
MAX_TOTAL_FRAMES = 240  # 2 minutes of video at 2 frames a second; a longer video gets its frames spread out
DOT = 0.01  # 1 cm: where several frames saw the same spot, the room keeps only the most confident point
CUBE = 0.05  # 5 cm cubes for the "how much of the room is filled" number
GRID = 256  # the surface's grid has at most this many cubes along each side (2 cm cubes in a room up to about 5 m)
MAX_TRIANGLES = 800_000  # about 15 MB for the browser
SMALLEST_PIECE = 200  # loose bits of surface with fewer triangles than this are floating noise
BRUSH = Path(os.environ.get("SIGHTSEEN_BRUSH", Path(__file__).parent / "vendor" / "brush" / "brush_app.exe"))
SPLAT_STEPS = 15_000  # Brush's training steps for the photo look: fewer is faster and blurrier
SPLAT_MAX = 500_000  # left alone, Brush grew take 2 to 2.84 million splats (670 MB); this keeps the file browser-sized
SPLAT_SH = 1  # how much a splat's colour can change with the viewing angle; Brush's default (3) is 3 times the file
SPLAT_MAX_SIDE = 1920  # Brush's own --max-resolution: frames for splat training are stored no bigger than this
# Give each part the whole-video run's camera positions for the frames they share, so MapAnything builds the part to
# agree with it before it's lined up (on take 3: worst join 8.8 cm down to 3.9). SIGHTSEEN_HAND_OVER=0 turns it off.
HAND_OVER_POSES = os.environ.get("SIGHTSEEN_HAND_OVER", "1") != "0"
# COLMAP's cameras in place of MapAnything's (take 3's splats: 21.4 to 31.2 dB on held-back frames). SIGHTSEEN_COLMAP=0
# keeps MapAnything's, as does COLMAP placing fewer than COLMAP_MIN_SHARE of the frames (it placed 83% of take 3's).
USE_COLMAP = os.environ.get("SIGHTSEEN_COLMAP", "1") != "0"
COLMAP_MIN_SHARE = 0.5
COLMAP_MAX_SIDE = 1024  # COLMAP finds features on a copy at most this big, so a 4K video stays quick on the CPU
# Frames COLMAP couldn't place go back in with MapAnything's camera lined up on their neighbours', so blank walls and
# fast turns aren't left black. SIGHTSEEN_ADD_BACK=0 leaves them out.
ADD_BACK = os.environ.get("SIGHTSEEN_ADD_BACK", "1") != "0"
ADD_BACK_NEIGHBOURS = 8  # placed frames nearest in the video, used to line each one up
ADD_BACK_MAX_GAP = 0.15  # metres: neighbours lining up worse than this means MapAnything's camera can't be trusted
SNAP_BACK = os.environ.get("SIGHTSEEN_SNAP_BACK", "0") != "0"  # off until it's proven on more than take 4
SNAP_REACH = (0.30, 0.12, 0.05)  # metres: how far ICP looks for a frame's matching points, coarse to fine
SNAP_VOXEL = 0.03  # metres between the points ICP uses
SNAP_NEAR = 0.04  # metres: a point this close to the placed frames' surface lies on it
SNAP_GAIN = 0.03  # a snap is kept when that share of the frame's points grows by this much
SNAP_MOVE = 1.0  # metres and degrees: a snap moving the camera further than this is a wrong match, not a fix
SNAP_TURN = 20.0  # (30 degrees let two of take 4's frames snap onto the wrong surfaces, and the frames snapped onto
# them after followed: the walls behind them went missing)
SNAP_LEAVE = 0.3  # a frame that can't be snapped, with less than this on the placed surface, is left out: its camera
# is wrong and its view would only add a second copy of what it saw
FLAT_DEGREES = 15  # surface facing within this many degrees of up or down counts as flat (floor, bed, desk, ceiling)
# A fitted patch must face within this of up to count: on take 3 the real floor, bed and ceiling sat 1.2 to 2.4 degrees
# off the cameras' up, and the noisy bits of floor 10 to 13
PLANE_DEGREES = 8
MIN_EYE, MAX_EYE = 0.9, 2.0  # metres: a phone held for filming sits in here, so a floor outside it is the wrong patch
LEVEL_SPREAD = 0.10  # metres: flat patches this close to the lowest (or highest) one are the same floor (or ceiling)
WALL_DEGREES = 20  # surface facing within this of a wall direction counts towards that wall
WALL_SHARE = 0.25  # a 2 cm slice needs this share of the busiest slice's surface to be a wall, not a stray bit
# metres (8 ft, the usual ceiling here): when the video never saw the ceiling, the shell is at least this tall, and
# room-shell.json says the height was assumed (take 4 filmed nothing above 1.83 m, which made a room too low to stand in)
ASSUMED_CEILING = 2.44
SHELL_TRIM = 0.06  # metres: splats this close to a wall or the ceiling go, and the clean shell shows there instead
FLOOR_TRIM = 0.03  # metres: the same for the floor, kept tighter so things standing on it keep their feet
BLOB = 0.30  # metres: splats wider than this are smears up close, not detail
FREE_SPACE = 0.08  # metres: splats further than this from the fused surface are floating haze
# The size MapAnything reads frames at, as pixels on the long side. 0 keeps its own size (518). Smaller frames let more
# of them into one run on an 8 GB GPU (about 81 at 280 px), so a video needs no parts and no joins.
INPUT_SIZE = int(os.environ.get("SIGHTSEEN_INPUT_SIZE", "0"))
INFER = dict(
    memory_efficient_inference=True,
    minibatch_size=1,  # smallest GPU memory use
    use_amp=True,
    amp_dtype="bf16",
    apply_mask=True,
    mask_edges=True,
    apply_confidence_mask=False,
    confidence_percentile=10,
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda" and not torch.cuda.is_bf16_supported():
    INFER["amp_dtype"] = "fp16"  # older GPUs (a T4, say) can't do bf16 maths

_loaded = {"key": None, "model": None}


def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loadedModel": _loaded["key"],
        "models": CHECKPOINTS,
    }


def _model(key):
    """Keeps one checkpoint on the GPU at a time: two don't fit in 8 GB."""
    if _loaded["key"] != key:
        _loaded.update(key=None, model=None)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        from mapanything.models import MapAnything

        model = MapAnything.from_pretrained(CHECKPOINTS[key]).to(DEVICE)
        model.eval()
        _loaded.update(key=key, model=model)
    return _loaded["model"]


def sharpness(frame):
    """How sharp a frame is: the variance of its Laplacian (edges stand out in a sharp frame, a blurred one is flat),
    measured on a copy at most 480 px on the long side so a 4K video doesn't slow it down."""
    small = cv2.resize(frame, None, fx=min(1.0, 480 / max(frame.shape[:2])), fy=min(1.0, 480 / max(frame.shape[:2])),
                       interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def extract_frames(video_path, frames_dir, fps, max_frames):
    """Takes about `fps` frames per second of video, never more than `max_frames`: the video is cut into that many
    equal slots and each slot gives its sharpest frame, so frames blurred by walking are left out."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Couldn't open the video. It may be damaged or in a format this PC can't read.")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = 10**9  # some files don't say how long they are; then each slot is 1/fps seconds of video
        slot_length = max(1, round(native_fps / fps))
    else:
        slots = int(min(max_frames, max(2, round(total / native_fps * fps))))
        slot_length = total / slots

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    paths, best, best_score, slot, index = [], None, -1.0, 0, 0

    def keep(frame):
        path = frames_dir / f"frame_{len(paths):04d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(str(path))

    while len(paths) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if int(index // slot_length) != slot:  # this frame starts the next slot: keep the last slot's sharpest
            keep(best)
            best, best_score, slot = None, -1.0, int(index // slot_length)
        score = sharpness(frame)
        if score > best_score:
            best, best_score = frame, score
        index += 1
    if best is not None and len(paths) < max_frames:
        keep(best)
    capture.release()

    if len(paths) < 2:
        raise ValueError("The video is too short: MapAnything needs at least 2 frames.")
    return paths


def plan(count, size):
    """Which frames go in each run: the whole-video run (`size` frames spread over all `count`), then equal parts of
    at most `size` frames each. A video that fits in one run is just that run."""
    if count <= size:
        return list(range(count)), []
    whole = sorted(set(np.linspace(0, count - 1, size).round().astype(int).tolist()))
    parts = [part.tolist() for part in np.array_split(np.arange(count), math.ceil(count / size))]
    return whole, parts


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _frame(pred):
    """One frame's points, colours, confidence and camera, kept as whole images so two runs' copies of the same
    frame line up pixel for pixel."""
    pts = _numpy(pred["pts3d"][0])  # (H, W, 3): metres, this run's first camera, OpenCV axes
    valid = _numpy(pred["mask"][0]).reshape(pts.shape[:2]) > 0.5
    conf = _numpy(pred["conf"][0]).reshape(pts.shape[:2])
    rgb = _numpy(pred["img_no_norm"][0])
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth = _numpy(pred["depth_z"][0]).reshape(pts.shape[:2])  # metres along the camera's view
    lens = _numpy(pred["intrinsics"][0])  # 3 x 3, in pixels of this frame
    pose = _numpy(pred["camera_poses"][0])  # 4 x 4, camera to this run's space
    return {"pts": pts, "valid": valid, "conf": conf, "rgb": rgb, "depth": depth, "lens": lens, "pose": pose}


def _rebuild(model, frames, indices, known=None):
    """One MapAnything run on some of the frames. `known` gives camera positions (frame number -> 4 x 4 camera to
    world, metres) MapAnything should take as given; the first frame must be one of them. Returns {frame number:
    that frame}, the seconds it took and the run's metric scale factor."""
    from mapanything.utils.image import load_images

    paths = [frames[i] for i in indices]
    views = load_images(paths, resize_mode="longest_side", size=INPUT_SIZE) if INPUT_SIZE else load_images(paths)
    for view, i in zip(views, indices):
        if known and i in known:
            view["camera_poses"] = torch.from_numpy(known[i].astype(np.float32))[None]
            view["is_metric_scale"] = torch.tensor([True])
    started = time.perf_counter()
    with torch.no_grad():
        predictions = model.infer(views, **INFER)
    seconds = time.perf_counter() - started
    scale_factor = float(_numpy(predictions[0]["metric_scaling_factor"]).reshape(-1)[0])
    run = {i: _frame(pred) for i, pred in zip(indices, predictions)}
    del predictions, views
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # VGGT-Long leaves out spots below a tenth of the run's median confidence when lining runs up
    floor = 0.1 * float(np.median(np.concatenate([f["conf"][f["valid"]] for f in run.values()])))
    for f in run.values():
        f["trusted"] = f["valid"] & (f["conf"] >= floor)
    return run, seconds, scale_factor


def _move(run, fit):
    """Moves a run's frames (points, cameras and depths) into the room's space."""
    resize, turn, move = fit
    for f in run.values():
        f["pts"] = (resize * f["pts"] @ turn.T + move).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = turn @ f["pose"][:3, :3]
        pose[:3, 3] = resize * turn @ f["pose"][:3, 3] + move
        f["pose"] = pose
        f["depth"] = f["depth"] * resize


def _umeyama(src, dst, weights):
    """The resize, turn and move that best put the src points onto dst, by weighted least squares (closed form)."""
    w = weights / weights.sum()
    mean_src, mean_dst = w @ src, w @ dst
    a, b = src - mean_src, dst - mean_dst
    u, d, vt = np.linalg.svd((b * w[:, None]).T @ a)
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1.0  # a turn, never a mirror image
    turn = u @ fix @ vt
    resize = float(np.trace(np.diag(d) @ fix) / (w @ (a * a).sum(axis=1)))
    move = mean_dst - resize * turn @ mean_src
    return resize, turn, move


def _fit(src, dst, weights, rounds=10):
    """VGGT-Long's join (eq. 1 to 3): fit, then refit with the spots that sit far off given less say (Huber),
    so a few wrong spots can't drag the whole part."""
    resize, turn, move = _umeyama(src, dst, weights)
    for _ in range(rounds):
        gaps = np.linalg.norm(dst - (resize * src @ turn.T + move), axis=1)
        limit = max(0.02, 2.0 * float(np.median(gaps)))  # a spot further off than this counts for limit / gap
        resize, turn, move = _umeyama(src, dst, weights * np.minimum(1.0, limit / np.maximum(gaps, 1e-9)))
    gaps = np.linalg.norm(dst - (resize * src @ turn.T + move), axis=1)
    return resize, turn, move, gaps


def _join(part, room, number):
    """Lines a part up with the room (the whole-video run), using the frames both have. Returns the fit (or None when
    there's too little to line up with) and the join's numbers."""
    shared = [i for i in part if i in room]
    src, dst, weights = [], [], []
    for i in shared:
        both = part[i]["trusted"] & room[i]["trusted"]
        src.append(part[i]["pts"][both])
        dst.append(room[i]["pts"][both])
        weights.append(np.sqrt(part[i]["conf"][both] * room[i]["conf"][both]))
    src, dst, weights = np.concatenate(src), np.concatenate(dst), np.concatenate(weights)

    report = {"part": number, "sharedFrames": len(shared), "spots": int(len(src))}
    if len(src) < 500:
        report["leftOut"] = True  # can't be lined up, so it stays out rather than landing in the wrong place
        return None, report
    if len(src) > 300_000:
        chosen = np.random.default_rng(number).choice(len(src), 300_000, replace=False)
        src, dst, weights = src[chosen], dst[chosen], weights[chosen]

    resize, turn, move, gaps = _fit(src.astype(np.float64), dst.astype(np.float64), weights.astype(np.float64))
    report.update(resize=round(resize, 4), gapCm=round(float(np.median(gaps)) * 100, 1),
                  within5cm=round(float(np.mean(gaps < 0.05)), 3))
    return (resize, turn.astype(np.float32), move.astype(np.float32)), report


def _cells(points, size):
    """Which `size` cube each point falls in, as one number per cube."""
    cells = np.floor(points / size).astype(np.int64)
    cells -= cells.min(axis=0)
    span = cells.max(axis=0) + 1
    return (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]


def _thin(points, colours, confidence, size):
    """One point per `size` cube, the most confident one: copies of the same spot from many frames only cost the
    browser, while an even spread keeps every part of the room equally solid."""
    order = np.argsort(-confidence, kind="stable")
    _, first = np.unique(_cells(points[order], size), return_index=True)
    keep = order[first]
    return points[keep], colours[keep]


def _fuse(room, path):
    """TSDF fusion: every frame's depth votes into one grid of small cubes, the grid keeps the weighted average (so
    frames that disagree a little blend into one wall), and marching cubes pulls the surface out as triangles.
    Writes the coloured mesh to `path` in three.js axes and returns its number of triangles."""
    import open3d as o3d

    sample = np.concatenate([f["pts"][f["trusted"]][::50] for f in room.values()])
    low = np.percentile(sample, 1, axis=0) - 0.15
    high = np.percentile(sample, 99, axis=0) + 0.15
    side = float((high - low).max())
    cube = max(0.02, side / GRID)
    cubes = int(math.ceil(side / cube))
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=cubes * cube, resolution=cubes, sdf_trunc=4 * cube,  # walls up to 4 cubes apart blend into one
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=low.reshape(3, 1).astype(np.float64))
    for f in room.values():
        height, width = f["depth"].shape
        depth = np.where(f["trusted"], f["depth"], 0.0).astype(np.float32)  # 0 = no depth here
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(f["rgb"])), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=2 * side, convert_rgb_to_intensity=False)
        lens = f["lens"]
        camera = o3d.camera.PinholeCameraIntrinsic(width, height, lens[0, 0], lens[1, 1], lens[0, 2], lens[1, 2])
        volume.integrate(rgbd, camera, np.linalg.inv(f["pose"].astype(np.float64)))

    mesh = volume.extract_triangle_mesh()
    pieces, sizes, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(sizes)[np.asarray(pieces)] < SMALLEST_PIECE)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > MAX_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(MAX_TRIANGLES)
    flip = np.array([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes (a half turn, so triangles keep their facing)
    write_ply(path, (np.asarray(mesh.vertices) * flip).astype(np.float32),
              (np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0) * 255.0).astype(np.uint8),
              np.asarray(mesh.triangles).astype(np.int32))
    return len(mesh.triangles)


def _quaternion(turn):
    """A 3 x 3 rotation as a unit quaternion (w, x, y, z)."""
    m = turn.astype(np.float64)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return w, x, y, z


def _turn(w, x, y, z):
    """A unit quaternion (w, x, y, z) as a 3 x 3 rotation, the other way round from _quaternion."""
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _full_size_lens(lens, seen, original):
    """MapAnything's lens numbers for the original frame instead of the copy it read. That copy was scaled until it
    covered the target size, then cropped in the middle (mapanything.utils.cropping.crop_resize_if_necessary), so
    undo the crop, then the scale. `seen` and `original` are (width, height)."""
    scale = max(seen[0] / original[0], seen[1] / original[1])
    crop_x = (original[0] * scale - seen[0]) / 2
    crop_y = (original[1] * scale - seen[1]) / 2
    return np.array([[lens[0, 0] / scale, 0, (lens[0, 2] + crop_x) / scale],
                     [0, lens[1, 1] / scale, (lens[1, 2] + crop_y) / scale],
                     [0, 0, 1]])


def _seen_lens(lens, seen, original):
    """The other way round: lens numbers for the original frame, made to fit the copy MapAnything read."""
    scale = max(seen[0] / original[0], seen[1] / original[1])
    crop_x = (original[0] * scale - seen[0]) / 2
    crop_y = (original[1] * scale - seen[1]) / 2
    return np.array([[lens[0, 0] * scale, 0, lens[0, 2] * scale - crop_x],
                     [0, lens[1, 1] * scale, lens[1, 2] * scale - crop_y],
                     [0, 0, 1]], dtype=np.float32)


def _solve_cameras(frames, indices, folder):
    """COLMAP on the frames: small features found in each frame and matched across every pair, then every camera solved
    together (incremental structure from motion), all with one lens since it's one phone. Returns COLMAP's biggest
    model, or None when it couldn't start one."""
    import pycolmap

    if folder.exists():
        shutil.rmtree(folder)
    (folder / "model").mkdir(parents=True)
    database = folder / "database.db"
    images = Path(frames[0]).parent
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"  # the camera Brush trains with
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = COLMAP_MAX_SIDE
    pycolmap.extract_features(database, images, image_names=[Path(frames[i]).name for i in indices],
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction)
    pycolmap.match_exhaustive(database)
    models = pycolmap.incremental_mapping(database, images, folder / "model")
    try:
        database.unlink()  # the matches: tens of MB nobody reads again
    except OSError:
        pass
    return max(models.values(), key=lambda model: model.num_reg_images()) if models else None


def _use_colmap(room, model, frames):
    """Swaps MapAnything's cameras for COLMAP's in every frame COLMAP placed, and leaves out the frames it didn't (their
    cameras would disagree). COLMAP has no sense of size, so each frame's MapAnything depth is compared with the depth
    of COLMAP's points seen in that frame: the median ratio over all frames turns COLMAP's units into metres, and each
    frame's own ratio corrects that frame's depth, so every depth agrees with COLMAP's cameras. The room's space stays
    the first placed frame's camera, in metres. Returns the numbers; the room is only changed when "used" is "colmap"."""
    names = {Path(frames[i]).name: i for i in room}
    placed = {names[image.name]: image for image in model.images.values() if image.has_pose and image.name in names}
    if len(placed) < COLMAP_MIN_SHARE * len(room):
        return {"used": "mapanything", "reason": f"COLMAP placed only {len(placed)} of {len(room)} frames"}

    ratios = {}  # frame number -> MapAnything's depth / COLMAP's depth at COLMAP's points
    for i, image in placed.items():
        seen = [(p.xy, p.point3D_id) for p in image.points2D if p.has_point3D()]
        if len(seen) < 20:
            continue
        pose, camera, depth = image.cam_from_world(), model.cameras[image.camera_id], room[i]["depth"]
        xyz = np.array([model.points3D[point].xyz for _, point in seen])
        far = xyz @ pose.rotation.matrix()[2] + pose.translation[2]
        height, width = depth.shape
        scale = max(width / camera.width, height / camera.height)
        u = np.floor(np.array([xy[0] for xy, _ in seen]) * scale - (camera.width * scale - width) / 2).astype(int)
        v = np.floor(np.array([xy[1] for xy, _ in seen]) * scale - (camera.height * scale - height) / 2).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (far > 0)
        u, v, far = u[inside], v[inside], far[inside]
        good = room[i]["trusted"][v, u]
        if good.sum() >= 20:
            ratios[i] = float(np.median(depth[v[good], u[good]] / far[good]))
    if not ratios:
        return {"used": "mapanything", "reason": "no COLMAP points landed on MapAnything's depth"}
    metres = float(np.median(list(ratios.values())))  # metres per COLMAP unit

    first = placed[min(placed)].cam_from_world()
    first_turn, first_shift = first.rotation.matrix(), np.asarray(first.translation)
    changes = {}
    for i, image in placed.items():
        pose, camera = image.cam_from_world(), model.cameras[image.camera_id]
        turn = pose.rotation.matrix()
        camera_to_room = np.eye(4, dtype=np.float32)
        camera_to_room[:3, :3] = first_turn @ turn.T
        camera_to_room[:3, 3] = metres * (first_turn @ (-turn.T @ np.asarray(pose.translation)) + first_shift)
        fx, fy, cx, cy = camera.params
        height, width = room[i]["depth"].shape
        lens = _seen_lens(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), (width, height),
                          (camera.width, camera.height))
        changes[i] = (camera_to_room, lens, metres / ratios[i] if i in ratios else 1.0)

    unplaced = [i for i in room if i not in placed]
    back = _add_back(room, changes, unplaced) if ADD_BACK else {}
    changes.update(back)
    left_out = [i for i in unplaced if i not in back]
    for i in left_out:
        del room[i]
    for i, (camera_to_room, lens, fix) in changes.items():
        f = room[i]
        f["depth"] = (f["depth"] * fix).astype(np.float32)
        rows, cols = np.indices(f["depth"].shape)
        in_camera = np.stack([(cols - lens[0, 2]) / lens[0, 0] * f["depth"],
                              (rows - lens[1, 2]) / lens[1, 1] * f["depth"], f["depth"]], axis=-1)
        f["pts"] = (in_camera @ camera_to_room[:3, :3].T + camera_to_room[:3, 3]).astype(np.float32)
        f["pose"], f["lens"] = camera_to_room, lens
    snapped = _snap_back(room, sorted(placed), sorted(back)) if SNAP_BACK and back else {}
    fixes = np.array([metres / ratio for ratio in ratios.values()])
    return {"used": "colmap", "framesPlaced": len(placed), "framesAddedBack": len(back), "framesLeftOut": len(left_out),
            "snapped": snapped,
            "errorPx": round(model.compute_mean_reprojection_error(), 3), "points": model.num_points3D(),
            "metresPerUnit": round(metres, 4),
            "depthFix": [round(float(np.percentile(fixes, 10)), 3), round(float(np.percentile(fixes, 90)), 3)]}


def _add_back(room, changes, unplaced):
    """COLMAP can't place frames of a blank wall or a fast blurred turn (on take 3: the door wall and the turn after
    it, 18 of 81 frames), and leaving them out leaves that side of the room black. Each one goes back with
    MapAnything's camera, moved onto COLMAP's by the resize, turn and move that best line up the nearest placed frames'
    two sets of cameras (their positions, and spots 1 m ahead and 1 m below them so the turn is pinned down too).
    Its lens is COLMAP's (one phone, one lens). Frames whose neighbours line up worse than ADD_BACK_MAX_GAP stay out.
    Returns frame number -> (camera_to_room, lens, depth resize), like the placed frames' changes."""
    placed = sorted(changes)
    if not placed:
        return {}
    marks = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.float64).T  # the camera, ahead, below
    back = {}
    for i in unplaced:
        nearest = sorted(placed, key=lambda j: abs(j - i))[:ADD_BACK_NEIGHBOURS]
        src = np.concatenate([(room[j]["pose"].astype(np.float64) @ marks)[:3].T for j in nearest])
        dst = np.concatenate([(changes[j][0].astype(np.float64) @ marks)[:3].T for j in nearest])
        resize, turn, move = _umeyama(src, dst, np.ones(len(src)))
        gap = float(np.median(np.linalg.norm(resize * src @ turn.T + move - dst, axis=1)))
        if gap > ADD_BACK_MAX_GAP:
            continue
        camera_to_room = np.eye(4, dtype=np.float32)
        camera_to_room[:3, :3] = turn @ room[i]["pose"][:3, :3]
        camera_to_room[:3, 3] = resize * turn @ room[i]["pose"][:3, 3] + move
        back[i] = (camera_to_room, changes[nearest[0]][1], resize)
    return back


def _snap_back(room, placed, back):
    """An added-back frame's camera is MapAnything's, lined up only by where the nearest placed frames' cameras are, so
    after a fast turn or a blank wall its view can land 0.3-0.7 m off (take 4: the desk close-ups, the bed's corner, the
    door and the wardrobe; the fused room held two desks and two beds). Each one is snapped onto the placed frames'
    own points: ICP, point to plane, reaching from 30 cm down to 5 cm, kept only when more of its points then lie on
    the placed frames' surface and the camera moved a believable amount. The frame overlapping the reference most goes
    first, and every frame joins the reference once done, so a frame that only sees what another added-back frame saw
    (the wardrobe, after the corner by it) snaps onto that one. A frame that can't be snapped (a wrong match: too far,
    too much turn) and lies mostly off the placed surface is marked "leftOut": it adds nothing to the fused room, since
    its view would only add a second copy of what it saw (it keeps its camera and its number, which later steps use).
    Moves each snapped frame's camera and points in place; returns frame number -> what happened."""
    import open3d as o3d

    registration = o3d.pipelines.registration

    def cloud(frames, normals=False):
        points = np.concatenate([room[i]["pts"][room[i]["trusted"]] for i in frames]).astype(np.float64)
        found = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(SNAP_VOXEL)
        if normals:
            found.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * SNAP_VOXEL, max_nn=30))
        return found

    reference, done, todo, report = cloud(placed, normals=True), list(placed), list(back), {}
    while todo:
        sources = {i: cloud([i]) for i in todo}
        before = {i: registration.evaluate_registration(sources[i], reference, SNAP_NEAR).fitness for i in todo}
        i = max(todo, key=lambda k: before[k])
        todo.remove(i)
        change = np.eye(4)
        for reach in SNAP_REACH:
            change = registration.registration_icp(sources[i], reference, reach, change,
                                                   registration.TransformationEstimationPointToPlane()).transformation
        after = registration.evaluate_registration(sources[i], reference, SNAP_NEAR, change).fitness
        camera = room[i]["pose"][:3, 3].astype(np.float64)
        moved = float(np.linalg.norm(change[:3, :3] @ camera + change[:3, 3] - camera))
        turned = float(np.degrees(np.arccos(np.clip((np.trace(change[:3, :3]) - 1) / 2, -1.0, 1.0))))
        kept = after >= before[i] + SNAP_GAIN and moved <= SNAP_MOVE and turned <= SNAP_TURN
        if kept:
            f = room[i]
            f["pts"] = (f["pts"] @ change[:3, :3].T.astype(np.float32) + change[:3, 3].astype(np.float32))
            f["pose"] = (change @ f["pose"].astype(np.float64)).astype(np.float32)
        left_out = not kept and before[i] < SNAP_LEAVE
        report[i] = {"on": round(before[i], 3), "onAfter": round(after, 3), "movedCm": round(100 * moved, 1),
                     "turnedDeg": round(turned, 1), "kept": bool(kept), "leftOut": bool(left_out)}
        if left_out:
            room[i]["leftOut"] = True
            continue
        done.append(i)
        reference = cloud(done, normals=True)
    return report


def _write_splat_data(room, points, colours, folder, frames):
    """Writes what a Gaussian splat trainer (Brush) starts from, in COLMAP's text format: every frame at its original
    size (more detail than MapAnything's smaller copy), each camera's lens and position, and the room's points as the
    starting splats. OpenCV axes throughout, which is COLMAP's own convention."""
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "images").mkdir(parents=True)
    sparse = folder / "sparse" / "0"
    sparse.mkdir(parents=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy"]
    images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then a line of 2D points (none here)"]
    for number, (i, f) in enumerate(sorted(room.items()), start=1):
        name = Path(frames[i]).with_suffix(".png").name  # Brush 0.3 fails on these JPGs ("early eof"); PNG loads fine
        original = cv2.imread(frames[i])
        height, width = original.shape[:2]
        lens = _full_size_lens(f["lens"], (f["rgb"].shape[1], f["rgb"].shape[0]), (width, height))
        shrink = min(1.0, SPLAT_MAX_SIDE / max(width, height))
        if shrink < 1.0:  # a 4K phone video: Brush trains at 1920 px at most, so don't store more
            width, height = round(width * shrink), round(height * shrink)
            original = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
            lens[:2] *= shrink
        cv2.imwrite(str(folder / "images" / name), original)
        cameras.append(f"{number} PINHOLE {width} {height} {lens[0, 0]:.4f} {lens[1, 1]:.4f} {lens[0, 2]:.4f} {lens[1, 2]:.4f}")
        to_camera = f["pose"][:3, :3].T.astype(np.float64)  # COLMAP stores world-to-camera
        shift = -to_camera @ f["pose"][:3, 3].astype(np.float64)
        qw, qx, qy, qz = _quaternion(to_camera)
        images.append(f"{number} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {shift[0]:.6f} {shift[1]:.6f} {shift[2]:.6f} {number} {name}")
        images.append("")
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(images) + "\n")
    start = np.random.default_rng(0).choice(len(points), min(len(points), 300_000), replace=False)
    lines = ["# POINT3D_ID X Y Z R G B ERROR (no tracks)"]
    for number, k in enumerate(start, start=1):
        x, y, z = points[k]
        r, g, b = colours[k]
        lines.append(f"{number} {x:.4f} {y:.4f} {z:.4f} {r} {g} {b} 0")
    (sparse / "points3D.txt").write_text("\n".join(lines) + "\n")


def write_cameras(out):
    """cameras.json: where the phone stood for every frame the splats trained on, in three.js axes, so the viewer can
    walk the video's path (splats only look like the video from near where it was filmed). Read back from splat-data/,
    so it also works for rooms built before this existed: `python reconstruct.py cameras <request folder>`."""
    sparse = Path(out) / "splat-data" / "sparse" / "0"
    lenses = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            number, _, width, height, fx, fy, *_ = line.split()
            lenses[number] = (int(width), int(height), float(fx), float(fy))
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV axes to three.js axes, for the room and for the camera alike
    spots = []
    for line in (sparse / "images.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 10 or line.startswith("#"):
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        to_camera = _turn(qw, qx, qy, qz)  # COLMAP stores the world-to-camera turn
        position = flip @ (-to_camera.T @ np.array([tx, ty, tz]))
        turn = flip @ to_camera.T @ flip
        width, height, fx, fy = lenses[parts[8]]
        spots.append({
            "frame": Path(parts[9]).stem,
            "position": [round(float(v), 4) for v in position],
            "quaternion": [round(float(v), 6) for v in _quaternion(turn)],  # w, x, y, z
            "fovX": round(math.degrees(2 * math.atan(width / 2 / fx)), 2),
            "fovY": round(math.degrees(2 * math.atan(height / 2 / fy)), 2),
        })
    spots.sort(key=lambda spot: spot["frame"])
    (Path(out) / "cameras.json").write_text(json.dumps(spots))
    return len(spots)


def _level(planes):
    """A floor or ceiling usually comes out as several patches a few cm apart (rugs, noise), so take the patches within
    LEVEL_SPREAD of the first one, use their height weighted by size, and fit one plane through all their points for
    the normal. Returns (height, normal, points) like a single plane, or None."""
    if not planes:
        return None
    group = [plane for plane in planes if abs(plane[0] - planes[0][0]) < LEVEL_SPREAD]
    heights = np.array([plane[0] for plane in group])
    sizes = np.array([plane[2] for plane in group])
    order = np.argsort(heights)
    middle = order[np.searchsorted(np.cumsum(sizes[order]), sizes.sum() / 2)]
    together = np.concatenate([plane[3] for plane in group])
    normal = np.linalg.svd(together - together.mean(axis=0), full_matrices=False)[2][2]
    if normal @ group[0][1] < 0:
        normal = -normal
    return float(heights[middle]), normal, int(sizes.sum())


def _flat_planes(points, up, rounds=12, min_points=200):
    """The big flat patches facing up or down (floor, bed, desk, ceiling), one RANSAC plane at a time. Returns
    (height along up, plane normal, how many points, the points) for each, lowest first."""
    import open3d as o3d

    o3d.utility.random.seed(0)  # RANSAC picks random points; the same room should give the same floor every time
    left = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    planes = []
    for _ in range(rounds):
        if len(left.points) < min_points:
            break
        (a, b, c, _), inliers = left.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        on_plane = np.asarray(left.points)[inliers]
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if normal @ up < 0:
            normal = -normal
        if len(inliers) >= min_points and normal @ up > math.cos(math.radians(PLANE_DEGREES)):
            planes.append((float(np.median(on_plane @ up)), normal, len(inliers), on_plane))
        left = left.select_by_index(inliers, invert=True)
    return sorted(planes, key=lambda plane: plane[0])


def write_room_frame(out):
    """room-frame.json: which way is up and where the floor is, so the viewer can stand the room upright with the floor
    at height 0 (the room's space is the first frame's camera, which was tilted). Walking on the website and standing
    in VR both need it. Up comes from the cameras: a phone's sideways axis stays level while it's carried, so up is the
    direction square to all of them (nerfstudio's "vertical" method). On take 3 that agreed with the ceiling and the
    bed to 1.5 degrees, better than the floor's own patches did, so the floor only sets the height: the lowest big flat
    patch facing up. Also the ceiling (the highest one) and how high the phone was held. Three.js axes, metres.
    `python reconstruct.py frame <request folder>` makes it for older rooms."""
    import open3d as o3d

    out = Path(out)
    spots = json.loads((out / "cameras.json").read_text())
    turns = np.array([_turn(*spot["quaternion"]) for spot in spots])
    positions = np.array([spot["position"] for spot in spots])
    up = np.linalg.svd(turns[:, :, 0])[2][2]  # the direction least along any camera's sideways axis
    if up @ turns[:, :, 1].mean(axis=0) < 0:
        up = -up

    if (out / "room-surface.ply").exists():
        mesh = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
        mesh.compute_vertex_normals()
        points, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    else:
        cloud = o3d.io.read_point_cloud(str(out / "room.ply"))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    flat = np.abs(normals @ up) > math.cos(math.radians(FLAT_DEGREES))
    planes = _flat_planes(points[flat], up)
    held = float(np.median(positions @ up))  # the phone's height, in the room's own space
    below = [plane for plane in planes if plane[0] < held]
    above = [plane for plane in planes if plane[0] > held]

    floor = _level(below)
    ceiling = _level(above[::-1])
    floor_height = floor[0] if floor else float(np.percentile(points @ up, 2))
    ceiling_height = ceiling[0] if ceiling else None

    ahead = turns[0] @ np.array([0.0, 0.0, -1.0])  # where the first frame looked, kept as "straight ahead"
    ahead -= (ahead @ up) * up
    ahead /= np.linalg.norm(ahead)
    level = np.array([np.cross(ahead, up), up, -ahead])  # rows: the upright room's x (right), y (up), z (back)
    eye = held - floor_height
    frame = {
        "quaternion": [round(float(v), 6) for v in _quaternion(level)],  # w, x, y, z
        "floorHeight": round(floor_height, 4),  # along up, in the room's own space; the viewer moves it to 0
        "eyeHeight": round(eye, 3),
        "ceilingHeight": round(ceiling_height - floor_height, 3) if ceiling_height is not None else None,
        "floorFound": floor is not None,
        "floorPoints": floor[2] if floor else 0,
        # How far the floor's own fitted plane leans from the cameras' up: a check, not used
        "floorTiltDegrees": round(math.degrees(math.acos(min(1.0, float(floor[1] @ up)))), 2) if floor else None,
        "trusted": floor is not None and MIN_EYE <= eye <= MAX_EYE,
    }
    (out / "room-frame.json").write_text(json.dumps(frame, indent=2))
    return frame


def _wall_at(along, side):
    """Where a wall sits along one of the room's axes: walls are the outermost big vertical surfaces, so it's the
    outermost 2 cm slice (on that side) holding at least WALL_SHARE of the busiest slice's surface, refined to the
    median of the points within 3 cm of it. Furniture fronts sit further in and are passed over."""
    slices = np.arange(along.min(), along.max() + 0.02, 0.02)
    counts, edges = np.histogram(along, bins=slices)
    busy = np.nonzero(counts >= WALL_SHARE * counts.max())[0]
    k = busy[-1] if side > 0 else busy[0]
    middle = (edges[k] + edges[k + 1]) / 2
    return float(np.median(along[np.abs(along - middle) < 0.03]))


def write_room_shell(out):
    """room-shell.json: the room's shell (four walls, floor and ceiling) as a box, found in the upright room, so the
    viewer can draw clean walls where nobody filmed them up close, hide what lies beyond them, and keep a walker
    inside. The furniture and everything filmed stay as the real splats. The walls' directions come from the surface
    that faces sideways (most of it is wall, and walls meet at right angles, so the angle that best fits all of them
    on a 90-degree turn wins); each wall's colour is the median of the surface on it.
    `python reconstruct.py shell <request folder>` makes it for older rooms. Needs room-frame.json."""
    import open3d as o3d

    out = Path(out)
    frame = json.loads((out / "room-frame.json").read_text())
    level = _turn(*frame["quaternion"])
    mesh = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices) @ level.T - np.array([0.0, frame["floorHeight"], 0.0])  # upright, floor at 0
    normals = np.asarray(mesh.vertex_normals) @ level.T
    colours = np.asarray(mesh.vertex_colors)

    sideways = np.abs(normals[:, 1]) < math.sin(math.radians(FLAT_DEGREES))
    turn = np.arctan2(normals[sideways, 2], normals[sideways, 0])
    yaw = float(np.angle(np.exp(4j * turn).mean()) / 4)  # the walls' direction, on a 90-degree turn
    axes = [np.array([math.cos(yaw), 0.0, math.sin(yaw)]), np.array([-math.sin(yaw), 0.0, math.cos(yaw)])]
    height = frame["ceilingHeight"] or max(float(np.percentile(points[:, 1], 99)), ASSUMED_CEILING)

    walls, faces = [], {}
    for axis, names in zip(axes, (("left", "right"), ("front", "back"))):
        facing = sideways & (np.abs(normals @ axis) > math.cos(math.radians(WALL_DEGREES)))
        along = points[facing] @ axis
        for side, name in zip((-1, 1), names):
            at = _wall_at(along, side)
            near = facing & (np.abs(points @ axis - at) < 0.05)
            walls.append(at)
            faces[name] = [int(v) for v in np.median(colours[near], axis=0) * 255] if near.any() else [200, 200, 200]
    floor = (points[:, 1] < 0.05) & (normals[:, 1] > math.cos(math.radians(FLAT_DEGREES)))
    ceiling = (points[:, 1] > height - 0.05) & (normals[:, 1] < -math.cos(math.radians(FLAT_DEGREES)))
    faces["floor"] = [int(v) for v in np.median(colours[floor], axis=0) * 255] if floor.any() else [150, 150, 150]
    faces["ceiling"] = [int(v) for v in np.median(colours[ceiling], axis=0) * 255] if ceiling.any() else [230, 230, 230]

    width, length = walls[1] - walls[0], walls[3] - walls[2]
    centre = axes[0] * (walls[0] + walls[1]) / 2 + axes[1] * (walls[2] + walls[3]) / 2 + np.array([0, height / 2, 0])
    shell = {
        "centre": [round(float(v), 4) for v in centre],
        "size": [round(width, 3), round(height, 3), round(length, 3)],  # along the walls' x, up, along their z
        "yawDegrees": round(math.degrees(yaw), 2),
        "faces": faces,  # RGB 0-255: left and right walls (x), front and back (z), floor, ceiling
        "ceilingAssumed": not frame["ceilingHeight"],
    }
    (out / "room-shell.json").write_text(json.dumps(shell, indent=2))
    return shell


def trim_splats(out):
    """Leaves the splats only what isn't the shell. Drops the ones beyond the walls, floor and ceiling (smears that
    training pushed out of the room), the ones within SHELL_TRIM of them (the shell draws those surfaces, clean, and
    seen up close a filmed wall's splats are the smear), and blobs wider than BLOB (they smear when you walk close).
    The untrimmed splats stay as room-splats-full.ply. Needs room-frame.json and room-shell.json.
    `python reconstruct.py trim <request folder>` trims an older room."""
    out = Path(out)
    full = out / "room-splats-full.ply"
    if not full.exists():
        shutil.copyfile(out / "room-splats.ply", full)
    data = full.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    properties = [line.split() for line in header.splitlines() if line.startswith("property")]
    if any(kind != "float" for _, kind, _ in properties):
        raise ValueError("room-splats-full.ply has non-float properties; this reader only knows Brush's layout")
    column = {name: k for k, (_, _, name) in enumerate(properties)}
    splats = np.frombuffer(data, dtype=np.float32, offset=end).reshape(-1, len(properties))

    frame = json.loads((out / "room-frame.json").read_text())
    shell = json.loads((out / "room-shell.json").read_text())
    level = _turn(*frame["quaternion"])
    centres = splats[:, [column["x"], column["y"], column["z"]]] * np.array([1.0, -1.0, -1.0])  # OpenCV to three.js
    upright = centres @ level.T - np.array([0.0, frame["floorHeight"], 0.0])
    yaw = math.radians(shell["yawDegrees"])
    offset = upright - np.array(shell["centre"])
    across = offset @ np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    along = offset @ np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
    width, height, length = shell["size"]
    inside = ((np.abs(across) < width / 2 - SHELL_TRIM) & (np.abs(along) < length / 2 - SHELL_TRIM)
              & (upright[:, 1] > FLOOR_TRIM) & (upright[:, 1] < height - SHELL_TRIM))
    widest = np.exp(splats[:, [column["scale_0"], column["scale_1"], column["scale_2"]]]).max(axis=1)
    # Real things have a surface, and the fused surface has every spot the frames' depth agreed on. Splats floating
    # further than FREE_SPACE from it are haze: training's way of explaining views, which looks like fog in between.
    import open3d as o3d

    surface = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
    gap = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres.astype(np.float64)))
                     .compute_point_cloud_distance(o3d.geometry.PointCloud(surface.vertices)))
    floating = gap > FREE_SPACE
    keep = inside & (widest < BLOB) & ~floating

    count = int(keep.sum())
    trimmed = header.replace(f"element vertex {len(splats)}\n", f"element vertex {count}\n").encode("ascii")
    (out / "room-splats.ply").write_bytes(trimmed + splats[keep].tobytes())
    return {"kept": count, "outsideOrOnShell": int((~inside).sum()), "blobs": int((inside & (widest >= BLOB)).sum()),
            "floating": int((inside & (widest < BLOB) & floating).sum())}


def _train_splats(out):
    """Gaussian splats with Brush: lots of small coloured blobs, adjusted until the room seen from each camera looks
    like that frame, so the room looks like the video. Trains on splat-data/, writes room-splats.ply and a log, and
    returns the seconds it took."""
    if not BRUSH.exists():
        raise FileNotFoundError(f"Brush isn't installed at {BRUSH} (see worker/setup.sh)")
    started = time.perf_counter()
    with open(out / "splats.log", "w", encoding="utf-8") as log:
        subprocess.run([str(BRUSH), str(out / "splat-data"), "--total-steps", str(SPLAT_STEPS),
                        "--max-splats", str(SPLAT_MAX), "--sh-degree", str(SPLAT_SH),
                        "--export-every", str(SPLAT_STEPS), "--export-path", str(out),
                        "--export-name", "room-splats.ply"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (out / "room-splats.ply").exists():
        raise RuntimeError("Brush finished without writing room-splats.ply (see splats.log)")
    return time.perf_counter() - started


def write_ply(path, points, colours, triangles=None):
    """Binary PLY with 32-bit floats (WebGL can't draw 64-bit ones): coloured points, and triangles for a surface."""
    count = len(points)
    data = np.empty(count, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    )
    if triangles is not None:
        faces = np.empty(len(triangles), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        faces["n"] = 3
        faces["a"], faces["b"], faces["c"] = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        header += f"element face {len(triangles)}\nproperty list uchar int vertex_indices\n"
    with open(path, "wb") as f:
        f.write((header + "end_header\n").encode("ascii"))
        f.write(data.tobytes())
        if triangles is not None:
            f.write(faces.tobytes())


def run(video_path, output_dir, model_key, fps, frames_per_part, stage):
    """One video, as many MapAnything runs as it needs. `stage(text)` reports progress to whoever is asking."""
    out = Path(output_dir)

    stage("taking frames from the video")
    frames = extract_frames(video_path, out / "frames", fps, MAX_TOTAL_FRAMES)
    whole, parts = plan(len(frames), frames_per_part)

    first_load = _loaded["key"] != model_key
    stage("loading the 3D model" + (" (the first load takes a minute)" if first_load else ""))
    model = _model(model_key)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if parts:
        stage(f"rebuilding the whole room from {len(whole)} frames spread over the video")
    else:
        stage(f"rebuilding the room from {len(whole)} frames")
    anchor, seconds, scale_factor = _rebuild(model, frames, whole)

    room = dict(anchor)  # frame number -> that frame, in the whole-video run's space
    joins = []
    for number, indices in enumerate(parts, start=1):
        stage(f"adding detail: part {number} of {len(parts)} ({len(indices)} frames)")
        known = {i: anchor[i]["pose"] for i in indices if i in anchor} if HAND_OVER_POSES else None
        if known:
            first = next(i for i in indices if i in known)
            indices = [first] + [i for i in indices if i != first]  # MapAnything needs a known camera first
        part, part_seconds, _ = _rebuild(model, frames, indices, known)
        seconds += part_seconds
        fit, report = _join(part, anchor, number)
        joins.append(report)
        if fit is None:
            continue  # this stretch keeps the whole-video run's frames
        _move(part, fit)
        room.update(part)  # a part's own copy of a shared frame replaces the whole-video run's
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else None

    cameras = {"used": "mapanything"}
    if USE_COLMAP:
        stage("lining up the cameras precisely (COLMAP)")
        started = time.perf_counter()
        try:
            model = _solve_cameras(frames, sorted(room), out / "colmap")
            if model is None:
                cameras = {"used": "mapanything", "reason": "COLMAP couldn't start a model"}
            else:
                cameras = _use_colmap(room, model, frames)
        except Exception as error:  # noqa: BLE001 - MapAnything's cameras are the backup, so COLMAP failing doesn't fail the room
            traceback.print_exc()
            cameras = {"used": "mapanything", "reason": f"{type(error).__name__}: {error}"}
        cameras["seconds"] = round(time.perf_counter() - started, 1)

    stage("making the solid surface")
    fused = {i: f for i, f in room.items() if not f.get("leftOut")}  # frames whose cameras couldn't be trusted add nothing
    try:
        triangles = _fuse(fused, out / "room-surface.ply")
        surface, surface_error = "room-surface.ply", None
    except Exception as error:  # noqa: BLE001 - the dots still work, so a failed surface doesn't fail the room
        traceback.print_exc()
        triangles, surface, surface_error = 0, None, f"{type(error).__name__}: {error}"

    stage("saving the room")
    points = np.concatenate([f["pts"][f["valid"]] for f in fused.values()])
    colours = np.concatenate([f["rgb"][f["valid"]] for f in fused.values()])
    confidence = np.concatenate([f["conf"][f["valid"]] for f in fused.values()])
    keep = confidence >= np.percentile(confidence, 10)  # drop the least confident tenth, as MapAnything's own mask does
    points, colours, confidence = points[keep], colours[keep], confidence[keep]
    points, colours = _thin(points, colours, confidence, DOT)
    if len(points) > MAX_POINTS:
        chosen = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
        points, colours = points[chosen], colours[chosen]
    filled = int(len(np.unique(_cells(points, CUBE))))
    _write_splat_data(room, points, colours, out / "splat-data", frames)
    write_cameras(out)
    room.clear()
    points = points * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # OpenCV axes to three.js axes
    write_ply(out / "room.ply", points, colours)
    try:
        room_frame = write_room_frame(out)
        room_frame["shell"] = write_room_shell(out)
    except Exception as error:  # noqa: BLE001 - the room still opens, just as filmed (tilted, no shell), without it
        traceback.print_exc()
        room_frame = {"error": f"{type(error).__name__}: {error}"}

    # A box around the 3D along the first frame's axes, ignoring the 2% of stray points at each end.
    # It's not the room's walls (those aren't lined up with the camera); the viewer's measuring tool is for that.
    low, high = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    size = high - low

    stage("making it look like the video (a few minutes)")
    (out / "room-splats.ply").unlink(missing_ok=True)  # a failed run mustn't leave an older room's splats behind
    (out / "room-splats-full.ply").unlink(missing_ok=True)  # nor an older untrimmed copy for trim_splats to read
    try:
        splat_seconds = _train_splats(out)
        splats, splats_error = "room-splats.ply", None
    except Exception as error:  # noqa: BLE001 - the surface and dots still work, so failed splats don't fail the room
        traceback.print_exc()
        splat_seconds, splats, splats_error = None, None, f"{type(error).__name__}: {error}"
    if splats and "shell" in room_frame:
        try:
            room_frame["trim"] = trim_splats(out)
        except Exception as error:  # noqa: BLE001 - the untrimmed splats still work
            traceback.print_exc()
            room_frame["trim"] = {"error": f"{type(error).__name__}: {error}"}

    result = {
        "frames": len(frames),
        "wholeVideoFrames": len(whole),
        "parts": len(parts),
        "seconds": round(seconds, 1),
        "peakVramGb": round(peak_gb, 2) if peak_gb is not None else None,
        "scaleFactor": round(scale_factor, 4),
        "points": int(len(points)),
        "filledCubes": filled,
        "triangles": triangles,
        "sizeX": round(float(size[0]), 3),
        "sizeY": round(float(size[1]), 3),
        "sizeZ": round(float(size[2]), 3),
        "ply": "room.ply",
        "surface": surface,
        "surfaceError": surface_error,
        "splats": splats,
        "splatSeconds": round(splat_seconds, 1) if splat_seconds is not None else None,
        "splatsError": splats_error,
        "model": model_key,
        "joins": joins,
        "cameras": cameras,
        "roomFrame": room_frame,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def explain(error):
    """Turns an exception into a sentence the student's page can show."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return ("The GPU ran out of memory. Lower sightseen.worker.max-frames in application.properties, "
                "restart Spring Boot and press Try again.")
    if isinstance(error, (ValueError, FileNotFoundError)):
        return str(error)
    return f"The 3D step failed: {type(error).__name__}: {error}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "cameras":
        print(write_cameras(sys.argv[2]), "camera spots written")
    elif len(sys.argv) == 3 and sys.argv[1] == "frame":
        print(json.dumps(write_room_frame(sys.argv[2]), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "shell":
        print(json.dumps(write_room_shell(sys.argv[2]), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "trim":
        print(json.dumps(trim_splats(sys.argv[2]), indent=2))
    else:
        sys.exit("usage: python reconstruct.py cameras|frame|shell|trim <request folder>")
