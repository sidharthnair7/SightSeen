"""Video in, 3D room out: frames from the video, MapAnything, then the room as a solid surface and as coloured points.

A GPU with 8 GB can only take about 24 frames in one MapAnything run. A longer video gets one run of 24 frames spread
over the whole video first: that run sets the room's shape and size (the one-run version measured right). The video is
then split into equal parts, and each part lines up with that whole-video run on the frames they share (the fit is
VGGT-Long's, arXiv 2507.16443, Section 3.1). Lining every part up with the same run means errors can't stack up from
part to part. Every join reports how well it fit.

All the frames are then fused into one surface (TSDF fusion: each frame's depth votes into a grid of small cubes, the
grid keeps the average, and marching cubes pulls out triangles), so the walls are solid instead of dots.

Last, Brush trains Gaussian splats on the frames and those cameras, so the room looks like the video.

Everything is written into the request's folder, next to the video:
  frames/            the frames MapAnything saw
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


def extract_frames(video_path, frames_dir, fps, max_frames):
    """Takes about `fps` frames per second of video, spread evenly, and never more than `max_frames`."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Couldn't open the video. It may be damaged or in a format this PC can't read.")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        wanted_count = int(min(max_frames, max(2, round(total / native_fps * fps))))
        wanted = set(np.linspace(0, total - 1, wanted_count).round().astype(int).tolist())
    else:
        # Some files don't say how many frames they have; take every nth frame instead
        step = max(1, round(native_fps / fps))
        wanted = None

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    paths, index = [], 0
    while len(paths) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if (wanted is not None and index in wanted) or (wanted is None and index % step == 0):
            path = frames_dir / f"frame_{len(paths):04d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(path))
        index += 1
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


def _rebuild(model, frames, indices):
    """One MapAnything run on some of the frames. Returns {frame number: that frame}, the seconds it took and the
    run's metric scale factor."""
    from mapanything.utils.image import load_images

    views = load_images([frames[i] for i in indices])
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


def _write_splat_data(room, points, colours, folder):
    """Writes what a Gaussian splat trainer (Brush) starts from, in COLMAP's text format: every frame as MapAnything
    saw it, each camera's lens and position, and the room's points as the starting splats. OpenCV axes throughout,
    which is COLMAP's own convention."""
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "images").mkdir(parents=True)
    sparse = folder / "sparse" / "0"
    sparse.mkdir(parents=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy"]
    images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then a line of 2D points (none here)"]
    for number, (i, f) in enumerate(sorted(room.items()), start=1):
        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(folder / "images" / name), cv2.cvtColor(f["rgb"], cv2.COLOR_RGB2BGR))
        height, width = f["rgb"].shape[:2]
        lens = f["lens"]
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
        part, part_seconds, _ = _rebuild(model, frames, indices)
        seconds += part_seconds
        fit, report = _join(part, anchor, number)
        joins.append(report)
        if fit is None:
            continue  # this stretch keeps the whole-video run's frames
        _move(part, fit)
        room.update(part)  # a part's own copy of a shared frame replaces the whole-video run's
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else None

    stage("making the solid surface")
    try:
        triangles = _fuse(room, out / "room-surface.ply")
        surface, surface_error = "room-surface.ply", None
    except Exception as error:  # noqa: BLE001 - the dots still work, so a failed surface doesn't fail the room
        traceback.print_exc()
        triangles, surface, surface_error = 0, None, f"{type(error).__name__}: {error}"

    stage("saving the room")
    points = np.concatenate([f["pts"][f["valid"]] for f in room.values()])
    colours = np.concatenate([f["rgb"][f["valid"]] for f in room.values()])
    confidence = np.concatenate([f["conf"][f["valid"]] for f in room.values()])
    keep = confidence >= np.percentile(confidence, 10)  # drop the least confident tenth, as MapAnything's own mask does
    points, colours, confidence = points[keep], colours[keep], confidence[keep]
    points, colours = _thin(points, colours, confidence, DOT)
    if len(points) > MAX_POINTS:
        chosen = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
        points, colours = points[chosen], colours[chosen]
    filled = int(len(np.unique(_cells(points, CUBE))))
    _write_splat_data(room, points, colours, out / "splat-data")
    room.clear()
    points = points * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # OpenCV axes to three.js axes
    write_ply(out / "room.ply", points, colours)

    # A box around the 3D along the first frame's axes, ignoring the 2% of stray points at each end.
    # It's not the room's walls (those aren't lined up with the camera); the viewer's measuring tool is for that.
    low, high = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    size = high - low

    stage("making it look like the video (a few minutes)")
    (out / "room-splats.ply").unlink(missing_ok=True)  # a failed run mustn't leave an older room's splats behind
    try:
        splat_seconds = _train_splats(out)
        splats, splats_error = "room-splats.ply", None
    except Exception as error:  # noqa: BLE001 - the surface and dots still work, so failed splats don't fail the room
        traceback.print_exc()
        splat_seconds, splats, splats_error = None, None, f"{type(error).__name__}: {error}"

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
