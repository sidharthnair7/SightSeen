"""Video in, 3D room out: frames from the video, MapAnything, then a coloured point cloud (room.ply) and its numbers.

A GPU with 8 GB can only take about 24 frames in one MapAnything run, so a longer video is split into equal parts
that each share half their frames with the part before. The shared frames line each part up with the room so far,
the way VGGT-Long does it (arXiv 2507.16443, Section 3.1). Every join reports how well it fit.

Everything is written into the request's folder, next to the video:
  frames/       the frames MapAnything saw
  room.ply      the 3D room as coloured points, in metres, in three.js axes (x right, y up, z towards the viewer)
  result.json   the run's numbers (the same dict run() returns)
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def split(count, size):
    """Equal parts of `size` frames, spread over `count` frames so each part shares at least half its frames
    with the part before (VGGT-Long's overlap). A video that fits in one run stays one part."""
    if count <= size:
        return [list(range(count))]
    step = size - size // 2
    parts = math.ceil((count - size) / step) + 1
    starts = np.linspace(0, count - size, parts).round().astype(int)
    return [list(range(start, start + size)) for start in starts]


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _frame(pred):
    """One frame's points, colours and confidence, kept as whole images so two parts' copies of the same frame
    line up pixel for pixel."""
    pts = _numpy(pred["pts3d"][0])  # (H, W, 3): metres, this part's first camera, OpenCV axes
    valid = _numpy(pred["mask"][0]).reshape(pts.shape[:2]) > 0.5
    conf = _numpy(pred["conf"][0]).reshape(pts.shape[:2])
    rgb = _numpy(pred["img_no_norm"][0])
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return {"pts": pts, "valid": valid, "conf": conf, "rgb": rgb}


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
    """Lines a new part up with the room so far, using the frames both have seen. Returns the fit (or None when
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


def write_ply(path, points, colours):
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
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def run(video_path, output_dir, model_key, fps, frames_per_part, stage):
    """One video, as many MapAnything runs as it needs. `stage(text)` reports progress to whoever is asking."""
    out = Path(output_dir)

    stage("taking frames from the video")
    frames = extract_frames(video_path, out / "frames", fps, MAX_TOTAL_FRAMES)
    parts = split(len(frames), frames_per_part)

    first_load = _loaded["key"] != model_key
    stage("loading the 3D model" + (" (the first load takes a minute)" if first_load else ""))
    model = _model(model_key)

    from mapanything.utils.image import load_images

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    room = {}  # frame number -> that frame's points, already moved into part 1's space
    joins, scale_factors, seconds = [], [], 0.0
    for number, indices in enumerate(parts, start=1):
        if len(parts) == 1:
            stage(f"rebuilding the room from {len(indices)} frames")
        else:
            stage(f"rebuilding part {number} of {len(parts)} ({len(indices)} frames)")
        views = load_images([frames[i] for i in indices])
        started = time.perf_counter()
        with torch.no_grad():
            predictions = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,  # smallest GPU memory use
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )
        seconds += time.perf_counter() - started
        scale_factors.append(float(_numpy(predictions[0]["metric_scaling_factor"]).reshape(-1)[0]))
        part = {i: _frame(pred) for i, pred in zip(indices, predictions)}
        del predictions, views
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # VGGT-Long leaves out spots below a tenth of the part's median confidence when lining parts up
        floor = 0.1 * float(np.median(np.concatenate([f["conf"][f["valid"]] for f in part.values()])))
        for f in part.values():
            f["trusted"] = f["valid"] & (f["conf"] >= floor)

        if number == 1:
            fit = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        else:
            stage(f"lining up part {number} with the room so far")
            fit, report = _join(part, room, number)
            joins.append(report)
            if fit is None:
                continue
        resize, turn, move = fit
        for i, f in part.items():
            if i not in room:
                f["pts"] = (resize * f["pts"] @ turn.T + move).astype(np.float32)
                room[i] = f
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else None

    stage("saving the room")
    points = np.concatenate([f["pts"][f["valid"]] for f in room.values()])
    colours = np.concatenate([f["rgb"][f["valid"]] for f in room.values()])
    confidence = np.concatenate([f["conf"][f["valid"]] for f in room.values()])
    room.clear()
    keep = confidence >= np.percentile(confidence, 10)  # drop the least confident tenth, as MapAnything's own mask does
    points, colours, confidence = points[keep], colours[keep], confidence[keep]
    points, colours = _thin(points, colours, confidence, DOT)
    if len(points) > MAX_POINTS:
        chosen = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
        points, colours = points[chosen], colours[chosen]
    filled = int(len(np.unique(_cells(points, CUBE))))
    points = points * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # OpenCV axes to three.js axes
    write_ply(out / "room.ply", points, colours)

    # A box around the 3D along the first frame's axes, ignoring the 2% of stray points at each end.
    # It's not the room's walls (those aren't lined up with the camera); the viewer's measuring tool is for that.
    low, high = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    size = high - low

    result = {
        "frames": len(frames),
        "parts": len(parts),
        "seconds": round(seconds, 1),
        "peakVramGb": round(peak_gb, 2) if peak_gb is not None else None,
        "scaleFactor": round(scale_factors[0], 4),
        "points": int(len(points)),
        "filledCubes": filled,
        "sizeX": round(float(size[0]), 3),
        "sizeY": round(float(size[1]), 3),
        "sizeZ": round(float(size[2]), 3),
        "ply": "room.ply",
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
