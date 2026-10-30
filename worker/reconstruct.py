"""Video in, 3D room out: frames from the video, MapAnything, then a coloured point cloud (room.ply) and its numbers.

Everything is written into the request's folder, next to the video:
  frames/       the frames MapAnything saw
  room.ply      the 3D room as coloured points, in metres, in three.js axes (x right, y up, z towards the viewer)
  result.json   the run's numbers (the same dict run() returns)
"""

import json
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
MAX_POINTS = 800_000  # enough to see and measure the room, small enough for a browser
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


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _gather(predictions):
    """Joins every frame's valid points into one cloud, with each point's colour and confidence."""
    points, colours, confidence = [], [], []
    for pred in predictions:
        pts = _numpy(pred["pts3d"][0])  # (H, W, 3): metres, first frame's camera, OpenCV axes
        valid = _numpy(pred["mask"][0]).reshape(pts.shape[:2]) > 0.5
        conf = _numpy(pred["conf"][0])
        rgb = _numpy(pred["img_no_norm"][0])
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        points.append(pts[valid])
        colours.append(rgb[valid])
        confidence.append(conf[valid])
    return np.concatenate(points), np.concatenate(colours), np.concatenate(confidence)


def write_ply(path, points, colours):
    count = len(points)
    data = np.empty(count, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    rgb = np.clip(colours * 255.0, 0, 255).astype(np.uint8)
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
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


def run(video_path, output_dir, model_key, fps, max_frames, stage):
    """One whole-video run. `stage(text)` reports progress to whoever is asking."""
    out = Path(output_dir)

    stage("taking frames from the video")
    frames = extract_frames(video_path, out / "frames", fps, max_frames)

    first_load = _loaded["key"] != model_key
    stage("loading the 3D model" + (" (the first load takes a minute)" if first_load else ""))
    model = _model(model_key)

    from mapanything.utils.image import load_images

    views = load_images(frames)
    stage(f"rebuilding the room from {len(frames)} frames")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
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
    seconds = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else None

    stage("saving the room")
    points, colours, confidence = _gather(predictions)
    keep = confidence >= np.percentile(confidence, 10)  # drop the least confident tenth, as MapAnything's own mask does
    points, colours = points[keep], colours[keep]
    if len(points) > MAX_POINTS:
        chosen = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
        points, colours = points[chosen], colours[chosen]
    points = points * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # OpenCV axes to three.js axes
    write_ply(out / "room.ply", points, colours)

    # A box around the 3D along the first frame's axes, ignoring the 2% of stray points at each end.
    # It's not the room's walls (those aren't lined up with the camera); the viewer's measuring tool is for that.
    low, high = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    size = high - low

    result = {
        "frames": len(frames),
        "seconds": round(seconds, 1),
        "peakVramGb": round(peak_gb, 2) if peak_gb is not None else None,
        "scaleFactor": round(float(_numpy(predictions[0]["metric_scaling_factor"]).reshape(-1)[0]), 4),
        "points": int(len(points)),
        "sizeX": round(float(size[0]), 3),
        "sizeY": round(float(size[1]), 3),
        "sizeZ": round(float(size[2]), 3),
        "ply": "room.ply",
        "model": model_key,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    del predictions, views
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
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
