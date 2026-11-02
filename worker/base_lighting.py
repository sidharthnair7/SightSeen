"""One base lighting for the whole room (the owner's Test 3): the phone changes its exposure and white balance as it
turns, so the same wall comes out brighter in one frame and greyer in the next, and a room painted from those frames
gets patches and seams. This finds, for every frame, the gain per colour channel that makes the room's surface agree
across frames, and writes the frames again under that one lighting.

How: spots on the fused surface (room-surface.ply) seen by several frames. In linear light (sRGB to the power 2.2) a
spot's colour in frame k is gain_k x the spot's own colour, so in logs it's a sum; alternating medians find the
gains and the spots' colours (medians, so a highlight or something in front doesn't drag a spot). The gains are
centred on the frames' average exposure. Disagreement = how far a spot's colour wanders between the frames that see
it (the spread of its log colour, as a percentage), before and after.

Writes base-lighting.json (each frame's gains, the disagreement before and after) and frames-base-light/ (each frame
divided by its gains, same names), which room_textures.py and room_furniture.py can paint from (images=...).
    python worker/base_lighting.py <request folder>
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from room_objects import _cameras

SPOTS = 40_000  # surface spots sampled
ON_SURFACE = 0.03  # metres: a spot is seen when it's this close to the surface the camera sees at its pixel
DARKEST, BRIGHTEST = 0.01, 0.92  # linear light: samples outside this (black, blown out) say nothing about exposure
LEAST_SEEN = 3  # a spot needs this many frames to count
ROUNDS = 12
NEIGHBOURS = 2  # frames either side for the smoothing median
TINT = 0.2  # a frame's colour channels may differ from its average gain by at most 20%
LOWEST, HIGHEST = 0.4, 2.5
FLIP = np.diag([1.0, -1.0, -1.0])


def _spread(log_colours, frames_of, spots_of, count):
    """Median over spots seen LEAST_SEEN+ times of the spread (standard deviation) of their log colour, as a percent."""
    seen = np.bincount(spots_of, minlength=count)
    mean = np.zeros((count, 3))
    np.add.at(mean, spots_of, log_colours)
    mean /= np.maximum(seen, 1)[:, None]
    square = np.zeros((count, 3))
    np.add.at(square, spots_of, (log_colours - mean[spots_of]) ** 2)
    deviation = np.sqrt(square / np.maximum(seen, 1)[:, None]).mean(axis=1)
    return float((np.exp(np.median(deviation[seen >= LEAST_SEEN])) - 1) * 100)


def solve(out):
    import open3d as o3d

    out = Path(out)
    started = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(out / "room-surface.ply"))
    vertices = np.asarray(mesh.vertices)
    chosen = np.random.default_rng(0).choice(len(vertices), min(SPOTS, len(vertices)), replace=False)
    spots = vertices[chosen] @ FLIP.T  # three.js axes back to OpenCV, the cameras' axes
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    cameras = _cameras(out / "splat-data")
    images = out / "splat-data" / "images"
    frames_of, spots_of, samples = [], [], []
    for k, camera in enumerate(cameras):
        picture = cv2.imread(str(images / camera["name"]))[:, :, ::-1].astype(np.float32) / 255
        height, width = picture.shape[:2]
        inside = spots @ camera["turn"].T + camera["shift"]
        ahead = inside[:, 2] > 0.05
        depth = np.where(ahead, inside[:, 2], 1.0)
        u = camera["lens"][0, 0] * inside[:, 0] / depth + camera["lens"][0, 2]
        v = camera["lens"][1, 1] * inside[:, 1] / depth + camera["lens"][1, 2]
        on = ahead & (u >= 2) & (u < width - 2) & (v >= 2) & (v < height - 2)
        if not on.any():
            continue
        # is the spot the surface this camera sees there? (a ray to it hits nothing first)
        distance = np.linalg.norm(inside[on], axis=1)
        directions = (inside[on] / distance[:, None]) @ camera["turn"] @ FLIP.T  # camera to room, then three.js
        origin = FLIP @ (-camera["turn"].T @ camera["shift"])
        rays = np.hstack([np.repeat(origin[None], len(directions), axis=0), directions]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible = np.abs(hit - distance) < ON_SURFACE
        index = np.nonzero(on)[0][visible]
        colour = cv2.remap(picture, u[index].astype(np.float32)[None], v[index].astype(np.float32)[None],
                           cv2.INTER_LINEAR)[0] if len(index) and len(index) < 32767 else None
        if colour is None:
            rows = -(-len(index) // 1024)
            pad = lambda x: np.pad(x.astype(np.float32), (0, rows * 1024 - len(index))).reshape(rows, 1024)  # noqa: E731
            colour = cv2.remap(picture, pad(u[index]), pad(v[index]), cv2.INTER_LINEAR).reshape(-1, 3)[:len(index)]
        linear = colour ** 2.2
        good = (linear.min(axis=1) > DARKEST) & (linear.max(axis=1) < BRIGHTEST)
        frames_of.append(np.full(good.sum(), k))
        spots_of.append(index[good])
        samples.append(linear[good])
    frames_of, spots_of = np.concatenate(frames_of), np.concatenate(spots_of)
    log_colours = np.log(np.concatenate(samples))

    gains = np.zeros((len(cameras), 3))
    albedo = np.zeros((len(spots), 3))
    for _ in range(ROUNDS):
        for c in range(3):
            residual = log_colours[:, c] - gains[frames_of, c]
            order = np.lexsort((residual, spots_of))  # medians per spot: sort by spot, then value
            albedo[:, c] = _group_median(spots_of[order], residual[order], len(spots))
            residual = log_colours[:, c] - albedo[spots_of, c]
            order = np.lexsort((residual, frames_of))
            gains[:, c] = _group_median(frames_of[order], residual[order], len(cameras))
        gains -= gains.mean(axis=0)  # the base lighting is the frames' average exposure

    # A phone's exposure drifts smoothly, so a frame far off its neighbours is a bad estimate (take 3's frame 65, a
    # blurred close-up with a rough camera, came out at 0.17 red): each frame takes the median of the frames around
    # it, and its colour channels may lean at most TINT away from their average
    solved = gains.copy()
    for k in range(len(cameras)):
        gains[k] = np.median(solved[max(0, k - NEIGHBOURS):k + NEIGHBOURS + 1], axis=0)
    average = gains.mean(axis=1, keepdims=True)
    gains = np.clip(average + np.clip(gains - average, -np.log(1 + TINT), np.log(1 + TINT)),
                    np.log(LOWEST), np.log(HIGHEST))
    gains -= gains.mean(axis=0)

    before = _spread(log_colours, frames_of, spots_of, len(spots))
    after = _spread(log_colours - gains[frames_of], frames_of, spots_of, len(spots))
    result = {
        "frames": {camera["name"]: [round(float(g), 4) for g in np.exp(gains[k])] for k, camera in enumerate(cameras)},
        "samples": int(len(log_colours)), "spots": int(len(np.unique(spots_of))),
        "disagreementBeforePercent": round(before, 2), "disagreementAfterPercent": round(after, 2),
        "gainRange": [round(float(np.exp(gains.min())), 3), round(float(np.exp(gains.max())), 3)],
    }

    lit = out / "frames-base-light"
    lit.mkdir(exist_ok=True)
    for k, camera in enumerate(cameras):
        picture = cv2.imread(str(images / camera["name"]))[:, :, ::-1].astype(np.float32) / 255
        corrected = np.clip((picture ** 2.2) / np.exp(gains[k]), 0, 1) ** (1 / 2.2)
        cv2.imwrite(str(lit / camera["name"]), (corrected * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
    result["seconds"] = round(time.perf_counter() - started, 1)
    (out / "base-lighting.json").write_text(json.dumps(result, indent=2))
    return result


def _group_median(groups, values, count):
    """The median of `values` within each group; groups and values already sorted by group, then value."""
    medians = np.zeros(count)
    starts = np.searchsorted(groups, np.arange(count))
    ends = np.searchsorted(groups, np.arange(count), side="right")
    filled = ends > starts
    middle = (starts + ends - 1) // 2
    medians[filled] = values[middle[filled]]
    return medians


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/base_lighting.py <request folder>")
    report = solve(sys.argv[1])
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
