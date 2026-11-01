"""Draws a room's Gaussian splats from each of its frames' cameras, so the Walk room can be painted from the splats'
pictures instead of the raw video frames: the splats were fitted to every frame at once, so their pictures share one
exposure and lose the phone's auto-exposure jumps and most of the motion blur.

Runs in WSL, in the gsplat venv (gsplat 1.4.0 on CUDA); Windows has no gsplat build:
    wsl.exe -e bash -lc "~/sightseen-gs/.venv/bin/python /mnt/c/.../worker/render_splats.py SPLATS SPLAT_DATA OUT_DIR"
SPLATS is a Brush .ply (OpenCV axes, the room's space), SPLAT_DATA the COLMAP text folder its cameras come from
(sparse/0/cameras.txt and images.txt), OUT_DIR gets one PNG per camera, with the frame's own name and size.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization


def read_splats(path):
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    names = [line.split()[-1] for line in data[:end].decode("ascii").splitlines() if line.startswith("property")]
    values = np.frombuffer(data, dtype=np.float32, offset=end).reshape(-1, len(names))
    column = {name: k for k, name in enumerate(names)}
    pick = lambda prefix, count: values[:, [column[f"{prefix}{k}"] for k in range(count)]]  # noqa: E731
    rest = pick("f_rest_", sum(name.startswith("f_rest_") for name in names))
    per_channel = rest.shape[1] // 3
    sh = np.concatenate([pick("f_dc_", 3)[:, None, :],  # 3DGS stores the rest channel by channel
                         rest.reshape(-1, 3, per_channel).transpose(0, 2, 1)], axis=1)
    return {
        "means": values[:, [column["x"], column["y"], column["z"]]],
        "quats": pick("rot_", 4),  # w, x, y, z
        "scales": np.exp(pick("scale_", 3)),
        "opacities": 1 / (1 + np.exp(-values[:, column["opacity"]])),
        "sh": sh, "degree": int(round(np.sqrt(sh.shape[1]))) - 1,
    }


def read_cameras(sparse):
    lenses = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            number, _, width, height, fx, fy, cx, cy = line.split()
            lenses[number] = (int(width), int(height), np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]]))
    cameras = []
    for line in (sparse / "images.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 10 and not line.startswith("#"):
            w, x, y, z, tx, ty, tz = map(float, parts[1:8])
            turn = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            view = np.eye(4)
            view[:3, :3], view[:3, 3] = turn, [tx, ty, tz]
            width, height, lens = lenses[parts[8]]
            cameras.append({"name": parts[9], "view": view, "lens": lens, "width": width, "height": height})
    return cameras


def main(splats_path, splat_data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splats = read_splats(splats_path)
    tensor = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")  # noqa: E731
    means, quats, scales = tensor(splats["means"]), tensor(splats["quats"]), tensor(splats["scales"])
    opacities, sh = tensor(splats["opacities"]), tensor(splats["sh"])
    cameras = read_cameras(Path(splat_data) / "sparse" / "0")
    with torch.no_grad():
        for camera in cameras:
            colours, _, _ = rasterization(means, quats, scales, opacities, sh, tensor(camera["view"])[None],
                                          tensor(camera["lens"])[None], camera["width"], camera["height"],
                                          sh_degree=splats["degree"])
            picture = (colours[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(out_dir / camera["name"]), picture[:, :, ::-1])
    print(f"{len(cameras)} frames drawn from {len(splats['means'])} splats into {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: render_splats.py SPLATS SPLAT_DATA OUT_DIR")
    main(*sys.argv[1:])
