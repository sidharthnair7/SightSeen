"""Microsoft's TRELLIS (MIT licence, open weights) on this PC's 8 GB RTX 3070: the fallback for rebuild_things.py's build
step while Meta's SAM 3D weights wait for approval. The same plan as sam3d_local.py: every model sits in ordinary
memory, and each step of the pipeline moves its own models onto the card and back off (the image encoder, then the
sparse structure flow and its decoder, then the detail flow, then the decoders).

TRELLIS gives no pose: each object comes back in its own axes (z up, inside a unit cube), and rebuild_things.place
fits it into the photo by its outline and the room's measured points. Its colours come from its own Gaussians (the
nearest four to each vertex), since its texture baker needs nvdiffrast, which isn't installed.

Runs inside WSL, in the SAM 3D conda env (~/sam3d, worker/wsl/setup_sam3d.sh), with TRELLIS cloned to ~/sam3d/TRELLIS
and its weights in ~/sam3d/weights/trellis-image-large:
    python worker/trellis_local.py <jobs.json>
The jobs file is sam3d_local.py's (pointmaps are ignored). Each outline's result goes to <out>/<n>.npz (vertices in
TRELLIS's own axes, faces, colours) and <out>/<n>.json (seconds, the card's peak memory, or the error).
"""
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")
TRELLIS = Path.home() / "sam3d" / "TRELLIS"
WEIGHTS = Path.home() / "sam3d" / "weights" / "trellis-image-large"
sys.path.insert(0, str(TRELLIS))
sys.modules.setdefault("rembg", types.ModuleType("rembg"))  # background removal: only for photos without an outline

import numpy as np  # noqa: E402
import torch  # noqa: E402

SH_C0 = 0.28209479177387814  # the zeroth spherical harmonic: a Gaussian's base colour is 0.5 + SH_C0 x its DC term


def _load():
    """TRELLIS with its models in ordinary memory and each step moving its own models onto the card."""
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.base import Pipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(WEIGHTS))  # loads onto the CPU
    Pipeline.device = property(lambda self: torch.device("cuda"))  # new tensors go to the card, wherever the models are
    models = pipeline.models
    steps = {
        "get_cond": ["image_cond_model"],
        "sample_sparse_structure": ["sparse_structure_flow_model", "sparse_structure_decoder"],
        "sample_slat": ["slat_flow_model"],
        "decode_slat": ["slat_decoder_mesh", "slat_decoder_gs"],
    }
    for name, used in steps.items():
        method = getattr(pipeline, name)

        def on_card(*args, _method=method, _used=used, **kwargs):
            parts = [models[key] for key in _used if key in models]
            for part in parts:
                part.to("cuda")
            try:
                return _method(*args, **kwargs)
            finally:
                for part in parts:
                    part.to("cpu")
                torch.cuda.empty_cache()

        setattr(pipeline, name, on_card)
    return pipeline


def _run(pipeline, photo, outline, seed):
    from PIL import Image
    from scipy.spatial import cKDTree

    rgba = Image.fromarray(np.concatenate([photo[..., :3], (outline.astype(np.uint8) * 255)[..., None]], axis=-1), "RGBA")
    outputs = pipeline.run(rgba, seed=seed, formats=["mesh", "gaussian"], preprocess_image=True)
    mesh, gaussians = outputs["mesh"][0], outputs["gaussian"][0]
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    centres = gaussians.get_xyz.detach().float().cpu().numpy()
    base = 0.5 + SH_C0 * gaussians.get_features.detach().float().cpu().numpy()[:, 0, :]
    distance, nearest = cKDTree(centres).query(vertices, k=4)
    weight = 1.0 / np.maximum(distance, 1e-4)
    colours = (base[nearest] * weight[..., None]).sum(axis=1) / weight.sum(axis=1, keepdims=True)
    return vertices.astype(np.float32), faces, (np.clip(colours, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    from PIL import Image

    plan = json.loads(Path(sys.argv[1]).read_text())
    seed = plan.get("seed", 42)
    started = time.perf_counter()
    pipeline = _load()
    print(f"loaded in {time.perf_counter() - started:.0f} s", flush=True)
    for job in plan["jobs"]:
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        photo = np.array(Image.open(job["photo"]).convert("RGB"))
        for n, outline_file in enumerate(job["outlines"]):
            if (out / f"{n}.json").exists():
                continue  # done in an earlier run
            outline = np.array(Image.open(outline_file).convert("L")) > 127
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            try:
                vertices, faces, colours = _run(pipeline, photo, outline, seed)
                np.savez(out / f"{n}.npz", vertices=vertices, faces=faces, colours=colours)
                result = {"vertices": len(vertices), "layout": {"canonical": "z-up"}}
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"[:800]}
            result["seconds"] = round(time.perf_counter() - began, 1)
            result["peakGb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            (out / f"{n}.json").write_text(json.dumps(result, indent=2))
            print(job["out"], n, json.dumps({k: v for k, v in result.items() if k != "layout"}), flush=True)


if __name__ == "__main__":
    main()
