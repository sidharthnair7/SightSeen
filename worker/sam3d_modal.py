"""SAM 3D Objects (Meta) on a Modal GPU, for rebuild_things.py's build step: the model needs 32 GB of graphics memory and
this PC's card has 8. Meta's weights come from Hugging Face, where they're gated (request access on the
facebook/sam-3d-objects page, then HF_TOKEN=... in .env); they're downloaded once into a Modal volume.

Each call takes one frame (the photo SAM 3D builds from, the outlines of the things in it, and the room's own depth
there as a pointmap) and returns each thing as a mesh already in that frame's OpenCV camera axes (x right, y down, z
ahead, metres when the pointmap is given): SAM 3D's GLB is turned back from its y-up export to the object's own axes,
moved by SAM 3D's layout into its PyTorch3D camera (x left, y up), and flipped to OpenCV's.

    python -m modal run worker/sam3d_modal.py::check     builds the image (the first time, about half an hour) and
                                                          checks everything imports
"""
import os
import sys
from pathlib import Path

import modal

COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"  # sam-3d-objects, Jun 2 2026
# 48 GB, enough for SAM 3D's 32 and cheaper than an A100. Modal runs no GPU at all without a card on file, so
# SIGHTSEEN_SAM3D_GPU="" leaves the app GPU-less, which is enough for check()
GPU = os.environ.get("SIGHTSEEN_SAM3D_GPU", "L40S") or None
KAOLIN = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

app = modal.App("sightseen-sam3d")
weights = modal.Volume.from_name("sightseen-sam3d-weights", create_if_missing=True)

# The image follows SAM 3D's own setup (doc/setup.md): Python 3.11, PyTorch 2.5.1 for CUDA 12.1, its requirements,
# then PyTorch3D and flash-attn, then kaolin and gsplat, then its Hydra patch. Left out: nvidia-pyindex (it only
# edits pip's settings) and the setup's pypi.ngc.nvidia.com index, which no longer resolves (Nov 2026).
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1", "libgomp1",
                 "libxrender1", "libxi6", "libxkbcommon0", "libsm6", "libxext6", "libxfixes3", "libxxf86vm1")
    .env({"PIP_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu121",
          "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "FORCE_CUDA": "1", "MAX_JOBS": "8"})
    .pip_install("torch==2.5.1+cu121", "torchvision==0.20.1+cu121", "torchaudio==2.5.1+cu121",
                 index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("hatchling", "hatch-requirements-txt", "wheel", "setuptools", "ninja")
    .run_commands(
        f"git clone https://github.com/facebookresearch/sam-3d-objects /opt/sam3d && cd /opt/sam3d && git checkout {COMMIT}",
        "cd /opt/sam3d && sed -i '/^nvidia-pyindex/d' requirements.txt",
        "cd /opt/sam3d && pip install -e '.[dev]'",
    )
    # The package itself is already installed (editable) by the step above: these two only add their requirements.
    # Modal's Python was built with clang, so C++ extensions (PyTorch3D) link with clang++: it's installed here
    .run_commands("apt-get update && apt-get install -y clang",
                  "cd /opt/sam3d && pip install --no-build-isolation -r requirements.p3d.txt")
    .run_commands(f"cd /opt/sam3d && PIP_FIND_LINKS={KAOLIN} pip install --no-build-isolation -r requirements.inference.txt",
                  "cd /opt/sam3d && python patching/hydra",
                  "pip install 'huggingface-hub[cli]<1.0'")
    # pkg_resources, which SAM 3D's imports still use, went out of setuptools in version 81
    .run_commands("pip install 'setuptools<81'")
)


def _token_secret():
    """The Hugging Face token from the repo's .env (on this PC), handed to the container as a secret."""
    token = os.environ.get("HF_TOKEN")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not token and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "HF_TOKEN" and value.strip():
                token = value.strip().strip('"')
    return modal.Secret.from_dict({"HF_TOKEN": token or ""})


@app.function(image=image, timeout=900)
def check():
    """Everything SAM 3D needs imports (no GPU, no weights), down to its notebook's Inference, the way load() uses it."""
    import importlib

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # as SAM 3D's notebook/inference.py sets it: no internal init
    os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")  # inference.py sets CUDA_HOME from it (no conda here)
    sys.path.append("/opt/sam3d/notebook")
    found = {}
    for module in ("torch", "pytorch3d", "kaolin", "gsplat", "xformers", "spconv", "moge", "utils3d", "sam3d_objects",
                   "inference"):
        try:
            imported = importlib.import_module(module)
            version = imported.__dict__.get("__version__")
            found[module] = str(version) if version else "ok"
        except Exception as error:
            found[module] = f"FAILED {type(error).__name__}: {error}"[:300]
    return found


@app.cls(image=image, gpu=GPU, volumes={"/weights": weights}, secrets=[_token_secret()], timeout=3600,
         scaledown_window=180)
class Sam3d:
    @modal.enter()
    def load(self):
        config = Path("/weights/hf/pipeline.yaml")
        if not config.exists():
            from huggingface_hub import snapshot_download

            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN isn't set: put it in .env after Hugging Face approves the access request")
            snapshot_download("facebook/sam-3d-objects", local_dir="/weights/hf-download", token=token, max_workers=4)
            os.replace("/weights/hf-download/checkpoints", "/weights/hf")  # as doc/setup.md lays it out
            weights.commit()
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # as SAM 3D's notebook/inference.py sets it
        os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")  # inference.py sets CUDA_HOME from it (no conda here)
        sys.path.append("/opt/sam3d/notebook")
        from inference import Inference

        self.inference = Inference(str(config), compile=False)

    @modal.method()
    def build(self, photo_png: bytes, outline_pngs: list, pointmap_npz: bytes = None, seed: int = 42) -> list:
        """Every outlined thing in one photo, each as a mesh in the photo's OpenCV camera axes (see the module doc)."""
        import io
        import time

        import numpy as np
        import torch
        from PIL import Image
        from sam3d_objects.utils.visualization import SceneVisualizer

        photo = np.array(Image.open(io.BytesIO(photo_png)).convert("RGB"))
        pointmap = None
        if pointmap_npz:
            pointmap = torch.from_numpy(np.load(io.BytesIO(pointmap_npz))["pointmap"]).float()  # H x W x 3, PyTorch3D axes
        y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # to_glb's z-up to y-up turn
        results = []
        for outline_png in outline_pngs:
            started = time.perf_counter()
            try:
                outline = np.array(Image.open(io.BytesIO(outline_png)).convert("L")) > 127
                output = self.inference(photo, outline, seed=seed, pointmap=pointmap)
                mesh = output["glb"]
                local = np.asarray(mesh.vertices, dtype=np.float64) @ y_up.T  # back to the object's own axes
                device = output["rotation"].device
                cloud = SceneVisualizer.object_pointcloud(
                    points_local=torch.tensor(local, dtype=torch.float32, device=device).unsqueeze(0),
                    quat_l2c=output["rotation"], trans_l2c=output["translation"], scale_l2c=output["scale"])
                in_camera = cloud.points_list()[0].detach().cpu().numpy() * np.array([-1.0, -1.0, 1.0])  # to OpenCV
                colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
                results.append({
                    "vertices": in_camera.astype(np.float32).tobytes(),
                    "faces": np.asarray(mesh.faces, dtype=np.int32).tobytes(),
                    "colours": colours.tobytes(),
                    "rotation": output["rotation"].detach().cpu().numpy().tolist(),
                    "translation": output["translation"].detach().cpu().numpy().tolist(),
                    "scale": output["scale"].detach().cpu().numpy().tolist(),
                    "seconds": round(time.perf_counter() - started, 1),
                })
            except Exception as error:
                results.append({"error": f"{type(error).__name__}: {error}"[:800],
                                "seconds": round(time.perf_counter() - started, 1)})
        return results
