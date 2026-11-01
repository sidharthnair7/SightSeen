"""SightSeen's 3D step on Modal: one MapAnything run over the whole video on a 24 GB GPU, so a room needs no parts and
no joins (an 8 GB GPU can only take about 24 frames in one run). Run it from worker/ with the helper's venv:

    .venv\\Scripts\\python.exe -m modal run modal_app.py --video <video file> --out <folder>

It writes what reconstruct.run writes (room.ply, room-surface.ply, splat-data/, result.json) into --out. The splats
are trained afterwards on this PC with Brush, which isn't in the cloud image yet.
"""

import io
import zipfile
from pathlib import Path

import modal

app = modal.App("sightseen")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")  # system libraries OpenCV and Open3D load
    .pip_install("torch", "torchvision")
    .pip_install("git+https://github.com/facebookresearch/map-anything.git")
    .pip_install("opencv-python-headless", "plotly==7.1.0", "dash==4.5.0")
    .pip_install("open3d==0.20.0", extra_options="--no-deps")
    .env({"HF_HOME": "/models"})
    .add_local_python_source("reconstruct")
)

# MapAnything's 4.9 GB checkpoint is downloaded into this volume once and reused by every later run
models = modal.Volume.from_name("sightseen-models", create_if_missing=True)


# Modal runs no GPU at all without a payment method on file (L4 and T4 were both refused on Nov 1), so this waits
# for the hackathon credits. An L4's 24 GB fits a whole video in one run.
@app.function(image=image, gpu="L4", volumes={"/models": models}, timeout=1800)
def rebuild(video: bytes, name: str, fps: float, frames_per_run: int) -> bytes:
    """The whole 3D step in the cloud. Returns the room's files as a zip (the video and its frames stay behind)."""
    import reconstruct

    work = Path("/tmp/room")
    work.mkdir(parents=True, exist_ok=True)
    (work / name).write_bytes(video)
    reconstruct.run(work / name, work, "apache", fps, frames_per_run, lambda text: print("stage:", text, flush=True))
    models.commit()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as room:
        for path in work.rglob("*"):
            if path.is_file() and path.name != name and "frames" not in path.relative_to(work).parts:
                room.write(path, path.relative_to(work))
    return buffer.getvalue()


@app.local_entrypoint()
def main(video: str, out: str, fps: float = 2.0, frames_per_run: int = 200):
    """frames_per_run is high on purpose: up to 200 frames go into a single run, so there are no parts."""
    files = rebuild.remote(Path(video).read_bytes(), Path(video).name, fps, frames_per_run)
    Path(out).mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(files)).extractall(out)
    print("saved to", out)
