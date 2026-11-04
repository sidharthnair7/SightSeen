"""SightSeen on Modal. Two ways in:

1. Hosted, for anyone with the link (the hackathon's judges): a small web app takes a room's video (from the
   student's page, or live from the landlord's link), a GPU rebuilds the room in 3D, and the rest of the pipeline
   (worker/pipeline.py: the walls' photos, the scale, the things rebuilt by SAM 3D on Modal's GPUs at once, the issues)
   runs after it. The room opens as soon as its walls are done; its things come in a few minutes later. The page stays
   on Cloudflare Pages and talks to this app's address (VITE_API_BASE when it's built). Splats (the Real look) aren't
   made here yet: Brush isn't in the cloud image, so hosted rooms show the Rebuilt look.

       .venv\\Scripts\\python.exe -m modal deploy sam3d_modal.py     (SAM 3D, once: the hosted pipeline calls it)
       .venv\\Scripts\\python.exe -m modal deploy modal_app.py       (this app; its address is printed)

   Modal runs no GPU without a card on file (the credits pay; set a spend limit). The keys come from the repo's .env
   (MODEL_API_KEY, HF_TOKEN) when it's deployed, as Modal secrets.

2. The 3D step alone, from this PC (one MapAnything run over the whole video on a 24 GB GPU):

       .venv\\Scripts\\python.exe -m modal run modal_app.py --video <video file> --out <folder>
"""

import io
import json
import os
import re
import secrets
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("sightseen")
WORKER = Path(__file__).resolve().parent
# Modal won't start an app with GPU functions until there's a card on file, so SIGHTSEEN_GPU="" leaves it GPU-less,
# which is enough for check() (`modal run modal_app.py` with no video)
GPU = os.environ.get("SIGHTSEEN_GPU", "L4") or None

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")  # system libraries OpenCV and Open3D load
    .pip_install("torch", "torchvision")
    .pip_install("git+https://github.com/facebookresearch/map-anything.git")
    .pip_install("opencv-python-headless", "plotly==7.1.0", "dash==4.5.0", "pycolmap==4.2.1")
    .pip_install("open3d==0.20.0", extra_options="--no-deps")
    # what the rest of the pipeline imports (worker/pipeline.py): the things' clustering, meshes, the Meta Model API
    .pip_install("scikit-learn", "scipy", "trimesh", "requests", "pillow", "meta-sam-parser==0.1.0")
    # Open3D loads EGL and a few X libraries when it's imported, even without a screen (as SAM 3D's image has them)
    .apt_install("libegl1", "libxrender1", "libxi6", "libxkbcommon0", "libsm6", "libxext6", "libxfixes3", "libxxf86vm1",
                 "libusb-1.0-0")
    .env({"HF_HOME": "/models"})
    # the pipeline's Python only: not Brush (vendor/, this PC's splat trainer, whose caches change while it runs)
    .add_local_dir(WORKER, "/root/worker", ignore=[".venv", "vendor", "target", "**/__pycache__", "*.pyc", "*.log"])
)
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi", "python-multipart")

# MapAnything's 4.9 GB checkpoint is downloaded into this volume once and reused by every later run
models = modal.Volume.from_name("sightseen-models", create_if_missing=True)
rooms = modal.Volume.from_name("sightseen-rooms", create_if_missing=True)  # /rooms/<request id>/: video and room
state = modal.Dict.from_name("sightseen-state", create_if_missing=True)  # request:<id> -> its record, token:<t> -> id


def _keys():
    """MODEL_API_KEY and HF_TOKEN from the repo's .env (on this PC), as a secret for the containers."""
    found = {}
    env = WORKER.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() in ("MODEL_API_KEY", "HF_TOKEN") and value.strip():
                found[name.strip()] = value.strip().strip('"')
    return modal.Secret.from_dict(found)


def _update(request_id, **fields):
    """A request's record, changed. Records live in the Dict so the web app sees progress straight away."""
    record = state.get(f"request:{request_id}") or {}
    record.update(fields)
    state[f"request:{request_id}"] = record
    return record


@app.function(image=image, gpu=GPU, volumes={"/models": models, "/rooms": rooms}, secrets=[_keys()], timeout=1800)
def reconstruct_room(request_id: str):
    """The 3D step on a GPU (MapAnything, then COLMAP for the cameras), then the rest of the room on a CPU."""
    sys.path.insert(0, "/root/worker")
    import reconstruct

    out = Path("/rooms") / request_id
    rooms.reload()
    video = next(out.glob("video.*"))
    _update(request_id, status="RECONSTRUCTING", stage="starting the GPU", failureMessage=None)
    try:
        result = reconstruct.run(video, out, "apache", 2.0, 200,
                                 lambda text: _update(request_id, stage=text))
    except Exception as error:  # noqa: BLE001 - the page shows why, in a sentence
        _update(request_id, status="FAILED", stage=None, failureMessage=reconstruct.explain(error))
        return
    finally:
        models.commit()
    rooms.commit()
    room = {k: result.get(k) for k in ("frames", "seconds", "peakVramGb", "scaleFactor", "points", "sizeX", "sizeY",
                                       "sizeZ")}
    _update(request_id, room=room, stage="putting the video's photos on the walls")
    finish_room.spawn(request_id)


@app.function(image=image, cpu=8, memory=16384, volumes={"/rooms": rooms}, secrets=[_keys()], timeout=3600)
def finish_room(request_id: str):
    """worker/pipeline.py on the room: it opens once its walls are done, and its things come in when they're built
    (SAM 3D on the deployed sightseen-sam3d app, every frame at once)."""
    sys.path.insert(0, "/root/worker")
    os.environ["SIGHTSEEN_SAM3D_DEPLOYED"] = "1"
    import pipeline

    out = Path("/rooms") / request_id
    rooms.reload()

    def walls_done():
        rooms.commit()
        record = state.get(f"request:{request_id}") or {}
        _update(request_id, status="READY", room={**(record.get("room") or {}), "version": int(time.time())},
                stage="rebuilding the room's things in 3D (a few minutes; the page updates when they're in)")

    try:
        pipeline.finish(out, runner="modal", stage=lambda text: _update(request_id, stage=text), walls_done=walls_done)
    finally:
        rooms.commit()
        record = state.get(f"request:{request_id}") or {}
        _update(request_id, status="READY", stage=None, room={**(record.get("room") or {}), "version": int(time.time())})


@app.function(image=image, cpu=4, memory=8192, volumes={"/rooms": rooms}, secrets=[_keys()], timeout=600)
def set_room_scale(request_id: str, units: float, metres: float) -> dict:
    """A real length for a room (worker/room_scale.py's set_measured): the walls and the issues made again."""
    sys.path.insert(0, "/root/worker")
    import room_scale

    rooms.reload()
    try:
        result = room_scale.set_measured(Path("/rooms") / request_id, units, metres)
    except ValueError as error:
        _update(request_id, scaleState=str(error))
        return {"error": str(error)}
    except Exception:  # noqa: BLE001 - the page says it didn't work; the log has the rest
        _update(request_id, scaleState="The room's size couldn't be changed. Try again.")
        raise
    rooms.commit()
    record = state.get(f"request:{request_id}") or {}
    _update(request_id, scaleState="done", room={**(record.get("room") or {}), "version": int(time.time())})
    return result


VISITORS = Path("/rooms/visitors")  # a visitor's own things: items/<id>/, each theirs, kept apart from any room
VISITORS_IN_VOLUME = "/visitors/items"  # the same place, as the web app names it (it doesn't mount the volume)


@app.function(image=image, cpu=2, memory=8192, volumes={"/rooms": rooms}, secrets=[_keys()], timeout=1500)
def build_visitor_item(item_id: str):
    """worker/items.py on one visitor's photo: SAM 3.1 outlines it, the deployed SAM 3D builds it at its real size."""
    sys.path.insert(0, "/root/worker")
    os.environ["SIGHTSEEN_SAM3D_DEPLOYED"] = "1"
    import items

    def told(meta):  # "done" waits for the commit below: the page fetches the model as soon as it reads it
        if meta.get("status") != "done":
            state[f"item:{item_id}"] = meta

    items.on_status = told
    rooms.reload()
    try:
        meta = items.build_item(VISITORS, item_id)
        rooms.commit()
    except Exception:  # noqa: BLE001 - the page says so instead of spinning
        meta = {**(state.get(f"item:{item_id}") or {}), "status": "failed", "why": "The build stopped. Try it again."}
    state[f"item:{item_id}"] = meta


def _within_today(kind: str, most: int) -> bool:
    """A public link spends GPU credits: a day's rooms and things are capped (a stranger can't drain them)."""
    key = f"count:{kind}:{datetime.now(timezone.utc).date()}"
    count = state.get(key) or 0
    if count >= most:
        return False
    state[key] = count + 1
    return True


ROOMS_A_DAY, THINGS_A_DAY = 25, 60
BIGGEST_VIDEO, BIGGEST_PHOTO = 800 * 1024 * 1024, 25 * 1024 * 1024
PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".webp")
SAFE_FILE = re.compile(r"[a-z0-9][a-z0-9.-]*\.(json|png|jpg|ply)")
VIDEO_TYPES = (".mp4", ".mov", ".m4v", ".webm", ".mkv")


@app.function(image=web_image, timeout=900)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    """The API the page uses, at the same paths as the Spring Boot server's (src/main/java), so the page doesn't
    change; errors come back as {"message": ...} like Spring's. Plain handlers: FastAPI runs them on threads, and
    Modal's own calls (the Dict, the volume, starting a GPU) wait.

    It reads and writes the rooms volume directly (read_file, batch_upload) and doesn't mount it: other containers
    write the rooms, so a mounted copy would need reloading, and a reload makes files vanish for a moment for every
    other request in the container."""
    import mimetypes
    import uuid

    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    class Start(BaseModel):
        label: str

    class Length(BaseModel):
        units: float
        metres: float

    api = FastAPI(title="SightSeen hosted")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def fail(status, message):
        return JSONResponse({"message": message}, status_code=status)

    def stored(path, missing):
        """A file from the volume as a response, or `missing` (a sentence) as a 404 if it isn't there."""
        try:
            chunks = rooms.read_file(path)
            first = next(chunks, b"")
        except (FileNotFoundError, modal.exception.NotFoundError):
            return fail(404, missing)
        except Exception:  # noqa: BLE001 - a hiccup in the volume: an answer the page can read, not a bare 500
            return fail(503, "That file couldn't be read just now. Try again.")

        def rest():
            yield first
            yield from chunks

        kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return StreamingResponse(rest(), media_type=kind, headers={"Cache-Control": "no-cache"})

    def base_of(request):
        return str(request.base_url).rstrip("/").replace("http://", "https://")

    def record_of(request_id):
        return state.get(f"request:{request_id}") if re.fullmatch(r"[0-9a-f-]{36}", request_id) else None

    def view(record, base):
        room = record.get("room") if record.get("status") == "READY" else None
        if room:
            room = {**room, "plyUrl": f"{base}/api/requests/{record['id']}/room.ply?v={room.get('version', 0)}"}
        return {"id": record["id"], "label": record["label"], "status": record["status"], "stage": record.get("stage"),
                "landlordPath": f"/film/{record['token']}", "roomPath": f"/room/{record['id']}",
                "createdAt": record["createdAt"], "videoSource": record.get("videoSource"), "model": "apache",
                "room": room, "failureMessage": record.get("failureMessage"), "scaleState": record.get("scaleState")}

    def receive(request_id, video, source, live):
        """Saves the video next to the room, marks how it came in, and starts the GPU."""
        ext = Path(video.filename or "").suffix.lower()
        if ext not in VIDEO_TYPES:
            return fail(400, "That isn't a video this can read: use an MP4, MOV or WebM.")
        if (video.size or 0) > BIGGEST_VIDEO:
            return fail(413, "That video is over 800 MB. A slow two-minute walk round the room is plenty.")
        if not _within_today("rooms", ROOMS_A_DAY):
            return fail(429, "The demo has built all the rooms it can today. Try the sample room, or come back tomorrow.")
        # A video replacing an earlier one: that one may be of another type, and may have been filmed live
        if (record_of(request_id) or {}).get("videoSource"):
            for old in [f"video{other}" for other in VIDEO_TYPES if other != ext] + ["video-live.json"]:
                try:
                    rooms.remove_file(f"/{request_id}/{old}")
                except Exception:  # noqa: BLE001 - it wasn't there
                    pass
        with rooms.batch_upload(force=True) as batch:
            batch.put_file(video.file, f"/{request_id}/video{ext}")
            if live:
                marker = {"recordedInPage": True, "at": datetime.now(timezone.utc).isoformat()}
                batch.put_file(io.BytesIO(json.dumps(marker).encode()), f"/{request_id}/video-live.json")
        record = _update(request_id, status="VIDEO_RECEIVED", stage="waiting for a GPU", videoSource=source,
                         failureMessage=None)
        reconstruct_room.spawn(request_id)
        return record

    @api.post("/api/requests")
    def create(body: Start, request: Request):
        label = body.label.strip()[:120]
        if not label:
            return fail(400, "Say which place it is.")
        request_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(9).replace("_", "x").replace("-", "y")
        record = {"id": request_id, "label": label, "token": token, "status": "WAITING_FOR_VIDEO",
                  "createdAt": datetime.now(timezone.utc).isoformat()}
        state[f"request:{request_id}"] = record
        state[f"token:{token}"] = request_id
        return view(record, base_of(request))

    @api.get("/api/requests/{request_id}")
    @api.get("/api/requests/{request_id}/request.json")
    def read(request_id: str, request: Request):
        record = record_of(request_id)
        return view(record, base_of(request)) if record else fail(404, "No request with that link.")

    @api.post("/api/requests/{request_id}/video")
    def upload(request_id: str, request: Request, video: UploadFile = File(...)):
        record = record_of(request_id)
        if not record:
            return fail(404, "No request with that link.")
        if record["status"] in ("VIDEO_RECEIVED", "RECONSTRUCTING"):
            return fail(409, "A video is already being built into a room. Wait for it to finish.")
        done = receive(request_id, video, "STUDENT_UPLOAD", False)
        return done if isinstance(done, JSONResponse) else view(done, base_of(request))

    @api.get("/api/film/{token}")
    def film(token: str):
        request_id = state.get(f"token:{token}")
        record = record_of(request_id) if request_id else None
        if not record:
            return fail(404, "This link doesn't lead anywhere. Ask the student for a new one.")
        return {"label": record["label"], "videoReceived": record["status"] != "WAITING_FOR_VIDEO",
                "building": record["status"] in ("VIDEO_RECEIVED", "RECONSTRUCTING")}

    @api.post("/api/film/{token}/video")
    def film_upload(token: str, live: bool = False, video: UploadFile = File(...)):
        request_id = state.get(f"token:{token}")
        record = record_of(request_id) if request_id else None
        if not record:
            return fail(404, "This link doesn't lead anywhere. Ask the student for a new one.")
        if record["status"] in ("VIDEO_RECEIVED", "RECONSTRUCTING"):
            return fail(409, "A video is already being built into a room.")
        done = receive(request_id, video, "LANDLORD_LINK", live)
        return done if isinstance(done, JSONResponse) else {"ok": True}

    @api.post("/api/requests/{request_id}/scale")
    def scale(request_id: str, body: Length):
        if not record_of(request_id):
            return fail(404, "No request with that link.")
        if not body.units > 0 or not 0.05 <= body.metres <= 20:
            return fail(400, "Give a real length between 5 cm and 20 m.")
        # It works on after answering: Modal cuts a request off at 150 s. The page checks scaleState.
        _update(request_id, scaleState="running")
        set_room_scale.spawn(request_id, body.units, body.metres)
        return {"pending": True}

    @api.post("/api/items")
    def add_item(photo: UploadFile = File(...), name: str = Form(...), widthCm: float = Form(...),
                 scale: float = Form(1.0)):
        what = " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())
        if not what or len(what) > 40:
            return fail(400, "Say what it is in a few words: desk, chair, monitor.")
        if not 5 <= widthCm <= 500:
            return fail(400, "Its longest side should be between 5 and 500 cm.")
        if not 0.2 <= scale <= 5:
            return fail(400, "This room's scale doesn't look right. Reload the page.")
        ext = Path(photo.filename or "").suffix.lower()
        if ext not in PHOTO_TYPES:
            return fail(400, "That isn't a photo this can read: use a JPG or PNG.")
        if (photo.size or 0) > BIGGEST_PHOTO:
            return fail(413, "That photo is over 25 MB.")
        if not _within_today("things", THINGS_A_DAY):
            return fail(429, "The demo has built all the things it can today. Come back tomorrow.")
        item_id = uuid.uuid4().hex[:12]
        upload_name = "upload" + (".jpg" if ext == ".jpeg" else ext)
        meta = {"id": item_id, "name": what, "widthCm": widthCm, "scale": scale, "photo": upload_name, "status": "queued",
                "createdAt": datetime.now(timezone.utc).isoformat()}
        with rooms.batch_upload(force=True) as batch:
            batch.put_file(photo.file, f"{VISITORS_IN_VOLUME}/{item_id}/{upload_name}")
            batch.put_file(io.BytesIO(json.dumps(meta).encode()), f"{VISITORS_IN_VOLUME}/{item_id}/item.json")
        state[f"item:{item_id}"] = meta
        build_visitor_item.spawn(item_id)
        return meta

    @api.get("/api/items/{item_id}")
    def read_item(item_id: str):
        meta = state.get(f"item:{item_id}") if re.fullmatch(r"[0-9a-f]{12}", item_id) else None
        return meta or fail(404, "That thing isn't here any more.")

    @api.get("/api/items/{item_id}/item.ply")
    def item_model(item_id: str):
        if not re.fullmatch(r"[0-9a-f]{12}", item_id):
            return fail(404, "That thing isn't built yet.")
        return stored(f"{VISITORS_IN_VOLUME}/{item_id}/item.ply", "That thing isn't built yet.")

    @api.get("/api/requests/{request_id}/items")
    @api.get("/api/requests/{request_id}/items.json")
    def items(request_id: str):
        return []  # a visitor's own things are theirs (/api/items): the page remembers which

    @api.post("/api/requests/{request_id}/rerun")
    def rerun(request_id: str, request: Request):
        record = record_of(request_id)
        if not record or record["status"] not in ("FAILED", "READY") or not record.get("videoSource"):
            return fail(400, "There's no video to build from yet.")
        if record.get("stage"):
            return fail(409, "This room is still being built. Wait for it to finish.")
        if not _within_today("rooms", ROOMS_A_DAY):
            return fail(429, "The demo has built all the rooms it can today. Come back tomorrow.")
        done = _update(request_id, status="VIDEO_RECEIVED", stage="waiting for a GPU", failureMessage=None)
        reconstruct_room.spawn(request_id)
        return view(done, base_of(request))

    @api.get("/api/requests/{request_id}/{name}")
    @api.get("/api/requests/{request_id}/assets/{name}")
    def file(request_id: str, name: str):
        if not record_of(request_id) or not SAFE_FILE.fullmatch(name):
            return fail(404, "No such file.")
        return stored(f"/{request_id}/{name}", "No such file.")

    return api


# The 3D step alone, from this PC (the room's files come back as a zip) -------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/models": models}, timeout=1800)
def rebuild(video: bytes, name: str, fps: float, frames_per_run: int) -> bytes:
    """The whole 3D step in the cloud. Returns the room's files as a zip (the video and its frames stay behind)."""
    sys.path.insert(0, "/root/worker")
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


@app.function(image=image, timeout=600)
def check() -> dict:
    """Everything the hosted pipeline needs imports (no GPU): the first deploy's quick test."""
    import importlib

    sys.path.insert(0, "/root/worker")
    found = {}
    for module in ("torch", "mapanything", "pycolmap", "open3d", "cv2", "sklearn", "scipy", "trimesh", "meta_sam_parser",
                   "reconstruct", "pipeline", "rebuild_things", "room_issues", "room_scale", "room_textures",
                   "room_materials"):
        try:
            importlib.import_module(module)
            found[module] = "ok"
        except Exception as error:  # noqa: BLE001 - the report is the point
            found[module] = f"FAILED {type(error).__name__}: {error}"[:300]
    return found


@app.local_entrypoint()
def main(video: str = "", out: str = "", fps: float = 2.0, frames_per_run: int = 200):
    """frames_per_run is high on purpose: up to 200 frames go into a single run, so there are no parts. With no video,
    it checks the hosted pipeline's image instead."""
    if not video:
        print(json.dumps(check.remote(), indent=2))
        return
    files = rebuild.remote(Path(video).read_bytes(), Path(video).name, fps, frames_per_run)
    Path(out).mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(files)).extractall(out)
    print("saved to", out)
