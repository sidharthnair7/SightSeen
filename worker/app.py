"""SightSeen's 3D helper: a small HTTP service that runs MapAnything for Spring Boot.

Spring Boot posts a job (where the video is, where to write the room) and then asks how it's going. There's one
GPU, so jobs run one at a time and the rest wait in a queue. Jobs live in memory: if this restarts, Spring Boot
finds out the next time it asks and tells the student to try again. The student's own things (worker/items.py) wait
in the same queue; their progress is in each item's own item.json, which Spring Boot reads.

Start it with run.ps1 (http://127.0.0.1:8001).
"""

import json
import os
import queue
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import items
import reconstruct

app = FastAPI(title="SightSeen 3D helper")


class JobIn(BaseModel):
    jobId: str
    videoPath: str
    outputDir: str
    model: str = "apache"
    fps: float = 2.0
    maxFrames: int = 24


class ItemIn(BaseModel):
    itemId: str
    requestDir: str


jobs: dict[str, dict] = {}
pending: "queue.Queue[JobIn | ItemIn]" = queue.Queue()
lock = threading.Lock()


def _update(job_id, **fields):
    with lock:
        jobs[job_id].update(fields)


def _view(job_id):
    with lock:
        return dict(jobs[job_id])


@app.get("/health")
def health():
    return reconstruct.health()


@app.post("/jobs", status_code=202)
def submit(job: JobIn):
    if job.model not in reconstruct.CHECKPOINTS:
        raise HTTPException(400, f"Unknown model '{job.model}'. Use one of: {', '.join(reconstruct.CHECKPOINTS)}.")
    if job.maxFrames < 2:
        raise HTTPException(400, "maxFrames must be at least 2.")
    with lock:
        jobs[job.jobId] = {"jobId": job.jobId, "state": "queued", "stage": "waiting for the GPU",
                           "message": None, "result": None}
    pending.put(job)
    return _view(job.jobId)


@app.post("/items", status_code=202)
def submit_item(item: ItemIn):
    """One of the student's things to build from its photo: items/<itemId>/ in the request folder holds the photo and
    item.json, which says how it's going."""
    pending.put(item)
    return {"itemId": item.itemId, "state": "queued"}


@app.get("/jobs/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return _view(job_id)


def _work():
    while True:
        job = pending.get()
        if isinstance(job, ItemIn):
            reconstruct.unload()  # TRELLIS needs the card
            try:
                items.build_item(job.requestDir, job.itemId)
            except Exception:  # noqa: BLE001 - build_item writes its own failures into item.json
                traceback.print_exc()
            continue
        _update(job.jobId, state="running")
        try:
            result = reconstruct.run(job.videoPath, job.outputDir, job.model, job.fps, job.maxFrames,
                                     lambda text, job_id=job.jobId: _update(job_id, stage=text))
            _update(job.jobId, state="done", stage="done", result=result)
        except Exception as error:  # noqa: BLE001 - every failure goes back to the student as a sentence
            traceback.print_exc()
            _update(job.jobId, state="failed", message=reconstruct.explain(error))


def _sweep():
    """The queue lives in memory: an item it held when this restarted will never finish, so it's marked failed and
    the student sees why instead of a spinner."""
    data = Path(os.environ.get("SIGHTSEEN_DATA", "data"))
    if not data.is_absolute():
        data = Path(__file__).resolve().parent.parent / data  # Spring Boot's folder, the repository
    for meta_file in data.glob("requests/*/items/*/item.json"):
        try:
            if json.loads(meta_file.read_text()).get("status") in ("queued", "outlining", "building"):
                items._status(meta_file.parent, status="failed",
                              why="The 3D helper restarted before it finished. Add it again.")
        except Exception:  # noqa: BLE001 - one bad file mustn't stop the helper starting
            traceback.print_exc()


_sweep()
threading.Thread(target=_work, daemon=True, name="gpu-worker").start()
