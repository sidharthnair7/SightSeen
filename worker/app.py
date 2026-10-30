"""SightSeen's 3D helper: a small HTTP service that runs MapAnything for Spring Boot.

Spring Boot posts a job (where the video is, where to write the room) and then asks how it's going. There's one
GPU, so jobs run one at a time and the rest wait in a queue. Jobs live in memory: if this restarts, Spring Boot
finds out the next time it asks and tells the student to try again.

Start it with run.ps1 (http://127.0.0.1:8001).
"""

import queue
import threading
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import reconstruct

app = FastAPI(title="SightSeen 3D helper")


class JobIn(BaseModel):
    jobId: str
    videoPath: str
    outputDir: str
    model: str = "apache"
    fps: float = 2.0
    maxFrames: int = 24


jobs: dict[str, dict] = {}
pending: "queue.Queue[JobIn]" = queue.Queue()
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


@app.get("/jobs/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return _view(job_id)


def _work():
    while True:
        job = pending.get()
        _update(job.jobId, state="running")
        try:
            result = reconstruct.run(job.videoPath, job.outputDir, job.model, job.fps, job.maxFrames,
                                     lambda text, job_id=job.jobId: _update(job_id, stage=text))
            _update(job.jobId, state="done", stage="done", result=result)
        except Exception as error:  # noqa: BLE001 - every failure goes back to the student as a sentence
            traceback.print_exc()
            _update(job.jobId, state="failed", message=reconstruct.explain(error))


threading.Thread(target=_work, daemon=True, name="gpu-worker").start()
