# SightSeen

See a rental room in 3D, at real size, from the landlord's video, before you sign.

Students renting from far away can't visit a place before signing. With SightSeen, the student starts a request and texts the landlord one link. The landlord films the room through that link, and SightSeen rebuilds it in 3D so the student can look around it and measure it.

## Meta models

- **MapAnything** (Meta, open weights) rebuilds the room in 3D, at metric scale, from frames of the video. SightSeen loads `facebook/map-anything-apache` (Apache 2.0) by default. `facebook/map-anything` (CC-BY-NC 4.0, non-commercial) is used only to compare the two on test rooms.
- Next: SAM 3.1 and Muse Spark through the Meta Model API, to find doors and outlets of known size and check the room's scale against them.

## How it fits together

- `src/`: Spring Boot (Java 25). Requests, the student's and the landlord's APIs, and the jobs sent to the 3D helper. Data lives in `data/` (an H2 file, the videos and the 3D rooms).
- `worker/`: the 3D helper (Python, FastAPI). It takes frames from a video, runs MapAnything on the GPU, and writes `room.ply` and `result.json` next to the video.
- `frontend/`: React, TypeScript, Tailwind and three.js. The student's request page and 3D view with a measuring tool, and the landlord's filming page.

## Run it

One-time setup of the 3D helper, from `worker/` in Git Bash (about 13 GB with both models):

```bash
bash setup.sh
```

Then three terminals:

1. 3D helper: `worker\run.ps1` (http://127.0.0.1:8001)
2. Spring Boot: `mvn spring-boot:run` (http://localhost:8080)
3. Frontend: `cd frontend`, then `npm install` and `npm run dev` (http://localhost:5173)

To film from a phone on the same Wi-Fi, open the student page at `http://<this PC's IP>:5173`, so the landlord link points at this PC.

Settings are in `src/main/resources/application.properties` (`sightseen.worker.*`: which model, frames per second of video, and the most frames in one run).

## Licence

To be chosen before submission.
