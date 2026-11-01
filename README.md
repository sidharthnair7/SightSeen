# SightSeen

See a rental room in 3D, at real size, from the landlord's video, before you sign.

Students renting from far away can't visit a place before signing. With SightSeen, the student starts a request and texts the landlord one link. The landlord films the room through that link, and SightSeen rebuilds it in 3D so the student can look around it and measure it.

## Meta models

- **MapAnything** (Meta, open weights) rebuilds the room in 3D, at metric scale, from frames of the video. SightSeen loads `facebook/map-anything-apache` (Apache 2.0) by default. `facebook/map-anything` (CC-BY-NC 4.0, non-commercial) is used only to compare the two on test rooms.
- Next: SAM 3.1 and Muse Spark through the Meta Model API, to find doors and outlets of known size and check the room's scale against them.

## How it fits together

- `src/`: Spring Boot (Java 25). Requests, the student's and the landlord's APIs, and the jobs sent to the 3D helper. Data lives in `data/` (an H2 file, the videos and the 3D rooms).
- `worker/`: the 3D helper (Python, FastAPI). It takes frames from a video and runs MapAnything on the GPU. One run can only take about 24 frames on an 8 GB card, so a longer video gets one run spread over the whole video first, which sets the room's shape and size. Then the video is cut into equal parts for detail. Each part gets the whole-video run's camera positions for the frames they share, so MapAnything builds it to agree, and then it's lined up with that run. Then COLMAP solves the cameras again, precisely, and MapAnything's depth gives them real size. If COLMAP can't place enough of the frames, MapAnything's cameras stay. Then every frame is merged into one solid surface (TSDF fusion with Open3D), and Brush trains Gaussian splats on the frames so the room looks like the video. It writes `room-splats.ply` (the room as it looks), `room-surface.ply` (the solid room the measuring tool uses), `room.ply` (the same room as dots) and `result.json` (the numbers, including how well each part lined up and whose cameras were used) next to the video.
- `frontend/`: React, TypeScript, Tailwind and three.js. The student's request page and 3D view with a measuring tool, and the landlord's filming page.

## Research behind it

I read these papers before and during the build. Here's what each one does, what SightSeen takes from it, and why I picked it over the other options.

**MapAnything** (Meta, [arXiv 2509.13414](https://arxiv.org/abs/2509.13414)). Turns photos into a 3D scene in metres, with every camera's position. It's the core of SightSeen.
- Why this one: a student needs real sizes, and MapAnything gives metres straight away. VGGT (also Meta) gives the shape but not the real size, because it's trained on scenes scaled to a standard size. MASt3R gives metric 3D too, but its code is non-commercial only (CC BY-NC-SA 4.0).
- On my own room it measured my desk at 77 to 79 cm. The tape says 77.
- SightSeen uses the Apache 2.0 checkpoint by default. The non-commercial one is only there to compare. I'll pick one before the end.

**SAM 3** (Meta, [arXiv 2511.16719](https://arxiv.org/abs/2511.16719)), as SAM 3.1 on the Meta Model API. Finds and outlines every object that matches a short phrase like "door", in photos and video.
- Why this one: UniMetric's paper used SAM 2.1, which needs a click or a box for each object. SAM 3 takes words, so SightSeen can ask for doors and outlets with nobody clicking.
- Status: planned, for the size check below.

**UniMetric** ([arXiv 2610.16945](https://arxiv.org/abs/2610.16945)). Checks and fixes a 3D model's size using everyday things with a known size range, like doors and outlets.
- Why: MapAnything got my desk right, but one desk isn't proof. I'd rather build more than show a student wrong sizes.
- To be fair to it: the paper tested nine other 3D models on mostly driving, roadside and robot scenes, not MapAnything and not a bedroom. So I'll judge it the same way, by tape against the 3D. Its code has no licence yet, so SightSeen will implement it from the paper's equations.
- Status: next.

**VGGT-Long** ([arXiv 2507.16443](https://arxiv.org/abs/2507.16443)). Cuts long videos into parts a 3D model can handle, then lines the parts up.
- What SightSeen takes: its way of lining up a part (Section 3.1). One fit per part, where points the model is unsure of count less and points that sit way off get ignored. Written from the paper, no code copied.
- What I changed: VGGT-Long lines each part up with the one before. When I tried that on my room, small errors stacked up and sizes drifted by up to about 5%. SightSeen lines every part up with one run over the whole video instead. Sizes came back to within about 1% of the one-run version. It also hands each part that run's camera positions first, since MapAnything can take known camera positions as input. On my room that cut the worst gap between parts from 8.8 cm to 3.9 cm.

**COLMAP** (Schönberger and Frahm, "Structure-from-Motion Revisited", [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)), through pycolmap (BSD-3-Clause). Finds the same small details in many frames and solves every camera together, so the views line up to under a pixel.
- Why: my splats were blurry and I couldn't tell why. I tested the trainer first (two different ones scored within about 1 dB of each other), then the cameras. On my room, MapAnything's guess of the lens changed from frame to frame (261 to 462 px, on a phone lens that never zoomed), and its camera directions were about 3 degrees off from COLMAP's. Same frames, same trainer, only the cameras swapped: on frames the trainer never saw, COLMAP's cameras scored 31.2 dB PSNR and MapAnything's 21.4. COLMAP won on all 17.
- What MapAnything still does: COLMAP has no sense of size, so MapAnything's depth turns COLMAP's units into metres. MapAnything's depth also builds the solid surface the measuring tool uses. And when COLMAP can't place enough of the frames (blank walls, fast turns), SightSeen keeps MapAnything's cameras.
- Why both: MapAnything alone gave the blur. COLMAP alone gives no metres and only sparse dots.
- Sizes still check out: with COLMAP's cameras my desk measures about 77 cm in the app, and the tape says 77. The same room built on MapAnything's cameras alone came out 15 to 19% bigger on every side, which matches its lens guess being about 20% too wide.

**G4Splat** ([arXiv 2510.12099](https://arxiv.org/abs/2510.12099)). Fills the parts of a room no camera saw. It finds flat surfaces like walls and floors and uses them to guide an AI video model.
- What SightSeen takes: the ideas, not the code. Find the gaps by what the cameras covered, and use flat surfaces to fill walls and floors.
- Why not run it: it takes about an hour per room on a 24 to 80 GB graphics card, on Linux. SightSeen runs on an 8 GB Windows card.

**The solid surface: TSDF fusion** (Open3D, MIT). Merges every frame's depth into one surface, so walls are solid instead of millions of loose dots.
- Why keep it now that there are splats: splats look more like a photo, but they need training for every room and they're hard to measure on. A surface keeps the measuring tool working, is light enough for VR, and gives real floors and walls to place things on later. The splats go on top of it: Brush (Apache 2.0) trains them and Spark (MIT) draws them in the browser.

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
