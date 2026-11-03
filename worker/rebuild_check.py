"""How much the Rebuilt look looks like the video: the room drawn from every frame's own camera (the shell's photos
plus the rebuilt things) next to the real frame. Two scores, so each round of rebuilding can be kept or dropped by
a number instead of by eye:
  - things: for every thing and every frame SAM 3.1 outlined it in, the overlap (IoU) of where the rebuilt thing shows
    from that camera with SAM's outline; averaged per thing, then over things.
  - picture: SSIM of the whole drawn frame against the real one (a quarter size), averaged over frames.
Writes checks/rebuild-check.json and checks/rebuild-check.jpg (six frames: real above, drawn below).
    python worker/rebuild_check.py <request folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

from rebuild_things import _mask, _room
from room_checks import FLIP, WalkRoom, _ssim
from room_objects import _cameras, _records

SHRINK = 4  # frames are drawn and compared at a quarter of the cameras' size


def check(out):
    import cv2
    import open3d as o3d

    out = Path(out)
    folder = out / "things"
    placed = json.loads((out / "room-things.json").read_text())["things"]
    answers = json.loads((folder / "sam-answers.json").read_text())
    room = WalkRoom(out, furniture="room-things-all.ply")
    level, lift = _room(out)

    # Each rebuilt thing as its own geometry, so a ray says which thing it hit
    scene = o3d.t.geometry.RaycastingScene()
    ids, middles = {}, {}
    for thing in placed:
        mesh = o3d.io.read_triangle_mesh(str(out / thing["file"]))
        ids[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))] = thing["slug"]
        middles[thing["slug"]] = np.asarray(mesh.vertices).mean(axis=0)
    names = {thing["slug"]: thing["name"] for thing in placed}

    overlaps, pictures, sheet = {}, [], []
    cameras = _cameras(out / "splat-data")
    for k, camera in enumerate(cameras):
        height, width = int(round(2 * camera["lens"][1, 2])), int(round(2 * camera["lens"][0, 2]))
        small = (height // SHRINK, width // SHRINK)
        rows, cols = np.mgrid[0:small[0], 0:small[1]]
        u, v = (cols.ravel() + 0.5) * SHRINK, (rows.ravel() + 0.5) * SHRINK
        origins, directions = room.camera_rays(camera, u, v)
        colour, _ = room.cast(origins, directions)
        drawn = (np.clip(colour, 0, 1).reshape(small[0], small[1], 3)[:, :, ::-1] * 255).astype(np.uint8)
        real = cv2.resize(cv2.imread(str(out / "splat-data" / "images" / camera["name"])), (small[1], small[0]),
                          interpolation=cv2.INTER_AREA)
        pictures.append(float(_ssim(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255,
                                    cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255)))
        if k % max(1, len(cameras) // 6) == 0 and len(sheet) < 6:
            sheet.append(np.vstack([real, drawn]))

        # Which rebuilt thing each pixel shows (the walls are left out: things in front of them are what's scored)
        rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        geometry = hit["geometry_ids"].numpy().astype(np.int64)  # uint32, with a huge value where nothing was hit
        shown = np.where(np.isfinite(hit["t_hit"].numpy()), geometry, -1).reshape(small)
        for slug, name in names.items():
            key = f"{camera['name']}|{name}"
            if key not in answers:
                continue
            masks = [cv2.resize(_mask(r, height, width).astype(np.uint8), (small[1], small[0]),
                                interpolation=cv2.INTER_NEAREST) > 0 for r in _records(answers[key])]
            masks = [m for m in masks if m.sum() > 5]
            if not masks:
                continue
            # The outline nearest where the thing projects is its own (three suitcases share a name)
            spot = ((middles[slug] + lift) @ level) @ FLIP
            spot = spot @ camera["turn"].T + camera["shift"]
            if spot[2] <= 0:
                continue
            px = camera["lens"][0, 0] * spot[0] / spot[2] + camera["lens"][0, 2]
            py = camera["lens"][1, 1] * spot[1] / spot[2] + camera["lens"][1, 2]
            centres = [np.argwhere(m).mean(axis=0) * SHRINK for m in masks]
            mask = masks[int(np.argmin([np.hypot(c[0] - py, c[1] - px) for c in centres]))]
            mine = shown == next(g for g, s in ids.items() if s == slug)
            union = (mine | mask).sum()
            if union:
                overlaps.setdefault(slug, []).append(float((mine & mask).sum() / union))
    per_thing = {slug: round(float(np.mean(v)), 3) for slug, v in overlaps.items()}
    result = {"things": round(float(np.mean(list(per_thing.values()))), 4) if per_thing else 0.0,
              "picture": round(float(np.mean(pictures)), 4), "frames": len(cameras),
              "thingsScored": len(per_thing), "perThing": dict(sorted(per_thing.items(), key=lambda kv: kv[1]))}
    (out / "checks").mkdir(exist_ok=True)
    (out / "checks" / "rebuild-check.json").write_text(json.dumps(result, indent=2))
    if sheet:
        cv2.imwrite(str(out / "checks" / "rebuild-check.jpg"), np.hstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python worker/rebuild_check.py <request folder>")
    print(json.dumps(check(sys.argv[1]), indent=2))
