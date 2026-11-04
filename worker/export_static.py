"""A static copy of one room for a free host (Cloudflare Pages; GitHub Pages works too): the viewer built in its static
mode (frontend/src/api.ts STATIC: no server, nothing to upload or rebuild) and the room's files at the same paths the
server serves them at, so the page needs no server at all. Rebuilt things with more than BUDGET triangles are thinned
for the web (a judge's browser loads every one of them); the stand-ins aren't, their paint is in their detail. Only the
things room-things.json lists go, and no file may go over a host's per-file cap.
    python worker/export_static.py <request id> [<out folder, default: site>]
Then upload the folder (.plans/HOSTING.md has the steps). Needs data/requests/<id>/request.json, the room's details
as the server returns them (GET /api/requests/<id>); the landlord's upload link is left out of the copy.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from rebuild_things import _write_thing_ply

ROOT = Path(__file__).resolve().parent.parent
FILES = ("room.ply", "room-surface.ply", "room-splats.ply", "room-frame.json", "cameras.json", "room-shell.json",
         "room-objects.json")  # at the room's own address
ASSETS = ("room-things.json", "room-materials.json", "room-shell-textures.json", "room-decals.json",
          "room-splat-objects.json", "room-issues.json", "video-live.json", "room-spots.json")  # at <room>/assets/ (the Walk look's room-furniture.ply is left out: debug only)
ASSET_GLOBS = ("room-material-*.jpg", "room-shell-*.jpg", "room-shell-*-seen.png", "room-decal-*.png", "issue-*.jpg")
BUDGET = 120_000  # triangles a rebuilt thing keeps at most on the web
VR_BUDGET = 30_000  # and in a headset's copy (vr-<file>): a Quest draws about a million triangles a frame, all told
CAP = 25 * 1024 * 1024  # bytes: Cloudflare Pages' largest file


def _thin(source, target, budget=BUDGET):
    """A rebuilt thing's PLY with at most BUDGET triangles (quadric decimation), its colours and its seen flags kept
    from the nearest original vertex; copied as it is when it's small enough. Returns the triangle counts."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh = o3d.io.read_triangle_mesh(str(source))
    before = len(mesh.triangles)
    if before <= budget:
        shutil.copyfile(source, target)
        return before, before
    data = source.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    names = [line.split()[2] for line in header.splitlines() if line.startswith("property") and "list" not in line]
    count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
    types = {"x": "<f4", "y": "<f4", "z": "<f4", "red": "u1", "green": "u1", "blue": "u1", "seen": "u1"}
    vertex = np.frombuffer(data, dtype=[(n, types[n]) for n in names], count=count, offset=end)
    original = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    thin = mesh.simplify_quadric_decimation(target_number_of_triangles=budget)
    points = np.asarray(thin.vertices)
    _, nearest = cKDTree(original).query(points)
    colours = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)[nearest]
    seen = vertex["seen"][nearest] if "seen" in names else np.zeros(len(points), np.uint8)
    _write_thing_ply(target, points, colours, np.asarray(thin.triangles), seen)
    return before, len(thin.triangles)


def export(request_id, target="site", api=""):
    """`api`: the hosted pipeline's address (worker/modal_app.py, deployed): the copy then lets anyone start a room of
    their own there, and its sample room stays here."""
    data = ROOT / "data" / "requests" / request_id
    site = Path(target) if Path(target).is_absolute() else ROOT / target
    # 1. The viewer in its static mode
    env = {**os.environ, "VITE_STATIC": "1", "VITE_STATIC_ROOM": request_id, "VITE_API_BASE": api.rstrip("/")}
    subprocess.run("npm run build", cwd=ROOT / "frontend", env=env, shell=True, check=True)
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(ROOT / "frontend" / "dist", site)
    room = site / "api" / "requests" / request_id
    (room / "assets").mkdir(parents=True)

    # 2. The room's details, without the landlord's upload link (a static copy can't take a video anyway)
    request = json.loads((data / "request.json").read_text())
    request.pop("landlordPath", None)
    request["label"] = "A sample room"  # what a visitor reads, not the filming take's name
    (room / "request.json").write_text(json.dumps(request))

    # 3. Its files where the viewer asks for them; the rebuilt things thinned
    thinned = {}
    for name in FILES:
        if (data / name).exists():
            shutil.copyfile(data / name, room / name)
    for name in ASSETS:
        if (data / name).exists():
            shutil.copyfile(data / name, room / "assets" / name)
    for pattern in ASSET_GLOBS:
        for path in sorted(data.glob(pattern)):
            shutil.copyfile(path, room / "assets" / path.name)
    for thing in json.loads((data / "room-things.json").read_text())["things"]:
        if thing.get("made"):  # a stand-in: its paint is in its detail
            shutil.copyfile(data / thing["file"], room / "assets" / thing["file"])
        else:
            thinned[thing["file"]] = _thin(data / thing["file"], room / "assets" / thing["file"])
        # A headset's own copy: the viewer in a headset's browser loads vr-<file> (stand-ins too: their paint blurs a
        # little, and the room's 2.5 million triangles become about 0.65 million)
        _thin(data / thing["file"], room / "assets" / f"vr-{thing['file']}", VR_BUDGET)

    # 4. The student's things already built (the static copy can show and move them, not build new ones)
    built = []
    for meta_file in sorted((data / "items").glob("*/item.json")) if (data / "items").exists() else []:
        meta = json.loads(meta_file.read_text())
        if meta.get("status") == "done" and (meta_file.parent / "item.ply").exists():
            (room / "items" / meta["id"]).mkdir(parents=True, exist_ok=True)
            thinned[f"items/{meta['id']}"] = _thin(meta_file.parent / "item.ply", room / "items" / meta["id"] / "item.ply")
            _thin(meta_file.parent / "item.ply", room / "items" / meta["id"] / "vr-item.ply", VR_BUDGET)
            built.append({k: v for k, v in meta.items() if k not in ("photo", "why")})
    (room / "items.json").write_text(json.dumps(sorted(built, key=lambda m: m.get("createdAt", ""))))

    files = [p for p in site.rglob("*") if p.is_file()]
    biggest = max(files, key=lambda p: p.stat().st_size)
    too_big = [str(p.relative_to(site)) for p in files if p.stat().st_size > CAP]
    report = {"site": str(site), "files": len(files),
              "megabytes": round(sum(p.stat().st_size for p in files) / 1e6, 1),
              "biggest": [str(biggest.relative_to(site)), round(biggest.stat().st_size / 1e6, 1)],
              "overTheCap": too_big,
              "thinned": {name: f"{a} -> {b} triangles" for name, (a, b) in thinned.items() if a != b},
              "items": len(built), "page": f"/room/{request_id}"}
    if too_big:
        raise SystemExit(f"over Cloudflare Pages' 25 MB a file: {too_big}")
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    api = args.pop(args.index("--api") + 1) if "--api" in args else ""
    if "--api" in args:
        args.remove("--api")
    if len(args) not in (1, 2):
        sys.exit("usage: python worker/export_static.py <request id> [<out folder>] [--api <hosted pipeline address>]")
    print(json.dumps(export(*args, api=api), indent=2))
