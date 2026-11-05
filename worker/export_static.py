"""A static copy of one room for a free host (Cloudflare Pages; GitHub Pages works too): the viewer built in its static
mode (frontend/src/api.ts STATIC: no server, nothing to upload or rebuild) and the room's files at the same paths the
server serves them at, so the page needs no server at all. Rebuilt things with more than BUDGET triangles are thinned
for the web (a judge's browser loads every one of them); the stand-ins aren't, their paint is in their detail. Only the
things room-things.json lists go, and no file may go over a host's per-file cap.
    python worker/export_static.py <request id> [<out folder, default: site>] [--api <address>] [--fake <request id>]
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
FILES = ("room-splats.ply", "room-frame.json", "cameras.json", "room-shell.json",
         "room-objects.json")  # at the room's own address (room.ply goes thinned; the debug looks' surface stays home)
ASSETS = ("room-things.json", "room-materials.json", "room-shell-textures.json", "room-decals.json",
          "room-splat-objects.json", "room-issues.json", "video-live.json", "room-spots.json", "room-trust.json",
          "room-ad-examples.json")  # at <room>/assets/ (the Walk look's room-furniture.ply is left out: debug only)
ASSET_GLOBS = ("room-material-*.jpg", "room-shell-*.jpg", "room-shell-*-seen.png", "room-decal-*.png", "issue-*.jpg",
               "presence-*.jpg", "ad-example-*.jpg")
CLOUD_POINTS = 300_000  # points the page's copy of the cloud keeps: enough to measure on, in the debug looks
BUDGET = 120_000  # triangles a rebuilt thing keeps at most on the web
VR_TRIANGLES = 600_000  # a headset's copies of a whole room (vr-<file>), all told: a Quest draws about a million a frame
VR_THING = 40_000  # and of one thing on its own
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


def headset_copies(paths, budget=VR_TRIANGLES, folder=None):
    """vr-<file> next to each model (or in `folder`): the same detail everywhere, `budget` triangles all told. Vertex
    clustering, because these models are painted in their vertices: decimation keeps shape and loses paint, and turns a flat
    painted blanket into long shards. Returns the triangles written."""
    import math

    import open3d as o3d
    from scipy.spatial import cKDTree

    meshes = {Path(path): o3d.io.read_triangle_mesh(str(path)) for path in paths}
    area = sum(mesh.get_surface_area() for mesh in meshes.values())
    voxel = 0.8 * math.sqrt(2 * area / budget) if area > 0 else 0.01  # 2 triangles a cell, less what clustering merges (measured)
    total = 0
    for path, mesh in meshes.items():
        thin = mesh.simplify_vertex_clustering(voxel_size=voxel, contraction=o3d.geometry.SimplificationContraction.Average)
        thin.remove_degenerate_triangles()
        thin.remove_duplicated_triangles()
        points = np.asarray(thin.vertices)
        colours = (np.asarray(thin.vertex_colors) * 255).round().astype(np.uint8)
        data = path.read_bytes()
        end = data.index(b"end_header\n") + len(b"end_header\n")
        header = data[:end].decode("ascii")
        names = [line.split()[2] for line in header.splitlines() if line.startswith("property") and "list" not in line]
        count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
        seen = np.zeros(len(points), np.uint8)
        if "seen" in names:
            types = {"x": "<f4", "y": "<f4", "z": "<f4", "red": "u1", "green": "u1", "blue": "u1", "seen": "u1"}
            vertex = np.frombuffer(data, dtype=[(n, types[n]) for n in names], count=count, offset=end)
            original = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
            seen = vertex["seen"][cKDTree(original).query(points)[1]]
        target = Path(folder or path.parent) / f"vr-{path.name}"
        _write_thing_ply(target, points, colours, np.asarray(thin.triangles), seen)
        total += len(thin.triangles)
    return total


def thin_cloud(source, target, most=CLOUD_POINTS):
    """A point cloud's PLY with at most `most` points, picked evenly at random (the same ones every time)."""
    data = Path(source).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
    if count <= most or "element face" in header:
        shutil.copyfile(source, target)
        return count, count
    sizes = {"float": 4, "double": 8, "uchar": 1, "int": 4, "uint": 4, "short": 2, "ushort": 2, "char": 1}
    stride = sum(sizes[line.split()[1]] for line in header.splitlines() if line.startswith("property"))
    body = np.frombuffer(data, dtype=np.uint8, offset=end, count=count * stride).reshape(count, stride)
    keep = np.sort(np.random.default_rng(0).choice(count, most, replace=False))
    smaller = header.replace(f"element vertex {count}", f"element vertex {most}")
    Path(target).write_bytes(smaller.encode("ascii") + body[keep].tobytes())
    return count, most


def export(request_id, target="site", api="", fake=""):
    """`api`: the hosted pipeline's address (worker/modal_app.py, deployed): the copy then lets anyone start a room of
    their own there, and its sample room stays here. `fake`: a second room, the fake tour that shows the scam checks."""
    site = Path(target) if Path(target).is_absolute() else ROOT / target
    # 1. The viewer in its static mode
    env = {**os.environ, "VITE_STATIC": "1", "VITE_STATIC_ROOM": request_id, "VITE_FAKE_ROOM": fake,
           "VITE_API_BASE": api.rstrip("/")}
    subprocess.run("npm run build", cwd=ROOT / "frontend", env=env, shell=True, check=True)
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(ROOT / "frontend" / "dist", site)
    report = _copy_room(request_id, site, "A sample room")
    if fake:
        _copy_room(fake, site, "A suspicious listing (example)")
    files = [p for p in site.rglob("*") if p.is_file()]
    biggest = max(files, key=lambda p: p.stat().st_size)
    too_big = [str(p.relative_to(site)) for p in files if p.stat().st_size > CAP]
    report.update({"site": str(site), "files": len(files),
                   "megabytes": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                   "biggest": [str(biggest.relative_to(site)), round(biggest.stat().st_size / 1e6, 1)],
                   "overTheCap": too_big})
    if too_big:
        raise SystemExit(f"over Cloudflare Pages' 25 MB a file: {too_big}")
    return report


def _copy_room(request_id, site, label):
    """One room's files into the copy, at the addresses the server gives them."""
    data = ROOT / "data" / "requests" / request_id
    room = site / "api" / "requests" / request_id
    (room / "assets").mkdir(parents=True)

    # 2. The room's details, without the landlord's upload link (a static copy can't take a video anyway)
    request = json.loads((data / "request.json").read_text())
    request.pop("landlordPath", None)
    request["label"] = label  # what a visitor reads, not the filming take's name
    (room / "request.json").write_text(json.dumps(request))

    # 3. Its files where the viewer asks for them; the rebuilt things thinned
    thinned = {}
    thin_cloud(data / "room.ply", room / "room.ply")
    for name in FILES:
        if (data / name).exists():
            shutil.copyfile(data / name, room / name)
    for name in ASSETS:
        if (data / name).exists():
            shutil.copyfile(data / name, room / "assets" / name)
    for pattern in ASSET_GLOBS:
        for path in sorted(data.glob(pattern)):
            shutil.copyfile(path, room / "assets" / path.name)
    things = json.loads((data / "room-things.json").read_text())["things"]
    for thing in things:
        if thing.get("made"):  # a stand-in: its paint is in its detail
            shutil.copyfile(data / thing["file"], room / "assets" / thing["file"])
        else:
            thinned[thing["file"]] = _thin(data / thing["file"], room / "assets" / thing["file"])
    # A headset's own copies: the viewer in a headset's browser loads vr-<file>
    print("headset triangles", headset_copies([data / thing["file"] for thing in things], folder=room / "assets"))

    # 4. The student's things already built (the static copy can show and move them, not build new ones)
    built = []
    for meta_file in sorted((data / "items").glob("*/item.json")) if (data / "items").exists() else []:
        meta = json.loads(meta_file.read_text())
        if meta.get("status") == "done" and (meta_file.parent / "item.ply").exists():
            (room / "items" / meta["id"]).mkdir(parents=True, exist_ok=True)
            thinned[f"items/{meta['id']}"] = _thin(meta_file.parent / "item.ply", room / "items" / meta["id"] / "item.ply")
            headset_copies([meta_file.parent / "item.ply"], VR_THING, room / "items" / meta["id"])
            built.append({k: v for k, v in meta.items() if k not in ("photo", "why")})
    (room / "items.json").write_text(json.dumps(sorted(built, key=lambda m: m.get("createdAt", ""))))
    return {"thinned": {name: f"{a} -> {b} triangles" for name, (a, b) in thinned.items() if a != b},
            "items": len(built), "page": f"/room/{request_id}"}


if __name__ == "__main__":
    args = sys.argv[1:]
    api = args.pop(args.index("--api") + 1) if "--api" in args else ""
    if "--api" in args:
        args.remove("--api")
    fake = args.pop(args.index("--fake") + 1) if "--fake" in args else ""
    if "--fake" in args:
        args.remove("--fake")
    if len(args) not in (1, 2):
        sys.exit("usage: python worker/export_static.py <request id> [<out folder>] [--api <address>] [--fake <request id>]")
    print(json.dumps(export(*args, api=api, fake=fake), indent=2))
    # Vercel serves files first, and every other path (a room's page) gets the app
    out = Path(args[1]) if len(args) == 2 and Path(args[1]).is_absolute() else ROOT / (args[1] if len(args) == 2 else "site")
    (out / "vercel.json").write_text(json.dumps({"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]}, indent=2))
