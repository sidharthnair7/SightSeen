"""The whole room after the 3D step, in one go: the steps that were run by hand on take 4. The walls get the video's
photos and clean materials, the scale is checked against wall plates, the room's things are found, rebuilt by SAM 3D
and set in the room, then the issues are looked for (after the things, so a wall hidden behind a wardrobe isn't
"never filmed"). Each step writes its progress to pipeline.json and to `stage` (the page shows it). A step that fails
is noted and the rest go on, so the room still opens with what's done.

A quick run rebuilds only the MOST biggest things: small clutter (a sock, a bottle) is left out, and a room is ready
in minutes instead of most of an hour.
    python worker/pipeline.py <request folder> [--runner local|modal] [--most 12]
"""
import json
import sys
import time
import traceback
from pathlib import Path

MOST = 12  # things a quick run rebuilds: the biggest ones


def _keep_biggest(out, most):
    """things/things.json keeps its `most` biggest things (by measured extent); the rest go to its "skipped" list.
    Returns how many are kept."""
    file = Path(out) / "things" / "things.json"
    found = json.loads(file.read_text())
    things = sorted(found["things"], key=lambda t: -t.get("extent", 0))
    if len(things) > most:
        found["skipped"] = found.get("skipped", []) + [
            {"name": t["name"], "why": f"a quick run rebuilds the {most} biggest things"} for t in things[most:]]
        found["things"] = things[:most]
        file.write_text(json.dumps(found, indent=2))
    return min(len(things), most)


def finish(out, runner="local", most=MOST, stage=print, walls_done=None):
    """Every step after reconstruct.run, in order (see the module's doc). `walls_done` is called once the walls are
    finished and sized (the hosted pipeline opens the room then, and the things come in later). Returns what
    pipeline.json says."""
    import rebuild_things
    import room_issues
    import room_materials
    import room_scale
    import room_textures

    out = Path(out)
    record = {"runner": runner, "most": most, "steps": [], "done": False}
    record_file = out / "pipeline.json"

    def save():
        record_file.write_text(json.dumps(record, indent=2))

    def step(name, words, work):
        stage(words)
        entry = {"step": name, "state": "running"}
        record["steps"].append(entry)
        save()
        started = time.perf_counter()
        try:
            result = work()
            entry["state"] = "done"
            if isinstance(result, dict):  # its plain numbers and words: the rest stays in the step's own files
                entry["result"] = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool)) or v is None}
        except Exception as error:  # noqa: BLE001 - one step failing mustn't lose the rest
            traceback.print_exc()
            entry.update(state="failed", error=f"{type(error).__name__}: {error}"[:300])
        entry["seconds"] = round(time.perf_counter() - started, 1)
        save()
        return entry["state"] == "done"

    def scale():
        result = room_scale.measure(out)
        if result.get("scale") != result.get("before"):  # a new scale, or back to MapAnything's own
            room_scale.rebuild(out)  # the walls, their photos and materials at the room's real size
        return {k: result.get(k) for k in ("scale", "before", "why")}

    def build():
        if runner == "local":
            import reconstruct
            reconstruct.unload()  # MapAnything off the card: SAM 3D needs all of it
        return rebuild_things.build(out, runner=runner)

    step("photos", "putting the video's photos on the walls", lambda: room_textures.make_textures(out))
    step("materials", "making the clean walls and floor", lambda: room_materials.make(out))
    step("scale", "checking the room's size against wall plates", scale)
    if walls_done:
        walls_done()
    if step("pick", "finding the room's things (Muse Spark and SAM 3.1)", lambda: rebuild_things.pick(out)):
        kept = _keep_biggest(out, most)
        if step("build", f"rebuilding the {kept} biggest things in 3D (SAM 3D)", build):
            step("place", "setting each thing in the room", lambda: rebuild_things.place(out))
    step("issues", "looking for damage and what was never filmed", lambda: room_issues.find(out))
    record["done"] = True
    save()
    return record


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    runner = args[args.index("--runner") + 1] if "--runner" in args else "local"
    most = int(args[args.index("--most") + 1]) if "--most" in args else MOST
    print(json.dumps(finish(args[0], runner=runner, most=most), indent=2))
