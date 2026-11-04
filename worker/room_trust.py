"""The scam checks. A rental scam shows a real room the scammer can't get into, usually with its real photos and tour.
So these ask two things a stolen video or stolen photos can't answer. The AI reads and compares; plain rules decide.

  the code  The landlord's link gives a one-time code to hold up on camera. Muse Spark reads every number held up in the
            video WITHOUT being told the code, and plain code compares what it read with the code. So the model can't
            agree with a number it wants to see, and an old tour can't have today's code in it.
  the ad    Each photo from the ad, against the video's frames. First plain geometry: SIFT features between the photo and
            each frame, and RANSAC keeps the ones that fit one 3D layout. If enough spread-out details line up, it's the
            same place, whatever any model thinks. Then Muse Spark compares what can't easily change (windows, doors,
            radiators, floor), with furniture and decor allowed to differ.
  depth     A "tour" panned across a photo has no parallax: every frame is a flat warp of the last (one homography explains
            the matches). Walking through a real room can't do that. Plain geometry. (A real tour replayed on a screen keeps its
            parallax, so this doesn't catch that one. The code does.)

What this shows: whoever filmed the video was in that room, and it's the room in the ad. It can't show they own it.

    python worker/room_trust.py code <request folder> <code>     (the code, and the depth check)
    python worker/room_trust.py ad <request folder> <photo> [<photo> ...]
Writes room-trust.json and presence-code.jpg; ad checks write <id>-ad.jpg and <id>-lines.jpg next to their results.
A check costs one Muse Spark call per ad photo, or one for the code: about a cent each. Depth costs nothing.
"""
import json
import os
import re
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rebuild_things import _mask, _spark
from room_objects import _outline, _records
from sam_check import load_env

# The code
EARLY = 15.0  # seconds at the start of the video looked at closely: the landlord is asked to show it first
EARLY_STEP = 0.5  # a frame every this many seconds there
LATE_FRAMES = 18  # and this many frames over the rest, in case they showed it later
READ_SIDE = 1024  # pixels on a frame's long side: enough to read a number on paper
NEAR_PLACED = 2.0  # seconds: the code's frame counts as in the room when a frame this close was placed in 3D

READ = (
    "These {n} photos are frames of a video of a room, each numbered in its top-left corner. Someone may hold up a piece "
    "of paper, a card or a phone screen showing a number of 3 to 6 digits, written by hand or printed. List every such "
    "number you can read. Read it digit by digit and don't guess: leave out any you can't read clearly. Ignore numbers "
    "that belong to the room (clocks, thermostats, door numbers, posters, labels on products). Answer with only a JSON "
    'list, empty if there is none: [{{"number": "<the digits>", "frames": [frame numbers where it is readable]}}]'
)

# Depth
DEPTH_PAIRS = 24  # pairs of frames looked at, at most
DEPTH_FRAMES = 48  # frames they're chosen from
DEPTH_GAPS = (16, 8, 4, 2)  # frames apart, widest first: a slow camera needs a wide gap to show any parallax
PAIR_MATCHES = 30  # matches a pair needs to count: a blank wall says nothing
FLAT_PAIR = 0.9  # a pair is flat when one homography explains this share of what a 3D layout does
FLAT_VIDEO = 0.8  # and the video is, when this share of its pairs are
FEW_PAIRS = 6  # fewer pairs than this and it says nothing at all

# The ad
SIFT_SIDE = 1280  # pixels on a picture's long side for its features
MOST_FEATURES = 3000
RATIO = 0.75  # Lowe's ratio test
ERROR_PIXELS = 1.5  # RANSAC: how far from fitting one layout a match may be
LINES_UP = 40  # matches fitting one layout that make it the same place
SPREAD = 0.06  # and the share of the ad photo they must span: one poster or logo matching isn't a room
MOST_FRAMES_ASKED = 12  # frames shown to Muse Spark with each photo
MOST_LINES = 60  # lines drawn in the evidence picture
MOST_PHOTOS = 5

COMPARE = (
    "The first photo, marked AD, is from a rental ad. The other {n} photos are frames of a video the landlord sent of the "
    "room they say is for rent, each numbered in its top-left corner. Is the AD photo the same room as the video?\n"
    "Furniture, bedding, decor and lighting can change between the two, and an ad photo is usually wider, brighter and "
    "tidier. Judge by what can't easily change: the windows (how many, their shape, which wall), doors, radiators and "
    "vents, built-in closets, the floor, ceiling features, the room's shape. Say \"no\" only if at least two things that "
    "can't easily change clearly differ, and \"yes\" only if you can name at least two that clearly match.\n"
    'Answer with only JSON: {{"same_room": "yes" or "no" or "unsure", "same": ["<short things that match>"], '
    '"different": ["<short things that clearly differ>"], "frame": <the number of the frame closest to the AD photo\'s '
    'view>, "why": "<one plain sentence for a student deciding whether to trust this ad>"}}'
)


def _key():
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY isn't set: put it in .env")
    return key


def _shrink(picture, side):
    import cv2

    scale = side / max(picture.shape[:2])
    return cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else picture


def _numbered(picture, number, side):
    """A JPEG of the picture, small enough to send, with its number in the top-left corner."""
    import cv2

    small = _shrink(picture, side).copy()
    cv2.rectangle(small, (0, 0), (58, 30), (0, 0, 0), -1)
    cv2.putText(small, str(number), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def _json_in(answer, opening, closing):
    try:
        return json.loads(answer[answer.index(opening): answer.rindex(closing) + 1])
    except ValueError:
        return None


# The code -----------------------------------------------------------------------------------------------------------
def _video_frames(video):
    """(seconds, picture) for the frames to read: closely at the start, then spread over the rest. Reads the whole file
    in order, because a video recorded in a browser often doesn't say how long it is or let you jump around in it."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("The video couldn't be opened.")
    rate = capture.get(cv2.CAP_PROP_FPS)
    rate = rate if rate and 1 < rate < 240 else 30.0
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total / rate if total and total > 0 else None
    step = max(2.0, (duration - EARLY) / LATE_FRAMES) if duration and duration > EARLY else 5.0
    picked, at, index = [], 0.0, 0
    while capture.grab():
        seconds = index / rate
        index += 1
        if seconds + 1e-6 >= at:
            found, picture = capture.retrieve()
            if found:
                picked.append((seconds, picture))
            at += EARLY_STEP if seconds < EARLY else step
    capture.release()
    return picked, index / rate


def _outlined(picture, key):
    """The frame with the paper outlined by SAM 3.1, when it finds one: evidence a student can see at a glance."""
    import cv2

    shown = _shrink(picture, 960).copy()
    path = Path(os.environ.get("TEMP", "/tmp")) / f"code-{uuid.uuid4().hex}.jpg"
    try:
        cv2.imwrite(str(path), shown)
        height, width = shown.shape[:2]
        for record in _records(_outline(path, "paper with a number", key) or "")[:1]:
            contours, _ = cv2.findContours(_mask(record, height, width).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(shown, contours, -1, (255, 220, 0), 3)
    except Exception:  # noqa: BLE001 - the outline is a nicety: the plain frame still shows the code
        pass
    finally:
        path.unlink(missing_ok=True)
    return shown


def _placed_nearby(out, seconds, duration):
    """Whether the room's own frames around that moment were placed in 3D with the rest of the video: then the paper was
    held up in this room, not filmed somewhere else and stuck on the front. None when the room doesn't say."""
    spots = out / "room-spots.json"
    frames = sorted((out / "frames").glob("frame_*.jpg"))
    if not spots.exists() or not frames or not duration:
        return None
    placed = set(json.loads(spots.read_text()).get("placed", []))
    each = duration / len(frames)
    return any(frame.stem in placed and abs((k + 0.5) * each - seconds) <= NEAR_PLACED for k, frame in enumerate(frames))


def find_depth(out):
    """Whether the video moves through real 3D space. Pairs of frames a moment apart: how much of what fits one 3D layout
    (the fundamental matrix) also fits one flat warp (a homography). A real room has some of each; a photo being panned is
    all flat. Flag less and be right: only when nearly every pair is flat, across enough of them."""
    import cv2

    frames = sorted((Path(out) / "frames").glob("frame_*.jpg"))
    frames = [frames[round(k)] for k in np.linspace(0, len(frames) - 1, min(len(frames), DEPTH_FRAMES))] if frames else []
    with ThreadPoolExecutor(max_workers=8) as pool:
        features = list(pool.map(lambda path: _features(cv2.imread(str(path))), frames))

    def share(start):
        """From this frame, the widest gap that still overlaps enough to judge."""
        for gap in DEPTH_GAPS:
            if start + gap < len(features):
                count, a, b, _ = _line_up(features[start], features[start + gap])
                if count >= PAIR_MATCHES and a is not None:
                    _, flat = cv2.findHomography(a, b, cv2.RANSAC, ERROR_PIXELS * 2)
                    return float(flat.sum()) / count if flat is not None else 0.0
        return None

    starts = range(0, len(features), max(1, len(features) // DEPTH_PAIRS))
    with ThreadPoolExecutor(max_workers=8) as pool:
        shares = [value for value in pool.map(share, starts) if value is not None]
    flat_pairs = sum(value >= FLAT_PAIR for value in shares)
    known = len(shares) >= FEW_PAIRS
    return {"flat": known and flat_pairs >= FLAT_VIDEO * len(shares), "known": known, "pairs": len(shares),
            "flatPairs": flat_pairs, "typical": round(float(np.median(shares)), 3) if shares else None}


def find_trust(out, code=None):
    """The checks a room's video gets when it's built. Writes and returns room-trust.json."""
    out = Path(out)
    report = {"depth": None, "code": None}
    try:
        report["depth"] = find_depth(out)
    except Exception:  # noqa: BLE001 - one check failing mustn't lose the other: the page says it didn't finish
        traceback.print_exc()
    (out / "room-trust.json").write_text(json.dumps(report, indent=2))
    if code:
        try:
            report["code"] = find_code(out, code)
        except Exception:  # noqa: BLE001 - see above
            traceback.print_exc()
            report["code"] = {"code": code, "seen": False, "failed": True}
        (out / "room-trust.json").write_text(json.dumps(report, indent=2))
    return report


def find_code(out, code, video=None):
    """Looks for the room's one-time code in its video. Returns what it found (find_trust keeps it)."""
    import cv2

    out = Path(out)
    key = _key()
    video = Path(video) if video else next(out.glob("video.*"))
    frames, duration = _video_frames(video)
    pictures = [_numbered(picture, k, READ_SIDE) for k, (_, picture) in enumerate(frames)]
    answer = _spark(pictures, READ.format(n=len(pictures)), key)
    parsed = _json_in(answer, "[", "]")
    read = [item for item in (parsed or []) if isinstance(item, dict)]
    numbers = [re.sub(r"\D", "", str(item.get("number", ""))) for item in read]
    report = {"code": code, "seen": False, "atSeconds": None, "file": None, "inRoom": None, "framesChecked": len(frames),
              "read": sorted({number for number in numbers if number and number != code}), "readBy": "Muse Spark",
              "readFailed": parsed is None,
              "decidedBy": "plain comparison: the model was never told the code"}
    for item, number in zip(read, numbers):
        shown = [f for f in item.get("frames", []) if isinstance(f, int) and 0 <= f < len(frames)]
        if number == code and shown:
            seconds, picture = frames[shown[0]]
            cv2.imwrite(str(out / "presence-code.jpg"), _outlined(picture, key), [cv2.IMWRITE_JPEG_QUALITY, 85])
            report.update(seen=True, atSeconds=round(seconds, 1), file="presence-code.jpg",
                          inRoom=_placed_nearby(out, seconds, duration))
            break
    return report


# The ad -------------------------------------------------------------------------------------------------------------
def _features(picture):
    import cv2

    gray = cv2.cvtColor(_shrink(picture, SIFT_SIDE), cv2.COLOR_BGR2GRAY)
    points, descriptors = cv2.SIFT_create(nfeatures=MOST_FEATURES).detectAndCompute(gray, None)
    return np.float32([p.pt for p in points]) if points else np.zeros((0, 2), np.float32), descriptors, gray.shape


def _line_up(ad, frame):
    """How many of the ad photo's details fit one 3D layout with this frame. Returns (count, the ad's points, the frame's
    points, how much of the ad photo they span)."""
    import cv2

    points_a, descriptors_a, shape_a = ad
    points_b, descriptors_b, shape_b = frame
    if descriptors_a is None or descriptors_b is None or len(points_a) < 12 or len(points_b) < 12:
        return 0, None, None, 0.0
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    # One to one: a frame with few details (a blank door frame) would otherwise take every match in the ad photo on one
    # point, and any layout "fits" points that all land in one place
    best = {}
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            match = pair[0]
            if match.trainIdx not in best or match.distance < best[match.trainIdx].distance:
                best[match.trainIdx] = match
    good = list(best.values())
    if len(good) < 12:
        return 0, None, None, 0.0
    a = points_a[[match.queryIdx for match in good]]
    b = points_b[[match.trainIdx for match in good]]
    _, inliers = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, ERROR_PIXELS, 0.999, 10000)
    if inliers is None:
        return 0, None, None, 0.0
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 8:
        return int(keep.sum()), None, None, 0.0
    a, b = a[keep], b[keep]
    # How much of each picture they span, the smaller of the two: details bunched in one spot of either aren't a room
    spread = min(float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]) / (shape[0] * shape[1]))
                 for points, shape in ((a, shape_a), (b, shape_b)))
    return len(a), a, b, spread


def _evidence(photo, frame, a, b, path):
    """The ad photo and the frame side by side, with the details that line up joined by lines (none when nothing does)."""
    import cv2

    left, right = _shrink(photo, SIFT_SIDE), _shrink(frame, SIFT_SIDE)
    height = 640
    scale_l, scale_r = height / left.shape[0], height / right.shape[0]
    left_small = cv2.resize(left, None, fx=scale_l, fy=scale_l, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_AREA)
    gap = 12
    canvas = np.full((height, left_small.shape[1] + gap + right_small.shape[1], 3), 16, np.uint8)
    canvas[:, :left_small.shape[1]] = left_small
    canvas[:, left_small.shape[1] + gap:] = right_small
    if a is not None:
        order = np.random.default_rng(0).permutation(len(a))[:MOST_LINES]
        for k in order:
            start = (int(a[k, 0] * scale_l), int(a[k, 1] * scale_l))
            end = (int(b[k, 0] * scale_r) + left_small.shape[1] + gap, int(b[k, 1] * scale_r))
            cv2.line(canvas, start, end, (255, 220, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, start, 3, (180, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 3, (180, 80, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def decide(inliers, spread, answer):
    """The plain rule over both checks. Geometry can say yes on its own; only the model can say no, and only with two
    named differences it was told to look for."""
    if inliers >= LINES_UP and spread >= SPREAD:
        return "match"
    verdict = str((answer or {}).get("same_room", "")).strip().lower()
    if verdict == "yes" and len((answer or {}).get("same", []) or []) >= 2:
        return "likely"
    if verdict == "no" and len((answer or {}).get("different", []) or []) >= 2:
        return "mismatch"
    return "unsure"


def check_ad(frames_dir, photos, results_dir, key=None, stage=print):
    """Each ad photo against a room's frames. Returns a list of results; their pictures go in results_dir."""
    import cv2

    key = key or _key()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_files:
        raise ValueError("This room has no frames to compare with.")
    stage("reading the video's frames")
    with ThreadPoolExecutor(max_workers=8) as pool:
        frame_pictures = list(pool.map(lambda path: cv2.imread(str(path)), frame_files))
        frame_features = list(pool.map(_features, frame_pictures))
    results = []
    for number, photo_file in enumerate(list(photos)[:MOST_PHOTOS]):
        stage(f"comparing ad photo {number + 1} with the video")
        photo = cv2.imread(str(photo_file))
        if photo is None:
            results.append({"id": uuid.uuid4().hex[:8], "status": "unsure", "inliers": 0, "same": [], "different": [],
                            "why": "That photo couldn't be read.", "photo": None, "evidence": None, "frame": None})
            continue
        ad = _features(photo)
        with ThreadPoolExecutor(max_workers=8) as pool:
            lined = list(pool.map(lambda frame: _line_up(ad, frame), frame_features))
        best = int(np.argmax([item[0] for item in lined]))
        inliers, a, b, spread = lined[best]
        # What Muse Spark sees: the frames that matched best, then spread over the video
        ranked = sorted(range(len(frame_files)), key=lambda k: -lined[k][0])[:3]
        spread_out = [round(k) for k in np.linspace(0, len(frame_files) - 1, MOST_FRAMES_ASKED)]
        asked = list(dict.fromkeys(ranked + spread_out))[:MOST_FRAMES_ASKED]
        marked = _shrink(photo, READ_SIDE).copy()
        cv2.rectangle(marked, (0, 0), (70, 30), (0, 0, 0), -1)
        cv2.putText(marked, "AD", (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        pictures = [cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()]
        pictures += [_numbered(frame_pictures[k], k_shown, READ_SIDE) for k_shown, k in enumerate(asked)]
        answer = _json_in(_spark(pictures, COMPARE.format(n=len(asked)), key), "{", "}") or {}
        status = decide(inliers, spread, answer)
        shown = answer.get("frame")
        closest = asked[shown] if isinstance(shown, int) and 0 <= shown < len(asked) else best
        chosen = best if status == "match" else closest
        identity = uuid.uuid4().hex[:8]
        cv2.imwrite(str(results_dir / f"{identity}-ad.jpg"), _shrink(photo, 720), [cv2.IMWRITE_JPEG_QUALITY, 85])
        _evidence(photo, frame_pictures[chosen], a if chosen == best and inliers >= 12 else None,
                  b if chosen == best and inliers >= 12 else None, results_dir / f"{identity}-lines.jpg")
        results.append({"id": identity, "status": status, "inliers": int(inliers), "spread": round(spread, 3),
                        "frame": frame_files[chosen].stem, "same": [str(s) for s in answer.get("same", []) or []][:5],
                        "different": [str(s) for s in answer.get("different", []) or []][:5],
                        "why": str(answer.get("why", "") or ""), "photo": f"{identity}-ad.jpg",
                        "evidence": f"{identity}-lines.jpg", "name": Path(photo_file).name})
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "code":
        print(json.dumps(find_trust(args[1], args[2]), indent=2))
    elif len(args) >= 3 and args[0] == "ad":
        print(json.dumps(check_ad(Path(args[1]) / "frames", args[2:], Path(args[1]) / "ad-check"), indent=2))
    else:
        sys.exit(__doc__)
