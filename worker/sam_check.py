"""One SAM 3.1 call through the Meta Model API, to see whether the key and credits work.

Usage (key in the repo's .env as MODEL_API_KEY=..., or set in the terminal):
    python worker\\sam_check.py <image.jpg> "door"

Costs one image: $2.50 per 1,000 images, so a quarter of a cent.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.meta.ai/v1/responses"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env():
    """Fill os.environ from the repo's .env, without overriding what the terminal already set."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and not name.strip().startswith("#") and value.strip():
            os.environ.setdefault(name.strip(), value.strip().strip('"'))


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python worker/sam_check.py <image.jpg> <phrase>")
    image, phrase = Path(sys.argv[1]), sys.argv[2]
    load_env()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("MODEL_API_KEY isn't set: put it in .env or in this terminal")

    data_url = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": "sam-3.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    }
    request = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, text = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read().decode()
    print(f"HTTP {status} in {time.time() - start:.1f} s")

    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        print(text[:2000])
        return
    if status != 200:
        print(json.dumps(reply, indent=2)[:2000])
        return
    # The answer is special-token text: one line per image, each object a box and a mask.
    output = "".join(
        part.get("text", "")
        for item in reply.get("output", [])
        for part in item.get("content", []) or []
    )
    print(f"{output.count('<|box')} '{phrase}' found")
    print(output[:600] if output else "(empty output: nothing matched)")


if __name__ == "__main__":
    main()
