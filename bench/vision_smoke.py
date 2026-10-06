#!/usr/bin/env python3
"""Send an image as an OpenAI data URL; save response, usage and latency.

Example: python3 bench/vision_smoke.py LABEL image.png --base http://127.0.0.1:8090
The question contains no expected image contents. Standard library only.
"""
import argparse
import base64
import datetime
import json
import mimetypes
from pathlib import Path
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label")
    ap.add_argument("image", type=Path)
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--prompt", default='Read the four-digit code at the top of the image and identify the three colored shapes from left to right. Return only JSON with keys "code" (string) and "objects" (array of objects with "color" and "shape"). Use English color and shape names.')
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    output = Path(__file__).resolve().parents[1] / "logs/strata-vision" / (args.label + ".json")
    if output.exists():
        ap.error(f"refusing to overwrite {output}")
    image_path = args.image.resolve()
    if not image_path.is_file():
        ap.error(f"missing {image_path}")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data_url = "data:" + mime + ";base64," + base64.b64encode(image_path.read_bytes()).decode()
    with urllib.request.urlopen(base + "/health", timeout=30) as r:
        health = json.load(r)
    if not health.get("images"):
        ap.error("server does not advertise vision on /health")
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    body = dict(model=model, messages=[dict(role="user", content=[
        dict(type="image_url", image_url=dict(url=data_url)),
        dict(type="text", text=args.prompt)])], temperature=0, max_tokens=args.max_tokens,
        chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none")
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        response = json.load(r)
    elapsed = time.perf_counter() - start
    result = dict(model=model, image=str(image_path), prompt=args.prompt, health=health,
                  elapsed_s=elapsed, response=response,
                  measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(response["choices"][0]["message"]["content"])
    print(json.dumps(dict(elapsed_s=elapsed, usage=response.get("usage"), saved=str(output))))


if __name__ == "__main__":
    main()
