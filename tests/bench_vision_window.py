#!/usr/bin/env python3
"""Measure 16 -> 17 -> text-only -> 18 -> text-only on an idle vision server.

Start the experimental server with DS4_VISION_KEEP_IMAGES=16. Run again on an
unmodified server to record the expected 17-image rejection. No model download
or server configuration is performed by this script.
"""
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def ask(args, history, label):
    payload = dict(model=args.model, messages=history, temperature=0,
                   reasoning_effort="none", max_tokens=args.max_tokens,
                   stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    last = None
    content = []
    usage = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        content.append(text)
    except urllib.error.HTTPError as exc:
        return {"label": label, "status": exc.code,
                "error": exc.read().decode(),
                "total_seconds": time.monotonic() - start}, None
    elapsed = time.monotonic() - start
    completion = usage.get("completion_tokens", 0)
    # SSE chunks can contain multiple tokens, so this is an estimate, not
    # the engine's decode benchmark. Preserve raw timings and usage as well.
    rate = ((completion - 1) / (last - first)
            if completion > 1 and first is not None and last > first else None)
    return {"label": label, "status": 200,
            "ttft_seconds": first - start if first is not None else None,
            "total_seconds": elapsed, "decode_tok_s_estimate": rate,
            "usage": usage, "content": "".join(content)}, {
                "role": "assistant", "content": "".join(content)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--archive-lines", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    fixtures = Path(__file__).parent / "vision-fixtures/glm53"
    names = ["text.png", "spatial.png", "screenshot.png", "earth.jpg",
             "diagram.png"]

    def image_turn(index):
        path = fixtures / names[index % len(names)]
        mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return {"role": "user", "content": [
            {"type": "text", "text": f"Image {index + 1}. Describe the latest image briefly."},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{data}"}}]}

    history = [{"role": "system", "content": "Answer briefly. " + "\n".join(
        f"Archive fact {i}: preserve the conversation text."
        for i in range(args.archive_lines))}]
    for i in range(16):
        history.append(image_turn(i))
        if i < 15:
            history.append({"role": "assistant", "content": "Image received."})
    results = []
    for label, index in [("16-cold", None), ("16-text", -1),
                         ("17-drop", 16), ("17-text", -1),
                         ("18-drop", 17), ("18-text", -1)]:
        if index == -1:
            history.append({"role": "user", "content": "Describe the latest image again briefly."})
        elif index is not None:
            history.append(image_turn(index))
        result, reply = ask(args, history, label)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if reply is None:
            raise SystemExit(1)
        history.append(reply)


if __name__ == "__main__":
    main()
