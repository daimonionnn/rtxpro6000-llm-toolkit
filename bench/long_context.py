#!/usr/bin/env python3
"""Measure 512-token generation after long prompts, using the decode SSE harness.

The first repetition has a fresh prefix; later repetitions reuse it. TG excludes
TTFT. Save actual prompt tokens, outputs and each timing, not just a median.
"""
import argparse
import datetime
import json
from pathlib import Path
import random
import urllib.request

from compare_decode import measure, PROMPTS
from prefill import make_text, TOKENS_PER_WORD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label")
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--sizes", type=int, nargs="+", default=[124000, 262000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    output = Path(__file__).resolve().parents[1] / "logs/strata-comparison" / (args.label + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        ap.error(f"refusing to overwrite {output}")
    data = dict(model=model, parameters=vars(args),
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                instruction=PROMPTS["prose"], runs=[])
    rng = random.Random(1234)
    for target in args.sizes:
        prompt = ("Background word corpus (do not repeat it):\n" +
                  make_text(int(target / TOKENS_PER_WORD), rng) +
                  "\nEnd of background. Your task:\n" + PROMPTS["prose"])
        for repeat in range(args.repeats):
            row = dict(target_tokens=target, repeat=repeat,
                       **measure(base, model, prompt, args.max_tokens))
            data["runs"].append(row)
            output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("target_tokens", "repeat", "ttft_s", "client_decode_tok_s", "usage", "finish_reason")}), flush=True)
    print(output)


if __name__ == "__main__":
    main()
