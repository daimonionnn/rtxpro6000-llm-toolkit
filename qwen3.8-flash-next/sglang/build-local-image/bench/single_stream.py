#!/usr/bin/env python3
"""Single-stream throughput for a served Qwen3.8-Flash-Next.

Reproduces the throughput column of BENCHMARKS.md. Token counts come from the
response `usage` field, not from counting stream chunks -- chunk counting
overstates on any engine that batches SSE frames.

    python3 bench/single_stream.py --base http://127.0.0.1:8000/v1

The model always reasons, so tok/s here is *total generated tokens* (thinking
plus answer) over wall time. That is the number to compare against a
non-thinking model's answer-only rate only if you say so out loud.
"""
import argparse
import json
import time
import urllib.request

WORKLOADS = {
    "english_prose": "Write four paragraphs on why NVMe latency, not capacity, "
                     "decides whether a large embedding table can stay off-GPU.",
    "coding": "Implement an LRU cache in Python with O(1) get and put, using a "
              "dict plus a doubly linked list. Include the node class and brief "
              "docstrings.",
    "reasoning": "A train leaves A at 60 km/h. Two hours later a second leaves A "
                 "at 90 km/h on the same track. Where and when does it catch up? "
                 "Show the algebra.",
    "long_context": None,  # filled in below
    "tool_style": "List the exact HTTP calls a client makes to stream a chat "
                  "completion from an OpenAI-compatible server, in order.",
}
WORKLOADS["long_context"] = (
    "Here is a log excerpt.\n\n"
    + "\n".join(f"[{i:05d}] worker {i%7} handled request {i} in {i%97} ms"
                for i in range(1200))
    + "\n\nWhich worker id appears most often, and what is its mean latency?"
)


def run(base, model, name, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    u = d.get("usage") or {}
    out = u.get("completion_tokens") or 0
    det = u.get("completion_tokens_details") or {}
    think = det.get("reasoning_tokens") or u.get("reasoning_tokens") or 0
    return {"workload": name, "wall_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": out, "reasoning_tokens": think,
            "tok_s": round(out / dt, 1) if dt else 0.0,
            "finish": d["choices"][0].get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--warmup", action="store_true",
                    help="send one throwaway request first (JIT kernels)")
    a = ap.parse_args()

    if a.warmup:
        run(a.base, a.model, "warmup", "Say OK.", 16, a.timeout)

    rows = []
    for name, prompt in WORKLOADS.items():
        row = run(a.base, a.model, name, prompt, a.max_tokens, a.timeout)
        rows.append(row)
        print(f"{row['workload']:16} {row['tok_s']:>7.1f} tok/s  "
              f"in={row['prompt_tokens']:>6} out={row['completion_tokens']:>6} "
              f"think={row['reasoning_tokens']:>6} {row['wall_s']:>7.2f}s "
              f"[{row['finish']}]")
    ok = [r["tok_s"] for r in rows if r["finish"] != "length"]
    if ok:
        print(f"\nmean over {len(ok)} completed: {sum(ok)/len(ok):.1f} tok/s")
    truncated = [r["workload"] for r in rows if r["finish"] == "length"]
    if truncated:
        print(f"truncated (raise --max-tokens): {', '.join(truncated)}")


if __name__ == "__main__":
    main()
