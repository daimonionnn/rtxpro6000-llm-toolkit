#!/usr/bin/env python3
"""Generate HumanEval+ / MBPP+ solutions from the running server, EvalPlus-compatible.

    uv run --no-project --with evalplus python bench/evalplus_codegen.py LABEL \
        [--dataset humaneval mbpp] [--thinking off|on] [--concurrency 4]
    bench/evalplus_evaluate.sh LABEL                  # score them in a sandbox

Writes logs/evalplus/<LABEL>/<dataset>.jsonl (sanitized, what gets scored) and
<dataset>.raw.jsonl (the model's full reply).

Requests are the ones `evalplus.codegen --backend openai` sends — same system
message, same instruction wrapper, same sanitizer — with two differences:

- thinking is set explicitly through chat_template_kwargs. The EvalPlus client
  cannot send it, and this model thinks by default, so a stock run measures
  whatever the server's default is and can leave `content` empty.
- requests run concurrently (default 4, the profiles' max-num-seqs). Greedy
  decoding makes the order irrelevant.

Resumes: tasks already present in the output file are skipped.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.sanitize import sanitize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Verbatim from evalplus.codegen / evalplus.gen.util.openai_request (0.3.1)
SYSTEM = "You are a helpful assistant good at coding."
INSTRUCTION = ("Please provide a self-contained Python script that solves the following "
               "problem in a markdown code block:")


def request(a, prompt):
    body = {
        "model": a.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"{INSTRUCTION}\n```python\n{prompt.strip()}\n```"}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": a.thinking == "on"},
    }
    req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            c = r["choices"][0]
            return c["message"].get("content") or "", c.get("finish_reason")
        except Exception as e:                       # server restart, timeout: retry
            if attempt == 4:
                raise
            print(f"  retry after {e!r}", file=sys.stderr)
            time.sleep(10)


def run(a, dataset):
    tasks = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    out_dir = os.path.join(ROOT, "logs", "evalplus", a.label)
    os.makedirs(out_dir, exist_ok=True)
    path, raw_path = (os.path.join(out_dir, f"{dataset}{s}.jsonl") for s in ("", ".raw"))
    done = set()
    if os.path.isfile(path):
        done = {json.loads(line)["task_id"] for line in open(path) if line.strip()}
    todo = [(tid, t) for tid, t in tasks.items() if tid not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{dataset}: {len(tasks)} tasks, {len(done)} already done, {len(todo)} to generate", flush=True)
    lock = threading.Lock()
    stats = {"n": 0, "length": 0, "empty": 0}
    t0 = time.time()

    def one(item):
        tid, task = item
        reply, finish = request(a, task["prompt"])
        solution = sanitize(reply, entrypoint=task["entry_point"])
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
            with open(raw_path, "a") as f:
                f.write(json.dumps({"task_id": tid, "solution": reply, "finish_reason": finish}) + "\n")
            stats["n"] += 1
            stats["length"] += finish == "length"
            stats["empty"] += not reply.strip()
            if stats["n"] % 25 == 0 or stats["n"] == len(todo):
                print(f"  {stats['n']}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, todo))
    print(f"{dataset}: wrote {path}  (truncated by max_tokens: {stats['length']}, empty replies: {stats['empty']}, "
          f"{time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="name for the profile under test, e.g. vllm-awq-w4a16-g32")
    ap.add_argument("--dataset", nargs="+", default=["humaneval", "mbpp"], choices=["humaneval", "mbpp"])
    ap.add_argument("--thinking", default="off", choices=["off", "on"])
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8090"))
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N remaining tasks (smoke test)")
    a = ap.parse_args()
    for d in a.dataset:
        run(a, d)


if __name__ == "__main__":
    main()
