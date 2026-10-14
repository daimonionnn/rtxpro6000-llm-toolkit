#!/usr/bin/env python3
"""Answer accuracy on GSM8K (math) and MMLU-Pro (knowledge/reasoning), seeded subsets.

    python3 bench/qa_accuracy.py run LABEL [--dataset gsm8k mmlu-pro] [--n-gsm8k 300] [--n-mmlu-pro 600]
    python3 bench/qa_accuracy.py compare LABEL_A LABEL_B

`run` sends a seeded random subset of each test split to the running server,
greedy, thinking off, with a short chain-of-thought instruction, and writes
logs/qa-accuracy/<LABEL>/<dataset>.jsonl (one line per question: the reply, the
extracted answer, whether it is correct). It resumes where it stopped. The same
seed selects the same questions for every profile, so `compare` can pair them
(McNemar exact test on the questions where the two profiles disagree).

Datasets come from the Hugging Face datasets-server API and are cached in
logs/qa-accuracy/.cache. Standard library only. These are relative numbers for
comparing profiles on this machine, not official leaderboard scores.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "qa-accuracy")
SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "default", "test"),
}
LETTERS = "ABCDEFGHIJ"


def fetch_split(name):
    path = os.path.join(OUT, ".cache", name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    dataset, config, split = SOURCES[name]
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=100))
        for attempt in range(8):                     # the API rate-limits long paginations
            try:
                page = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                                        timeout=120))
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                time.sleep(5 * 2 ** attempt)
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            break
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rows, open(path, "w"))
    return rows


def prompt_for(name, row):
    if name == "gsm8k":
        return (row["question"] + "\n\nSolve the problem step by step. "
                "End with a final line in the form `#### <number>`.")
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (f"Question: {row['question']}\n\nOptions:\n{options}\n\n"
            "Think step by step, then end with a final line in the form `Answer: <letter>`.")


def expected(name, row):
    if name == "gsm8k":
        return row["answer"].split("####")[-1].strip().replace(",", "")
    return row["answer"]


def extract(name, text):
    if name == "gsm8k":
        m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        v = nums[-1].replace(",", "")
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return v
    m = re.findall(r"Answer:\s*\**\(?([A-J])\b", text)
    return m[-1] if m else None


def ask(a, text):
    body = {"model": a.model, "messages": [{"role": "user", "content": text}],
            "max_tokens": a.max_tokens, "temperature": 0, "seed": 1234, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            c = r["choices"][0]
            return c["message"].get("content") or "", c.get("finish_reason"), r.get("usage", {})
        except Exception:
            if attempt == 4:
                raise


def run(a):
    for name in a.dataset:
        rows = fetch_split(name)
        n = getattr(a, "n_" + name.replace("-", "_"))
        idx = sorted(random.Random(a.seed).sample(range(len(rows)), min(n, len(rows))))
        path = os.path.join(OUT, a.label, name + ".jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        done = {json.loads(l)["idx"] for l in open(path)} if os.path.exists(path) else set()
        todo = [i for i in idx if i not in done]
        lock = threading.Lock()

        def one(i):
            row = rows[i]
            reply, finish, usage = ask(a, prompt_for(name, row))
            got, want = extract(name, reply), expected(name, row)
            ok = got is not None and (got == want if name == "mmlu-pro" else
                                      _num_eq(got, want))
            rec = dict(idx=i, expected=want, got=got, correct=ok, finish=finish,
                       completion_tokens=usage.get("completion_tokens"),
                       category=row.get("category"), reply=reply)
            with lock, open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return ok

        with cf.ThreadPoolExecutor(a.concurrency) as pool:
            for k, _ in enumerate(pool.map(one, todo), 1):
                if k % 50 == 0:
                    print(f"{name}: {k}/{len(todo)}", flush=True)
        recs = [json.loads(l) for l in open(path)]
        right = sum(r["correct"] for r in recs)
        print(f"{name}: {right}/{len(recs)} = {right / len(recs):.3f}  "
              f"(unparsed {sum(r['got'] is None for r in recs)}, truncated "
              f"{sum(r['finish'] == 'length' for r in recs)})", flush=True)


def _num_eq(got, want):
    try:
        return math.isclose(float(got), float(want), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return got == want


def compare(a):
    for name in SOURCES:
        paths = [os.path.join(OUT, l, name + ".jsonl") for l in (a.label_a, a.label_b)]
        if not all(os.path.exists(p) for p in paths):
            continue
        ra, rb = ({json.loads(l)["idx"]: json.loads(l)["correct"] for l in open(p)} for p in paths)
        common = sorted(set(ra) & set(rb))
        only_a = sum(ra[i] and not rb[i] for i in common)
        only_b = sum(rb[i] and not ra[i] for i in common)
        k, n = min(only_a, only_b), only_a + only_b
        p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
        print(f"{name}: n={len(common)}  {a.label_a} {sum(ra[i] for i in common)}  "
              f"{a.label_b} {sum(rb[i] for i in common)}  only-{a.label_a} {only_a}  "
              f"only-{a.label_b} {only_b}  McNemar exact p={p:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--dataset", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    r.add_argument("--n-gsm8k", type=int, default=300)
    r.add_argument("--n-mmlu-pro", type=int, default=600)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8090"))
    r.add_argument("--model", default=os.environ.get("MODEL", "Qwen3.8-Flash-Next"))
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--concurrency", type=int, default=4)
    c = sub.add_parser("compare")
    c.add_argument("label_a")
    c.add_argument("label_b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
