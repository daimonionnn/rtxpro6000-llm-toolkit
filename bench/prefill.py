#!/usr/bin/env python3
"""Measure prefill throughput at several prompt sizes, cold and prefix-cached.

    python3 bench/prefill.py [size ...]        # default: 4096 32768 131072

Prefill time is measured as time-to-first-token with max_tokens=1, which is what
a caller actually waits for (it includes tokenization and HTTP transfer; on this
setup that overhead is small - the engine's own per-chunk "input throughput"
lines in the server log agree with these numbers to within a few percent).

Each cold run uses freshly generated text (with a per-run random nonce) and POSTs
/flush_cache first, so neither the radix/prefix cache nor HiCache/NIXL storage
from an earlier run can inflate the result. The cached run then repeats the
identical prompt to show the other extreme.

Standard library only.
"""
import json, os, random, string, sys, time, urllib.error, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8090")
MODEL = "Qwen3.8-Flash-Next"
TOKENS_PER_WORD = 1.154  # measured for this tokenizer on this word mix

WORDS = """the of and to in a is that for it as was with be by on not he this are or his
from at which but have an they one you had we all her she there been their system memory
cache token model server request buffer kernel thread process value compute latency
throughput context window prefill decode attention layer weight matrix vector index batch
queue stream client parser config runtime driver device python function return class
method object string number list array result error module import export static const""".split()


def flush():
    # The server refuses (HTTP 400) while any request is still running or waiting.
    # With HiCache a finished request can count as busy for a moment while its
    # prefix is written through to storage, so retry briefly instead of failing.
    req = urllib.request.Request(BASE + "/flush_cache", method="POST")
    for attempt in range(30):
        try:
            urllib.request.urlopen(req, timeout=60).read()
            return
        except urllib.error.HTTPError as err:
            if err.code != 400 or attempt == 29:
                raise
            time.sleep(1)


RUN_NONCE = os.urandom(6).hex()


def make_text(n_words, rng):
    # A unique tag per prompt so no earlier prompt can be a prefix of this one.
    # The per-run nonce matters with HiCache/NIXL: /flush_cache clears GPU and
    # host tiers but not the files on disk, and the word sequence is seeded, so
    # without it a repeated benchmark run restores its "cold" prompts from storage.
    tag = RUN_NONCE + "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
    out = [tag]
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 18 == 17:
            out.append(".\n")
    return " ".join(out)


def measure(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    ttft = prompt_tokens = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if ttft is None and chunk.get("choices"):
                ttft = time.time() - start
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"]["prompt_tokens"]
    return ttft, prompt_tokens


def main(sizes):
    rng = random.Random(1234)
    print(f"{'target':>8}  {'run':<14}  {'prompt tok':>10}  {'TTFT s':>8}  {'tok/s':>10}")
    print("-" * 58)
    for target in sizes:
        prompt = make_text(int(target / TOKENS_PER_WORD), rng)
        flush()
        for label in ("cold", "prefix-cached"):
            ttft, ptok = measure(prompt)
            print(f"{target:>8}  {label:<14}  {ptok:>10}  {ttft:>8.3f}  {ptok / ttft:>10.1f}")
        print()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [4096, 32768, 131072])
