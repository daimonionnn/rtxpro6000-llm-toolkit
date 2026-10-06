#!/usr/bin/env python3
"""Sequential, repeatable chat decode comparison; save responses and SSE timings.

Usage: python3 bench/compare_decode.py LABEL [--base http://127.0.0.1:8090]
Uses the server's advertised model ID. Thinking is explicitly disabled for both
llama.cpp-compatible servers and Strata. No generated code is executed.
"""
import argparse
import datetime
import json
import pathlib
import time
import urllib.request

PROMPTS = {
    "prose": "Explain how a relational database uses transactions, indexes and write-ahead logging. Write a detailed tutorial of at least 1000 words with concrete examples.",
    "code": "Write a complete Python implementation of an LRU cache with a doubly linked list and dictionary, including type hints, detailed comments and example usage. Then explain its complexity. Produce at least 1000 words of code and explanation.",
    "slovak": "Po slovensky podrobne vysvetli, ako funguje vyrovnávacia pamäť procesora, RAM a virtuálna pamäť. Použi konkrétne príklady a správnu slovenskú gramatiku. Napíš aspoň 1000 slov.",
}


def measure(base, model, prompt, max_tokens):
    body = dict(model=model, messages=[dict(role="user", content=prompt)],
                temperature=0, seed=1234, max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none")
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    content, reasoning, usage, timings, finish = [], [], {}, {}, None
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    content.append(text)
                    reasoning.append(thought)
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    tokens = usage.get("completion_tokens")
    # SSE chunks may contain several tokens (especially with speculation), so
    # this client rate is approximate. Keep engine timings separately if offered.
    rate = (tokens - 1) / (last - first) if tokens and tokens > 1 and last and last > first else None
    return dict(ttft_s=first-start if first else None, elapsed_s=elapsed,
                client_decode_tok_s=rate, usage=usage, engine_timings=timings,
                finish_reason=finish, content="".join(content), reasoning="".join(reasoning))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--base", default="http://127.0.0.1:8090")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]["id"]
    output = pathlib.Path(__file__).resolve().parents[1] / "logs" / "strata-comparison" / (args.label + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"refusing to overwrite {output}; choose a new label")
    data = dict(label=args.label, base=base, model=model,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                parameters=vars(args), prompts=PROMPTS, runs=[])
    warmup = measure(base, model, "What is 17 + 25? Answer briefly.", 64)
    data["warmup"] = warmup
    for repeat in range(args.repeats):
        for name, prompt in PROMPTS.items():
            row = dict(repeat=repeat, prompt=name, **measure(base, model, prompt, args.max_tokens))
            data["runs"].append(row)
            output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("repeat", "prompt", "ttft_s", "client_decode_tok_s", "usage", "finish_reason")}), flush=True)
    print(output)


if __name__ == "__main__":
    main()
