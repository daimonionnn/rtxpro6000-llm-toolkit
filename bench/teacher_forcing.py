#!/usr/bin/env python3
"""Teacher-forced reference logits from a llama.cpp-family server, for Strata's quant_quality.py.

    python3 bench/teacher_forcing.py ids --tokenizer DIR --out SET.ids.json [--seg-tokens 1000] [--v1 english,code]
    python3 bench/teacher_forcing.py run --ids SET.ids.json --out REF.tsv [--base URL]

ids  tokenizes the Slovak texts in bench/fixtures/sk-teacher-forcing/ with Strata's tokenizer
     (the pack's tokenizer directory) and, with --v1, appends segments of Strata's own
     data/quant_quality/v1 set, so a Slovak gap can be told apart from a general one.
run  reads every segment one token at a time through the server: each request is the
     segment's first p+1 tokens with max_tokens 1 and 64 logprobs, so with the prompt cache
     only the one new token is evaluated, on the decode path. The top 64 log-probabilities
     come from the raw logits (no sampler). Token ids only come back from the OpenAI
     /v1/completions endpoint, which takes the prompt as text: each prefix is detokenized
     and checked with /tokenize to give back exactly the same ids; a position where it
     does not (a prefix ending inside a multi-byte character) is skipped and counted, as is
     one whose most likely next token is an incomplete UTF-8 byte (no logprobs come back). Rows are written in Strata's STRATA_LOGPOS
     format (pos, target, logp(target), argmax, logp(argmax), argmax==target, two unused
     NaN fields, 64 id:logp), plus REF.tsv.seg.json, so

         python tools/quant_quality.py compare REF.tsv STRATA_RUN.tsv --segments

     compares Strata against this server. A target outside the top 64 gets the 64th
     log-probability, so the reference perplexity is a lower bound; top-1 and KL are not
     affected. A request that evaluated more than one token (a rolled-back recurrent
     state, a dropped cache) is reported, because it took the batched prompt path.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "bench/fixtures/sk-teacher-forcing"
SK_SEGMENTS = ["story", "expository", "formal"]
TOPK = 64


def cmd_ids(a):
    strata = pathlib.Path(a.strata_dir)
    sys.path.insert(0, str(strata / "tools"))
    import strata_tokenizer as ST  # noqa: E402
    tp = pathlib.Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    segs = []
    for name in SK_SEGMENTS:
        ids = tok.encode((FIXTURES / f"{name}.txt").read_text(encoding="utf-8"), parse_special=False)
        if len(ids) < a.seg_tokens:
            raise SystemExit(f"{name}: only {len(ids)} tokens (< {a.seg_tokens})")
        segs.append({"name": f"sk-{name}", "ids": ids[:a.seg_tokens]})
        print(f"sk-{name}: {len(ids)} tokens, kept {a.seg_tokens}")
    if a.v1:
        v1 = {s["name"]: s for s in json.load(open(strata / "data/quant_quality/v1/set.ids.json"))["segments"]}
        for name in a.v1.split(","):
            segs.append({"name": name, "ids": v1[name]["ids"][:a.seg_tokens]})
            print(f"{name}: from the v1 set, kept {len(segs[-1]['ids'])}")
    pathlib.Path(a.out).write_text(json.dumps({"version": "sk-teacher-forcing", "seg_tokens": a.seg_tokens,
                                               "segments": segs}))
    print(f"wrote {a.out}")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def cmd_run(a):
    st = json.load(open(a.ids))
    out = pathlib.Path(a.out)
    base = a.base.rstrip("/")
    segs, rows, batched, skipped = [], 0, 0, 0
    with out.open("w") as f:
        for s in st["segments"]:
            ids, first, t0 = s["ids"], rows, time.time()
            # the first request evaluates the start of the segment as a prompt; the rest add one token each
            for p in range(a.start - 1, len(ids) - 1):
                text = post(base + "/detokenize", {"tokens": ids[:p + 1]})["content"]
                if post(base + "/tokenize", {"content": text, "add_special": False})["tokens"] != ids[:p + 1]:
                    skipped += 1
                    continue
                r = post(base + "/v1/completions", {"prompt": text, "max_tokens": 1, "logprobs": TOPK,
                                                    "temperature": 0, "cache_prompt": True})
                n_eval = (r.get("timings") or {}).get("prompt_n")
                if p >= a.start and n_eval is not None and n_eval != 1:
                    batched += 1
                    if batched <= 5:
                        print(f"  {s['name']} pos {p}: {n_eval} tokens evaluated", flush=True)
                lp = r["choices"][0].get("logprobs") or {}
                if not lp.get("content"):
                    # the most likely token is an incomplete UTF-8 byte: the server holds it back, logprobs too
                    skipped += 1
                    continue
                top = lp["content"][0]["top_logprobs"]
                pairs = [(int(x["id"]), max(float(x["logprob"]), -1e4)) for x in top][:TOPK]
                tgt = ids[p + 1]
                d = dict(pairs)
                lt = d.get(tgt, pairs[-1][1])
                am, la = pairs[0]
                f.write(f"{p}\t{tgt}\t{lt:.9f}\t{am}\t{la:.9f}\t{int(am == tgt)}\tnan\tnan\t"
                        + "\t".join(f"{i}:{v:.6f}" for i, v in pairs) + "\n")
                rows += 1
            segs.append({"name": s["name"], "first_line": first, "end_line": rows})
            print(f"{s['name']}: {rows - first} positions, {time.time() - t0:.1f} s", flush=True)
    pathlib.Path(str(out) + ".seg.json").write_text(json.dumps(
        {"segments": segs, "base": a.base, "ids": a.ids, "batched_requests": batched,
         "skipped_positions": skipped}, indent=1))
    print(f"wrote {out}; requests that evaluated more than one token: {batched}, positions skipped: {skipped}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seg-tokens", type=int, default=1000)
    p.add_argument("--v1", default="")
    p.add_argument("--strata-dir", default=os.path.expanduser("~/development/Strata"))
    p = sub.add_parser("run")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8090"))
    p.add_argument("--start", type=int, default=1, help="first position scored (its prefix goes in as one prompt)")
    a = ap.parse_args()
    return cmd_ids(a) if a.cmd == "ids" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
