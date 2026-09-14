# Benchmarks

Standalone scripts for measuring whichever profile is serving. They talk to
the OpenAI-compatible API on `http://127.0.0.1:8090` (override with the `BASE`
environment variable), work with SGLang, vLLM and TabbyAPI alike, and use the
Python standard library only.

| Script | Measures |
|---|---|
| `prefill.py` | Prefill speed: time-to-first-token at several prompt sizes, cold and prefix-cached |
| `evalplus_codegen.py` + `evalplus_evaluate.sh` | Code ability: HumanEval+ and MBPP+ pass@1, scored in a sandbox |
| `language_samples.py` | Output quality in a non-English language: fixed prompts per profile, compared blind |

## `prefill.py` — prefill speed

```bash
python3 bench/prefill.py                  # 4K, 32K and 128K tokens
python3 bench/prefill.py 8192 65536       # or your own sizes
BASE=http://127.0.0.1:9000 python3 bench/prefill.py
```

For each size it sends a freshly generated prompt with `max_tokens=1` and reports
the time to the first streamed token, then sends the identical prompt again:

```
  target  run             prompt tok    TTFT s       tok/s
----------------------------------------------------------
   32768  cold                 31882     3.361      9484.8
   32768  prefix-cached        31882     0.728     43800.7
```

- **TTFT is what a caller waits for.** It includes tokenization and HTTP; on this
  machine that overhead is within a few percent of the engine's own per-chunk
  throughput lines.
- **Cold runs stay cold.** Every run gets a random nonce at the start of its
  prompts, and before each size the script flushes the prefix cache where the
  engine offers an endpoint (SGLang `POST /flush_cache`, retried while HiCache is
  still writing; vLLM `/reset_prefix_cache` in dev mode only). Without the nonce, a
  repeated run on a HiCache/NIXL profile would restore its "cold" prompts from disk.
- Prompt sizes are approximate: the text is generated from a word list at 1.154
  tokens per word, calibrated for the Qwen3.8-Flash-Next tokenizer; the `prompt
  tok` column is the server's exact count.
- The first request after a server start includes warmup (kernel compilation,
  CUDA graph paths). Run it twice and use the second run.

Decode speed is not covered here; the model docs measure it with a streamed
request of a few hundred tokens.

## `evalplus_codegen.py` + `evalplus_evaluate.sh` — HumanEval+ and MBPP+

```bash
# with the profile running: generate 164 + 378 solutions (a few minutes)
uv run --no-project --with evalplus python bench/evalplus_codegen.py vllm-awq-w4a16-g32

# score them: base and plus pass@1 per dataset
bench/evalplus_evaluate.sh vllm-awq-w4a16-g32
```

**Generation** sends each task the way `evalplus.codegen --backend openai` does —
the same system message, instruction wrapper and sanitizer — greedy, 4 requests at
a time, and writes `logs/evalplus/<LABEL>/{humaneval,mbpp}.jsonl` plus `.raw.jsonl`
with the full replies and finish reasons. The one change from stock EvalPlus:
**thinking is set explicitly** (`--thinking off` by default) through
`chat_template_kwargs`, which the EvalPlus client cannot send; this model thinks by
default, so a stock run measures whatever the server defaults to and can come back
with empty `content`. It resumes where it stopped; `--limit N` runs only N tasks as
a smoke test, `--max-tokens` defaults to 4096.

**Scoring** runs `evalplus.evaluate` in the official `ganler/evalplus` image
(pinned by digest) with `--network none`, a memory and PID limit, and your user
id: the model-written code never runs on the host. The datasets come from the
generation step's download and are copied to `logs/evalplus/.cache`, where EvalPlus
also caches ground-truth results. It needs complete sets — a `--limit` run only
checks that the pipeline works.

- The two numbers per dataset are **base** (the original tests) and **plus**
  (EvalPlus's extended tests); plus is the stricter one.
- These are relative numbers for comparing profiles on this machine. On
  `sglang-nvfp4-nvme` this harness scores HumanEval 0.982 / HumanEval+ 0.963, while
  the upstream recipe's author reports 0.939 / 0.921 for the same configuration
  with a different harness.

## `language_samples.py` — non-English output, compared blind

English benchmarks miss most of the damage quantization does to lower-resource
languages. This script collects answers to a fixed prompt set from each profile
and merges them into a sheet where nobody can tell which profile wrote what.

```bash
# 1. With each profile running in turn, collect its answers
python3 bench/language_samples.py collect vllm-awq-w4a16-g32
python3 bench/language_samples.py collect exllamav3-exl3-5.05bpw

# 2. Merge everything collected into a blind sheet and a separate key
python3 bench/language_samples.py blind
python3 bench/language_samples.py blind vllm-awq-w4a16-g32 exllamav3-exl3-5.05bpw --seed 38
```

**`collect LABEL`** sends every prompt of the language set one at a time, greedy
(temperature 0), thinking off, and writes
`logs/language-samples/<lang>/<LABEL>.json` with each answer, its finish reason,
token count and time. `LABEL` is any name for the profile under test.

| Option | Default | Meaning |
|---|---|---|
| `--lang` | `sk` | prompt set |
| `--base` | `$BASE` or `http://127.0.0.1:8090` | server |
| `--model` | `Qwen3.8-Flash-Next` | model id sent with each request |
| `--max-tokens` | 800 | per answer |

**`blind [LABEL ...]`** takes the named collections (all of them when none are
named) and writes `blind-<timestamp>.md`, where the answers to each prompt are
shuffled and labelled A, B, C… independently per prompt, plus
`blind-<timestamp>-key.md` mapping the letters back. `--seed` makes the shuffle
repeatable.

**Prompt set.** `sk` has ten Slovak prompts, each aimed at something quantization
tends to break first: an explanation, a formal email, noun inflection in five
cases, numeral agreement, idioms, a non-literal translation, a summary, a code
explanation, a short story with dialogue, and a grammar correction. To add a
language, add a list to `PROMPTS` in the script.

**Judging.** Read the sheet before opening the key. For the Slovak comparison in
[RESULTS.md](../RESULTS.md#non-english-slovak-blind-check)
the sheet was copied alone to a separate directory and graded by an LLM that saw
nothing else — every error quoted with a fix and a severity, a 1–10 score per
answer and a ranking per prompt — and the letters were mapped to profiles only
afterwards.

Keep in mind:

- One greedy answer per prompt is a small sample; a single early token can change
  an answer's whole course. Treat differences as a direction.
- Collect every profile in a comparison with the same script version, so all of
  them answer the same prompts.
- Results live under `logs/`, which is not tracked.
