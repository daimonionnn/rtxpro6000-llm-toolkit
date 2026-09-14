# SGLang, NVFP4, PLE table streamed from NVMe

Profile `sglang-nvfp4-nvme`, directory `qwen3.8-flash-next/sglang/nvfp4-nvme/`,
started with `scripts/start-qwen3.8-flash-next-sglang-nvfp4-nvme.sh`.

The yepapa-nest recipe as vendored in `sglang/build-local-image/`: RadixArk NVFP4
checkpoint, local Docker image `sglang-flashnext-sm120:local`, NEXTN speculation,
FP8 KV cache, and the 47.7 GiB PLE table read per token from NVMe instead of held
in RAM. The lowest host-RAM footprint of all profiles, and the slowest prefill of
the SGLang ones. Setup: [setup.md](../setup.md) sections 1–4.

Measured 2026-09-12: one RTX PRO 6000 Blackwell Workstation (96 GB, SM120), driver
610.57.04, the image built by `build.sh`, launch flags exactly as in
`serve-nvfp4-nvme.sh`.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights (experts + BF16 part) | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **NVMe**, read per token with io_uring (`O_DIRECT`, not cached in RAM) |
| **Model total** | **125.9 GiB** | 78.2 GiB VRAM + 47.7 GiB NVMe |
| KV cache — 231,936 tokens, FP8 | 2.9 GiB | VRAM |
| Mamba cache — 25 slots, **FP32** state | 5.3 GiB | VRAM |
| Free after CUDA graph capture | 5.5 GiB | VRAM |
| **VRAM in use** | **91.3 GiB** of 95.6 GiB | |
| **Host RAM** | **~0** beyond page cache | |
| Disk besides the checkpoint | Docker image 36.9 GB | |

KV, mamba and free figures come from the startup log (SGLang labels them GB but
counts GiB); "VRAM in use" from `nvidia-smi`.

## Code benchmarks

HumanEval 0.982 / HumanEval+ 0.963, MBPP 0.923 / MBPP+ 0.791 (greedy, thinking
off; [RESULTS.md](../../../RESULTS.md#code-humaneval-and-mbpp)).

Slovak blind check: 61 of 100, last of seven — mostly one grammar-correction answer
that rambled into its 800-token limit ([RESULTS.md](../../../RESULTS.md#non-english-slovak-blind-check)).

## Single-stream throughput

Prompt: *"Implement an LRU cache in Python with O(1) get and put. Include a short
docstring and 3 unit tests."* Temperature 0. Token counts from the response
`usage` field.

| Run | tok/s |
|---|---|
| 1 | 200.5 |
| 2 | 221.6 |
| 3 | 214.1 |

An earlier single measurement of the same prompt with `max_tokens=1200` gave
**214.0 tok/s** non-thinking and **206.7 tok/s** thinking (the thinking figure
includes its reasoning tokens).

The upstream repo reports 180.9 non-thinking and 148.2 thinking for its coding
workload, so this build is ahead in both modes — plausibly a newer source tree
and a newer driver than the author had.

### Read these numbers carefully

Three runs is not a throughput curve. The spread here (200–222) is wide enough
that differences under roughly 10% between configurations should not be trusted
without more samples. The prompt is also not the upstream repo's, so the
comparison against 180.9 is indicative, not like-for-like.

A very short generation measures worse: a 119-token completion came out at
162.6 tok/s, because time-to-first-token and ramp dominate.

## Prefill

Measured with `bench/prefill.py`: time-to-first-token at `max_tokens=1`, freshly
generated text each time, `/flush_cache` before every cold run so the prefix
cache cannot help.

| Prompt | Cold TTFT | Cold tok/s | Prefix-cached TTFT | Cached tok/s |
|---|---|---|---|---|
| 4,020 tok | 0.550 s | 7,314 | 0.616 s | 6,525 |
| 31,969 tok | 4.367 s | 7,320 | 0.191 s | 167,612 |
| 127,694 tok | 18.047 s | 7,076 | 0.745 s | 171,305 |

A second sweep gave 5,461 / 7,221 at 4K, 6,599 / 7,288 at 32K and 7,063 / 7,023
at 128K, so the first 4K run is warmup and everything else sits in a band around
**7,000–7,300 tok/s**.

**Prefill is flat and linear here.** With `--chunked-prefill-size 8192` every
chunk costs about the same, so wall time is roughly `prompt_tokens / 7,100`
regardless of prompt length. The engine's own per-chunk figures in the server log
agree — `Prefill batch ... input throughput (token/s)` reports 6,900–7,400 for
8,192-token chunks — which also means HTTP transfer and tokenization contribute
little to the TTFT above.

Upstream reports a *rising* curve on its hardware (7,914 at 8K, 9,395 at 32K,
9,990 at 65K). This build is lower at 32K and above and does not rise. The
measurements are not like-for-like — different prompts and a different source
tree — so treat it as a gap worth knowing about rather than a regression to chase.

### The prefix cache is what actually matters for an agent

Re-sending an identical 32K prompt drops TTFT from 4.4 s to 0.19 s, and 128K from
18.0 s to 0.75 s — roughly **23×**. In a coding agent loop, where each turn
re-sends a long and mostly unchanged prefix, this is the difference between a
usable and an unusable setup.

At 4K there is no benefit; the prompt is small enough that fixed overhead
dominates either way.

The caveat from upstream applies: the prefix cache holds 233K tokens here against
roughly 1.8M for a dense 27B at the same memory, so long shared prefixes get
evicted and re-prefilled sooner than they would on a smaller model.

## Memory breakdown, as the engine reports it

| Item | GB |
|---|---|
| Target model weights (NVFP4) | 80.62 |
| NEXTN draft (MTP) weights | 4.37 |
| Mamba cache, 25 slots | 5.34 |
| KV cache, 231,936 tokens (fp8) | 2.88 |
| CUDA graph capture (verify + draft decode + draft extend) | 0.85 |
| Free after startup | 5.49 |
| **Total in use (nvidia-smi)** | **93,534 / 97,887 MiB** (91.3 / 95.6 GiB) |

The mamba cache at 25 slots breaks down as conv_state 0.05, ssm_state 2.74,
intermediate_ssm_state 2.53, intermediate_conv_window 0.02.

The 47.68 GiB PLE table (320,001,536 rows across 10 files) is **not** in this
table — it is read from NVMe and never resident.

## Concurrency

`max_running_requests` is capped to **5**, and the engine says why:

```
max_running_requests is capped to 2 by the mamba state cache
(max_mamba_cache_size=10, 5 state slots per request).
```

Five state slots per request, 25 slots configured. Raising `--max-mamba-cache-size`
takes memory from the KV pool; there is no free lunch. Upstream measured
aggregate throughput flat from c4 onward at roughly 370–410 tok/s.

If the workload is many concurrent users rather than one deep agent loop, a dense
27B on the same card is roughly 3× better. This setup is tuned for the opposite
case.

## Cold start

About 95 seconds with a warm page cache: ~42 s loading the 206 target shards,
~7 s for the MTP draft, then FlashInfer autotune and CUDA graph capture.
