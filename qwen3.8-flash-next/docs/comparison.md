# Comparing the launchers

Every launcher in this toolkit runs **the same checkpoint with the same weights**:
RadixArk/Qwen3.8-Flash-Next-NVFP4 @ `7b719225242a`. What differs is where the PLE
table lives, a handful of runtime precisions, and the context window. This page
puts the architecture, quantization, speed and quality question side by side.

All measurements were taken on this machine (one RTX PRO 6000 Blackwell 96 GB,
driver 610.57.04) on 2026-09-12 and 2026-09-13. How each launcher was made to work
is in [ple-ram-experiment.md](ple-ram-experiment.md).

## Architecture

Common to all launchers. Read from the checkpoint's `config.json`.

| | |
|---|---|
| Model | Qwen3.8-Flash-Next (`qwen4_exp`), **176B MoE, ~6B active** |
| Layers | 48: **36 linear attention** (GDN) + **12 full attention** (every 4th layer) |
| Full attention | QSA sparse attention; the indexer selects 2,048 tokens (compress ratio 4) |
| Attention heads | 24 query, 2 KV, head_dim 256; hidden size 2,560 |
| Linear attention heads | 16 key, 48 value |
| MoE | **512 routed experts, 10 active** per token, plus 1 shared expert; expert FFN 640 |
| PLE | N-gram embedding table (3-grams) at layer 2 — **51B parameters**, 47.7 GiB in FP8 |
| Other | hyper-connections ×4, 1 MTP layer (drives NEXTN speculation), vision encoder |
| Native context | 262,144 tokens; vocabulary 248,320 |

**Comments**

- **Most of the parameter count is not compute.** Of 176B parameters, 51B are
  the PLE lookup table and most of the rest are 512 experts of which only 10 run
  per token. That is why a 176B model decodes at the speed of a small one, and why
  the PLE table can leave the GPU at all: it is a lookup, not matrix math.
- **Decode speed barely depends on context length.** Three layers in four carry a
  fixed-size recurrent state instead of a growing KV cache, and the other quarter
  attend to a selected 2,048 tokens. Measured: 228–247 tok/s from an empty prompt
  to 221K tokens of context (see Speed below). For a long-running agent this
  matters more than the empty-context figure.
- **The KV cache is small per token** — only 12 layers keep one, with 2 KV heads
  each. That is what lets hundreds of thousands of tokens fit in a few GB.

## Quantization and runtime settings

| Launcher | Weights | KV cache | Mamba SSM state | RoPE / window | KV pool (tokens) |
|---|---|---|---|---|---|
| **`sglang-nvfp4-nvme`** (yepapa-nest recipe) | NVFP4 mix¹ | FP8 | **FP32** | native / 262K | 231,936 |
| **`sglang-nvfp4-ram`** (our image, PLE in RAM) | NVFP4 mix¹ | FP8 | BF16 | native / 262K | 498,624 |
| **`sglang-nvfp4-ram-official` (cookbook defaults)** (official image, cookbook as published) | NVFP4 mix¹ | **BF16** | BF16 | native / 262K | 76,864 |
| **`sglang-nvfp4-ram-official` with FP8 KV** (official image, FP8 KV) | NVFP4 mix¹ | FP8 | BF16 | native / 262K | 498,624 — **crashes** |
| **`sglang-nvfp4-ram-official`** (official image, tuned for one agent) | NVFP4 mix¹ | **BF16** | BF16 | native / 262K | 256,832 |
| **`sglang-nvfp4-ram-pennyroyal`** (jpezzulli fork) | NVFP4 mix¹ | FP8 | BF16 | **YaRN ×2** / 524K | **831,872** |

¹ **NVFP4 mix**, identical everywhere: routed experts in **NVFP4 W4A4** (4-bit
float weights *and* activations, one FP8 E4M3 scale per block of 16 values, FP32
global scale); attention, linear attention, shared expert, router,
hyper-connections, MTP draft, vision encoder, embeddings and lm_head in **BF16**;
the PLE table in **FP8 E4M3**. `quant_info.py` prints this breakdown from the
checkpoint at every launch.

**Comments**

- **The checkpoint is not "4-bit" throughout.** Only the routed experts are NVFP4,
  but they hold most of the non-PLE parameters, which is what brings the weights
  down to ~80 GB of VRAM.
- **FP32 mamba state in `sglang-nvfp4-nvme` is the model's own default**
  (`mamba_ssm_dtype: float32` in `config.json`), not a choice. The RAM profiles set
  `--mamba-ssm-dtype bfloat16` because halving that state is part of how they fit.
  Memory confirms it: 110 MB per state slot in `sglang-nvfp4-nvme`, 55 MB in the others.
- **FP8 KV here is uncalibrated.** The checkpoint ships no KV scales, and the log
  warns `Defaulting to scaling factors of 1.0. This may lead to less accurate
  results!` Upstream PR #36644 (unmerged on 2026-09-13) would add per-layer descale.
- **KV precision decides the pool size** almost by itself: the same memory holds
  498,624 FP8 tokens or 256,832 BF16 tokens (2b and 2c differ only in that).
- **`sglang-nvfp4-ram-official` with FP8 KV is listed as a warning.** The official image has no fix for FP8 KV
  in chunked prefill, and the first 57K-token prompt killed its scheduler with
  `Unsupported rhs dtype fp8e4nv`.
- **YaRN in `sglang-nvfp4-ram-pennyroyal` is static.** `--json-model-override-args` sets factor 2 for
  every request, so short prompts get rescaled positions too. That is the price of
  the 524K window.
- **Speculative decoding does not change output.** All launchers use NEXTN with 4
  draft tokens; verification against the target is exact, so the draft's precision
  (`modelopt_fp4` in most, `unquant` in `sglang-nvfp4-ram-pennyroyal`) affects speed, not results.

## Where the model lives: VRAM, RAM, NVMe

The checkpoint is the same file set for every profile — **125.9 GiB (135.3 GB)
on disk, 180.0B parameters, 6.01 bits per parameter on average**. Exact counts,
read from the safetensors headers:

| Component | Parameters | Precision | Bits / param | On disk |
|---|---|---|---|---|
| Routed experts (48 layers × 512) | 120.8B | NVFP4 W4A4 | 4.50¹ | 63.3 GiB |
| PLE n-gram table | 51.2B | FP8 E4M3 | 8.00 | 47.7 GiB |
| Attention, linear attention, shared experts, router, embeddings, lm_head, MTP, vision | 8.0B | BF16 | 16.00 | 14.9 GiB |
| **Total** | **180.0B** | | **6.01** | **125.9 GiB** |

¹ 4 bits per value plus one FP8 scale per 16 values (0.5 bits) plus FP32 global
scales.

Only 10 of 512 experts run per token, so of the ~6B parameters active per token
2.36B are routed-expert weights at 4.5 bits and the rest are BF16.

What differs between profiles is **where the PLE table lives** and **how the free
VRAM is spent**. KV cache, mamba cache and free-after-graphs figures are from each
profile's startup log (SGLang labels them GB but counts in GiB); "VRAM in use" is
`nvidia-smi`, converted from MiB. The card has 95.6 GiB (97,887 MiB).

### `sglang-nvfp4-nvme` — `qwen3.8-flash-next/sglang/nvfp4-nvme/`

| | Size | Where |
|---|---|---|
| Model weights (experts + BF16 part) | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **NVMe**, read per token with io_uring (`O_DIRECT`, not cached in RAM) |
| **Model total** | **125.9 GiB (135.3 GB)** | 78.2 GiB VRAM + 47.7 GiB NVMe |
| KV cache — 231,936 tokens, FP8 | 2.9 GiB | VRAM |
| Mamba cache — 25 slots, **FP32** state | 5.3 GiB | VRAM |
| Free after CUDA graph capture | 5.5 GiB | VRAM |
| **VRAM in use** | **91.3 GiB** of 95.6 GiB (93,534 MiB) | |
| **Host RAM** | **~0** beyond page cache | |
| Disk besides the checkpoint | Docker image `sglang-flashnext-sm120:local` 36.9 GB (36.1 GB shared with its base image) | |

### `sglang-nvfp4-ram` — `qwen3.8-flash-next/sglang/nvfp4-ram/`

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB (135.3 GB)** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 498,624 tokens, FP8 | 6.2 GiB | VRAM |
| Mamba cache — 12 slots, BF16 state | 1.8 GiB | VRAM |
| Free after CUDA graph capture | 4.6 GiB | VRAM |
| **VRAM in use** | **92.1–92.9 GiB** of 95.6 GiB | |
| **Host RAM** | **~65 GB** | |
| Disk besides the checkpoint | same Docker image as `sglang-nvfp4-nvme` | |

### `sglang-nvfp4-ram-official` (2c, as started by `scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official.sh`) — `qwen3.8-flash-next/sglang/nvfp4-ram-official/`

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB (135.3 GB)** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 256,832 tokens, **BF16** | 6.4 GiB | VRAM |
| Mamba cache — 12 slots, BF16 state | 1.8 GiB | VRAM |
| Free after CUDA graph capture | 4.6 GiB | VRAM |
| **VRAM in use** | not recorded; same budget as `sglang-nvfp4-ram` | |
| **Host RAM** | **~65 GB** | |
| Disk besides the checkpoint | Docker image `lmsysorg/sglang:dev-qwen38-next-local` 33 GB | |

### `sglang-nvfp4-ram-pennyroyal` — `qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/`

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB (135.3 GB)** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 831,872 tokens, FP8 | 10.3 GiB | VRAM |
| Mamba cache — 24 slots, BF16 state, no intermediate buffer | 1.4 GiB | VRAM |
| Free after CUDA graph capture | 3.8–3.9 GiB | VRAM |
| **VRAM in use** | **91.4–93.1 GiB** of 95.6 GiB | |
| **Host RAM** | **~65 GB**, **~97 GB** with HiCache (32 GB host tier) | |
| Disk besides the checkpoint | fork + venv 11 GB, kernel cache 0.2 GB, NIXL 14 MB; HiCache files grow with use until the disk reaches 92% | |

**Comments**

- **Rows do not add up exactly.** The KV, mamba and free figures come from the
  startup log, "VRAM in use" from `nvidia-smi` at a later moment, and CUDA
  context, activations and CUDA graphs (~1 GiB) are not listed. The 78.2 GiB of
  weights already includes the MTP draft head, which is part of the BF16 tensors.
- **The model itself is identical in all four** — same parameters, same bits,
  78.2 GiB on the GPU. Size therefore cannot separate the profiles' quality; only
  the KV cache precision (FP8 or BF16), the mamba state precision (FP32 or BF16)
  and `sglang-nvfp4-ram-pennyroyal`'s YaRN can.
- The "extra" VRAM each profile shows is state, not model: the KV pool and the
  mamba cache trade against each other inside the same ~14 GiB.
- NVMe versus RAM moves 47.7 GiB between host RAM and the SSD. It changes speed
  (prefill 1.5–2× faster from RAM), not what the model computes.

## Estimating quality from size

Bits per parameter is a rough but useful proxy for how much a quantization can
lose. It is only meaningful together with *which* parts are compressed:
quantizing attention, routers or embeddings hurts more than quantizing experts.

| Checkpoint | On disk | Avg bits / param | Routed experts | Attention, router, shared, embeddings | PLE table |
|---|---|---|---|---|---|
| Qwen BF16 (original) | 360 GB² | 16 | 16 | 16 | 16 |
| unsloth GGUF Q8_0 | 175.3 GiB | 8.4 | 8 | 8 | 8 |
| Qwen FP8 | ~190 GB (estimate³) | ~8.4 | 8 | 16 | 8 |
| wtdcode AWQ W4A16 | 180.8 GB | 8.0 | **INT4, group 128 (4.1)**, activations 16-bit | 16 | **16** |
| **RadixArk NVFP4 — all our profiles** | **135.3 GB** | **6.0** | **FP4, group 16, FP8 scales (4.5)**, activations 4-bit | **16** | 8 |
| unsloth GGUF UD-Q4_K_XL | 103.7 GiB | 4.9 | ~4–5, mixed | mixed, higher for sensitive tensors | quantized |
| unsloth GGUF UD-IQ4_XS | 87.2 GiB | 4.2 | ~4 | mixed | quantized |
| unsloth GGUF UD-Q3_K_XL | 83.8 GiB | 4.0 | ~3–4 | mixed | quantized |
| unsloth GGUF UD-IQ3_XXS | 76.3 GiB | 3.6 | ~3 | mixed | quantized |

² From the RadixArk model card ("360 GB BF16 source").
³ Not downloaded; estimated as 8-bit experts and PLE with the rest in BF16.

**Comments**

- **Our checkpoint spends its bits where they matter.** Every sensitive tensor —
  attention, the router, shared experts, embeddings, the MTP head — stays
  byte-identical to BF16 (the quantizer's audit compared 118.4 GB of them). Only
  the experts, which individually see few tokens, are compressed, and with a fine
  16-value block and floating-point scales.
- **Rule of thumb for this kind of MoE:** experts at ~4.5 bits with sensitive parts
  untouched typically cost around a point on hard reasoning and little elsewhere.
  That matches the measurements below.
- **AWQ W4A16 is not simply lower quality.** Its experts get fewer bits (4.1, coarser
  groups) but its activations stay 16-bit, and its PLE table is BF16. The best
  independent agentic result found (below) used it.
- **GGUF that fits entirely in 96 GB VRAM is ~3.6–4.2 bits on average and
  quantizes the sensitive parts too** — expect a visible drop compared with ours.

## Published quality results

### From Qwen (vendor, BF16 model)

[Qwen/Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):

| Benchmark | Qwen3.8-Flash-Next | Qwen3.8-27B | DeepSeek-V4-Flash | Claude-Opus-4.6 |
|---|---|---|---|---|
| SWE-bench Pro | **62.5** | 61.7 | 56.0 | 53.4 |
| SWE-bench Multilingual | **81.0** | 73.8 | — | 77.5 |
| DeepSWE 1.1 | **58.7** | 42.2 | 54.4 | — |
| LiveCodeBench v6 | **91.9** | 90.3 | 90.6 | 88.8 |
| GPQA Diamond | **91.7** | 89.2 | 90.8 | 91.3 |
| IFBench | **81.3** | 79.5 | 79.2 | 62.5 |
| CoWorkBench | **73.9** | 70.7 | 45.1 | 68.2 |
| JobBench | **55.7** | 33.4 | 41.3 | 36.6 |

The card also lists 125B main parameters with 6B active, plus the 51B n-gram
table and 4B MTP, and a native context of 262,144 tokens extensible to 1M.

### From the quantizer, on this exact NVFP4 checkpoint

RadixArk's `qualification-notes.md` and metric files ship inside
`models/Qwen3.8-Flash-Next-NVFP4/`:

| Eval | Protocol | BF16 reference | **NVFP4 (our checkpoint)** |
|---|---|---|---|
| GSM8K | all 1,319, temp 0.6 | 97.12–97.50 (three runs) | **97.27** — inside the BF16 band |
| AIME 2026 | 30 problems × 8, temp 1.0, thinking | 100 (240/240) | **98.75 pass@1** (237/240), majority@8 100 |

Their note: *"this quantization preserves single-turn accuracy (GSM8K/AIME
in-band); long agentic generations tend to run longer than BF16."* The AIME run
was on the checkpoint revision with a BF16 PLE table; the current FP8 PLE revision
scored GSM8K 97.27 against 97.35 for the BF16 PLE one.

They also warn that a loader which only upcasts the FP8 PLE bytes without applying
the table's scale "will serve wrong PLE embeddings silently". All three SGLang
builds here multiply by `weight_scale` (checked in `qwen4_exp.py` of each).

### Independent, on local hardware

| Source | Setup | Result |
|---|---|---|
| [yepapa-nest recipe](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000) — our `sglang-nvfp4-nvme` configuration, RTX PRO 6000 | HumanEval+, greedy | **0.939 / 0.921** non-thinking, 0.957 / 0.927 thinking; GSM8K 0.98; tools 6/7 |
| [jpezzulli fork](https://github.com/jpezzulli/sglang-rtxpro6000) — our `sglang-nvfp4-ram-pennyroyal`, RTX PRO 6000 | own reasoning, tool, vision and agent suite | reasoning **98.52/100**; tool calls 29/30 exact; vision and a six-turn agent task passed |
| [E2Studio local LLM benchmark](https://wonderrico.github.io/local_llm_benchmark/benchmark-main.html?filter=3.8) | first 100 Django tasks of SWE-bench Verified, mini-swe-agent | **98/100** with AWQ W4A16 + INT4 PLE on vLLM — the top score on that board; Qwen3.8-27B 74–81, DeepSeek V4 Flash 81–92 |
| [eesel AI review](https://www.eesel.ai/blog/qwen38-flash-next-review), citing Artificial Analysis | hosted API | Intelligence Index 56, #5 of 111 in its class; flagged **very verbose** (~200M output tokens on the index vs a 110M median) |

Also reported: Humanity's Last Exam 35.9 against Claude Opus 4.6 Max's 40.0, and
Agents' Last Exam pass@1 24.3 against DeepSeek-V4-Flash's 25.2
([search summary of review sites](https://www.eesel.ai/blog/qwen38-flash-next-review) —
not verified against a primary source). Qwen itself describes the model as an
intentionally under-trained preview of the Qwen4 architecture.

## Opinion: how good is it?

**Very good for its cost, and the quantization used here gives up little.**

- **Strongest where this toolkit is aimed:** agentic coding. It leads its vendor
  table on SWE-bench Pro, SWE-bench Multilingual and LiveCodeBench, and the one
  independent agentic benchmark on local hardware put it at the top of its board,
  well ahead of Qwen3.8-27B.
- **The NVFP4 checkpoint is close to BF16.** GSM8K lands inside the BF16 band;
  AIME 2026 is 1.25 points lower (three problems out of 240). A loss of roughly
  a point on hard reasoning is the expected size for 4.5-bit experts with
  everything sensitive left in BF16.
- **Weak spots to plan for:** it is verbose and thinks at length — the quantizer
  notes long agentic generations run even longer than BF16 — so budget
  `max_tokens` generously or turn thinking off for routine steps. It is a
  preview, and tool calling is good but not perfect (6/7 and 29/30 in the two
  independent batteries). On the hardest general-knowledge exams frontier models
  still lead.
- **Between our profiles the differences should be small**, since they run the
  same weights. The unmeasured risks are `sglang-nvfp4-ram-pennyroyal`'s static YaRN at short context
  and the uncalibrated FP8 KV cache in `sglang-nvfp4-nvme`, `sglang-nvfp4-ram` and `sglang-nvfp4-ram-pennyroyal`.
  That is what the HumanEval+ TODO in the README is meant to settle.

## Speed

Cold prefill: freshly generated text, prefix cache flushed first. Decode: warm
runs after a warmup.

| Launcher | Decode | Prefill 4K | Prefill 32K | Prefill 128K |
|---|---|---|---|---|
| `sglang-nvfp4-nvme` | 214–222 tok/s | 0.55 s · 7,314 tok/s | 4.37 s · 7,320 | 18.0 s · 7,076 |
| `sglang-nvfp4-ram` | 236–249 | 0.31 s · 12,911 | 2.90 s · 11,010 | 10.7 s · 11,892 |
| `sglang-nvfp4-ram-official` (cookbook defaults) | 248–263 | 0.30 s · 13,217 | 2.94 s · 10,828 | not run |
| `sglang-nvfp4-ram-official` | **258–260** | 0.30 s · 13,377 | 2.45 s · 12,997 | 10.4 s · 12,217 |
| `sglang-nvfp4-ram-pennyroyal` | 235–254 | **0.27 s** · 14,992 | 2.62 s · 12,174 | **9.68 s** · 13,171 |
| `sglang-nvfp4-ram-pennyroyal` + HiCache | 228–247² | 0.29 s · ~13,900 | **2.36 s** · 13,506 | 9.89 s · 12,888 |

² Measured with a different prompt and across context lengths; see below.

**`sglang-nvfp4-ram-pennyroyal` + HiCache, the configuration currently in use:**

| Context behind the request | TTFT (cold) | Decode |
|---|---|---|
| 36 tokens | 0.08 s | 228 tok/s |
| 7.3K | 0.52 s | 244–247 tok/s |
| 29K | 2.13 s | 231–243 tok/s |
| 116K | 8.7 s | 235–239 tok/s |
| 221K | 18.1 s | 235–236 tok/s |

| Prompt | Cold | From prefix cache | After a server restart (NIXL) |
|---|---|---|---|
| 57K | 4.34 s | 0.28 s | 0.53 s |
| 128K | 9.89 s | 0.45 s | — |
| 220K | 17.7 s | 1.01 s | **1.51 s** |
| 255K | 21.5 s | 0.83 s | — |
| 492K (`sglang-nvfp4-ram-pennyroyal` without HiCache) | 57.3 s | — | — |

The 57K and 220K rows come from the persistence test (needle prompts, one run
each); 128K and 255K from `bench/prefill.py`. Different prompts and runs, so the
cached times are not strictly comparable across rows — 255K from cache measured
faster than 220K for that reason, not because it is.

**Comments**

- **Decode differences between the RAM profiles are noise.** Each figure is three
  or four warm runs, and the ranges overlap. Do not pick a launcher on decode.
- **Prefill is where RAM beats NVMe clearly** — 1.5× to 2× cold. A prefill step
  gathers a PLE row for every prompt token, and the NVMe path reads each from the
  SSD with `O_DIRECT`, bypassing the page cache. Decode needs only one row per
  step, which is why its gain is small.
- **Cold prefill slows only gently with length:** about 12% from 4K to 255K in
  `sglang-nvfp4-ram-pennyroyal`, and still ~8,600 tok/s at 492K.
- **In practice the prefix cache decides waiting time.** An agent resends a long,
  mostly unchanged prefix every turn; with the cache a 220K prompt returns in about
  a second instead of 18. HiCache/NIXL extends that across restarts.
- **Concurrency is 4–5 requests at most in every long-context configuration**
  (the published cookbook cell, 2a, trades context for 16). These setups are for
  one agent, not many users.

## Alternatives considered but not run

| Checkpoint | Quantization | Size | Why not |
|---|---|---|---|
| Qwen/Qwen3.8-Flash-Next | BF16 original | 360 GB (model card) | Does not fit one 96 GB card |
| Qwen/Qwen3.8-Flash-Next-FP8 | FP8, near-lossless | ~125 GB of non-PLE weights (estimate) | Does not fit one 96 GB card |
| nvidia/…-NVFP4 (ModelOpt mixed) | NVFP4 experts, FP8 PLE, **FP8 MTP experts** | not downloaded | Smaller draft, so more KV (cookbook: ~170K vs ~78K for RadixArk at 16 requests); needs the loader in the official image. A candidate for a later test. |
| wtdcode/…-AWQ-W4A16 | **INT4 weight-only**, group 128, activations BF16; PLE in BF16 | 180.8 GB (PLE shard 102.4 GB) | No fast FP4 path on Blackwell, and the BF16 PLE table needs ~95 GB of RAM |
| unsloth GGUF (LM Studio, llama.cpp) | K-quants; largest that fits in VRAM ≈ UD-IQ3_XXS | 76.3 GiB (Q4_K_XL: 103.7 GiB) | ~3-bit to stay on the GPU; no NEXTN speculation or PLE offload, much smaller KV |


**Comments**

- **NVFP4 is the best quantization that runs entirely on this card.** Anything
  more precise does not fit in 96 GB.
- **AWQ W4A16 is not simply worse.** It leaves activations in BF16, while NVFP4
  quantizes activations to 4 bits too; NVFP4 compensates with much finer blocks
  (16 vs 128) and floating-point FP8 scales. Which loses less on this model is an
  open question, but AWQ would be slower here and much heavier on host RAM.
- **GGUF was a real option.** LM Studio's llama.cpp runtime 2.37.0 knows the
  architecture (`qwen4exp`). The quant that fits entirely in VRAM is about 3-bit,
  though, and llama.cpp offers neither the NEXTN speculation nor the long KV pool.

## Which one is highest quality?

**Not measured yet.** The needle tests confirm long-context retrieval works in every
launcher; they say nothing about reasoning or code. With identical weights,
quality can only differ through three settings:

| | KV cache | Mamba SSM state | RoPE |
|---|---|---|---|
| `sglang-nvfp4-nvme` | FP8, uncalibrated | **FP32** | native |
| `sglang-nvfp4-ram` | FP8, uncalibrated | BF16 | native |
| `sglang-nvfp4-ram-official` | **BF16** | BF16 | native |
| `sglang-nvfp4-ram-pennyroyal` | FP8, uncalibrated | BF16 | **YaRN ×2** |

**Comments**

- **`sglang-nvfp4-ram-official`** keeps the full-precision KV cache, so the 12 full-attention layers
  see exact keys and values.
- **`sglang-nvfp4-nvme`** keeps the recurrent state of the 36 linear-attention layers
  in FP32 — the most precise — but pays with uncalibrated FP8 KV.
- **`sglang-nvfp4-ram`** is one step lower than one of those on each axis.
- **`sglang-nvfp4-ram-pennyroyal`** has `sglang-nvfp4-ram`'s precisions plus static YaRN on every prompt. On
  paper it carries the most risk at short context — and it is the only launcher
  with a window beyond 262K.
- These effects are probably small, and which dominates cannot be reasoned out.
  The only quality figures that exist are the yepapa-nest author's for the NVMe
  `sglang-nvfp4-nvme` configuration: **HumanEval+ 0.939 / 0.921** non-thinking and
  0.957 / 0.927 thinking, **GSM8K 0.98**.
- Running HumanEval+ on the four launchers above is the open TODO in
  [README.md](../README.md#todo).
