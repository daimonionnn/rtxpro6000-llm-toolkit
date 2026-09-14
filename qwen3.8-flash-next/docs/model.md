# Qwen3.8-Flash-Next: architecture, checkpoints, quantization

What the model is, what the checkpoints used by this toolkit's profiles keep at
which precision, which runtime settings can change output, and what has been
published about its quality. Measured speed and quality of each profile are in
[RESULTS.md](../../RESULTS.md); per-profile details in [profiles/](profiles/).

## Architecture

Read from the checkpoint's `config.json`.

| | |
|---|---|
| Model | Qwen3.8-Flash-Next (`qwen4_exp`), **180B parameters, MoE with ~6B active** |
| Layers | 48: **36 linear attention** (GDN) + **12 full attention** (every 4th layer) |
| Full attention | QSA sparse attention; the indexer selects 2,048 tokens (compress ratio 4) |
| Attention heads | 24 query, 2 KV, head_dim 256; hidden size 2,560 |
| Linear attention heads | 16 key, 48 value |
| MoE | **512 routed experts, 10 active** per token, plus 1 shared expert; expert FFN 640 |
| PLE | N-gram embedding table (3-grams) at layer 2 — **51.2B parameters** |
| Other | hyper-connections ×4, 1 MTP layer (drives NEXTN / MTP speculation), vision encoder |
| Native context | 262,144 tokens; vocabulary 248,320 |

**Comments**

- **Most of the parameter count is not compute.** Of 180B parameters, 51B are the
  PLE lookup table and 121B are 512 experts of which only 10 run per token. That is
  why a 180B model decodes at the speed of a small one, and why the PLE table can
  leave the GPU at all: it is a lookup, not matrix math.
- **Decode speed barely depends on context length.** Three layers in four carry a
  fixed-size recurrent state instead of a growing KV cache, and the other quarter
  attend to a selected 2,048 tokens. Measured: 228–247 tok/s from an empty prompt
  to 221K tokens on `sglang-nvfp4-ram-pennyroyal`.
- **The KV cache is small per token** — only 12 layers keep one, with 2 KV heads
  each. Hundreds of thousands of tokens fit in a few GB.

## Checkpoints and where their bits go

Parameter counts and sizes read from the safetensors headers by
`qwen3.8-flash-next/quant_info.py`, which every launcher runs at start.

| Checkpoint | Routed experts (120.8B) | Attention, router, shared expert, embeddings, MTP, vision (8.0B) | PLE table (51.2B) | Size | Profiles |
|---|---|---|---|---|---|
| RadixArk NVFP4 | **NVFP4 W4A4**, group 16, FP8 scales — 4.50 bits, 63.3 GiB | BF16, 14.9 GiB | FP8, 47.7 GiB | 125.9 GiB | the four `sglang-nvfp4-*` |
| wtdcode AWQ W4A16 | INT4 weight-only, group 128, symmetric — 4.13 bits, 58.0 GiB | BF16, 14.9 GiB | BF16, 95.4 GiB | 168.3 GiB | `vllm-awq-w4a16` |
| cyankiwi AWQ INT4 | INT4 weight-only, **group 32, asymmetric** — 4.63 bits, 65.0 GiB | BF16, 14.9 GiB | BF16, 95.4 GiB | 175.3 GiB | `vllm-awq-w4a16-g32` |
| turboderp EXL3 5.05 bpw | EXL3 5.05 bpw, 70.8 GiB | **EXL3** (head 6 bpw) + BF16 norms, 7.5 GiB | EXL3 6 bpw, 36.4 GiB | 114.6 GiB | `exllamav3-exl3-5.05bpw` |
| Qwen FP8 | FP8 W8A8, 128×128 blocks — 8.00 bits, 112.5 GiB | BF16 (MTP experts FP8), 12.6 GiB | FP8, 47.7 GiB | 172.8 GiB | `vllm-fp8-offload` |
| *Qwen BF16 original (not used)* | 16 | 16 | 16 | ~335 GiB | — |

In the NVFP4 checkpoint "4 bits" means one FP4 value per weight plus one FP8 E4M3
scale per 16 values and an FP32 global scale; the activations entering the experts
are quantized to 4 bits too. The AWQ and EXL3 checkpoints quantize weights only.

**Published KL divergence against BF16**, from the EXL3 quantizer's chart (the same
in-domain English and code trace for every format): EXL3 5.05 bpw **0.0040**, EXL3
4.05 bpw 0.0067, NVFP4 W4A16 0.0100, NVFP4 W4A4 0.0241. The AWQ checkpoints were
not on it.

**Comments**

- **Every checkpoint except EXL3 keeps the sensitive tensors in BF16** — attention,
  linear attention, the router, shared experts, embeddings — and compresses only
  the routed experts, which individually see few tokens.
- **Activations matter.** W4A4 more than doubles KL divergence over W4A16 with the
  same weights. On code this did not show (see [RESULTS.md](../../RESULTS.md#code-humaneval-and-mbpp));
  in the Slovak check the weight-only group-32 checkpoint held up best.
- **8-bit weights do not fit one card.** The FP8 experts alone are 112.5 GiB, so any
  8-bit profile keeps part of them in host RAM and is bound by PCIe
  ([profiles/vllm-fp8-offload.md](profiles/vllm-fp8-offload.md)).
- **GGUF that fits entirely in 96 GB VRAM is ~3.6–4.2 bits on average** (unsloth
  UD-IQ3_XXS 76.3 GiB to UD-IQ4_XS 87.2 GiB) and quantizes the sensitive tensors too.

## Runtime settings that can change output

Beyond the checkpoint, three settings differ between profiles:

| Profile | KV cache | Linear-attention (SSM) state | RoPE |
|---|---|---|---|
| `sglang-nvfp4-nvme` | FP8, uncalibrated | **FP32** (model default) | native |
| `sglang-nvfp4-ram` | FP8, uncalibrated | BF16 | native |
| `sglang-nvfp4-ram-official` | **BF16** | BF16 | native |
| `sglang-nvfp4-ram-pennyroyal` | FP8, uncalibrated | BF16 | **YaRN ×2 on every prompt** |
| `vllm-awq-w4a16`, `vllm-awq-w4a16-g32`, `vllm-fp8-offload` | BF16 | model default | native |
| `exllamav3-exl3-5.05bpw` | FP16 | engine default | native |

**Comments**

- **FP32 SSM state in `sglang-nvfp4-nvme` is the model's own default**
  (`mamba_ssm_dtype: float32`). The SGLang RAM profiles set `bfloat16` because
  halving that state is part of how they fit: 110 MB per state slot against 55 MB.
- **FP8 KV here is uncalibrated.** The checkpoint ships no KV scales, and the log
  warns `Defaulting to scaling factors of 1.0`. Upstream PR #36644 would add
  per-layer descale.
- **FP8 KV crashes the official image** on long prompts (`Unsupported rhs dtype
  fp8e4nv`), so `sglang-nvfp4-ram-official` runs BF16 KV; the local image carries a
  patch for it.
- **YaRN in `sglang-nvfp4-ram-pennyroyal` is static.** `--json-model-override-args`
  sets factor 2 for every request, so short prompts get rescaled positions too —
  the price of its 524K window.
- **Speculative decoding does not change output.** Verification against the target
  model is exact; the draft's precision affects speed only.
- Of these, only `sglang-nvfp4-nvme` has been benchmarked on code; given that NVFP4
  already matches the 16-bit-activation formats there, the rest are unlikely to
  matter much for code, but remain unmeasured.

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

### From the quantizer, on the NVFP4 checkpoint

RadixArk's `qualification-notes.md` and metric files ship inside
`models/Qwen3.8-Flash-Next-NVFP4/`:

| Eval | Protocol | BF16 reference | NVFP4 |
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

The AWQ group-32 checkpoint publishes no quality numbers; Intel's AutoRound W4A16
(group 128, not used here) reports MMLU 85.60 against 86.51 for BF16.

### Independent, on local hardware

| Source | Setup | Result |
|---|---|---|
| [yepapa-nest recipe](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000) — the `sglang-nvfp4-nvme` configuration, RTX PRO 6000 | HumanEval+, greedy | **0.939 / 0.921** non-thinking, 0.957 / 0.927 thinking; GSM8K 0.98; tools 6/7 |
| [jpezzulli fork](https://github.com/jpezzulli/sglang-rtxpro6000) — `sglang-nvfp4-ram-pennyroyal`, RTX PRO 6000 | own reasoning, tool, vision and agent suite | reasoning **98.52/100**; tool calls 29/30 exact; vision and a six-turn agent task passed |
| [flash-next-rtxpro6000-bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench) — vLLM, EXL3 and SGLang, RTX PRO 6000 | tool-eval-bench, 88 scenarios × 8 | vLLM NVFP4/FP8 87.9, EXL3 5.05 bpw 86.9, 4.05 bpw 87.4, SGLang NVFP4 88.0–88.9; Qwen3.8-27B NVFP4 91.0, BF16 89.0 |
| [E2Studio local LLM benchmark](https://wonderrico.github.io/local_llm_benchmark/benchmark-main.html?filter=3.8) | first 100 Django tasks of SWE-bench Verified, mini-swe-agent | **98/100** with AWQ W4A16 + INT4 PLE on vLLM — top of that board; Qwen3.8-27B 74–81 |
| [eesel AI review](https://www.eesel.ai/blog/qwen38-flash-next-review), citing Artificial Analysis | hosted API | Intelligence Index 56, #5 of 111 in its class; flagged **very verbose** |

Also reported: Humanity's Last Exam 35.9 against Claude Opus 4.6 Max's 40.0
([review summary](https://www.eesel.ai/blog/qwen38-flash-next-review), not verified
against a primary source). Qwen describes the model as an intentionally
under-trained preview of the Qwen4 architecture.

## Opinion: how good is it?

**Very good for its cost; the quantizations here give up little on code and
somewhat more on less common languages.**

- **Strongest where this toolkit is aimed:** agentic coding. It leads its vendor
  table on SWE-bench Pro, SWE-bench Multilingual and LiveCodeBench, and on this
  machine every quantization solved more HumanEval+/MBPP+ tasks than Qwen3.6-27B at
  full precision.
- **Quantization shows in non-English output first.** Code scores do not separate
  NVFP4, AWQ and EXL3; the Slovak check does, and even FP8 keeps the model's own
  grammar slips.
- **Weak spots to plan for:** it is verbose and thinks at length — budget
  `max_tokens` generously or turn thinking off for routine steps. It is a preview;
  tool calling is good but trails Qwen3.8-27B in independent runs. On the hardest
  general-knowledge exams frontier models still lead.

## Alternatives considered

| Checkpoint | Quantization | Size | Status |
|---|---|---|---|
| Qwen/Qwen3.8-Flash-Next | BF16 original | ~335 GiB | Not run: does not fit one 96 GB card |
| Qwen/Qwen3.8-Flash-Next-FP8 | FP8 W8A8, routed experts only | 172.8 GiB | **Run** as `vllm-fp8-offload`; for 8-bit weights on one card use ik_llama.cpp instead |
| wtdcode/…-AWQ-W4A16 | INT4 weight-only, group 128 | 168.3 GiB | **Run** as `vllm-awq-w4a16` |
| cyankiwi/…-AWQ-INT4 | INT4 weight-only, group 32 asymmetric, multilingual calibration | 175.3 GiB | **Run** as `vllm-awq-w4a16-g32` |
| turboderp/…-exl3 5.05 bpw | EXL3 trellis, all linear layers | 114.6 GiB | **Run** as `exllamav3-exl3-5.05bpw` |
| nvidia/…-NVFP4 (ModelOpt mixed) | NVFP4 experts, FP8 PLE, FP8 MTP experts | not downloaded | Smaller draft, so more KV (cookbook: ~170K vs ~78K at 16 requests); still W4A4 experts |
| Intel/…-W4A16-AutoRound | INT4, group 128 | ~181 GB | Not run: same group size as `vllm-awq-w4a16` |
| unsloth GGUF (llama.cpp) | K-quants; largest that fits in VRAM ≈ UD-IQ4_XS | 87.2 GiB (Q8_0 175 GiB) | Not run here: ~4-bit to stay on the GPU; Q8_0 with experts in RAM runs under ik_llama.cpp in a separate toolkit |
