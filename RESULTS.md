# Results

Every profile in the toolkit, measured on the same machine — one RTX PRO 6000
Blackwell 96 GB, 244 GB RAM ([hardware](README.md#hardware)) — between 2026-09-12
and 2026-09-14. How the benchmarks work: [bench/README.md](bench/README.md).
Configuration, memory breakdown and full measurements per profile:
`<model>/docs/profiles/<profile>.md`.

Profile ids below drop the model prefix where the model is obvious from the
section; the start script is always `scripts/start-<model>-<profile>.sh`.

## Recommendations

| If you want | Use | Why |
|---|---|---|
| The most context and the fastest prefill | `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal` | 524K window, 831K-token KV pool, 235–254 tok/s; prefix cache survives restarts with HiCache. Depends on one person's fork and applies YaRN to every prompt. |
| Fast and conservative | `qwen3.8-flash-next-sglang-nvfp4-ram` (or `-official` for no local build) | 236–260 tok/s, 262K window, plain Docker |
| The best non-English output at interactive speed | `qwen3.8-flash-next-vllm-awq-w4a16-g32` | level with FP8 in a blind Slovak check, ~110 tok/s, 262K window |
| The fastest single-stream decode with 16-bit activations | `qwen3.8-flash-next-exllamav3-exl3-5.05bpw` | 119 tok/s on prose, 216 on code; loads in under a minute. Prefill ~1.6× slower than vLLM, last in the Slovak check. |
| 8-bit weights (FP8 / Q8_0) | **ik_llama.cpp**, not a profile here | `vllm-fp8-offload` must keep experts in RAM and decodes at ~16 tok/s; a Q8_0 GGUF under ik_llama.cpp on this machine reaches ~36–40 tok/s |
| A dense model | `qwen3.6-27b-sglang-bf16` | full BF16 precision, but it solved fewer code tasks than every Flash-Next quantization and decodes slower (62–87 tok/s) |

**On quality in one line:** the Flash-Next quantizations are indistinguishable on
code and all beat Qwen3.6-27B at BF16 there; they differ in non-English output,
where weight-only INT4 with group 32 held up best.

## Speed and capacity

| Model | Profile | Runtime | Weights | KV cache, tokens | Context | TTFT 4K / 32K / 128K | Decode, tok/s | VRAM in use | Host RAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `sglang-nvfp4-nvme` | SGLang, local image | NVFP4 W4A4; PLE from NVMe | 231,936 FP8 | 262,144 | 0.55 / 4.37 / 18.0 s | 214–222 | 91.3 GiB | ~0 |
| | `sglang-nvfp4-ram` | SGLang, local image | NVFP4 W4A4 | 498,624 FP8 | 262,144 | 0.31 / 2.90 / 10.7 s | 236–249 | 92.1–92.9 GiB | ~65 GB |
| | `sglang-nvfp4-ram-official` | SGLang, official image | NVFP4 W4A4 | 256,832 BF16 | 262,144 | 0.30 / 2.45 / 10.4 s | 258–260 | not recorded | ~65 GB |
| | `sglang-nvfp4-ram-pennyroyal` | SGLang fork, native | NVFP4 W4A4 | **831,872 FP8** | **524,288** | **0.27 / 2.62 / 9.68 s** | 235–254 | 91.4–93.1 GiB | ~65 GB (+32 GB HiCache) |
| | `vllm-awq-w4a16` | vLLM preview image | INT4 W4A16 g128 | 605,187 BF16 | 262,144 | 0.36 / 3.28 / 12.1 s | 102–103 | 87.1–90.5 GiB | ~115 GB |
| | `vllm-awq-w4a16-g32` | vLLM preview image | INT4 W4A16 g32 | 315,039 BF16 | 262,144 | 0.78¹ / 3.36 / 12.1 s | 110 | not recorded | ~115 GB |
| | `exllamav3-exl3-5.05bpw` | ExLlamaV3 1.5.0 / TabbyAPI | EXL3 5.05 bpw | 262,144 FP16 | 262,144 | 0.68 / 4.83 / 19.4 s | **119 prose, 216 code** | 92.0–93.4 GiB | ~43 GB |
| | `vllm-fp8-offload` | vLLM preview image | FP8, 50 GiB of experts in RAM | 313,483 BF16 | 262,144 | 8K 11.9 s · 69K 86.3 s | 15.6–16.6 | ~88 GiB | ~131 GB |
| Qwen3.6-27B | `sglang-bf16` | SGLang, official image | BF16 | not recorded | 262,144 | not measured | 87 code, 62 prose | 86.5 GiB | ~0 (no offload) |

¹ Prefix-cached; the first cold 4K request after startup took 1.83 s with warmup.

- **Decode barely falls with context on Flash-Next**: three of four layers keep a
  fixed-size recurrent state and the rest attend to 2,048 selected tokens —
  228–247 tok/s from empty to 221K on `sglang-nvfp4-ram-pennyroyal`, 108 tok/s at
  195K on EXL3.
- **Decode differences between the SGLang RAM profiles are noise**; context, KV
  pool and prefill are solid.
- **TTFT in practice is the prefix cache**: an agent resends a long, mostly
  unchanged prefix, and a cached 128K prompt returns in 0.4–0.75 s on every
  profile except `vllm-fp8-offload` (10 s at 69K).
- **Speculative decoding** (NEXTN / MTP) is on in the SGLang and ExLlamaV3 profiles
  and off in the vLLM ones; its gain depends on how predictable the text is, which
  is why code decodes faster than prose where it is on.

## Code: HumanEval+ and MBPP+

164 + 378 tasks, greedy, thinking off, the EvalPlus prompts, scored in the
official EvalPlus image without network (2026-09-14). **Base** runs the original
tests, **plus** EvalPlus's stricter extended tests.

| Model | Profile | HumanEval | HumanEval+ | MBPP | MBPP+ | Plus tests passed, of 542 |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `sglang-nvfp4-nvme` | **0.982** | **0.963** | 0.923 | 0.791 | 457 |
| | `vllm-awq-w4a16` | 0.976 | 0.951 | 0.934 | 0.788 | 454 |
| | `vllm-awq-w4a16-g32` | 0.970 | 0.951 | **0.937** | 0.796 | 457 |
| | `exllamav3-exl3-5.05bpw` | 0.976 | 0.957 | **0.937** | **0.802** | **460** |
| Qwen3.6-27B | `sglang-bf16` | 0.976 | 0.927 | 0.931 | 0.778 | 446 |

**Comments**

- **The four Flash-Next quantizations are indistinguishable.** Task by task,
  `vllm-awq-w4a16-g32` and each of the others disagree on 11–16 of 542 tasks, split
  about evenly (sign test p = 0.55–1.0). NVFP4's 4-bit activations did not
  measurably lose to the 16-bit-activation formats.
- **Every Flash-Next quantization beat Qwen3.6-27B at full BF16 precision** by 8–14
  tasks: 22–23 tasks only Flash-Next solved against 9–15 only the 27B solved
  (p = 0.02 for EXL3, 0.08–0.26 for the others). The gap is in the plus tests;
  base pass rates are close.
- **Compare within this table, not across sources.** On `sglang-nvfp4-nvme` this
  harness scores HumanEval 0.982 / 0.963, against 0.939 / 0.921 published for the
  same configuration with a different harness.
- Code only, thinking off, one greedy sample per task. Knowledge, long reasoning
  with thinking, and agentic tool use are not covered.

## Non-English: Slovak blind check

Ten Slovak prompts — an explanation, a formal email, noun inflection, numeral
agreement, idioms, a translation, a summary, a code explanation, a short story, a
grammar correction — answered by four Flash-Next profiles with
`bench/language_samples.py` (greedy, thinking off). The answers were shuffled into
a sheet labelled A–D per prompt, and a separate LLM grader that saw only the sheet
listed every error with a quote, fix and severity, scored each answer 1–10 and
ranked the four per prompt. Letters were mapped back to profiles afterwards.

| Profile | Weights | Score, sum of 10 | Error penalty | Mean rank¹ | Ranked first¹ |
|---|---|---|---|---|---|
| `vllm-fp8-offload` | FP8 experts | **71** | 48 | 2.25 | **4** |
| `vllm-awq-w4a16-g32` | INT4 group 32 | **70** | **46** | **2.00** | 2 |
| `vllm-awq-w4a16` | INT4 group 128 | 62 | 63 | 2.50 | 2 |
| `exllamav3-exl3-5.05bpw` | EXL3 5.05 bpw | 61 | 58 | 3.25 | 0 |

¹ Over the 8 prompts whose answers differed; numeral agreement and the translation
were identical in all four.

| Prompt | FP8 | AWQ g32 | AWQ g128 | EXL3 |
|---|---|---|---|---|
| explain | 9 | 8 | 6 | 8 |
| formal-email | 6 | 7 | 8 | 7 |
| inflection | 9 | 6 | 3 | 2 |
| numbers-agreement | 10 | 10 | 10 | 10 |
| idioms | 6 | 4 | 3 | 5 |
| translate | 8 | 8 | 8 | 8 |
| summary | 8 | 9 | 8 | 7 |
| code-explain | 7 | 8 | 8 | 6 |
| story | 4 | 7 | 5 | 6 |
| grammar-fix | 4 | 3 | 3 | 2 |

**Comments**

- **AWQ group 32 matched FP8; group 128 and EXL3 5.05 bpw trailed.** The gap comes
  mostly from three prompts (inflection, idioms, story): a direction, not a
  measurement — one greedy answer per prompt, where one early token can change the
  rest of an answer.
- **EXL3's place contradicts its KL divergence**, the best of these formats on
  in-domain English text. Unlike the vLLM checkpoints it also quantizes attention,
  linear attention and the shared experts; whether that matters more for Slovak was
  not tested.
- **The worst errors are the model's own.** All four, FP8 included, wrote the
  non-word „vereta“, left „tri jablka“ uncorrected (should be „jablká“) and called
  „čo“ a conjunction. Recurring across the sheet: missing vocalized prepositions
  („v fáze“, „z štandardnej“), Czech forms („v Pythonu“, „plácl“, „marné“), dropped
  reflexive „sa“, non-words („previn“, „pstružina“).
- An LLM grader is not a native speaker; individual calls can be wrong. A larger
  prompt set, several samples per prompt or a human read of the same sheet would
  firm this up.

## Not measured yet

- **Code benchmarks** on `qwen3.8-flash-next-sglang-nvfp4-ram`, `-official`,
  `-pennyroyal` (they differ from `sglang-nvfp4-nvme` only in KV / SSM precision and,
  for pennyroyal, YaRN) and `vllm-fp8-offload` (~1 h per run at its speed).
- **Slovak check** on the SGLang NVFP4 profiles and Qwen3.6-27B.
- **Qwen3.6-27B**: prefill, KV pool, long-context retrieval.
- Anything with thinking on, knowledge benchmarks, agentic tool use.
