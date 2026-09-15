# Results

Every profile in the toolkit, measured on the same machine — one RTX PRO 6000
Blackwell 96 GB, 244 GB RAM ([hardware](README.md#hardware)) — between 2026-09-12
and 2026-09-15. How the benchmarks work: [bench/README.md](bench/README.md).
Configuration, memory breakdown and full measurements per profile:
`<model>/docs/profiles/<profile>.md`.

Profile ids below drop the model prefix where the model is obvious from the
section; the start script is always `scripts/start-<model>-<profile>.sh`.

One setup is not a profile here: `ik_llama.cpp Q8_0`, the Q8_0 GGUF of
Qwen3.8-Flash-Next served by ik_llama.cpp from
[ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit), measured
through its API with the same benchmarks as the profiles (2026-09-15), as the
reference for 8-bit weights.

## Recommendations

| If you want | Use | Why |
|---|---|---|
| The most context and the fastest prefill | `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal` | 524K window, 831K-token KV pool, 235–254 tok/s; prefix cache survives restarts with HiCache. Depends on one person's fork and applies YaRN to every prompt. |
| Fast and conservative | `qwen3.8-flash-next-sglang-nvfp4-ram` (or `-official` for no local build) | 236–260 tok/s, 262K window, plain Docker |
| Little free host RAM | `qwen3.8-flash-next-sglang-nvfp4-nvme` | reads the PLE table from NVMe instead of holding ~65 GB in RAM; 214–222 tok/s, but the smallest KV pool and the slowest prefill of the SGLang profiles |
| The best non-English output at interactive speed | `qwen3.8-flash-next-vllm-awq-w4a16-g32` | first in three blind Slovak checks (seven and nine Flash-Next profiles, and against the dense 27B models), level with FP8 in the two runs with FP8, 3 points behind ik_llama.cpp Q8_0; ~110 tok/s, 262K window |
| The fastest single-stream decode with 16-bit activations | `qwen3.8-flash-next-exllamav3-exl3-5.05bpw` | 119 tok/s on prose, 216 on code; loads in under a minute. Prefill ~1.6× slower than vLLM, lower half in every Slovak check. |
| No refusals | `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` | leoncca's uncensored AWQ g32 with a small vLLM patch; 461 of 542 code tests, the best measured, no loss against the original. One unexplained crash under the first concurrent load. |
| — not this one | `qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated` | dealignai's abliterated NVFP4 lost 16 code tasks net against the same profile without abliteration (p = 0.02) |
| — superseded | `qwen3.8-flash-next-vllm-awq-w4a16` | INT4 group 128: the largest vLLM KV pool (605K), but in the bottom two of every Slovak check; group 32 is slightly faster (110 against 102 tok/s) |
| 8-bit weights (FP8 / Q8_0), the best Slovak measured | **ik_llama.cpp with a Q8_0 GGUF**, not a profile here | first of four in its blind Slovak check with the fewest weighted errors (73 against 70 for AWQ g32), but no better on code (454 of 542). 38 tok/s, 131K window, ~105 GiB host RAM. `vllm-fp8-offload` also keeps experts in RAM and decodes at ~16 tok/s. |
| A dense model | `qwen3.8-27b-sglang-bf16` or `qwen3.6-27b-sglang-bf16` | full BF16 precision, but both solved fewer code tasks than every measured Flash-Next quantization except the abliterated one, scored 21–27 points below AWQ g32 in the Slovak check, and decode slower (57–87 tok/s) |

**On quality in one line:** the seven original-weight Flash-Next profiles and
ik_llama.cpp Q8_0 are indistinguishable on code, and they and the uncensored AWQ g32
all solved more tasks than both dense 27B models at BF16; in Slovak the differences
between Flash-Next setups are small — ik_llama.cpp Q8_0 narrowly best, weight-only
INT4 group 32 the most consistent of the profiles — and the dense 27B models are
clearly behind.

## Speed and capacity

| Model | Profile | Runtime | Weights | KV cache, tokens | Context | TTFT 4K / 32K / 128K | Decode, tok/s | VRAM in use | Host RAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `sglang-nvfp4-nvme` | SGLang, local image | NVFP4 W4A4; PLE from NVMe | 231,936 FP8 | 262,144 | 0.55 / 4.37 / 18.0 s | 214–222 | 91.3 GiB | ~0 |
| | `sglang-nvfp4-ram` | SGLang, local image | NVFP4 W4A4 | 498,624 FP8 | 262,144 | 0.31 / 2.90 / 10.7 s | 236–249 | 92.1–92.9 GiB | ~65 GB |
| | `sglang-nvfp4-ram-official` | SGLang, official image | NVFP4 W4A4 | 256,832 BF16 | 262,144 | 0.30 / 2.45 / 10.4 s | 258–260 | not recorded | ~65 GB |
| | `sglang-nvfp4-ram-official-abliterated` | SGLang, official image | NVFP4 W4A4, abliterated | 256,832 BF16 | 262,144 | not measured | 200–216 code, 133 prose | 91.5 GiB | ~65 GB |
| | `sglang-nvfp4-ram-pennyroyal` | SGLang fork, native | NVFP4 W4A4 | **831,872 FP8** | **524,288** | **0.27 / 2.62 / 9.68 s** | 235–254 | 91.4–93.1 GiB | ~65 GB (+32 GB HiCache) |
| | `vllm-awq-w4a16` | vLLM preview image | INT4 W4A16 g128 | 605,187 BF16 | 262,144 | 0.36 / 3.28 / 12.1 s | 102–103 | 87.1–90.5 GiB | ~115 GB |
| | `vllm-awq-w4a16-g32` | vLLM preview image | INT4 W4A16 g32 | 315,039 BF16 | 262,144 | 0.78¹ / 3.36 / 12.1 s | 110 | not recorded | ~115 GB |
| | `vllm-awq-w4a16-g32-uncensored` | vLLM preview image + PLE patch | INT4 W4A16 g32, uncensored | 339,153 BF16 | 262,144 | not measured | 110 | 87.4 GiB | ~50 GB (est.) |
| | `exllamav3-exl3-5.05bpw` | ExLlamaV3 1.5.0 / TabbyAPI | EXL3 5.05 bpw | 262,144 FP16 | 262,144 | 0.68 / 4.83 / 19.4 s | **119 prose, 216 code** | 92.0–93.4 GiB | ~43 GB |
| | `vllm-fp8-offload` | vLLM preview image | FP8, 50 GiB of experts in RAM | 313,483 BF16 | 262,144 | 8K 11.9 s · 69K 86.3 s | 15.6–16.6 | ~88 GiB | ~131 GB |
| Qwen3.8-Flash-Next, not a profile | `ik_llama.cpp Q8_0` | ik_llama.cpp `d5f53d9f`, native | Q8_0 GGUF; routed experts of 17 of 48 layers in RAM | 131,072 Q8_0 | 131,072 | 2.20 / 17.4 / 86.6 s² | 38 | 92.2 GiB | ~105 GiB |
| Qwen3.6-27B | `sglang-bf16` | SGLang, official image | BF16 | not recorded | 262,144 | not measured | 87 code, 62 prose | 86.5 GiB | ~0 (no offload) |
| Qwen3.8-27B | `sglang-bf16` | SGLang, official image | BF16 | 295,344 BF16 | 262,144 | not measured | 85 code, 57 prose | 84.4 GiB | ~0 (no offload) |

¹ Prefix-cached; the first cold 4K request after startup took 1.83 s with warmup.
² The third size is 124K tokens, the largest prompt that fits its 131K window; cached
prompts return in 0.12–0.26 s. `llama-server -ngl 99 -ncmoe 17 -c 131072 -fa on
-ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 8 -tb 24 -thp --parallel 1`, one request at
a time, no speculative decoding.

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
official EvalPlus image without network (2026-09-14; the uncensored and abliterated
checkpoints and Qwen3.8-27B 2026-09-15). **Base** runs the original
tests, **plus** EvalPlus's stricter extended tests.

| Model | Profile | HumanEval | HumanEval+ | MBPP | MBPP+ | Plus tests passed, of 542 |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | `sglang-nvfp4-nvme` | 0.982 | 0.963 | 0.923 | 0.791 | 457 |
| | `sglang-nvfp4-ram` | 0.970 | 0.957 | 0.929 | 0.794 | 457 |
| | `sglang-nvfp4-ram-official` | 0.976 | 0.951 | 0.929 | 0.799 | 458 |
| | `sglang-nvfp4-ram-pennyroyal` | 0.963 | 0.951 | 0.918 | 0.794 | 456 |
| | `vllm-awq-w4a16` | 0.976 | 0.951 | 0.934 | 0.788 | 454 |
| | `vllm-awq-w4a16-g32` | 0.970 | 0.951 | **0.937** | 0.796 | 457 |
| | `exllamav3-exl3-5.05bpw` | 0.976 | 0.957 | **0.937** | **0.802** | **460** |
| | `vllm-fp8-offload` | — | — | — | — | not measured (1–2 h per run at 16 tok/s) |
| Qwen3.8-Flash-Next, uncensored | `vllm-awq-w4a16-g32-uncensored` | **0.988** | **0.970** | **0.942** | 0.799 | **461** |
| | `sglang-nvfp4-ram-official-abliterated` | 0.933 | 0.890 | 0.921 | 0.783 | 442 |
| Qwen3.8-Flash-Next, not a profile | `ik_llama.cpp Q8_0` | 0.970 | 0.951 | 0.926 | 0.788 | 454 |
| Qwen3.6-27B | `sglang-bf16` | 0.976 | 0.927 | 0.931 | 0.778 | 446 |
| Qwen3.8-27B | `sglang-bf16` | 0.970 | 0.933 | 0.910 | 0.780 | 448 |

**Comments**

- **All seven original-weight Flash-Next profiles are indistinguishable on code.** 439 of the 542
  plus tests were solved by every one of them and 69 by none; only 34 tasks vary.
  No pair differs significantly (task-by-task sign test, lowest p = 0.11 over all
  21 pairs). That covers the quantization (NVFP4 W4A4, INT4 g128 and g32, EXL3) and
  the runtime settings that differ between the SGLang profiles: FP8 or BF16 KV
  cache, FP32 or BF16 SSM state, and pennyroyal's YaRN ×2 on every prompt.
- **All seven original-weight Flash-Next profiles beat Qwen3.6-27B at full BF16 precision** by 10–14
  tasks: 20–23 tasks only Flash-Next solved against 9–15 only the 27B solved
  (p = 0.02 for EXL3, 0.06–0.10 for the SGLang profiles and g32, 0.26 for g128).
  Each comparison alone is at most borderline; seven of seven pointing the same way
  is the stronger evidence. The gap is in the plus tests; base pass rates are close.
- **The two refusal-removed checkpoints go opposite ways.**
  - `vllm-awq-w4a16-g32-uncensored` (orcarouter's abliteration, leoncca's AWQ g32)
    passed 461 plus tests, the most of any profile; against `vllm-awq-w4a16-g32` it
    solved 10 tasks the original missed and missed 6 it solved (p = 0.45). No
    measurable cost.
  - `sglang-nvfp4-ram-official-abliterated` (dealignai) passed 442, the fewest of any
    Flash-Next profile and below both dense 27B models. Against `sglang-nvfp4-ram-official` —
    same launcher, same quantization format, only the abliteration differs — it
    lost 30 tasks and gained 14 (p = 0.02). Its author's MMLU (−0.18 pp) did not
    catch this.
- **8-bit weights did not help on code.** `ik_llama.cpp Q8_0` passed 454, the same
  as INT4 g128 and within noise of every original-weight profile: against each it
  solved 5–9 tasks the other missed and missed 9–12 the other solved (lowest
  p = 0.21, against EXL3). All eight solved 437 tasks, none solved 68.
- **Qwen3.8-27B is no better than Qwen3.6-27B on these benchmarks**: 448 against 446,
  18 tasks only the newer model solved against 16 only the older one solved
  (p = 0.86). Against `vllm-awq-w4a16-g32-uncensored` it solved 11 tasks the MoE
  missed and missed 24 it solved (p = 0.04).
- **Compare within this table, not across sources.** On `sglang-nvfp4-nvme` this
  harness scores HumanEval 0.982 / 0.963, against 0.939 / 0.921 published for the
  same configuration with a different harness.
- Code only, thinking off, one greedy sample per task. Knowledge, long reasoning
  with thinking, and agentic tool use are not covered.

## Non-English: Slovak blind check

Ten Slovak prompts — an explanation, a formal email, noun inflection, numeral
agreement, idioms, a translation, a summary, a code explanation, a short story, a
grammar correction — answered with `bench/language_samples.py` (greedy, thinking
off). The answers were shuffled into a sheet labelled per prompt, and a separate LLM
grader that saw only the sheet listed every error with a quote, fix and severity,
scored each answer 1–10 and ranked the answers per prompt. Letters were mapped back
to profiles afterwards.

**Every Flash-Next profile except `vllm-fp8-offload`, one grader** (2026-09-15;
FP8 was graded only in the earliest four-way run, below):

| Profile | Weights | Score, sum of 10 | Error penalty | Mean rank¹ | Ranked first / last¹ |
|---|---|---|---|---|---|
| `vllm-awq-w4a16-g32` | INT4 W4A16 g32 | **75** | 36 | 4.00 | 0 / 0 |
| `sglang-nvfp4-ram-official` | NVFP4 W4A4, BF16 KV | 72 | 42 | 5.57 | 1 / 1 |
| `sglang-nvfp4-ram-official-abliterated` | NVFP4 W4A4, abliterated | 71 | **35** | 4.29 | 1 / 1 |
| `vllm-awq-w4a16-g32-uncensored` | INT4 W4A16 g32, uncensored | 70 | 42 | **3.71** | **4** / 0 |
| `sglang-nvfp4-ram` | NVFP4 W4A4 | 69 | 41 | 5.71 | 0 / 0 |
| `sglang-nvfp4-ram-pennyroyal` | NVFP4 W4A4, YaRN ×2 | 68 | 43 | 5.14 | 0 / 1 |
| `exllamav3-exl3-5.05bpw` | EXL3 5.05 bpw | 67 | 53 | 5.57 | 1 / 3 |
| `sglang-nvfp4-nvme` | NVFP4 W4A4, FP32 SSM | 66 | 60² | 5.14 | 0 / 1 |
| `vllm-awq-w4a16` | INT4 W4A16 g128 | 65 | 50 | 5.86 | 0 / 0 |

¹ Rank 1–9 over the 7 prompts where the answers differed in substance; numeral
agreement, the translation and the summary were near-identical across profiles.
² Includes a grammar-correction answer that rambled into its 800-token limit.

| Prompt | AWQ g32 | official | abliterated | g32 uncensored | ram | pennyroyal | EXL3 | nvme | AWQ g128 |
|---|---|---|---|---|---|---|---|---|---|
| explain | 9 | 6 | 9 | 9 | 7 | 8 | 9 | 9 | 7 |
| formal-email | 8 | 8 | 5 | 9 | 6 | 5 | 8 | 6 | 9 |
| inflection | 7 | **10** | 6 | 4 | 4 | 4 | 3 | 6 | 4 |
| numbers-agreement | 10 | 10 | 8 | 10 | 10 | 10 | 10 | 10 | 10 |
| idioms | 5 | 3 | 6 | 4 | 4 | 5 | 6 | 2 | 4 |
| translate | 8 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 8 |
| summary | 9 | 9 | 9 | 7 | 9 | 9 | 8 | 9 | 9 |
| code-explain | 9 | 8 | 9 | 9 | 9 | 9 | 6 | 9 | 7 |
| story | 6 | 5 | 4 | 8 | 7 | 5 | 7 | 4 | 4 |
| grammar-fix | 4 | 4 | 6 | 2 | 5 | 5 | 2 | 2 | 3 |

**Earlier runs of the same method** graded the same answers again with fresh
graders: seven profiles on 2026-09-14 (`vllm-awq-w4a16-g32` 68, `sglang-nvfp4-ram-official`
68, pennyroyal 65, ram 64, EXL3 64, g128 63, nvme 61) and four with the FP8
reference (`vllm-fp8-offload` 71, g32 70, g128 62, EXL3 61). Graders differ in how
strict they are — the nine-way grader scored every profile 3–7 points higher — but
the order held: g32 first and official second in both multi-profile runs, g128,
nvme and EXL3 at the bottom. Compare scores within one run only.

**Dense Qwen3.6-27B and Qwen3.8-27B at BF16 against two Flash-Next profiles, one
grader** (2026-09-15):

| Profile | Weights | Score, sum of 10 | Error penalty | Mean rank, 1–4 | Ranked first / last |
|---|---|---|---|---|---|
| `qwen3.8-flash-next-vllm-awq-w4a16-g32` | INT4 W4A16 g32 | **77** | **42** | **1.9** | **4** / 0 |
| `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` | INT4 W4A16 g32, uncensored | 70 | 58 | 2.3 | 2 / 2 |
| `qwen3.8-27b-sglang-bf16` | BF16 | 56 | 88 | 2.8 | 2 / 4 |
| `qwen3.6-27b-sglang-bf16` | BF16 | 50 | 86 | 3.0 | 2 / 4 |

| Prompt | Flash-Next AWQ g32 | Flash-Next g32 uncensored | Qwen3.8-27B | Qwen3.6-27B |
|---|---|---|---|---|
| explain | 9 | 9 | 8 | 9 |
| formal-email | 8 | 9 | 3 | 6 |
| inflection | 7 | 4 | 2 | 1 |
| numbers-agreement | 10 | 10 | 10 | 1 |
| idioms | 6 | 5 | 2 | 3 |
| translate | 8 | 8 | 9 | 6 |
| summary | 9 | 6 | 7 | 7 |
| code-explain | 9 | 9 | 7 | 5 |
| story | 8 | 8 | 4 | 5 |
| grammar-fix | 3 | 2 | 4 | 7 |

The two Flash-Next answers are the same greedy outputs as in the nine-way run and
scored 77 and 70 here against 75 and 70 in the nine-way run — this grader was about
as strict.

**ik_llama.cpp Q8_0 against FP8 and both AWQ g32 checkpoints, one grader**
(2026-09-15):

| Setup | Weights | Score, sum of 10 | Error penalty | Mean rank, 1–4¹ | Ranked first / last¹ |
|---|---|---|---|---|---|
| `ik_llama.cpp Q8_0` | Q8_0 GGUF, Q8_0 KV | **73** | **36** | **2.00** | 2 / 1 |
| `qwen3.8-flash-next-vllm-awq-w4a16-g32` | INT4 W4A16 g32 | 70 | 47 | 2.62 | 1 / 0 |
| `qwen3.8-flash-next-vllm-fp8-offload` | FP8 | 69 | 53 | 2.88 | 2 / 4 |
| `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` | INT4 W4A16 g32, uncensored | 68 | 50 | 2.50 | **3** / 3 |

¹ Over the 8 prompts where the answers differed; all four numeral-agreement and
translation answers were identical.

| Prompt | ik_llama.cpp Q8_0 | AWQ g32 | FP8 | g32 uncensored |
|---|---|---|---|---|
| explain | 9 | 9 | 8 | 9 |
| formal-email | 8 | 7 | 6 | 8 |
| inflection | 4 | 6 | **9** | 4 |
| numbers-agreement | 10 | 10 | 10 | 10 |
| idioms | 5 | 4 | 6 | 3 |
| translate | 8 | 8 | 8 | 8 |
| summary | 8 | 9 | 8 | 7 |
| code-explain | 9 | 8 | 7 | 9 |
| story | **8** | 6 | 4 | **8** |
| grammar-fix | 4 | 3 | 3 | 2 |

The three vLLM answers are the same outputs as in the earlier runs: g32 70 and FP8 69
here against 70 and 71 in the first four-way run, a grader of similar strictness.

**Comments**

- **AWQ group 32 came first in every run of the profiles here but one**, and in that
  one matched FP8 (70 against 71); only ik_llama.cpp Q8_0 scored above it (73 against 70).
  Among the profiles here it is the most consistent choice for Slovak, not a clear
  winner: the spread is a few prompts' worth, with one greedy
  answer per prompt.
- **Removing refusals did not clearly hurt Slovak.** The abliterated NVFP4 checkpoint
  made the fewest weighted errors in the nine-way run and was the only profile that
  avoided the non-word „vereta“; the uncensored AWQ g32 checkpoint had the best mean
  rank and the most first places there, though it scored 5–7 points below the
  original g32 in both runs that graded the two side by side. On code the two
  checkpoints differ sharply (see above) — the Slovak check is too small to show that.
- **ik_llama.cpp Q8_0 wrote the best Slovak of its run, narrowly.** It led AWQ g32
  by 3 points with the fewest weighted errors (36 against 47–53) and the best mean
  rank; it was first or second on every differing prompt except inflection, where it
  was last. It is not a different class: its grammar correction also opened with the
  non-word „vereta“ and missed „jablká“, its idioms answer had „previn“ and
  „Hádzat“, and it got two of the five cases wrong where FP8 got all five right. FP8, also 8-bit, placed third —
  bit width alone does not explain the lead.
- **The dense 27B models write clearly worse Slovak than Flash-Next, despite full
  BF16 precision**: 14–27 points below the two AWQ g32 checkpoints, 1.5–2× their
  weighted errors, each last on four of ten prompts. Qwen3.8-27B scored 6 points
  above Qwen3.6-27B with about as many weighted errors — no clear step between the
  two. Both corrected „tri jablka“ → „jablká“, which every Flash-Next
  profile missed, but otherwise make the errors Flash-Next makes, only more often: Czech words
  („Omluvy“, „predem“, „srozumiteľnejšou“, „Kľudne“), gender slips („veľká pstruh“,
  „opotrebovanú prút“), non-words („nadmierať“, „odraza“, „vysielá“), invented
  grammar rules in the correction task, and both glossed „mať maslo na hlave“ as
  “stupid” or “naive”. Qwen3.6-27B alone wrote digits with plural verbs
  („sú 5 žien“). The engine does not explain the gap: `sglang-nvfp4-ram-official`
  runs on the same SGLang image and placed second of nine.
- **NVFP4 is not clearly worse at this sample size.** `sglang-nvfp4-ram-official`
  placed second again, with the only fully correct inflection answer in both runs.
- **EXL3 5.05 bpw placed in the lower half in every run**, despite the lowest
  published KL divergence on English and code. Unlike the vLLM checkpoints it also
  quantizes attention, linear attention and the shared experts.
- **The worst Flash-Next errors are the model's, not the quantization's.** Every
  Flash-Next profile missed „tri jablka“ → „jablká“, and all but one wrote „vereta“. Recurring across the sheet: missing
  vocalized prepositions („v fáze“, „z štandardnej“), missing reflexive „sa“
  („ospravedlniť Vás“, „sťažujú mestu“), Czech forms („v Pythonu“, „Házať“), gender
  agreement slips („Vašu pochopenie“, „pstruha“, „prázdny vedro“), and non-words
  („chybujúca“, „pevnom“, „vyjavela“).
- An LLM grader is not a native speaker; individual calls can be wrong. More
  prompts, several samples per prompt, or a human read of the same sheet would
  firm this up.

## Not measured yet

- **Code benchmarks** on `qwen3.8-flash-next-vllm-fp8-offload` (1–2 h per run at
  its speed).
- **Slovak check** with more prompts or several samples per prompt, to separate the
  Flash-Next profiles.
- **Dense 27B models**: prefill and long-context retrieval; the KV pool of
  Qwen3.6-27B.
- Anything with thinking on, knowledge benchmarks, agentic tool use.
