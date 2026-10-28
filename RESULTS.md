# Results

An overview of every model in the toolkit on the same machine: one RTX PRO 6000
Blackwell 96 GB and 256 GB of RAM ([hardware](README.md#hardware)). Each row links to
the details; this page only compares. How the benchmarks work: [bench/README.md](bench/README.md).

| Model | Details |
|---|---|
| Qwen3.8-Flash-Next | [qwen3.8-flash-next/docs/results.md](qwen3.8-flash-next/docs/results.md) (all profiles: speed, memory, code, Slovak), [Strata](qwen3.8-flash-next/strata/README.md) |
| DeepSeek-V4.1-Flash-1017 | [deepseek-v4.1-flash/README.md](deepseek-v4.1-flash/README.md) |
| Qwen3.6-27B, Qwen3.8-27B (dense references) | [qwen3.6-27b](qwen3.6-27b/README.md), [qwen3.8-27b](qwen3.8-27b/README.md) |

Single stream, thinking off, greedy where quality is measured. Results from different
dates used different engine versions and RAM speeds (DDR5-6667 until 2026-10-25, 6400
since); compare within a table.

## Recommendations

| If you want | Use | Why |
|---|---|---|
| The default: near-full quality at high speed | [`qwen3.8-flash-next-strata-q8-vision`](qwen3.8-flash-next/strata/README.md#upgrade-to-027-speed-ab-2026-10-27) | Q8_0 on Strata 0.2.7 with MTP: 159 prose / 176 code / 122 Slovak tok/s, ~6,700 tok/s cold prefill, image input, 128K (256K profile); GSM8K 97.0 %, MMLU-Pro 78.5 %, 457 of 542 code tests. ~37 GiB of pinned RAM. |
| The best answers, at a lower speed | [`deepseek-v4.1-flash-strata-ud-q8kxl`](deepseek-v4.1-flash/README.md#strata-ud-q8_k_xl) | DeepSeek's own MXFP4 experts: the best Slovak measured, GSM8K 98.3 %, MMLU-Pro 81.0 %; ~29 tok/s, ~1,000 tok/s prefill at 30K, up to 512K, images. Needs ~224 GB of RAM. For coding sessions and complex tasks, not for an agent making many quick calls. |
| DeepSeek, faster and smaller | [`deepseek-v4.1-flash-strata-ist-iq2xs`](deepseek-v4.1-flash/README.md#ista-daslab-iq2_xs-225-bpw) | ISTA-DASLab's official 2.25 bpw set: 40–59 tok/s, 2,118 tok/s prefill at 30K, Slovak 84, up to 256K, images; ~140 GB of RAM. |
| The fastest prefill, the largest Qwen3.8 context | `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal` | 2.6 s at 32K, 9.7 s at 128K; 524K window, 831K-token KV pool, 235–254 tok/s; prefix cache survives restarts with HiCache. Depends on one person's fork and applies YaRN to every prompt. |
| Fast and conservative | `qwen3.8-flash-next-sglang-nvfp4-ram` (or `-official` for no local build) | 236–260 tok/s, 262K window, plain Docker |
| Little free host RAM | `qwen3.8-flash-next-sglang-nvfp4-nvme` | reads the PLE table from NVMe instead of holding ~65 GB in RAM; 214–222 tok/s, but the smallest KV pool and the slowest prefill of the SGLang profiles |
| The best Slovak among the Qwen3.8 profiles | `qwen3.8-flash-next-vllm-awq-w4a16-g32` | first in three blind Slovak checks of the Qwen3.8 profiles; ~110 tok/s, 262K window |
| The fastest decode with 16-bit activations | `qwen3.8-flash-next-exllamav3-exl3-5.05bpw` | 119 tok/s on prose, 216 on code; loads in under a minute. Prefill ~1.6× slower than vLLM, lower half in every Slovak check. |
| No refusals | `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` | leoncca's uncensored AWQ g32 with a small vLLM patch; 461 of 542 code tests, the best measured, no loss against the original. One unexplained crash under the first concurrent load. |
| No refusals, 8-bit weights | [`strata-q8-abliterated-vision`](qwen3.8-flash-next/strata/README.md#abliterated-q8_0) | windowsxp811203's abliterated Q8_0 on Strata: no measured loss against the original Q8_0 (code 455 vs 457 of 542, GSM8K 96.7 vs 97.0 %, MMLU-Pro 80.2 vs 78.5 %), vision works. |
| Qwen3.8 at 8 bits outside Strata | **ik_llama.cpp with a Q8_0 GGUF**, not a profile here | first of four in its blind Slovak check of Qwen3.8 setups (73 against 70 for AWQ g32), but no better on code (454 of 542); 38 tok/s, 131K window, ~105 GiB host RAM. `vllm-fp8-offload` also keeps experts in RAM and decodes at ~16 tok/s. |
| Qwen3.8 with fewer bits on Strata | [`strata-q6`](qwen3.8-flash-next/strata/README.md#strata-q6_kq8_0) | Q6_K/Q8_0 experts, 7.21 bpw; less pinned expert RAM (16 against 35 GiB) and faster generation than Q8 in the 2026-10-06 runs, but a slower first prefill; no scored quality run. |
| A dense model | `qwen3.8-27b-sglang-bf16` or `qwen3.6-27b-sglang-bf16` | full BF16 precision, but both solved fewer code tasks than every measured Flash-Next quantization except the abliterated NVFP4, scored 21–27 points below AWQ g32 in the Slovak check, and decode slower (57–87 tok/s) |
| — not this one | `qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated` | dealignai's abliterated NVFP4 lost 16 code tasks net against the same profile without abliteration (p = 0.02) |
| — superseded | `qwen3.8-flash-next-vllm-awq-w4a16` | INT4 group 128: the largest vLLM KV pool (605K), but in the bottom two of every Slovak check; group 32 is slightly faster |

## Qwen3.8-Flash-Next

180B MoE, ~6B active, 262K native context. TTFT is cold, decode the median of
512-token answers; host RAM excludes the OS file cache. Code: EvalPlus plus tests
passed, of 542. Slovak: the nine-profile blind sheet of 2026-09-15 (one grader, sum of
ten prompts). [All columns, footnotes and the 2026-10-06 Strata Q8 / Q6 rows](qwen3.8-flash-next/docs/results.md#speed-and-capacity).

| Profile | Engine, weights | Context | TTFT 32K / 128K | Decode, tok/s | Host RAM | Code | Slovak |
|---|---|---:|---|---|---:|---:|---:|
| `strata-q8-vision` (default) | Strata 0.2.7, Q8_0, MTP | 131,072 | 4.79 / 18.0 s¹ | 159 prose, 176 code, 122 Slovak | ~37 GiB | 457² | 64 / 66⁴ |
| `strata-q8-abliterated-vision` | Strata, abliterated Q8_0 | 131,072 | — | 101 / 125 / 81² | — | 455² | 62 / 64⁴ |
| `sglang-nvfp4-ram-pennyroyal` | SGLang fork, NVFP4 | **524,288** | **2.62 / 9.68 s** | 235–254 | ~65 GB | 456 | 68 |
| `sglang-nvfp4-ram` | SGLang, NVFP4 | 262,144 | 2.90 / 10.7 s | 236–249 | ~65 GB | 457 | 69 |
| `sglang-nvfp4-ram-official` | SGLang official image, NVFP4 | 262,144 | 2.45 / 10.4 s | 258–260 | ~65 GB | 458 | 72 |
| `sglang-nvfp4-nvme` | SGLang, NVFP4, PLE from NVMe | 262,144 | 4.37 / 18.0 s | 214–222 | ~0 | 457 | 66 |
| `vllm-awq-w4a16-g32` | vLLM, INT4 g32 | 262,144 | 3.36 / 12.1 s | 110 | ~115 GB | 457 | **75** |
| `vllm-awq-w4a16-g32-uncensored` | vLLM, INT4 g32, uncensored | 262,144 | — | 110 | ~50 GB | **461** | 70 |
| `vllm-awq-w4a16` | vLLM, INT4 g128 | 262,144 | 3.28 / 12.1 s | 102–103 | ~115 GB | 454 | 65 |
| `exllamav3-exl3-5.05bpw` | ExLlamaV3, EXL3 5.05 bpw | 262,144 | 4.83 / 19.4 s | 119 prose, **216 code** | ~43 GB | 460 | 67 |
| `vllm-fp8-offload` | vLLM, FP8, experts in RAM | 262,144 | 69K: 86.3 s | 15.6–16.6 | ~131 GB | — | 71³ |
| `sglang-nvfp4-ram-official-abliterated` | SGLang, NVFP4, abliterated | 262,144 | — | 200–216 code, 133 prose | ~65 GB | 442 | 71 |

¹ 120K. ² Strata 0.1.41 (2026-10-14). ³ In an earlier four-profile sheet with a different
grader (AWQ g32 70 there). ⁴ Two graders on a 2026-10-14 sheet with ik_llama.cpp Q8_0
(74 / 72) and AWQ g32 (69 / 76); both Strata runs placed last there
([details](qwen3.8-flash-next/strata/README.md#full-test-against-the-original-q8_0-2026-10-14)).

## DeepSeek-V4.1-Flash-1017

552B MoE (16B active for output) with a ~190B-parameter Engram table; 128K context,
DSpark on for Strata, warm decode means. [Details](deepseek-v4.1-flash/README.md).

| Profile | Engine, experts | Decode | Prefill, cold ~30K | Host RAM | Slovak |
|---|---|---|---:|---:|---:|
| `strata-ud-q8kxl` | Strata 0.2.7, UD-Q8_K_XL (original MXFP4) | ~29 tok/s | ~1,000 tok/s | ~224 GB | **86.5**¹ |
| `strata-ist-iq2xs` | Strata 0.2.7, ISTA-DASLab IQ2_XS 2.25 bpw | 40–59 tok/s | **2,118 tok/s** | ~140 GB | 84¹ |
| `xolotl-xq-v2` | Xolotl, XQ v2 (PTQ1_0 / Q2_0 / Q3_K) | **60–68 tok/s** (Slovak ~40) | 668 tok/s | 72 GiB arena | 77¹ |
| `llamacpp-ud-q3kxl`² | llama.cpp PR, UD-Q3_K_XL | 12.9–13.0 tok/s | 443 tok/s | ~195 GB | 86¹ |
| `ik-mxfp4`² | ik_llama.cpp, original MXFP4 | not measured | not measured | ~217 GB | 81¹ |

Profile ids drop the `deepseek-v4.1-flash-` prefix. Strata also has 256K, 512K (UD-Q8_K_XL)
and vision variants of both. ¹ One grader on the 2026-10-27/28 sheets (the 2.25 bpw set on
a second sheet with the same rubric). ² Checkpoint no longer on disk; figures as measured.

On llama.cpp DeepSeek decodes about 11× slower than Qwen3.8 on Strata, bounded by reading
~5 GB of experts from dual-channel DDR5 per token; on Strata about 5× (UD-Q8_K_XL) and 3×
(2.25 bpw), which keep more experts on the GPU.

## Across models

### Slovak

Ten Slovak prompts aimed at what quantization breaks first (inflection, numeral agreement,
idioms, a grammar fix, …), greedy, scored blind by an LLM grader; a stand-in for smaller,
morphologically rich languages ([why Slovak](README.md#results)). Scores are comparable
within one sheet only.

**The one sheet with both families, two graders** (2026-10-23, `logs/language-samples/sk/blind-20261023-051943*`):

| Setup | Score /100, graders 1 / 2 |
|---|---:|
| DeepSeek-V4.1 UD-Q3_K_XL (llama.cpp) | **83 / 84.5** |
| DeepSeek-V4.1 original MXFP4 (ik_llama.cpp) | 75 / 79.5 |
| Qwen3.8-Flash-Next Q8_0 (ik_llama.cpp) | 67 / 70.5 |
| Qwen3.8-Flash-Next AWQ g32 (vLLM) | 67 / 68 |
| Qwen3.8-Flash-Next abliterated Q8_0 (Strata 0.1.44.1) | 63 / 65 |

The DeepSeek answers had about half the weighted errors. Later DeepSeek-only sheets
(one grader) placed Strata UD-Q8_K_XL level with UD-Q3_K_XL (86.5 / 86) and the
2.25 bpw set at 84. The Qwen3.8-only sheets are in
[qwen3.8-flash-next/docs/results.md](qwen3.8-flash-next/docs/results.md#non-english-slovak-blind-check).

### Knowledge, math and code

The same seeded questions and EvalPlus tasks for every setup, thinking off.

| | GSM8K (300) | MMLU-Pro (600) | HumanEval / + (164) | MBPP / + (378) |
|---|---:|---:|---:|---:|
| DeepSeek-V4.1 UD-Q8_K_XL, Strata 0.2.7 | **98.3 %** | **81.0 %** | 0.963 / 0.896 | not finished |
| Qwen3.8 Q8_0, Strata 0.1.41 | 97.0 % | 78.5 % | 0.970 / 0.951 | 0.931 / 0.796 |
| Qwen3.8 abliterated Q8_0, Strata | 96.7 % | 80.2 % | 0.976 / 0.951 | 0.934 / 0.791 |
| Qwen3.8 uncensored AWQ g32, vLLM | — | — | **0.988 / 0.970** | **0.942** / 0.799 |
| Qwen3.6-27B BF16, SGLang | — | — | 0.976 / 0.927 | 0.931 / 0.778 |
| Qwen3.8-27B BF16, SGLang | — | — | 0.970 / 0.933 | 0.910 / 0.780 |

Only DeepSeek's MMLU-Pro lead over the original Qwen3.8 Q8_0 approaches significance
(McNemar p = 0.058). These short, single-function code tasks with thinking off do not
reflect published agentic coding results, where DeepSeek-V4.1-Flash-1017 leads
([README](README.md#results)). Code details and the per-task tests:
[Qwen3.8](qwen3.8-flash-next/docs/results.md#code-humaneval-and-mbpp),
[DeepSeek](deepseek-v4.1-flash/README.md#quality-checks-ud-q8_k_xl-on-strata).

## Not measured yet

- DeepSeek: MBPP+ for UD-Q8_K_XL; GSM8K / MMLU-Pro / EvalPlus for the 2.25 bpw set; a
  second grader for the 2026-10-27/28 Slovak sheets.
- Qwen3.8 Strata 0.2.7: GSM8K / MMLU-Pro / EvalPlus and a blind Slovak grading (the last
  scored runs used 0.1.41–0.1.44.1).
- Anything with thinking on, agentic tool use, long-context retrieval.
- [Qwen3.8-Flash-Next open items](qwen3.8-flash-next/docs/results.md#not-measured-yet).
