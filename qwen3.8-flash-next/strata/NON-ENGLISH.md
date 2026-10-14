# Strata and non-English output

In the 2026-10-14 blind Slovak check, both Strata Q8_0 profiles scored below
ik_llama.cpp with the same Q8_0 weights. This page records the evidence, the
candidate causes found in the engine, and how to separate them. Nothing here has
been confirmed yet; the open tasks are in the [TODO](../README.md#todo).

## The observation

Ten Slovak prompts from `bench/language_samples.py`, greedy, thinking off, one
answer per prompt. The answers were shuffled into one sheet
(`logs/language-samples/sk/blind-20261014-035124.md`) and graded by two LLM
graders that saw only the sheet:

| Configuration | Grader 1 | Grader 2 |
|---|---:|---:|
| ik_llama.cpp Q8_0 (answers from 2026-09-15) | **74** | 72 |
| vLLM AWQ g32 (cyankiwi) | 69 | **76** |
| vLLM AWQ g32 uncensored (leoncca) | 70 | 68 |
| Strata 0.1.41, original Q8_0 | 64 | 66 |
| Strata 0.1.41, abliterated Q8_0 | 62 | 64 |

Scores are sums of ten 1–10 grades. The two Strata profiles are within noise of
each other, so the abliteration is not the cause. They are 6–10 points behind
ik_llama.cpp, mostly on the story and noun-inflection prompts. Code, GSM8K and
MMLU-Pro showed no such gap
([full test](README.md#full-test-against-the-original-q8_0-2026-10-14)).

Limits of the evidence: one greedy answer per prompt; two prompts carry most of the
difference; the ik_llama.cpp answers come from an earlier run (without MTP, which
should not change greedy text); the graders are LLMs, not native speakers.

### After the upgrade to 0.1.44.1

Same ten prompts, abliterated Q8_0, `STRATA_ROUTE_TAIL_SKIP=0`, a fresh three-way
sheet (`logs/language-samples/sk/blind-20261014-121550.md`) and two new graders:

| Configuration | Grader 3 | Grader 4 | Error penalty, 3 / 4 |
|---|---:|---:|---:|
| ik_llama.cpp Q8_0 (answers from 2026-09-15) | **72** | **73** | 40 / 48 |
| Strata 0.1.44.1, abliterated Q8_0 | 67 | 70 | **43 / 44** |
| Strata 0.1.41, abliterated Q8_0 (same answers as above) | 64 | 63 | 53 / 62 |

The 0.1.44.1 answers differ from the 0.1.41 ones on seven of ten prompts. Both
graders scored them higher, with fewer weighted errors, and the gap to
ik_llama.cpp narrowed from 8–10 to 3–5 points. Which of the 622 commits between the
two versions is responsible is not known, and the sample is still one answer per
prompt.

### Original BF16 bytes for the small tensors

The Q8_0 GGUF stores the hyper-connection projections, `ssm_alpha`, `ssm_beta` and
`ple_value` as Q8_0; the model itself is BF16. `bf16_dense_patch.py` copies these 363
tensors (1.22 GiB) from the BF16 GGUF of the same abliterated checkpoint into the
pack, by HTTP range requests, so the engine reads the original weights instead of
rounded Q8_0 values. The replaced values differed from the old ones by 0.67% relative
RMS (median, largest 1.03%), the size of Q8_0 quantization error, which also
confirmed that names, shapes and element order match.

Four-way sheet (`blind-20261014-123938.md`), two new graders:

| Configuration | Grader 5 | Grader 6 |
|---|---:|---:|
| ik_llama.cpp Q8_0 | **73** | **72** |
| Strata 0.1.44.1, abliterated, small tensors rounded from Q8_0 | 69 | 68 |
| Strata 0.1.44.1, abliterated, small tensors original BF16 | 67 | 64 |
| Strata 0.1.41, abliterated | 64 | 63 |

The original BF16 tensors did not improve the Slovak score; the 2–4 point drop comes
mostly from one prompt (code explanation, 7 / 6 against 9 / 9) and is within the
noise of one answer per prompt. GSM8K was unchanged (291 against 290 of 300,
p = 1.0), and decode speed too (108.7 / 120.5 / 85.7 tok/s). The rounding of these
tensors is therefore not a measurable cause of the gap. The running pack was
regenerated with `iq_pack.py` afterwards (byte-identical to the rounded version),
the variant that scored higher in this check.

## Ruled out

- **Tokenizer and chat template.** Strata exports its own tokenizer
  (`tools/strata_tokenizer.py`), but the same three `compare_decode.py` requests
  count 44 / 59 / 74 prompt tokens on both engines, including the template.
  An exact token-ID comparison has not been run.
- **Sampling.** The benchmarks request temperature 0 with no penalties.
- **Small-tensor precision** (see above): replacing the rounded tensors with the
  original BF16 ones did not change the Slovak score measurably.
- **Router precision.** `ffn_gate_inp` and `ffn_gate_inp_shexp` are stored as F32
  holding BF16 values; the conversion in the pack is exact
  (`packs/q8_0/conversions.json`).

## Candidate causes in Strata 0.1.41

Ordered by how likely they look. Each is small on its own; lower-resource
languages tend to show accumulated numerical drift first.

### 1. Paths that the engine reports as inexact

The engine log names them at start-up:

- **Prefill CPU share**, on by default for prompt chunks under 1,024 tokens: the
  idle CPU computes some experts the GPU would stream. The log states "answers can
  differ slightly … mean KL ~0.004". `STRATA_PREFILL_CPU_SHARE=0` turns it off.
- **Experts computed in three places.** Cached experts run on the GPU, missed ones
  partly on the CPU pool (AVX2 here, no AVX-512) and partly copied over PCIe
  (`pcie_frac 0.55`). The log notes that the GPU "rounds differently from the CPU".
  About 69% of the expert slots fit in VRAM (16,966 of 24,576), so every token mixes
  these paths.
- **BF16 rounding of 363 small tensors** (`iq_pack.py --compat-bf16`). The largest
  errors are in the hyper-connection projections, which every token passes through:

  | Tensor (per layer) | Source | Largest absolute error |
  |---|---|---:|
  | `hc_ffn_up` | Q8_0 → BF16 | 0.0144 |
  | `hc_attn_up` | Q8_0 → BF16 | 0.0078 |
  | `output_hc_up` | Q8_0 → BF16 | 0.0070 |
  | `hc_ffn_down` | Q8_0 → BF16 | 0.0033 |
  | `hc_attn_down` | Q8_0 → BF16 | 0.0020 |
  | `ssm_alpha`, `ssm_beta`, `hc_*_inject`, `ple_value` | Q8_0 → BF16 | ≤ 0.0010 |

  This is about one BF16 unit in the last place, but it is applied to every token
  on top of the Q8_0 quantization. Strata 0.1.44 adds `STRATA_HC_Q8=1`, which keeps
  these projections in Q8_0 and dequantizes them on the fly.

### 2. Speculative decoding (MTP)

Greedy verification should reproduce non-speculative output, but drafts are checked
in windows of 1–6 tokens through different numerical paths. Upstream commit
`c81f9984` (2026-10-09, after our build) fixes pipelined one-token windows that
advanced the model state before the host's verdict. Whether this configuration hits
that path is unknown; if it does, the state can drift. `--spec 0` disables drafting
for an A/B test.

### 3. Reasoning recovery (thinking on only)

When the server detects repeated reasoning, it resumes with temperature ≥ 1.0 and
presence penalty ≥ 1.5 (`serve/server.py`, "repeated reasoning detected"). A
presence penalty that high pushes against tokens already used, which in an
inflected language includes common endings. This does not affect the benchmarks
(thinking off), but it can affect agents that leave thinking on. No server log from
daily use survives to show how often it fires.

### 4. KV cache format

Strata uses INT8 KV; the ik_llama.cpp run used Q8_0 KV. Both are 8-bit, so a large
effect is not expected.

## Route tail skip (Strata 0.1.42+)

Strata 0.1.42 and later enable `STRATA_ROUTE_TAIL_SKIP=7` by default on CUDA: in a
verify window, a missed expert that every token ranks 7th–10th is neither copied nor
computed, and its contribution becomes zero. Strata's own measurements put the KL
for chat and non-English prompts at 0.004–0.009, about twice that of code. The skip
stays off only when 90% or more of the experts are in VRAM; this machine holds about
69%, so it would be active. Since the upgrade to 0.1.44.1 on 2026-10-14 the launchers
set `STRATA_ROUTE_TAIL_SKIP=0` (computing every expert, as 0.1.41 did) unless the
variable is already set; measure before enabling it.

## How to separate the causes

1. **Engine fidelity.** Strata 0.1.44 ships `tools/quant_quality.py`: teacher-forced
   top-1 agreement and KL against a reference. Run it on Slovak text with
   ik_llama.cpp Q8_0 as the reference, then with each switch below. Teacher forcing
   removes the single-sample noise of the blind sheet.
2. **One switch at a time**, against the default:
   `STRATA_PREFILL_CPU_SHARE=0`, `--spec 0`, `STRATA_HC_Q8=1` (0.1.44+), and on
   0.1.44+ `STRATA_ROUTE_TAIL_SKIP=0`.
3. **A larger blind check.** Fresh Slovak answers from ik_llama.cpp and Strata in the
   same session, several samples per prompt or more prompts, at least two graders.
