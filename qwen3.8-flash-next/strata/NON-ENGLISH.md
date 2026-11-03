# Strata and non-English output

In the 2026-10-14 blind Slovak check, both Strata Q8_0 profiles scored below
ik_llama.cpp with the same Q8_0 weights. This page records the evidence, the
candidate causes found in the engine, and how they were separated.

**Status, 2026-11-03: the gap is gone on Strata 0.2.7.** A fresh blind check with three
answers per engine and two graders puts Strata level with ik_llama.cpp, and teacher
forcing finds no Slovak-specific drift and no switch that matters
([measurement below](#strata-027-against-ik_llamacpp-2026-11-03)).

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

## Strata 0.2.7 against ik_llama.cpp, 2026-11-03

Both engines read the same Q8_0 GGUF: Strata 0.2.7 (the default
`strata-q8-vision-256k` profile) and ik_llama.cpp `13cffc00` through
[ikllama/](../ikllama/README.md).

### Blind sheet

Ten prompts, three answer sets per engine: greedy, and two samples at temperature 0.7,
top-p 0.8, top-k 20, no presence penalty, seeds 1 and 2, all sent explicitly so both
engines sample alike. Thinking off. Six answers per prompt, shuffled, graded 0-10 by
two LLM graders that saw only the sheet
(`logs/language-samples/sk/blind-20261103-015803*.md`):

| Answer set | Grader 1 | Grader 2 | Mean |
|---|---:|---:|---:|
| ik_llama.cpp, greedy | 74.0 | 70.5 | 72.25 |
| ik_llama.cpp, sample 1 | 73.0 | 68.0 | 70.5 |
| ik_llama.cpp, sample 2 | 68.5 | 66.5 | 67.5 |
| Strata 0.2.7, greedy | 73.5 | 73.5 | **73.5** |
| Strata 0.2.7, sample 1 | 67.5 | 67.0 | 67.25 |
| Strata 0.2.7, sample 2 | 73.0 | 71.5 | 72.25 |

The engines average 70.1 (ik_llama.cpp) and 71.0 (Strata); two samples of one engine
differ by up to 5 points, more than the engines do. Both fail the same two prompts
(case inflection of "mesto", and the grammar fix, where most answers keep "jablka").
Strata's worst set (sample 1) holds the two lowest inflection and grammar-fix answers,
and ran to the 800-token limit on both.

**Chinese characters.** None in any of the 100 answers, including two more sets per
engine at presence penalty 1.5 (the model card's instruct default, suspected of
language mixing). Thinking was off; the reports of Chinese output were not reproduced
by this test.

### Teacher forcing

Six segments of 1,000 tokens, read one token at a time: three Slovak texts written for
this test (`bench/fixtures/sk-teacher-forcing/`, a story with dialogue, an expository
text, an official letter) and the English, code and multilingual segments of Strata's
own quant-quality set. The reference is ik_llama.cpp with F16 KV, read through its server
by `bench/teacher_forcing.py` (5,967 positions; 27 skipped where a prefix ends inside a
multi-byte character). Strata runs through `tools/quant_quality.py` (decode windows, no
prefill CPU share, route tail skip off), switched one setting at a time with
`bench/strata_quant_quality.py`. Top-1 agreement and KL over the reference's 64 most
likely tokens; perplexity on the 5,868 positions where the target is in every run's top
64 (`bench/teacher_forcing_ppl.py`):

| Run | Top-1, all | KL, all | KL, Slovak story / expository / formal | KL, English / code | Perplexity |
|---|---:|---:|---|---|---:|
| ik_llama.cpp, Q8_0 KV (the reference's own KV-format change) | 96.85% | 0.0097 | 0.0121 / 0.0083 / 0.0073 | 0.0128 / 0.0107 | 2.478 |
| Strata, experts on the CPU, F16 KV (the tool's default) | 95.83% | 0.0180 | 0.0237 / 0.0110 / 0.0157 | 0.0203 / 0.0210 | 2.484 |
| Strata, experts on the GPU, F16 KV | 95.64% | 0.0180 | 0.0203 / 0.0131 / 0.0177 | 0.0188 / 0.0208 | 2.481 |
| Strata, experts on the GPU, F16 KV, `STRATA_HC_Q8=1` | 95.66% | 0.0170 | 0.0204 / 0.0106 / 0.0160 | 0.0199 / 0.0189 | 2.485 |
| Strata, experts on the GPU, INT8 KV (as served) | 95.73% | 0.0176 | 0.0195 / 0.0125 / 0.0162 | 0.0197 / 0.0235 | 2.487 |

The reference's perplexity is 2.480.

- **Strata differs from ik_llama.cpp about twice as much as ik_llama.cpp differs from
  itself with Q8_0 instead of F16 KV** (KL 0.017-0.018 against 0.0097), and evenly: the
  Slovak segments are not further off than English or code.
- **None of the switches moves it.** Expert placement (CPU or GPU), the hyper-connection
  projections in Q8_0 instead of BF16, and INT8 instead of F16 KV all stay within
  0.0170-0.0180, so none of candidates 1 and 4 above is the source.
- **The drift does not cost quality.** Perplexity on the same text is within 0.3% of the
  reference for every run: lower than ik_llama.cpp's on English and the multilingual
  segment, 1.1-1.8% higher on the Slovak letter, 0-1.1% on the story. KL against ik_llama.cpp measures
  distance from ik_llama.cpp, not from the BF16 model; without a BF16 reference it cannot
  say which engine is closer to it.
- Not covered: the batched prompt path (prefill CPU share), MTP drafting (teacher forcing
  does not draft), and thinking on with reasoning recovery.
