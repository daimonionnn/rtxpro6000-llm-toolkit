# Measurements

All numbers below were taken on the same physical card — one RTX PRO 6000
Blackwell Workstation (96 GB, SM120) — with the same prompt battery.

Qwen3.8-Flash-Next is measured in **both modes**, because they behave like two
different models: turning thinking off costs some accuracy and buys a lot of
throughput. The dense Qwen3.8-27B is shown in both modes for the same reason.

## Read this first: the comparison is not clean

The dense Qwen3.8-27B rows were served by **vLLM 0.25.0**; the Flash-Next rows
by **SGLang** with the build in this repository. Different engines, and days
apart (2026-08-15 vs 2026-08-27/28).

That matters differently per metric:

- **HumanEval+ / GSM8K** are accuracy at temperature 0. Engine-independent —
  compare freely.
- **tok/s and TTFT** are engine-dependent. Read them as "what this card
  delivered in each configuration", not as a model-vs-model verdict.
- **Concurrency** shapes differ structurally, not incidentally: Flash-Next's
  recurrent-state slots and KV pool draw on one budget, a dense model's do not.

Anyone rerunning the 27B side under SGLang would produce a cleaner table, and
we would be glad to see it.

## Single card, single stream

Thinking is selected per request with `chat_template_kwargs.enable_thinking`
and `reasoning_effort`.

| | **Flash-Next**<br>non-thinking | **Flash-Next**<br>thinking (xhigh) | **27B NVFP4**<br>non-thinking | **27B NVFP4**<br>thinking (high) |
| --- | --- | --- | --- | --- |
| Params | 176B MoE / 6B active | ← | 27B dense | ← |
| GPU memory | 78.2 GB weights + 47.7 GB PLE **on NVMe** | ← | 21.8 GB | ← |
| KV cache | 233,856 tok (fp8) | ← | 1,799,860 tok | ← |
| English prose | **154.7** tok/s | 125.7 | 119.4 | 96.8 |
| Korean PRD | **128.7** | 128.4 | 99.7 | 101.2 |
| Coding | **180.9** | 148.2 | 78.1 | 118.3 |
| Korean chat | 118.2 | **124.2** | 92.2 | 98.5 |
| Tool call | **178.6** | 127.0 | 158.9 | 113.8 |
| TTFT | 0.092 s | 0.334 s | **0.055 s** | 10.14 s |
| Prefill 8K / 32K / 65K | 7,914 / 9,395 / 9,990 tok/s | 8,049 / 9,535 / 10,229 | — | — |
| HumanEval+ (base/plus) | 0.939 / 0.921 | 0.957 / 0.927 | 0.915 / 0.890 | **0.982 / 0.939** |
| GSM8K (flex/strict) | **0.98 / 0.98** | ← (measured non-thinking) | 0.95 / 0.94 | not measured |
| Tool battery | 6/7 | 6/7 | **7/7** | **7/7** |
| Needle (9 positions) | **9/9** | **9/9** | — | — |

Flash-Next thinking tok/s **includes its thinking tokens** — total generated
tokens per second, not answer tokens.

Prefill is within 2% between the two modes, as expected — thinking affects
decode, not the forward pass over the prompt.

### What the table says

**Same conditions, both non-thinking:** Flash-Next wins **all five** workloads
and both accuracy scores (HumanEval+ 0.939/0.921 vs 0.915/0.890, GSM8K 0.98 vs
0.95). Coding is the widest gap — 180.9 against 78.1 tok/s.

**Thinking is a trade, not an upgrade.** Turning it on moves HumanEval+ from
0.939/0.921 to 0.957/0.927 (+1.8 / +0.6 points) and costs 22% of coding
throughput (180.9 → 148.2 tok/s) plus 3.6× the TTFT. Worth it for hard
single-shot problems, not for volume.

**The dense 27B still wins two things.** A *thinking* 27B is more accurate on
code than Flash-Next in either mode (0.939 plus-score), and its TTFT
non-thinking is 0.055 s. It also passes 7/7 on tools where Flash-Next gets 6/7 —
it prefers sequential calls over emitting two in parallel.

What Flash-Next brings is a 176B model's breadth on hardware that should not fit
one, at speeds a dense 27B does not reach.

## Concurrency

| Streams | Flash-Next (non-thinking) | Flash-Next (thinking) | 27B NVFP4 (non-thinking) |
| --- | --- | --- | --- |
| 1 | 117 tok/s | 132 | — |
| 2 | 240 | 245 | — |
| 4 | 134 ⚠️ | 408 | — |
| 8 | 385 | 397 | 846 |
| 12 | 393 | 373 | 1,079 |
| 16 | 264 ⚠️ | 391 | **1,112** |

Zero errors at every level in all three columns — the ⚠️ cells completed
successfully, they are just out of line with their neighbours.

**Do not read a curve shape off the non-thinking column.** It is non-monotonic
(c4 dips to 134, c16 to 264, with wall times of 7.6 s and 15.5 s against 5.3 s
at c8), which single-run measurement noise explains and a real throughput curve
does not. The thinking column from the same harness is smooth.

What both columns agree on: **aggregate throughput is flat from c4 onward,
around 370–410 tok/s.** `--max-mamba-cache-size 25` yields roughly 5 concurrent
streams, and past that requests queue rather than overlap. Raising it takes
memory from the KV pool. If your workload is many concurrent users rather than
one deep agent loop, the dense 27B is the better answer on this card — by
roughly 3×.

## Tuning decisions, and what was rejected

| Knob | Setting | Why |
| --- | --- | --- |
| `--cuda-graph-backend-decode` | `breakable` | With PR #36749 this runs together with NEXTN: 57.6 → 167 tok/s single-stream. Full and `tc_piecewise` graphs are incompatible with the PLE device-to-host copy during capture |
| `--speculative-num-draft-tokens` | 4 | Architectural cap from the QSA compression ratio. Higher values do not help |
| `--kv-cache-dtype` | `fp8_e4m3` | Doubles the KV pool. Requires patch 4 (or upstream #36644) or long prompts crash on the 2nd chunk |
| `--max-mamba-cache-size` | 25 | ≈5 concurrent streams; the trade is against KV pool size |
| `--chunked-prefill-size` | 8192 | 7.9K–10K tok/s prefill across 8K–65K prompts |
| Hierarchical cache | **off** | Attaches and reopens 2.4× faster, but a needle probe fails after restore — silent context loss. See README |

## How these were measured

- Single-stream figures are five fixed prompts (English prose, a Korean PRD,
  a coding task, Korean chat, a tool call), temperature 0, token counts taken
  from the response `usage` field rather than counting stream chunks.
- HumanEval+ is the EvalPlus set, greedy, `base / plus` pass@1, through a proxy
  that pins the thinking mode and coerces null content.
- GSM8K is 5-shot, greedy, 100 items, `flexible-extract / strict-match`.
- The tool battery is seven cases: single auto-select, multi-tool select,
  parallel calls, multi-turn follow-up, argument schema/enum conformance, a
  negative case where no tool should fire, and a forced call.
- Needle is a 9-point sweep: 8K / 32K / 65K tokens × 10% / 50% / 90% depth.
- The harness itself is part of a private homelab repo and is not published
  here; `bench/single_stream.py` reproduces the throughput column with no
  dependencies beyond the standard library.

The non-thinking battery ran in 8 minutes on the image built by `build.sh`; the
thinking battery was measured the day before on the same image minus the
optional `max_thinking_tokens` patch. The launch flags in `serve.sh` are exactly
the ones that produced these numbers. Raw engine logs are not included — they
contain host paths.
