# SGLang official image, abliterated NVFP4, PLE table in RAM

Profile `sglang-nvfp4-ram-official-abliterated`, started with
`scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated.sh`. It
reuses the launcher in `qwen3.8-flash-next/sglang/nvfp4-ram-official/` unchanged —
image, memory settings, BF16 KV cache, NEXTN speculation — with a different
checkpoint. Everything in [sglang-nvfp4-ram-official.md](sglang-nvfp4-ram-official.md)
applies.

## Checkpoint

[dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
@ `be794b99`: an abliterated build of Qwen3.8-Flash-Next — refusal behaviour removed
by a direct weight edit, with no fine-tuning — in NVFP4.

Its layout is identical to the RadixArk NVFP4 checkpoint the other SGLang profiles
use: the same `config.json`, the same ModelOpt 0.46 NVFP4 configuration (group 16,
the same 13 excluded modules), the same 296,475 tensor names including the FP8 PLE
table with its scale, and the same 135.3 GB. That is why it loads in the existing
launcher without changes.

Published by its author (not re-measured here):

| | Base | Abliterated |
|---|---|---|
| MMLU, 57 subjects × 40 questions | 82.11 % | 81.93 % |
| HarmBench real-harm behaviours complied with (greedy, reasoning off / low / xhigh) | — | 100 % |

MTP speculation and image and video input are reported to work.

> **Refusals are removed.** The model answers requests the original declines. Output
> is the operator's responsibility.

## Setup

```bash
hf download dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --revision be794b990578ef3031eccf9f28e675a289a09ee9 \
  --local-dir models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4
scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated.sh
```

Serves `http://127.0.0.1:8090/v1` as model `Qwen3.8-Flash-Next`, like the other
Flash-Next profiles, so clients need no change. `scripts/status.sh` shows which of
the two checkpoints is running: `start_profile` passes the profile id to the shared
launcher, which labels the container with it.

## Measurements

2026-09-15, same machine.

| | |
|---|---|
| KV cache | 256,832 tokens, BF16 (as `sglang-nvfp4-ram-official`) |
| VRAM in use | 91.5 GiB |
| Decode, LRU-cache prompt with code | 216.0 / 203.3 / 200.4 tok/s |
| Decode, Slovak prose | 133 tok/s |
| Tool calls (`qwen3_coder`), thinking on and off | parsed into `tool_calls` |
| HumanEval+ / MBPP+ | 0.890 / 0.783 — 442 of 542 plus tests (below) |
| Slovak blind check | 71 of 100, third of nine (below) |

### Code benchmarks

| | This profile | `sglang-nvfp4-ram-official` |
|---|---|---|
| HumanEval / HumanEval+ | 0.933 / 0.890 | 0.976 / 0.951 |
| MBPP / MBPP+ | 0.921 / 0.783 | 0.929 / 0.799 |
| Plus tests passed, of 542 | **442** | 458 |

Same launcher, same quantization format; only the abliteration differs. Task by
task it lost 30 plus tests and gained 14 (sign test p = 0.02) — the fewest passes
of any Flash-Next profile, below both dense 27B models at BF16 (Qwen3.6-27B 446,
Qwen3.8-27B 448). The author's MMLU check
(−0.18 pp) does not show this. For a refusal-free Flash-Next,
[`vllm-awq-w4a16-g32-uncensored`](vllm-awq-w4a16-g32-uncensored.md) kept its code
ability.

Slovak blind check: third of nine (71 of 100) with the fewest weighted errors, and
the only profile that avoided the non-word „vereta“. The damage shows on code, not
in this small language check ([RESULTS.md](../../../RESULTS.md#non-english-slovak-blind-check)).
