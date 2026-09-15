# SGLang, BF16

Profile `sglang-bf16`, directory `qwen3.6-27b/sglang/bf16/`, started with
`scripts/start-qwen3.6-27b-sglang-bf16.sh`.

Qwen/Qwen3.6-27B @ `6a9e13bd` with its original BF16 weights on the official SGLang
image `lmsysorg/sglang:dev-qwen38-next-local` (the one
`qwen3.8-flash-next-sglang-nvfp4-ram-official` uses; its SGLang has the `qwen3_5`
model and MTP code). Measured 2026-09-14.

## Setup

```bash
hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir models/Qwen3.6-27B
docker pull lmsysorg/sglang:dev-qwen38-next-local      # 33 GB, if not already present
scripts/start-qwen3.6-27b-sglang-bf16.sh
```

Serves `http://127.0.0.1:8090/v1` as model `Qwen3.6-27B`.

## Configuration

Settings follow a published single-card BF16 27B setup (Qwen3.8-27B on an RTX PRO
6000):

| Setting | Value | Why |
|---|---|---|
| `--speculative-algorithm NEXTN`, 3 steps, 4 draft tokens | from the model's MTP layer | 3 steps was that setup's optimum for BF16 |
| `--speculative-attention-mode decode`, `--cuda-graph-max-bs 8` | | without the cap, speculation sizes CUDA graphs for 48 and they take ~4 GB of pool |
| `--mamba-ssm-dtype bfloat16` | | halves the linear-attention state |
| KV cache | BF16 | FP8 KV needs calibrated scales an unquantized checkpoint does not carry; without them output is corrupted |
| `--mem-fraction-static` | 0.88 | 0.94 left too little headroom for transient allocations in the reference setup |
| `--max-running-requests` | 4 | one agent |
| `SGLANG_SANITIZE_NAN_LOGITS`, `expandable_segments` | on | as in the reference setup |

`MEMFRAC`, `MAXRUN`, `SPEC_STEPS`, `CTX` and `EXTRA_ARGS` can be overridden from the
environment.

## Where it lives

| | Size | Where |
|---|---|---|
| Weights | 51.7 GiB, BF16 | VRAM |
| KV cache and linear-attention state | the rest of the 0.88 fraction | VRAM |
| **VRAM in use** | **86.5 GiB** of 95.6 GiB | |
| Host RAM | nothing offloaded | |

The KV pool size was not recorded.

## Measurements

| | |
|---|---|
| Decode, LRU-cache prompt with code | 87.0 / 86.8 / 86.8 tok/s |
| Decode, Slovak prose | 62.0 / 62.0 tok/s |
| NEXTN acceptance on prose | accept length ~2.5 of 4 |
| HumanEval / HumanEval+ | 0.976 / 0.927 |
| MBPP / MBPP+ | 0.931 / 0.778 |
| Plus tests passed | 446 of 542 |
| Slovak blind check, score of 100 | 50 (Qwen3.8-27B 56, Flash-Next AWQ g32 77) |

Code benchmarks: `bench/evalplus_*`, greedy, thinking off. Against the
Qwen3.8-Flash-Next profiles: [RESULTS.md](../../../RESULTS.md).

In the four-way Slovak blind check (2026-09-15) it placed last, 27 points below
Flash-Next AWQ g32. It wrote digits with plural verbs („sú 5 žien“) where words and
singular agreement were asked for, used Czech forms („srozumiteľnejšou“, „Kľudne“),
and misread „mať maslo na hlave“; it gave the best grammar correction of the four.

Not measured: prefill, KV pool size, long-context retrieval.
