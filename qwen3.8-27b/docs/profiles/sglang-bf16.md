# SGLang, BF16

Profile `sglang-bf16`, directory `qwen3.8-27b/sglang/bf16/`, started with
`scripts/start-qwen3.8-27b-sglang-bf16.sh`.

Qwen/Qwen3.8-27B @ `1d4bf0f2` with its original BF16 weights on the official SGLang
image `lmsysorg/sglang:dev-qwen38-next-local`. Same architecture (`qwen3_5`) and
launcher settings as [Qwen3.6-27B](../../../qwen3.6-27b/docs/profiles/sglang-bf16.md).
Measured 2026-09-15.

## Setup

```bash
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir models/Qwen3.8-27B
scripts/start-qwen3.8-27b-sglang-bf16.sh
```

Serves `http://127.0.0.1:8090/v1` as model `Qwen3.8-27B`. The configuration table in
the Qwen3.6-27B profile document applies unchanged: NEXTN speculation with 3 steps,
BF16 KV cache (no FP8 without calibrated scales), BF16 SSM state,
`--mem-fraction-static 0.88`, 4 concurrent requests.

## Where it lives

| | Size | Where |
|---|---|---|
| Weights | 51.7 GiB, BF16, 18 shards | VRAM |
| KV cache — **295,344 tokens**, BF16 | 18.0 GiB | VRAM |
| **VRAM in use** | **84.4 GiB** of 95.6 GiB | |
| Host RAM | nothing offloaded | |

## Measurements

| | |
|---|---|
| Decode, LRU-cache prompt with code | 85.1 / 84.9 tok/s |
| Decode, Slovak prose | 56.8 tok/s |
| HumanEval / HumanEval+ | 0.970 / 0.933 |
| MBPP / MBPP+ | 0.910 / 0.780 |
| Plus tests passed | 448 of 542 |
| Slovak blind check, score of 100 | 56 (Qwen3.6-27B 50, Flash-Next AWQ g32 77) |

Task by task against Qwen3.6-27B (446) it solved 18 tasks the older model missed
and missed 16 it solved (p = 0.86) — no measurable difference on this benchmark.

In the four-way Slovak blind check it placed third, 6 points above Qwen3.6-27B with
about as many weighted errors (88 against 86) and 14–21 points below the two
Flash-Next AWQ g32 checkpoints. Typical errors: Czech words („Omluvy“, „predem“),
gender slips („veľká pstruh“), non-words („nadmierať“, „odraza“) and invented
grammar rules. Details and the Flash-Next comparison: [RESULTS.md](../../../RESULTS.md#non-english-slovak-blind-check).
