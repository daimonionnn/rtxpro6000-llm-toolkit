# Qwen3.6-27B

Qwen3.6-27B, a dense 27B model with 262K native context, at full BF16 precision on
one RTX PRO 6000 Blackwell 96 GB. In this toolkit it serves as the dense reference
for the Qwen3.8-Flash-Next quantizations: the whole model fits on the card without
any quantization.

## Checkpoint

| Checkpoint | Weights | Size | Used by |
|---|---|---|---|
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) @ `6a9e13bd` | BF16, unquantized | 51.7 GiB (15 shards) | `sglang-bf16` |

Architecture `qwen3_5`: 64 layers, Gated DeltaNet linear attention with full
attention every 4th layer, a vision encoder and one MTP layer used for speculative
decoding.

## Profiles

Serves `http://127.0.0.1:8090/v1` as model `Qwen3.6-27B`, like every profile in the
toolkit one at a time. Start with `scripts/start-qwen3.6-27b-sglang-bf16.sh`.

| Profile | Runtime | Weights | KV cache | Context | Decode, tok/s | VRAM in use |
|---|---|---|---|---|---|---|
| `sglang-bf16` | SGLang, `lmsysorg/sglang:dev-qwen38-next-local` | BF16 | BF16 | 262,144 | **87 code, 62 prose** (NEXTN, 3 steps) | 86.5 GiB |

- **KV cache stays BF16.** FP8 KV needs calibrated scales, which an unquantized
  checkpoint does not carry; without them the output is corrupted.
- NEXTN draft acceptance is much higher on code than on prose (accept length ~2.5
  of 4 on Slovak prose), hence the two decode figures.
- Settings follow a published single-card BF16 27B setup; `MEMFRAC` (0.88),
  `MAXRUN` (4), `SPEC_STEPS` (3), `CTX` and `EXTRA_ARGS` can be overridden from the
  environment.

## Code benchmarks

HumanEval+ and MBPP+ with `bench/evalplus_codegen.py` and
`bench/evalplus_evaluate.sh`, greedy, thinking off, 2026-09-14:

| | HumanEval | HumanEval+ | MBPP | MBPP+ | Plus tests passed, of 542 |
|---|---|---|---|---|---|
| `qwen3.6-27b-sglang-bf16` | 0.976 | 0.927 | 0.931 | 0.778 | 446 |
| Qwen3.8-Flash-Next quantizations (4 profiles) | 0.970–0.982 | 0.951–0.963 | 0.923–0.937 | 0.788–0.802 | 454–460 |

The 27B at full precision solved 8–14 fewer tasks than each Flash-Next
quantization, mostly on the stricter plus tests. Per-profile numbers and the
task-by-task comparison are in the
[Flash-Next README](../qwen3.8-flash-next/README.md#code-benchmarks).

## Layout

```
qwen3.6-27b/
├── README.md
└── sglang/
    └── bf16/                     launcher + stop
```
