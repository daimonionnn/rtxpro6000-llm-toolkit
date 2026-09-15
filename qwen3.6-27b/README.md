# Qwen3.6-27B

Qwen3.6-27B, a dense 27B model with 262K native context, at full BF16 precision on
one RTX PRO 6000 Blackwell 96 GB. In this toolkit it serves as the dense reference
for the Qwen3.8-Flash-Next quantizations: the whole model fits on the card without
any quantization.

## Checkpoint

| Checkpoint | Weights | Size | Profiles |
|---|---|---|---|
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) @ `6a9e13bd` | BF16, unquantized | 51.7 GiB (15 shards) | `sglang-bf16` |

Architecture `qwen3_5`: 64 layers, Gated DeltaNet linear attention with full
attention every 4th layer, a vision encoder and one MTP layer used for speculative
decoding.

## Profiles

One at a time with every other profile, on `http://127.0.0.1:8090/v1` as model
`Qwen3.6-27B`.

| Profile | Runtime | Weights | KV cache | Context | Decode, tok/s | HumanEval+ / MBPP+ |
|---|---|---|---|---|---|---|
| [`sglang-bf16`](docs/profiles/sglang-bf16.md) | SGLang, official image | BF16 | BF16 | 262,144 | 87 code, 62 prose | 0.927 / 0.778 |

Setup, configuration and measurements: [docs/profiles/sglang-bf16.md](docs/profiles/sglang-bf16.md).
Against the Qwen3.8-Flash-Next profiles: [RESULTS.md](../RESULTS.md) — every
Flash-Next quantization solved more of the code tasks, and in the Slovak blind check
it scored 50 of 100 against 77 for Flash-Next AWQ g32 (Qwen3.8-27B: 56).

## Layout

```
qwen3.6-27b/
├── README.md
├── docs/
│   └── profiles/sglang-bf16.md
└── sglang/
    └── bf16/                     launcher + stop
```
