# Qwen3.8-27B

Qwen3.8-27B, a dense 27B model with 262K native context, at full BF16 precision on
one RTX PRO 6000 Blackwell 96 GB — the newer sibling of
[Qwen3.6-27B](../qwen3.6-27b/README.md), with the same `qwen3_5` architecture and
the same launcher settings. Like 3.6-27B it serves as a dense reference for the
Qwen3.8-Flash-Next quantizations.

## Checkpoint

| Checkpoint | Weights | Size | Profiles |
|---|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2` | BF16, unquantized | 51.7 GiB (18 shards) | `sglang-bf16` |

## Profiles

One at a time with every other profile, on `http://127.0.0.1:8090/v1` as model
`Qwen3.8-27B`.

| Profile | Runtime | Weights | KV cache | Context | Decode, tok/s | HumanEval+ / MBPP+ | Slovak check |
|---|---|---|---|---|---|---|---|
| [`sglang-bf16`](docs/profiles/sglang-bf16.md) | SGLang, official image | BF16 | 295,344 BF16 | 262,144 | 85 code, 57 prose | 0.933 / 0.780 | 56 (3.6-27B 50, Flash-Next AWQ g32 77) |

Setup, configuration and measurements: [docs/profiles/sglang-bf16.md](docs/profiles/sglang-bf16.md).
Against Qwen3.6-27B and the Qwen3.8-Flash-Next profiles: [RESULTS.md](../RESULTS.md).

## Layout

```
qwen3.8-27b/
├── README.md
├── docs/
│   └── profiles/sglang-bf16.md
└── sglang/
    └── bf16/                     launcher + stop
```
