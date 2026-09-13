# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (180B MoE, ~6B active, 262K native context) on one RTX PRO 6000
Blackwell 96 GB. Most of the parameter count is a 51B N-gram (PLE) lookup table
that can live outside the GPU, and 512 experts of which 10 run per token — see
[docs/comparison.md](docs/comparison.md#architecture).

## Checkpoints

Each is kept once under `models/` at the toolkit root and shared by every profile
that uses it.

| Checkpoint | Routed experts | Rest | PLE table | Size | Used by |
|---|---|---|---|---|---|
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ `7b719225242a` | NVFP4 W4A4, 4.50 bits | BF16 | FP8, 47.7 GiB | 125.9 GiB | the four SGLang profiles |
| [wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) @ `0939125b` | INT4 W4A16, 4.13 bits | BF16 | BF16, 95.4 GiB | 168.3 GiB | `vllm-awq-w4a16` |

## Profiles

Only one can hold the GPU at a time; every one serves `http://127.0.0.1:8090/v1` as
model `Qwen3.8-Flash-Next`. Start scripts are
`scripts/start-qwen3.8-flash-next-<profile>.sh` at the toolkit root.

| | `sglang-nvfp4-nvme` | `sglang-nvfp4-ram` | `sglang-nvfp4-ram-official` | `sglang-nvfp4-ram-pennyroyal` | `vllm-awq-w4a16` |
|---|---|---|---|---|---|
| Directory | `sglang/nvfp4-nvme/` | `sglang/nvfp4-ram/` | `sglang/nvfp4-ram-official/` | `sglang/nvfp4-ram-pennyroyal/` | `vllm/awq-w4a16/` |
| Engine / runtime | SGLang, local Docker image | SGLang, local Docker image | SGLang, `lmsysorg/sglang:dev-qwen38-next-local` | SGLang pennyroyal fork, native venv | vLLM, `vllm/vllm-openai:qwen38-flash-next` |
| Checkpoint | NVFP4 | NVFP4 | NVFP4 | NVFP4 | AWQ W4A16 |
| PLE table | streamed from NVMe | pinned RAM | pinned RAM | pinned RAM | RAM (BF16) |
| **KV cache** | 231,936 (FP8) | 498,624 (FP8) | 256,832 (BF16)¹ | **831,872 (FP8)** | 605,187 (BF16) |
| **Context window** | 262,144 | 262,144 | 262,144 | **524,288** (YaRN ×2) | 262,144 |
| TTFT 4K / 32K / 128K | 0.55 / 4.37 / 18.0 s | 0.31 / 2.90 / 10.7 s | 0.30 / 2.45 / 10.4 s | **0.27 / 2.62 / 9.68 s** | 0.36 / 3.28 / 12.1 s |
| Decode, warm | 214–222 tok/s | 236–249 | 258–260 | 235–254 | 102–103 |
| Needle test | not run | 12/12 to 220K | 12/12 to 220K | **15/15 to 492K** | 12/12 to 220K |
| Prefix cache across restarts | no | no | no | **yes** with HiCache (220K in 1.5 s) | no |
| Concurrency | 5 | 4 | 4 | 4 | 4 |
| VRAM in use | 91.3 GiB | 92.1–92.9 GiB | not recorded | 91.4–93.1 GiB | 87.1–90.5 GiB |
| Host RAM | ~0 | ~65 GB | ~65 GB | ~65 GB, +32 GB with HiCache | ~115 GB |

¹ As started by its script (4 requests, BF16 KV). The launcher's own defaults
reproduce the published cookbook cell (16 requests, 76,864 tokens). FP8 KV crashes
this image on long prompts.

Decode differences between the SGLang RAM profiles are inside the measurement
spread; context size, KV size, prefill and the needle results are solid.

**Which one:** for a single long-context agent, `sglang-nvfp4-ram-pennyroyal` has the
most room and the fastest prefill, but it applies YaRN to every prompt
(short-context quality not measured), runs outside Docker and depends on one
person's fork. `sglang-nvfp4-ram` is the conservative SGLang choice;
`sglang-nvfp4-ram-official` is the only one with no local patches or builds.
`vllm-awq-w4a16` decodes at less than half the speed but runs the other
quantization — BF16 activations and a BF16 PLE table — so it is the one to compare
output quality against.

## TODO

### HumanEval+ across the profiles

The four SGLang profiles run the same weights and differ in three runtime settings
that can affect quality; `vllm-awq-w4a16` runs a differently quantized checkpoint.
None of it has been measured — the needle tests check retrieval from long context,
not code quality.

| Profile | Experts | KV cache | Mamba SSM state | RoPE |
|---|---|---|---|---|
| `sglang-nvfp4-nvme` | NVFP4 W4A4 | FP8, uncalibrated (scale 1.0) | **FP32** (model default) | native |
| `sglang-nvfp4-ram` | NVFP4 W4A4 | FP8, uncalibrated | BF16 | native |
| `sglang-nvfp4-ram-official` | NVFP4 W4A4 | **BF16** | BF16 | native |
| `sglang-nvfp4-ram-pennyroyal` | NVFP4 W4A4 | FP8, uncalibrated | BF16 | **YaRN ×2 on every prompt** |
| `vllm-awq-w4a16` | **INT4 W4A16** (BF16 activations) | BF16 | model default | native |

- [ ] Run HumanEval+ (EvalPlus, 164 problems, greedy, base / plus pass@1) on the five profiles
- [ ] Non-thinking, temperature 0 — the conditions of the only published figure
- [ ] Optionally repeat with thinking (`reasoning_effort: xhigh`) for the best one or two
- [ ] Record results in `docs/comparison.md` and revise "Which one" above

Reference, from the yepapa-nest recipe's author (`sglang-nvfp4-nvme` configuration,
same card): **0.939 / 0.921** non-thinking, 0.957 / 0.927 thinking. Their harness
pinned the thinking mode and coerced null content, since thinking is on by default
and can leave `content` empty.

## Layout

```
qwen3.8-flash-next/
├── README.md
├── quant_info.py                 checkpoint precision report, run by every launcher
├── docs/
├── sglang/
│   ├── nvfp4-nvme/               launcher + stop
│   ├── nvfp4-ram/                launcher + stop
│   ├── nvfp4-ram-official/       launcher + stop
│   ├── nvfp4-ram-pennyroyal/     launcher + stop, build, NIXL build, host shim
│   ├── build-local-image/        Docker image for nvfp4-nvme and nvfp4-ram (vendored yepapa-nest recipe)
│   └── pennyroyal-fork/          fork checkout + venv, created by its build.sh (not tracked)
└── vllm/
    └── awq-w4a16/                launcher + stop
```

## Documentation

| Document | Covers |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation: prerequisites, checkpoints, each profile |
| [docs/comparison.md](docs/comparison.md) | Architecture, quantization, VRAM / RAM / NVMe footprint per SGLang profile, quality estimate from size, published test results, speed |
| [docs/vllm-awq.md](docs/vllm-awq.md) | The vLLM AWQ W4A16 profile: footprint, measurements, the single-GPU executor fix |
| [docs/ple-ram-experiment.md](docs/ple-ram-experiment.md) | How the SGLang RAM profiles were made to fit, with every result and failure |
| [docs/benchmarks.md](docs/benchmarks.md) | `sglang-nvfp4-nvme` measurements and memory breakdown |
| [docs/upstream-fixes.md](docs/upstream-fixes.md) | What we changed relative to the upstream recipe, and why |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Traps, error messages, and what to do about them |

## Sources

- NVMe recipe: https://github.com/yepapa-nest/qwen38-flashnext-rtx6000 (commit `2ef81d5`)
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- Fork: https://github.com/jpezzulli/sglang-rtxpro6000 (tag `pennyroyal-v2.5.0`)
- Model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
