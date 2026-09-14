# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell 96 GB: 180B parameters, a MoE
with ~6B active, 262K native context. Most of the parameter count is a 51B N-gram
(PLE) lookup table that can live outside the GPU and 512 experts of which 10 run
per token, so it decodes at the speed of a small model and barely slows with
context. Architecture, checkpoints and published quality:
[docs/model.md](docs/model.md).

## Checkpoints

Each is kept once under `models/` at the toolkit root and shared by the profiles
that use it.

| Checkpoint | Routed experts | Rest | PLE table | Size | Profiles |
|---|---|---|---|---|---|
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ `7b719225242a` | NVFP4 W4A4, 4.50 bits | BF16 | FP8 | 125.9 GiB | `sglang-nvfp4-*` |
| [wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) @ `0939125b` | INT4 W4A16 g128, 4.13 bits | BF16 | BF16 | 168.3 GiB | `vllm-awq-w4a16` |
| [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) @ `d39638a0` | INT4 W4A16 g32 asymmetric, 4.63 bits | BF16 | BF16 | 175.3 GiB | `vllm-awq-w4a16-g32` |
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) `5.05bpw_h6_ng6` @ `7cef615f` | EXL3 5.05 bpw | EXL3 5.05 bpw, head 6 | 6 bpw | 114.6 GiB | `exllamav3-exl3-5.05bpw` |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) @ `236dfdf2` | FP8 W8A8, 128×128 blocks | BF16 | FP8 | 172.8 GiB | `vllm-fp8-offload` |

## Profiles

One at a time, each on `http://127.0.0.1:8090/v1` as model `Qwen3.8-Flash-Next`.
A profile `<engine>-<variant>` lives in `<engine>/<variant>/`, starts with
`scripts/start-qwen3.8-flash-next-<engine>-<variant>.sh` at the toolkit root, and
is documented in `docs/profiles/<engine>-<variant>.md`.

| Profile | Runtime | Weights | PLE table | Context | Decode, tok/s | Host RAM |
|---|---|---|---|---|---|---|
| [`sglang-nvfp4-nvme`](docs/profiles/sglang-nvfp4-nvme.md) | SGLang, local image | NVFP4 W4A4 | streamed from NVMe | 262,144 | 214–222 | ~0 |
| [`sglang-nvfp4-ram`](docs/profiles/sglang-nvfp4-ram.md) | SGLang, local image | NVFP4 W4A4 | pinned RAM | 262,144 | 236–249 | ~65 GB |
| [`sglang-nvfp4-ram-official`](docs/profiles/sglang-nvfp4-ram-official.md) | SGLang, official image | NVFP4 W4A4 | pinned RAM | 262,144 | 258–260 | ~65 GB |
| [`sglang-nvfp4-ram-pennyroyal`](docs/profiles/sglang-nvfp4-ram-pennyroyal.md) | SGLang fork, native | NVFP4 W4A4 | pinned RAM | **524,288** | 235–254 | ~65 GB |
| [`vllm-awq-w4a16`](docs/profiles/vllm-awq-w4a16.md) | vLLM preview image | INT4 W4A16 g128 | RAM | 262,144 | 102–103 | ~115 GB |
| [`vllm-awq-w4a16-g32`](docs/profiles/vllm-awq-w4a16-g32.md) | vLLM preview image | INT4 W4A16 g32 | RAM | 262,144 | 110 | ~115 GB |
| [`exllamav3-exl3-5.05bpw`](docs/profiles/exllamav3-exl3-5.05bpw.md) | ExLlamaV3 / TabbyAPI | EXL3 5.05 bpw | RAM | 262,144 | **119 prose, 216 code** | ~43 GB |
| [`vllm-fp8-offload`](docs/profiles/vllm-fp8-offload.md) | vLLM preview image | FP8, 50 GiB of experts in RAM | RAM | 262,144 | 15.6–16.6 | ~131 GB |

KV pool, prefill, VRAM, code benchmarks and the Slovak check for every profile, with
recommendations: [RESULTS.md](../RESULTS.md).

**Which one, in short:**

- **Most context, fastest prefill:** `sglang-nvfp4-ram-pennyroyal` — a personal fork
  with YaRN on every prompt; `sglang-nvfp4-ram` or `-official` for plain Docker.
- **Best non-English output:** `vllm-awq-w4a16-g32`, level with FP8 in a blind
  Slovak check.
- **Fastest single-stream decode with 16-bit activations:** `exllamav3-exl3-5.05bpw`.
- **8-bit weights:** use ik_llama.cpp with a Q8_0 GGUF, not `vllm-fp8-offload`.
- On code, all measured quantizations are within noise of each other.

## TODO

### Why ik_llama.cpp moves offloaded experts faster than vLLM

Both keep part of the routed experts in host RAM and compute them on the GPU, but
ik_llama.cpp's Q8_0 decodes ~2.3x faster than `vllm-fp8-offload` with a comparable
amount of expert weight offloaded.

- [ ] Run the ik_llama.cpp Q8_0 profile and sample `nvidia-smi dmon -s t` (PCIe
      rx/tx) and CPU load during decode and prefill, as was done for `vllm-fp8-offload`
      (~23 GB/s, ~1.7 GB per token at `OFFLOAD_GIB=60`)
- [ ] Compare bytes crossing PCIe per token; confirm whether the gap is vLLM's
      in-place UVA reads versus ik_llama's copy of the selected experts
- [ ] Record the result in `docs/profiles/vllm-fp8-offload.md`

### Benchmarks on the remaining profiles

| Profile | Differs from the measured ones in |
|---|---|
| `sglang-nvfp4-ram` | SSM state BF16 instead of `sglang-nvfp4-nvme`'s FP32 |
| `sglang-nvfp4-ram-official` | BF16 KV cache instead of uncalibrated FP8 |
| `sglang-nvfp4-ram-pennyroyal` | YaRN ×2 applied to every prompt |
| `vllm-fp8-offload` | FP8 experts; ~16 tok/s makes a code run take ~1 h |

- [ ] Code benchmarks (`bench/evalplus_codegen.py` + `bench/evalplus_evaluate.sh`) on these
- [ ] Slovak samples (`bench/language_samples.py`) on the SGLang profiles
- [ ] Optionally the best one or two with thinking on (`--thinking on`)

## Layout

```
qwen3.8-flash-next/
├── README.md
├── quant_info.py                 checkpoint precision report, run by every launcher
├── docs/
│   ├── model.md                  architecture, checkpoints, quantization, published results
│   ├── setup.md                  installation, per profile
│   ├── troubleshooting.md
│   ├── upstream-fixes.md
│   ├── ple-ram-experiment.md     how the SGLang RAM profiles came about (history)
│   └── profiles/<profile>.md     one per profile
├── sglang/
│   ├── nvfp4-nvme/               launcher + stop
│   ├── nvfp4-ram/                launcher + stop
│   ├── nvfp4-ram-official/       launcher + stop
│   ├── nvfp4-ram-pennyroyal/     launcher + stop, build, NIXL build, host shim
│   ├── build-local-image/        Docker image for nvfp4-nvme and nvfp4-ram (vendored yepapa-nest recipe)
│   └── pennyroyal-fork/          fork checkout + venv, created by its build.sh (not tracked)
├── vllm/
│   ├── awq-w4a16/                launcher + stop
│   ├── awq-w4a16-g32/            launcher + stop
│   └── fp8-offload/              launcher + stop
└── exllamav3/
    └── exl3-5.05bpw/             launcher + stop, TabbyAPI config template, sampler preset
```

## Documentation

| Document | Covers |
|---|---|
| [docs/model.md](docs/model.md) | Architecture; what each checkpoint keeps at which precision; runtime settings that can change output; published quality results; alternatives considered |
| [docs/setup.md](docs/setup.md) | Prerequisites, checkpoint downloads, starting each profile |
| [docs/profiles/](docs/profiles/) | One document per profile: configuration, where the model lives in VRAM / RAM / disk, measurements, caveats |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Traps, error messages, and what to do about them |
| [docs/upstream-fixes.md](docs/upstream-fixes.md) | What the local SGLang image changes relative to the upstream recipe, and why |
| [docs/ple-ram-experiment.md](docs/ple-ram-experiment.md) | How the SGLang RAM profiles were made to fit, with every result and failure |
| [../RESULTS.md](../RESULTS.md) | All profiles of all models side by side, benchmarks, recommendations |

## Sources

- NVMe recipe: https://github.com/yepapa-nest/qwen38-flashnext-rtx6000 (commit `2ef81d5`)
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- Fork: https://github.com/jpezzulli/sglang-rtxpro6000 (tag `pennyroyal-v2.5.0`)
- TabbyAPI: https://github.com/theroyallab/tabbyAPI; single-card benchmark of vLLM, EXL3 and SGLang: https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench
- Model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
