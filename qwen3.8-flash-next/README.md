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
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) @ `236dfdf2` | FP8 W8A8, 128×128 blocks | BF16 | FP8, 47.7 GiB | 172.8 GiB | `vllm-fp8-offload` |
| [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) @ `d39638a0` | INT4 W4A16, group 32 asymmetric, 4.63 bits | BF16 | BF16, 95.4 GiB | 175.3 GiB | `vllm-awq-w4a16-g32` |
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) `5.05bpw_h6_ng6` @ `7cef615f` | EXL3 5.05 bpw | EXL3 5.05 bpw, head 6 | 6 bpw, 36.4 GiB | 114.6 GiB | `exllamav3-exl3-5.05bpw` |

## Profiles

Only one can hold the GPU at a time; every one serves `http://127.0.0.1:8090/v1` as
model `Qwen3.8-Flash-Next`. A profile `<engine>-<variant>` lives in
`<engine>/<variant>/` and starts with `scripts/start-qwen3.8-flash-next-<engine>-<variant>.sh`
at the toolkit root.

| Profile | Runtime | Weights | PLE table | KV cache, tokens | Context | TTFT 4K / 32K / 128K | Decode, tok/s | Host RAM |
|---|---|---|---|---|---|---|---|---|
| `sglang-nvfp4-nvme` | SGLang, local image | NVFP4 W4A4 | streamed from NVMe | 231,936 FP8 | 262,144 | 0.55 / 4.37 / 18.0 s | 214–222 | ~0 |
| `sglang-nvfp4-ram` | SGLang, local image | NVFP4 W4A4 | pinned RAM | 498,624 FP8 | 262,144 | 0.31 / 2.90 / 10.7 s | 236–249 | ~65 GB |
| `sglang-nvfp4-ram-official` | SGLang, `lmsysorg/sglang:dev-qwen38-next-local` | NVFP4 W4A4 | pinned RAM | 256,832 BF16¹ | 262,144 | 0.30 / 2.45 / 10.4 s | 258–260 | ~65 GB |
| `sglang-nvfp4-ram-pennyroyal` | SGLang pennyroyal fork, native | NVFP4 W4A4 | pinned RAM | **831,872 FP8** | **524,288** (YaRN ×2) | **0.27 / 2.62 / 9.68 s** | 235–254 | ~65 GB, +32 GB with HiCache |
| `vllm-awq-w4a16` | vLLM, `vllm/vllm-openai:qwen38-flash-next` | INT4 W4A16 g128 | RAM, BF16 | 605,187 BF16 | 262,144 | 0.36 / 3.28 / 12.1 s | 102–103 | ~115 GB |
| `vllm-awq-w4a16-g32` | vLLM, same image | INT4 W4A16 g32 | RAM, BF16 | 315,039 BF16 | 262,144 | 0.78² / 3.36 / 12.1 s | 110 | ~115 GB |
| `exllamav3-exl3-5.05bpw` | ExLlamaV3 1.5.0 via TabbyAPI | EXL3 5.05 bpw | RAM, 6 bpw | 262,144 FP16 | 262,144 | 0.68 / 4.83 / 19.4 s | **119 prose, 216 code** (MTP) | ~43 GB |
| `vllm-fp8-offload` | vLLM, same image | FP8, 50 GiB of experts in RAM | RAM, FP8 | 313,483 BF16 | 262,144 | 8K 11.9 s · 69K 86.3 s | 15.6–16.6 | ~131 GB |

¹ As started by its script (4 requests, BF16 KV). The launcher's own defaults
reproduce the published cookbook cell (16 requests, 76,864 tokens). FP8 KV crashes
this image on long prompts.
² Prefix-cached; the first cold 4K request after startup took 1.83 s with warmup.

Also measured:

| Profile | Needle test | Prefix cache across restarts | VRAM in use | Details |
|---|---|---|---|---|
| `sglang-nvfp4-nvme` | not run | no | 91.3 GiB | [benchmarks.md](docs/benchmarks.md) |
| `sglang-nvfp4-ram` | 12/12 to 220K | no | 92.1–92.9 GiB | [ple-ram-experiment.md](docs/ple-ram-experiment.md) |
| `sglang-nvfp4-ram-official` | 12/12 to 220K | no | not recorded | [ple-ram-experiment.md](docs/ple-ram-experiment.md) |
| `sglang-nvfp4-ram-pennyroyal` | **15/15 to 492K** | **yes** with HiCache (220K in 1.5 s) | 91.4–93.1 GiB | [ple-ram-experiment.md](docs/ple-ram-experiment.md) |
| `vllm-awq-w4a16` | 12/12 to 220K | no | 87.1–90.5 GiB | [vllm-awq.md](docs/vllm-awq.md) |
| `vllm-awq-w4a16-g32` | 3/3 at 170K | no | not recorded | [vllm-awq-g32.md](docs/vllm-awq-g32.md) |
| `exllamav3-exl3-5.05bpw` | 3/3 at 170K | no | 92.0–93.4 GiB | [exllamav3-exl3.md](docs/exllamav3-exl3.md) |
| `vllm-fp8-offload` | 2/2 at 8K and 69K | no | ~88 GiB | [vllm-fp8-offload.md](docs/vllm-fp8-offload.md) |

Decode differences between the SGLang RAM profiles are inside the measurement
spread; context size, KV size, prefill and the needle results are solid.

## Code benchmarks

HumanEval+ (164 tasks) and MBPP+ (378) with `bench/evalplus_codegen.py` and
`bench/evalplus_evaluate.sh`, measured 2026-09-14: greedy, thinking off, the
EvalPlus prompts, scored in the official EvalPlus image. **Base** runs the original
tests, **plus** EvalPlus's stricter extended tests.

| Profile | HumanEval | HumanEval+ | MBPP | MBPP+ | Plus tests passed, of 542 |
|---|---|---|---|---|---|
| `sglang-nvfp4-nvme` | **0.982** | **0.963** | 0.923 | 0.791 | 457 |
| `vllm-awq-w4a16` | 0.976 | 0.951 | 0.934 | 0.788 | 454 |
| `vllm-awq-w4a16-g32` | 0.970 | 0.951 | **0.937** | 0.796 | 457 |
| `exllamav3-exl3-5.05bpw` | 0.976 | 0.957 | **0.937** | **0.802** | **460** |
| Qwen3.6-27B BF16 (`qwen3.6-27b-sglang-bf16`, for reference) | 0.976 | 0.927 | 0.931 | 0.778 | 446 |

**Comments**

- **The four quantizations are indistinguishable on code.** Compared task by task,
  `vllm-awq-w4a16-g32` and each of the others disagree on only 11–16 of 542 tasks,
  split about evenly (sign test p = 0.55–1.0). W4A4 NVFP4 did not measurably lose to
  the 16-bit-activation formats here.
- **All four beat the dense Qwen3.6-27B at full BF16 precision**, by 8–14 tasks:
  22–23 tasks only Flash-Next solves against 9–15 only the 27B solves (p = 0.02
  for EXL3, 0.08–0.26 for the others). The gap is in the stricter plus tests;
  base pass rates are close.
- These are this harness's numbers: on `sglang-nvfp4-nvme` it scores HumanEval
  0.982 / 0.963, against 0.939 / 0.921 published for the same configuration with a
  different harness. Compare rows within the table, not across sources.
- Code only, thinking off. Knowledge, long reasoning with thinking, and
  non-English output are not covered (for Slovak see
  [comparison.md](docs/comparison.md#non-english-spot-check-slovak)).

**Which one:**

- **Fastest, most context:** the NVFP4 SGLang profiles. For a single long-context
  agent, `sglang-nvfp4-ram-pennyroyal` has the most room and the fastest prefill,
  but it applies YaRN to every prompt (short-context quality not measured), runs
  outside Docker and depends on one person's fork. `sglang-nvfp4-ram` is the
  conservative SGLang choice; `sglang-nvfp4-ram-official` is the only one with no
  local patches or builds. W4A4 quantization is also the lossiest format here.
- **Higher-precision 4–5 bit, entirely on the GPU:** `exllamav3-exl3-5.05bpw` has the
  lowest published KL divergence of anything that fits, decodes fastest on a single
  stream (119 tok/s on prose, 216 on code with MTP) and loads in under a minute;
  prefill is ~1.6x slower than vLLM. `vllm-awq-w4a16-g32` and `vllm-awq-w4a16` keep
  attention and the shared experts in BF16 and prefill fastest of the 16-bit-
  activation profiles; group 32 costs no speed against group 128.
- **For 8-bit weights (FP8 / Q8_0) on one card, use ik_llama.cpp, not vLLM.**
  `vllm-fp8-offload` runs Qwen's own FP8 checkpoint, but with part of the experts in
  host RAM it decodes at ~16 tok/s and prefills at ~800 tok/s
  ([docs/vllm-fp8-offload.md](docs/vllm-fp8-offload.md)). A Q8_0 GGUF under
  ik_llama.cpp on the same machine, also computing the offloaded experts on the
  GPU, decodes at ~36–40 tok/s and prefills at ~1,650 tok/s. The vLLM profile stays
  for reference; it is not a practical backend.
- **Code quality does not separate the quantizations:** HumanEval+ and MBPP+ are
  within noise across NVFP4, both AWQ checkpoints and EXL3, and all four score above
  Qwen3.6-27B BF16 ([Code benchmarks](#code-benchmarks)).
- **Non-English output:** in a blind-graded ten-prompt Slovak check,
  `vllm-awq-w4a16-g32` matched the FP8 checkpoint, while `vllm-awq-w4a16` and
  `exllamav3-exl3-5.05bpw` trailed; all four share the model's own errors
  ([comparison.md](docs/comparison.md#non-english-spot-check-slovak)). A small
  sample — a direction, not a measurement.

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
- [ ] Record the result in `docs/vllm-fp8-offload.md`

### Code benchmarks on the remaining profiles

HumanEval+ and MBPP+ were run on four profiles ([Code benchmarks](#code-benchmarks)
above). Still open:

| Profile | Differs from the measured ones in |
|---|---|
| `sglang-nvfp4-ram` | Mamba SSM state BF16 instead of `sglang-nvfp4-nvme`'s FP32 |
| `sglang-nvfp4-ram-official` | BF16 KV cache instead of uncalibrated FP8 |
| `sglang-nvfp4-ram-pennyroyal` | YaRN ×2 applied to every prompt |
| `vllm-fp8-offload` | FP8 experts — the precision reference; ~16 tok/s makes a run take ~1 h |

- [ ] Run `bench/evalplus_codegen.py` + `bench/evalplus_evaluate.sh` on these
- [ ] Optionally repeat the best one or two with thinking on (`--thinking on`)

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
| [docs/setup.md](docs/setup.md) | Installation: prerequisites, checkpoints, each profile |
| [docs/comparison.md](docs/comparison.md) | Architecture, quantization, VRAM / RAM / NVMe footprint per SGLang profile, quality estimate from size, published test results, speed, Slovak spot check |
| [docs/vllm-awq.md](docs/vllm-awq.md) | The vLLM AWQ W4A16 profile: footprint, measurements, the single-GPU executor fix |
| [docs/vllm-awq-g32.md](docs/vllm-awq-g32.md) | The vLLM AWQ group-32 profile: checkpoint, footprint, measurements against group 128 |
| [docs/exllamav3-exl3.md](docs/exllamav3-exl3.md) | The ExLlamaV3 / TabbyAPI EXL3 profile: checkpoint, configuration, measurements, client-visible differences |
| [docs/vllm-fp8-offload.md](docs/vllm-fp8-offload.md) | The vLLM FP8 profile: expert offload to pinned RAM, measurements, why it is PCIe-bound |
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
- TabbyAPI: https://github.com/theroyallab/tabbyAPI; single-card EXL3 benchmark: https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench
