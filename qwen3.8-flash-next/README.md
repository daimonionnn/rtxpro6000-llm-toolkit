# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell 96 GB: 180B parameters, a MoE
with ~6B active, 262K native context. Most of the parameter count is a 51B N-gram
(PLE) lookup table that can live outside the GPU and 512 experts of which 10 run
per token, so it decodes at the speed of a small model and barely slows with
context. Architecture, checkpoints and published quality:
[docs/model.md](docs/model.md).

## Checkpoints

Checkpoints under `models/` at the toolkit root are shared by the profiles that
use them. Strata Q8 reuses the existing GGUF in the user's LM Studio model
directory; `--model` can point its launcher at another location.

| Checkpoint | Routed experts | Rest | PLE table | Size | Profiles |
|---|---|---|---|---|---|
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ `7b719225242a` | NVFP4 W4A4, 4.50 bits | BF16 | FP8 | 125.9 GiB | `sglang-nvfp4-*` |
| [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) @ `be794b99` — abliterated | NVFP4 W4A4, 4.50 bits | BF16 | FP8 | 125.9 GiB | `sglang-nvfp4-ram-official-abliterated` |
| [wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) @ `0939125b` | INT4 W4A16 g128, 4.13 bits | BF16 | BF16 | 168.3 GiB | `vllm-awq-w4a16` |
| [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) @ `d39638a0` | INT4 W4A16 g32 asymmetric, 4.63 bits | BF16 | BF16 | 175.3 GiB | `vllm-awq-w4a16-g32` |
| [leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) @ `fa561462` — uncensored | INT4 AWQ g32, zero points, 4.65 bits | BF16 | FP8 | 128.6 GiB | `vllm-awq-w4a16-g32-uncensored` |
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) `5.05bpw_h6_ng6` @ `7cef615f` | EXL3 5.05 bpw | EXL3 5.05 bpw, head 6 | 6 bpw | 114.6 GiB | `exllamav3-exl3-5.05bpw` |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) @ `236dfdf2` | FP8 W8A8, 128×128 blocks | BF16 | FP8 | 172.8 GiB | `vllm-fp8-offload` |
| [lmstudio-community/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF), Q8_0 | Q8_0, 8.50 bits including scales | mixed GGUF; small projections converted to BF16 | Q8_0, mmap | 175.30 GiB | `strata-q8`, `strata-q8-256k`, `strata-q8-vision*` |
| [lmstudio-community/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF) @ `158fc825df3e`, Q6_K | Q6_K gate/up + Q8_0 down, 7.21 bits including scales | mixed GGUF; small projections converted to BF16 | Q8_0, mmap | 156.13 GiB | `strata-q6`, `strata-q6-256k`, `strata-q6-vision*` |

Vision variants additionally share the original 907,542,592-byte BF16 mmproj
already installed under LM Studio. [Encoder preparation and usage](strata/README.md#vision-profiles).

## Profiles

The toolkit default (`scripts/start.sh`) is **Strata Q8 vision 128K**, profile
`qwen3.8-flash-next-strata-q8-vision`, with up to 4,096 image tokens. On the
workstation, `strata-server.service` starts it automatically at boot.

One at a time, each on `http://127.0.0.1:8090/v1` by default. All 18 profiles
start with `scripts/start-qwen3.8-flash-next-<engine>-<variant>-<context>k.sh` at the toolkit
root and are managed by `scripts/status.sh` and `scripts/stop.sh`.
SGLang, vLLM and TabbyAPI advertise model `Qwen3.8-Flash-Next`. Strata advertises
its full profile ID: `qwen3.8-flash-next-strata-q8` or `...-q6`, with `-256k`
for the larger-context variants. Vision variants insert `-vision` after the
quantization name, e.g. `qwen3.8-flash-next-strata-q8-vision-256k`. Strata launchers share the `strata/` directory
and [documentation](strata/README.md); other profiles are documented in
`docs/profiles/<engine>-<variant>.md`.

Script suffixes label defaults: `128k` = 131,072 tokens, `256k` = 262,144,
`512k` = 524,288 (Pennyroyal and its HiCache launcher). Context overrides remain
available. Profile IDs and advertised API model names are unchanged by the
filename convention.

| Profile | Runtime | Weights | PLE table | Context | Decode, tok/s | Host RAM |
|---|---|---|---|---|---|---|
| [`sglang-nvfp4-nvme`](docs/profiles/sglang-nvfp4-nvme.md) | SGLang, local image | NVFP4 W4A4 | streamed from NVMe | 262,144 | 214–222 | ~0 |
| [`sglang-nvfp4-ram`](docs/profiles/sglang-nvfp4-ram.md) | SGLang, local image | NVFP4 W4A4 | pinned RAM | 262,144 | 236–249 | ~65 GB |
| [`sglang-nvfp4-ram-official`](docs/profiles/sglang-nvfp4-ram-official.md) | SGLang, official image | NVFP4 W4A4 | pinned RAM | 262,144 | 258–260 | ~65 GB |
| [`sglang-nvfp4-ram-pennyroyal`](docs/profiles/sglang-nvfp4-ram-pennyroyal.md) | SGLang fork, native | NVFP4 W4A4 | pinned RAM | **524,288** | 235–254 | ~65 GB |
| [`sglang-nvfp4-ram-official-abliterated`](docs/profiles/sglang-nvfp4-ram-official-abliterated.md) | SGLang, official image | NVFP4 W4A4, abliterated | pinned RAM | 262,144 | 200–216 | ~65 GB |
| [`vllm-awq-w4a16`](docs/profiles/vllm-awq-w4a16.md) | vLLM preview image | INT4 W4A16 g128 | RAM | 262,144 | 102–103 | ~115 GB |
| [`vllm-awq-w4a16-g32`](docs/profiles/vllm-awq-w4a16-g32.md) | vLLM preview image | INT4 W4A16 g32 | RAM | 262,144 | 110 | ~115 GB |
| [`vllm-awq-w4a16-g32-uncensored`](docs/profiles/vllm-awq-w4a16-g32-uncensored.md) | vLLM preview image + PLE patch | INT4 AWQ g32, uncensored | RAM, FP8 | 262,144 | 110 | ~50 GB (est.) |
| [`exllamav3-exl3-5.05bpw`](docs/profiles/exllamav3-exl3-5.05bpw.md) | ExLlamaV3 / TabbyAPI | EXL3 5.05 bpw | RAM | 262,144 | **119 prose, 216 code** | ~43 GB |
| [`vllm-fp8-offload`](docs/profiles/vllm-fp8-offload.md) | vLLM preview image | FP8, 50 GiB of experts in RAM | RAM | 262,144 | 15.6–16.6 | ~131 GB |
| [`strata-q8`](strata/README.md#run) | Strata 0.1.39, native + Q8 PLE patch; MTP | Q8_0 | mmap / OS file cache | 131,072 | 101.7 prose, 103.5 code, 75.1 Slovak | 34.83 GiB pinned experts + PLE cache |
| [`strata-q6`](strata/README.md#strata-q6_kq8_0) | Strata 0.1.39, native + Q8/Q6 patches; MTP | Q6_K/Q8_0, 7.21 bpw | mmap / OS file cache | 131,072 | 117.1 prose, 142.5 code, 81.3 Slovak | 16.04 GiB pinned experts + PLE cache |
| [`strata-q8-256k`](strata/README.md#256k-profiles) | Strata 0.1.39, native + Q8 PLE patch; MTP | Q8_0 | mmap / OS file cache | 262,144 | 89.4 prose, 104.9 code, 73.1 Slovak | 36.75 GiB pinned experts + PLE cache |
| [`strata-q6-256k`](strata/README.md#256k-profiles) | Strata 0.1.39, native + Q8/Q6 patches; MTP | Q6_K/Q8_0, 7.21 bpw | mmap / OS file cache | 262,144 | 119.5 prose, 144.4 code, 93.2 Slovak | 17.95 GiB pinned experts + PLE cache |
| [`strata-q8-vision`](strata/README.md#vision-profiles) | Strata 0.1.39 + patches; BF16 GPU vision, MTP | Q8_0 | mmap / OS file cache | 131,072 | not benchmarked | 36.55 GiB pinned experts + PLE cache |
| [`strata-q8-vision-256k`](strata/README.md#vision-profiles) | Strata 0.1.39 + patches; BF16 GPU vision, MTP | Q8_0 | mmap / OS file cache | 262,144 | not benchmarked | 38.46 GiB pinned experts + PLE cache |
| [`strata-q6-vision`](strata/README.md#vision-profiles) | Strata 0.1.39 + patches; BF16 GPU vision, MTP | Q6_K/Q8_0, 7.21 bpw | mmap / OS file cache | 131,072 | not benchmarked | 17.75 GiB pinned experts + PLE cache |
| [`strata-q6-vision-256k`](strata/README.md#vision-profiles) | Strata 0.1.39 + patches; BF16 GPU vision, MTP | Q6_K/Q8_0, 7.21 bpw | mmap / OS file cache | 262,144 | not benchmarked | 19.66 GiB pinned experts + PLE cache |

KV pool, prefill, VRAM, code benchmarks and the Slovak check for every profile, with
recommendations: [RESULTS.md](../RESULTS.md).

Strata rates above are 2026-10-06 medians of three 512-token responses to short
prompts; its adaptive
expert cache warms during the test. Its RAM figures count only pinned experts,
not all process memory or the PLE table's OS file cache. Its configurations have
not had a new EvalPlus run or independently scored Slovak comparison.

**Which one, in short:**

- **Most context, fastest prefill:** `sglang-nvfp4-ram-pennyroyal` — a personal fork
  with YaRN on every prompt; `sglang-nvfp4-ram` or `-official` for plain Docker.
- **Best non-English output at interactive speed:** `vllm-awq-w4a16-g32` — first in
  three blind Slovak checks, level with FP8 in the two runs with FP8; the differences between profiles are small,
  and both dense 27B models at BF16 placed well below it.
- **Fastest single-stream decode with 16-bit activations:** `exllamav3-exl3-5.05bpw`.
- **8-bit weights:** `strata-q8` improves generation speed over ik Q8 in the
  2026-10-06 comparison; `strata-q6` uses a 7.21-bpw expert mixture and is faster
  for generation. Q8 has shorter latency in the initial 4–120K prefill sweep; Q6 is
  slightly faster at ~255K. The older Q8 quality
  reference scored 73 in Slovak and 454 of 542 code tasks in the separate public
  [ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit) setup;
  Strata's output quality has not been independently scored.
- **Without refusals:** `vllm-awq-w4a16-g32-uncensored` — the most code tests passed
  of any profile (needs a small vLLM patch). `sglang-nvfp4-ram-official-abliterated`
  lost measurable code ability to its abliteration.
- On code, the seven profiles with the original weights are within noise of each
  other.

## Strata Q8 / Q6 experiment

Registered launchers and local compatibility patches are in
[strata/README.md](strata/README.md). The [2026-10-06 comparison](strata/COMPARISON.md)
measures both engines with MTP and a 128K context:

| Configuration | Prose | Code | Slovak | Cold TTFT, ~120K |
|---|---:|---:|---:|---:|
| ik_llama.cpp Q8 | 50.9 tok/s | 55.9 tok/s | 46.8 tok/s | 94.7 s |
| Strata Q8 | 101.7 tok/s | 103.5 tok/s | 75.1 tok/s | 21.7 s |
| ik_llama.cpp Q6 | 60.9 tok/s | 68.5 tok/s | 60.0 tok/s | 87.0 s |
| Strata Q6 | 117.1 tok/s | 142.5 tok/s | 81.3 tok/s | 27.4 s |

Q6 uses Q6_K gate/up, Q8_0 down and a Q8_0 PLE table: 7.21 bits per routed
weight including scales. Strata's adaptive expert cache warms across requests.
The ten Slovak samples per configuration are not a quality score. All eight Strata launchers are registered; the compatibility patches remain
experimental.

All Strata profiles use 8,192-token automatic prefill chunks and INT8 KV.
Q8 has quantized MMQ expert prefill; Q6 currently dequantizes expert matrices
to FP16 before its batched products. The ik_llama.cpp reference engine and
launchers are maintained in
[ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit).

### 256K variants

Use `scripts/start-qwen3.8-flash-next-strata-q8-256k.sh` or
`...-q6-256k.sh` after preparing the engine and pack. Both allocate a native
262,144-token context with INT8 KV and MTP, using 8,192-token prefill chunks.

| Profile | Fresh TTFT, ~255K | Effective prefill | TG after ~120K | TG after ~255K |
|---|---:|---:|---:|---:|
| `strata-q8-256k` | 46.58 s | 5,471 tok/s | 127.4 tok/s | 117.8 tok/s |
| `strata-q6-256k` | 44.00 s | 5,791 tok/s | 139.9 tok/s | 153.1 tok/s |

Long TG is the median of three 512-token prose responses; the first prefix is
fresh and two repeats are cached. The word corpus, prior cache warming and
prompt differ from the short decode tests. Actual long prompts contain about
254,875 tokens, leaving room for output. Both completed without OOM.
[Full timings and memory](strata/COMPARISON.md#256k-context-profiles).

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

### Remaining benchmarks

- [x] Code benchmarks on all profiles except `vllm-fp8-offload`, which is skipped:
      ik_llama.cpp Q8_0 covers 8-bit weights (454 of 542, within noise)
- [x] Blind Slovak grading of every profile (`vllm-fp8-offload` in the two four-way
      runs), plus ik_llama.cpp Q8_0 as the 8-bit reference
- [ ] A larger Slovak prompt set or several samples per prompt, to separate profiles
      that are now within a few prompts of each other
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
│   ├── awq-w4a16-g32-uncensored/ launcher + stop, PLE patch, index filter
│   └── fp8-offload/              launcher + stop
├── exllamav3/
│   └── exl3-5.05bpw/             launcher + stop, TabbyAPI config template, sampler preset
└── strata/                      Q8/Q6 Python launchers, compatibility patches, comparison
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
| [strata/README.md](strata/README.md) | Q8/Q6 text and vision profiles, pinned build, compatibility patches, model preparation and launch commands |
| [strata/VISION.md](strata/VISION.md) | Image smoke checks, encoder and expert memory, lifecycle validation |
| [strata/COMPARISON.md](strata/COMPARISON.md) | Q8/Q6 versus ik_llama.cpp: decode, prefill, context, memory and validation |
| [../RESULTS.md](../RESULTS.md) | All profiles of all models side by side, benchmarks, recommendations |

## Sources

- NVMe recipe: https://github.com/yepapa-nest/qwen38-flashnext-rtx6000 (commit `2ef81d5`)
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- Fork: https://github.com/jpezzulli/sglang-rtxpro6000 (tag `pennyroyal-v2.5.0`)
- TabbyAPI: https://github.com/theroyallab/tabbyAPI; single-card benchmark of vLLM, EXL3 and SGLang: https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench
- Model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
