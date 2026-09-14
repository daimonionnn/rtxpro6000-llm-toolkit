# rtxpro6000-llm-toolkit

Launch profiles, scripts and measurements for running large language models on a
**single NVIDIA RTX PRO 6000 Blackwell 96 GB**, with SGLang, vLLM or ExLlamaV3.

Every profile serves an OpenAI-compatible API on `http://127.0.0.1:8090/v1`. Only
one profile can hold the GPU at a time; the scripts in `scripts/` start, stop and
report on them.

## Models

| Model | Profiles | Engines | Details |
|---|---|---|---|
| **Qwen3.8-Flash-Next** — 180B MoE, ~6B active, 262K context | 8 | SGLang, vLLM, ExLlamaV3 | [qwen3.8-flash-next/README.md](qwen3.8-flash-next/README.md) |
| **Qwen3.6-27B** — dense 27B, 262K context, BF16 reference | 1 | SGLang | [qwen3.6-27b/README.md](qwen3.6-27b/README.md) |

Each model directory has a README with its checkpoints, profiles and open TODOs;
the Flash-Next one also has `docs/` with setup, troubleshooting, the model's
architecture and one document per profile (`docs/profiles/<profile>.md`).

## Results

**[RESULTS.md](RESULTS.md)** puts every profile of every model side by side: speed,
context and memory, HumanEval+ / MBPP+, a blind Slovak check, and recommendations.
In short:

- **Most context and fastest prefill:** `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal`
  (524K window); `-sglang-nvfp4-ram` for plain Docker.
- **Best non-English output:** `qwen3.8-flash-next-vllm-awq-w4a16-g32`.
- **On code** the Flash-Next quantizations are within noise of each other, and all
  beat Qwen3.6-27B at BF16.
- **8-bit weights on one card:** ik_llama.cpp, not vLLM.

## Scripts

| Script | Does |
|---|---|
| `scripts/start-<model>-<engine>-<variant>.sh` | Start one profile (listed below) |
| `scripts/stop.sh` | Stop whichever profile is running (`--rm` also removes a Docker container) |
| `scripts/status.sh` | What is running, from which directory, with which checkpoint, context and KV cache |
| `scripts/start-ui.sh` / `scripts/stop-ui.sh` | Chat UI in the background on http://127.0.0.1:5173/ |

Current profiles:

| Start script | Profile |
|---|---|
| `start-qwen3.8-flash-next-sglang-nvfp4-nvme.sh` | SGLang, local Docker image, NVFP4, PLE table streamed from NVMe |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram.sh` | SGLang, local Docker image, NVFP4, PLE table in RAM |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-official.sh` | SGLang, official lmsysorg image, NVFP4, PLE table in RAM |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal.sh` | SGLang pennyroyal fork (native), NVFP4, PLE table in RAM, 524K context |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal-hicache.sh` | the same with HiCache/NIXL prefix persistence |
| `start-qwen3.8-flash-next-vllm-awq-w4a16.sh` | vLLM, official image, AWQ W4A16, PLE table in RAM |
| `start-qwen3.8-flash-next-vllm-awq-w4a16-g32.sh` | vLLM, official image, AWQ W4A16 group 32, PLE table in RAM |
| `start-qwen3.8-flash-next-exllamav3-exl3-5.05bpw.sh` | ExLlamaV3 via TabbyAPI, EXL3 5.05 bpw, n-gram table in RAM, MTP |
| `start-qwen3.8-flash-next-vllm-fp8-offload.sh` | vLLM, official image, official FP8, 50 GiB of experts and the PLE table in RAM |
| `start-qwen3.6-27b-sglang-bf16.sh` | Qwen3.6-27B · SGLang, official image, BF16, NEXTN speculation |

```bash
scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal.sh
scripts/status.sh
scripts/stop.sh
```

The start scripts call the launcher in the profile's directory, set `MODEL_DIR` to
its checkpoint under `models/`, and refuse to start while another profile is
running. Launcher variables can be overridden from the environment, e.g.
`CHUNKED=8192 scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram.sh`.

Docker profiles run as the container `rtxpro6000-llm` with `--restart
unless-stopped`: a server left running comes back after a reboot and takes most of
the VRAM. One stopped with `scripts/stop.sh` stays stopped.

> **Note:** reasoning models here have thinking **on by default**. With a small
> `max_tokens` the whole budget can go to `reasoning_content` and leave `content`
> empty; send `"chat_template_kwargs": {"enable_thinking": false}` for plain answers.

### Chat UI

```bash
scripts/start-ui.sh          # http://127.0.0.1:5173
```

Shows time-to-first-token, decode and prefill speed and token counts for every
turn, and works with any profile. See [ui/README.md](ui/README.md).

### Benchmarks

`bench/prefill.py` measures prefill (cold and prefix-cached), `bench/evalplus_*`
scores code ability (HumanEval+ / MBPP+) in a sandbox, and
`bench/language_samples.py` compares non-English output between profiles blind.
See [bench/README.md](bench/README.md).

## Layout

```
.
├── README.md
├── RESULTS.md                    all profiles side by side, benchmarks, recommendations
├── scripts/                      start-<profile>.sh, stop.sh, status.sh, start-ui.sh, stop-ui.sh
│   └── _lib.sh                   the profile registry (PROFILES) shared by the scripts
├── common/                       stop-docker.sh, shared by every Docker profile
├── qwen3.8-flash-next/           one directory per model
│   ├── README.md
│   ├── docs/                     model.md, setup, troubleshooting, profiles/<profile>.md
│   ├── quant_info.py
│   ├── sglang/<variant>/         launcher + stop per profile
│   ├── vllm/<variant>/
│   └── exllamav3/<variant>/
├── qwen3.6-27b/                  README.md, docs/profiles/, sglang/bf16/
├── bench/                        README.md, prefill.py, evalplus_*, language_samples.py — any engine
├── ui/                           README.md, local chat UI
├── models/                       checkpoints (not tracked)
└── logs/                         build and serve logs (not tracked)
```

## Adding a model or profile

1. Put the launcher in `<model>/<engine>/<variant>/`, with a `stop.sh` — for a
   Docker profile a two-line wrapper around `common/stop-docker.sh`. Run the
   container as `rtxpro6000-llm` with the label
   `rtxpro6000-llm.profile=<model>-<engine>-<variant>` so `status.sh` and `stop.sh`
   recognise it.
2. Add a line to `PROFILES` in `scripts/_lib.sh`: id, directory, launcher,
   checkpoint directory under `models/`, description.
3. Add `scripts/start-<id>.sh`: source `_lib.sh` and call `start_profile <id>`.
4. Document it in `<model>/docs/profiles/<engine>-<variant>.md`, add a row to the
   model's README, and its measurements to `RESULTS.md`.

## Hardware

- **NVIDIA RTX PRO 6000 Blackwell Workstation**, 96 GB, SM120 (compute capability
  12.0). The Server and Max-Q editions have the same memory and architecture.
- 244 GB RAM; checkpoints on NVMe
- An AMD Radeon AI PRO R9700 32 GB in the same machine. It is not used: CUDA and
  ROCm cannot share one tensor-parallel group. Its ROCm install does affect native
  (non-Docker) builds — see the Qwen3.8-Flash-Next troubleshooting notes.
- Driver 610.57.04, kernel 7.0.0-31, Docker 29.5.3, CUDA 13.3 and 13.4 toolkits

## License

Apache License 2.0 — see [LICENSE](LICENSE).

`qwen3.8-flash-next/sglang/build-local-image/` is a vendored copy of the
yepapa-nest recipe, also Apache-2.0, with its own `LICENSE` and the changes listed
in its `UPSTREAM.md`. Software fetched at build time — SGLang and vLLM images, the
pennyroyal fork, NIXL — and the model checkpoints are not part of this repository
and keep their own licenses.
