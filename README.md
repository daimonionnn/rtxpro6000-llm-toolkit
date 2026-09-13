# rtxpro6000-llm-toolkit

Launch profiles, scripts and measurements for running large language models on a
**single NVIDIA RTX PRO 6000 Blackwell 96 GB**, with SGLang or vLLM.

Every profile serves an OpenAI-compatible API on `http://127.0.0.1:8090/v1`. Only
one profile can hold the GPU at a time; the scripts in `scripts/` start, stop and
report on them.

## Models

| Model | Profiles | Engines | Details |
|---|---|---|---|
| **Qwen3.8-Flash-Next** — 180B MoE, ~6B active, 262K context | 5 | SGLang, vLLM | [qwen3.8-flash-next/README.md](qwen3.8-flash-next/README.md) |

Each model directory has its own README with checkpoints, a comparison of its
profiles, measurements and open TODOs, and a `docs/` folder with setup,
troubleshooting and the full results.

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
turn, and works with any profile. See [docs/ui.md](docs/ui.md).

## Layout

```
.
├── README.md
├── scripts/                      start-<profile>.sh, stop.sh, status.sh, start-ui.sh, stop-ui.sh
│   └── _lib.sh                   the profile registry (PROFILES) shared by the scripts
├── common/                       stop-docker.sh, shared by every Docker profile
├── qwen3.8-flash-next/           one directory per model
│   ├── README.md
│   ├── docs/
│   ├── quant_info.py
│   ├── sglang/<variant>/         launcher + stop per profile
│   └── vllm/<variant>/
├── bench/prefill.py              prefill benchmark (cold vs prefix-cached), any engine
├── ui/                           local chat UI
├── docs/ui.md
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
4. Document it in the model's README and `docs/`.

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
