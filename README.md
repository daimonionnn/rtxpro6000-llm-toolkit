# sglang-rtxpro6000-toolkit

Serving large models with SGLang on a **single RTX PRO 6000 Blackwell 96 GB**.

Currently: Qwen3.8-Flash-Next (176B MoE, 6B active) with the RadixArk NVFP4
checkpoint and NEXTN speculative decoding. The 126 GiB checkpoint lives once in
`models/` and is shared by every launcher.

## Launchers

Four ways to run the same checkpoint, one directory each. Only one can hold the
GPU at a time, and every one serves `http://127.0.0.1:8090/v1` as model
`Qwen3.8-Flash-Next`. Start and stop them with the scripts in `scripts/`;
**`scripts/status.sh` shows which one is running and from where.**

| | NVMe baseline | Variant 1 | Variant 2 | Variant 3 |
|---|---|---|---|---|
| Directory | `sglang/v0-nvme/` | `sglang/v1-ram/` | `sglang/v2-official-image/` | `sglang/v3-pennyroyal/` |
| Launcher | `serve-nvfp4-nvme.sh` | `serve-nvfp4-ram.sh` | `serve-nvfp4-ram.sh` | `serve-nvfp4-ram.sh` |
| Runtime | Docker, locally built image | Docker, locally built image | Docker, stock `lmsysorg/sglang:dev-qwen38-next-local` | native venv, jpezzulli fork |
| PLE table (47.7 GiB) | streamed from NVMe | pinned RAM | pinned RAM | pinned RAM |
| **KV cache** | 231,936 (FP8) | 498,624 (FP8) | 256,832 (BF16)¹ | **831,872 (FP8)** |
| **Context window** | 262,144 | 262,144 | 262,144 | **524,288** (YaRN ×2) |
| TTFT 4K / 32K / 128K | 0.55 / 4.37 / 18.0 s | 0.31 / 2.90 / 10.7 s | 0.30 / 2.45 / 10.4 s | **0.27 / 2.62 / 9.68 s** |
| Decode, warm | 214–222 tok/s | 236–249 | 258–260 | 235–254 |
| Needle test | not run | 12/12 to 220K | 12/12 to 220K | **15/15 to 492K** |
| Prefix cache across restarts | no | no | no | **yes** with `HICACHE=1` (220K in 1.5 s) |
| Concurrency | 5 | 4 | 4 | 4 |
| Host RAM | ~0 | ~65 GB | ~65 GB | ~65 GB, +32 GB with HiCache |
| Stop with | `./stop.sh` in the same directory | same | same | same |

¹ With `MAXRUN=4 MAMBA_SLOTS=12
KV_DTYPE=auto`. The launcher's defaults reproduce the published cookbook cell
(16 requests, 76,864 tokens). FP8 KV crashes this image on long prompts.

Decode differences between the RAM variants are inside the measurement spread.
Context size, KV size, prefill and the needle results are solid. Full results,
settings and the reasoning behind them are in
[docs/comparison.md](docs/comparison.md) and
[docs/ple-ram-experiment.md](docs/ple-ram-experiment.md).

**Which one:** for a single long-context agent, Variant 3 has the most room by a
wide margin. But it applies YaRN to every prompt (short-context quality not yet
measured), runs outside Docker, and depends on one person's fork. Variant 1 is
the conservative choice with twice the baseline's KV. Variant 2 is the only one
with no local patches or builds.

## Quick start

All day-to-day control is in `scripts/`, named after the profiles:

| Script | Does |
|---|---|
| `scripts/start-v0-nvme.sh` | Start the NVMe baseline |
| `scripts/start-v1-ram.sh` | Start Variant 1 |
| `scripts/start-v2-official-image.sh` | Start Variant 2, tuned for one agent (4 requests, BF16 KV) |
| `scripts/start-v3-pennyroyal.sh` | Start Variant 3 |
| `scripts/start-v3-pennyroyal-hicache.sh` | Start Variant 3 with HiCache/NIXL persistence |
| `scripts/stop.sh` | Stop whichever variant is running (`--rm` also removes a Docker container) |
| `scripts/status.sh` | What is running, from which directory, with which context and KV cache |
| `scripts/start-ui.sh` / `scripts/stop-ui.sh` | Chat UI in the background on http://127.0.0.1:5173/ |

```bash
scripts/start-v3-pennyroyal-hicache.sh
scripts/status.sh
scripts/stop.sh
```

The start scripts only call the launcher in `sglang/<profile>/`. They set
`MODEL_DIR` to `models/Qwen3.8-Flash-Next-NVFP4` and refuse to start while another
variant is running. Every launcher variable can still be overridden, e.g.
`CHUNKED=8192 scripts/start-v1-ram.sh`.

Each launcher prints what runs at which precision before starting, refuses a
checkpoint that is not NVFP4, and waits until the server is ready. The Variant 3
launcher also refuses to start while another server holds the GPU.

The Docker launchers use `--restart unless-stopped`: a server left running comes
back after a reboot and takes ~92 GiB of VRAM. One stopped with `stop.sh` stays
stopped. Variant 3 never restarts by itself.

Smoke test:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Hello"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

> **Note:** thinking is **on by default** in every variant. With a small
> `max_tokens` the whole budget goes to `reasoning_content` and `content` comes
> back empty. See [docs/troubleshooting.md](docs/troubleshooting.md).

### Chat UI

```bash
scripts/start-ui.sh          # http://127.0.0.1:5173
```

Shows time-to-first-token, decode and prefill speed, and token counts for every
turn. Works with any launcher. See [docs/ui.md](docs/ui.md).

## TODO

### HumanEval+ across the launchers

All launchers run the same weights. They differ in three runtime settings that
can affect quality, and none of it has been measured — the needle tests check
retrieval from long context, not code quality.

| Launcher | KV cache | Mamba SSM state | RoPE |
|---|---|---|---|
| NVMe baseline | FP8, uncalibrated (scale 1.0) | **FP32** (model default) | native |
| Variant 1 | FP8, uncalibrated | BF16 | native |
| Variant 2c | **BF16** | BF16 | native |
| Variant 3 | FP8, uncalibrated | BF16 | **YaRN ×2 on every prompt** |

On paper Variant 2c has the most precise KV cache, the NVMe baseline the most
precise recurrent state, and Variant 3 the most risk at short context. Which
effect dominates is unknown.

- [ ] Run HumanEval+ (EvalPlus, 164 problems, greedy, base / plus pass@1) on those four
- [ ] Non-thinking, temperature 0 — the conditions of the only published figure
- [ ] Optionally repeat with thinking (`reasoning_effort: xhigh`) for the best one or two
- [ ] Record results in `docs/ple-ram-experiment.md` and revise "Which one" above

Reference, from the yepapa-nest recipe's author (NVMe baseline configuration,
same card): **0.939 / 0.921** non-thinking, 0.957 / 0.927 thinking. Their harness
pinned the thinking mode and coerced null content, since thinking is on by default
and can leave `content` empty. Estimated effort: 45–60 minutes including restarts.

## Quantization

The checkpoint is not uniformly 4-bit. The routed experts are NVFP4 W4A4; the
attention, shared experts, router and MTP draft are BF16; the PLE table is FP8.
KV cache precision is a launch setting: FP8 in Variants 1 and 3, BF16 in
Variant 2.

## Layout

```
.
├── README.md
├── scripts/                               start / stop / status per profile, chat UI start / stop
├── bench/prefill.py                       prefill benchmark (cold vs prefix-cached)
├── ui/                                    local chat UI with live TTFT / tok-s stats
├── docs/                                  documentation (below)
├── models/
│   └── Qwen3.8-Flash-Next-NVFP4/          126 GiB, 206 shards, shared by all launchers
├── sglang/
│   ├── v0-nvme/                           NVMe baseline: launcher, stop
│   ├── v1-ram/                            Variant 1: launcher, stop
│   ├── v2-official-image/                 Variant 2: launcher, stop
│   ├── v3-pennyroyal/                     Variant 3: launcher, stop, build, NIXL build, host shim
│   ├── build-local-image/                 builds the Docker image for v0 and v1 (yepapa-nest recipe clone)
│   ├── common/                            shared: quant_info.py, stop-docker.sh
│   └── pennyroyal-fork/                   Variant 3 source + venv (cannot be moved: absolute paths)
└── logs/                                  build and serve logs
```

## Documentation

| Document | Covers |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation from scratch: prerequisites, checkpoint, each launcher |
| [docs/comparison.md](docs/comparison.md) | All launchers side by side: architecture, quantization, VRAM / RAM / NVMe footprint per variant, quality estimate from size, published test results, speed |
| [docs/ple-ram-experiment.md](docs/ple-ram-experiment.md) | How the RAM variants were made to fit, with every result and failure |
| [docs/benchmarks.md](docs/benchmarks.md) | NVMe baseline measurements and memory breakdown |
| [docs/upstream-fixes.md](docs/upstream-fixes.md) | What we changed relative to the upstream recipe, and why |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Traps, error messages, and what to do about them |
| [docs/ui.md](docs/ui.md) | The chat UI: what each stat means, how to point it elsewhere |

## Hardware

- **NVIDIA RTX PRO 6000 Blackwell Workstation**, 96 GB, SM120 (compute cap 12.0)
- 244 GB RAM; checkpoint on NVMe (Samsung 990 PRO 4 TB)
- AMD Radeon AI PRO R9700 32 GB. **Not usable for this model:** CUDA and ROCm
  cannot share one TP group. Its ROCm install does affect Variant 3; see
  troubleshooting.
- Driver 610.57.04, kernel 7.0.0-31, Docker 29.5.3, CUDA 13.3 and 13.4 toolkits

## Sources

- NVMe recipe: https://github.com/yepapa-nest/qwen38-flashnext-rtx6000 (commit `2ef81d5`)
- Official recipe: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- Fork: https://github.com/jpezzulli/sglang-rtxpro6000 (tag `pennyroyal-v2.5.0`)
- Checkpoint: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 @ `7b719225242a`
