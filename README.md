# rtxpro6000-llm-toolkit

Launch profiles, scripts and measurements for running large language models on a
**single NVIDIA RTX PRO 6000 Blackwell 96 GB**, with SGLang, vLLM, ExLlamaV3 or
Strata — and llama.cpp, ik_llama.cpp, Xolotl and Strata's DeepSeek engine for DeepSeek-V4.1-Flash.

> **Host RAM matters as much as the GPU.** This machine has 256 GB of RAM
> (4 × 64 GB DDR5-6400, 244 GiB usable). The Qwen3.8-Flash-Next profiles use from
> almost none to ~131 GB of it (Strata 16–38 GiB of pinned experts plus file cache,
> SGLang ~65 GB, vLLM 115–131 GB; per profile in [RESULTS.md](RESULTS.md)). The
> DeepSeek-V4.1-Flash profiles at 3–8-bit quality need nearly all of it: the model is
> 485–508 GB, the 96 GB card holds only a fraction of its experts, and 195–225 GB of
> experts live in RAM (Strata's UD-Q8_K_XL leaves about 20–35 GB for everything else).
> An RTX PRO 6000 with a typical 64 GB of RAM cannot run them: they either fail to start
> or read the experts from the SSD for every token, which is far too slow to use. The
> ~2-bit sets are smaller: Xolotl XQ v2 (152 GB) pins a 72 GiB RAM arena, and Strata with
> ISTA-DASLab's 2.25 bpw set keeps about 140 GB of RAM busy. Neither was tried with less
> RAM. See [Hardware](#hardware).

The default profile is **Strata Q8 vision 128K**
(`qwen3.8-flash-next-strata-q8-vision`), launched by `scripts/start.sh`.
It uses a 131,072-token context and up to 4,096 image tokens. On this workstation,
`strata-server.service` also starts it automatically at boot.

By default every profile serves an OpenAI-compatible API on
`http://127.0.0.1:8090/v1`. Only one profile can hold the GPU at a time. The scripts in `scripts/` manage the
registered profiles, including Strata Q8/Q6 at 128K and 256K. Strata runs in
the foreground; Ctrl+C or `scripts/stop.sh` stops it. See [Strata setup](qwen3.8-flash-next/strata/README.md).

## Models

| Model | Profiles | Engines | Details |
|---|---|---|---|
| **Qwen3.8-Flash-Next** — 180B MoE, ~6B active, 262K native context | 18 | SGLang, vLLM, ExLlamaV3, Strata | [qwen3.8-flash-next/README.md](qwen3.8-flash-next/README.md) |
| **Qwen3.6-27B** — dense 27B, 262K context, BF16 reference | 1 | SGLang | [qwen3.6-27b/README.md](qwen3.6-27b/README.md) |
| **Qwen3.8-27B** — dense 27B, 262K context, BF16 reference | 1 | SGLang | [qwen3.8-27b/README.md](qwen3.8-27b/README.md) |
| **DeepSeek-V4.1-Flash-1017** — 552B MoE, 16B active for output, Engram on SSD | 14 | llama.cpp PR, ik_llama.cpp, Xolotl, Strata (native) | [deepseek-v4.1-flash/README.md](deepseek-v4.1-flash/README.md) |

Each model directory has a README with its checkpoints and profiles, and one
document per profile in `docs/profiles/<profile>.md`; the Flash-Next one also has
setup, troubleshooting, the model's architecture and open TODOs. The ten Strata
Qwen3.8 profiles share [their setup and comparison](qwen3.8-flash-next/strata/README.md);
the DeepSeek ones are in [deepseek-v4.1-flash/README.md](deepseek-v4.1-flash/README.md#strata-ud-q8_k_xl).

## Results

**[RESULTS.md](RESULTS.md)** compares the models on one page: recommendations, one table
per model (speed, memory, code, Slovak) and the quality checks that ran across models. The
details are with each model: [Qwen3.8-Flash-Next](qwen3.8-flash-next/docs/results.md),
[DeepSeek-V4.1-Flash](deepseek-v4.1-flash/README.md), [Qwen3.6-27B](qwen3.6-27b/README.md),
[Qwen3.8-27B](qwen3.8-27b/README.md). In short:

- **The default and the best everyday model: Qwen3.8-Flash-Next on Strata**
  (`qwen3.8-flash-next-strata-q8-vision`, started by `scripts/start.sh`). Q8_0 weights,
  so close to full quality, and still fast: 159 tok/s on prose, 176 on code and 122 in
  Slovak with MTP, about 6,700 tok/s cold prefill at 32–120K, image input, 128K context
  (256K profile). GSM8K 97.0 %, MMLU-Pro 78.5 %, HumanEval+ 0.951 (measured on Strata
  0.1.41). It needs about 37 GiB of RAM beside the card and suits agents that make
  many calls.
- **Second: DeepSeek-V4.1-Flash on Strata**, two profiles. `deepseek-v4.1-flash-strata-ud-q8kxl`
  runs the experts as DeepSeek released them (MXFP4) and writes the best Slovak measured
  here (86.5 against 63–70 for the best Qwen3.8 setups, one grader); GSM8K 98.3 %,
  MMLU-Pro 81.0 %; ~29 tok/s, ~1,000 tok/s prefill at 30K, ~224 GB of RAM.
  `deepseek-v4.1-flash-strata-ist-iq2xs` uses ISTA-DASLab's official 2.25 bpw set:
  40–59 tok/s, 2,118 tok/s prefill at 30K, Slovak 84, ~140 GB of RAM. Both decode three
  to six times slower than Qwen3.8, so they are not the everyday agent model, but suit
  coding sessions and complex, demanding tasks where the answer matters more than the
  wait. Both take images and up to 256K tokens (UD-Q8_K_XL also 512K). Published
  results favour DeepSeek on agentic coding: DeepSWE 1.1 77.0 for V4.1-Flash-1017
  ([DeepSeek, self-reported](https://kingy.ai/blog/deepseek-v4-1-flash-1017-specs-benchmarks-pricing/))
  against 58.7 for Qwen3.8-Flash-Next ([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)),
  and [BenchLM](https://benchlm.ai/compare/deepseek-v4-1-flash-vs-qwen3-8-flash-next)'s
  coding score 61.1 against 51.8; Qwen3.8 leads there on agentic tasks overall (64.7 against
  61.7). Those runs use full-precision weights, thinking and agent harnesses; the local
  checks here use these quantizations with thinking off, and on short single-function
  HumanEval+ tasks DeepSeek scored lower (0.896 against 0.951; MBPP+ pending).
- **Slovak stands in for non-English languages.** It is a hard case: six cases, three
  genders, numeral agreement that changes the noun's case ("dve ženy", "päť žien"), verb
  aspect, and only about five million speakers, so little training data, with Czech close
  enough that weak setups drift into it. Quantization damage shows here first (wrong case
  endings, invented idioms, Czech words), so the blind Slovak check is a sensitive test
  for smaller, morphologically rich languages, European ones in particular. It says less
  about non-Latin scripts (tokenization) or agglutinative languages such as Finnish or
  Turkish, and it is small: ten prompts, one greedy answer each, an LLM grader.
- **Fastest prefill and the largest Qwen3.8 context:** `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal`
  (2.6 s at 32K, 9.7 s at 128K; 524K window, 831K-token KV pool, NVFP4 weights);
  `-sglang-nvfp4-ram` for plain Docker.
- **No refusals:** `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` — no measurable
  loss on code, a few points below g32 in Slovak; dealignai's abliterated NVFP4 lost
  code ability. `strata-q8-abliterated-vision` is the 8-bit option on Strata.
- **Dense Qwen3.6-27B and Qwen3.8-27B at BF16** trail Flash-Next on code and
  clearly in Slovak; 3.8 is no measurable step over 3.6.

### Strata

[Strata](https://github.com/Niko1221/Strata) serves both main models from one checkout
(0.2.7): Qwen3.8-Flash-Next with its Qwen engine and DeepSeek-V4.1-Flash with its own
DeepSeek engine (`strata-dsv41`). Both take images: Qwen3.8 through a GPU encoder with the
BF16 mmproj (up to 4,096 image tokens), DeepSeek through the engine's own BF16 vision
tower. Setup, profiles and measurements: [Qwen3.8 on Strata](qwen3.8-flash-next/strata/README.md),
[DeepSeek on Strata](deepseek-v4.1-flash/README.md#strata-ud-q8_k_xl).

## Scripts

| Script | Does |
|---|---|
| `scripts/start.sh` | Start the default profile: Strata Q8 vision 128K (`qwen3.8-flash-next-strata-q8-vision`) |
| `scripts/start-<model>-<engine>-<variant>-<context>k.sh` | Start one profile with its labelled default context (listed below) |
| `scripts/stop.sh` | Stop the running registered profile (`--rm` also removes a Docker container) |
| `scripts/status.sh` | Which registered profile is running, from which directory, with which checkpoint, context and KV cache |
| `scripts/start-ui.sh` / `scripts/stop-ui.sh` | Chat UI in the background on http://127.0.0.1:5173/ |

Current profiles:

Context suffixes use 1K = 1,024 tokens: `-128k` = 131,072, `-256k` = 262,144,
and `-512k` = 524,288 tokens. They describe the launch defaults; explicit
`CTX`, `CONTEXT_LENGTH` or Strata `--context` overrides still apply.

| Launch script or command | Profile |
|---|---|
| `start-qwen3.8-flash-next-sglang-nvfp4-nvme-256k.sh` | SGLang, local Docker image, NVFP4, PLE table streamed from NVMe |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-256k.sh` | SGLang, local Docker image, NVFP4, PLE table in RAM |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-official-256k.sh` | SGLang, official lmsysorg image, NVFP4, PLE table in RAM |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated-256k.sh` | the same launcher with dealignai's abliterated NVFP4 |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal-512k.sh` | SGLang pennyroyal fork (native), NVFP4, PLE table in RAM, 524K context |
| `start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal-hicache-512k.sh` | the same with HiCache/NIXL prefix persistence |
| `start-qwen3.8-flash-next-vllm-awq-w4a16-256k.sh` | vLLM, official image, AWQ W4A16, PLE table in RAM |
| `start-qwen3.8-flash-next-vllm-awq-w4a16-g32-256k.sh` | vLLM, official image, AWQ W4A16 group 32, PLE table in RAM |
| `start-qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored-256k.sh` | vLLM, official image + FP8 PLE patch, leoncca's uncensored AWQ W4A16 group 32 |
| `start-qwen3.8-flash-next-exllamav3-exl3-5.05bpw-256k.sh` | ExLlamaV3 via TabbyAPI, EXL3 5.05 bpw, n-gram table in RAM, MTP |
| `start-qwen3.8-flash-next-vllm-fp8-offload-256k.sh` | vLLM, official image, official FP8, 50 GiB of experts and the PLE table in RAM |
| [start-qwen3.8-flash-next-strata-q8-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-128k.sh) | Strata Q8_0, native engine, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-128k.sh](scripts/start-qwen3.8-flash-next-strata-q6-128k.sh) | Strata Q6_K/Q8_0, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-256k.sh](scripts/start-qwen3.8-flash-next-strata-q8-256k.sh) | Strata Q8_0, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-256k.sh](scripts/start-qwen3.8-flash-next-strata-q6-256k.sh) | Strata Q6_K/Q8_0, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-vision-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh) | Strata Q8, BF16 GPU vision, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-vision-256k.sh](scripts/start-qwen3.8-flash-next-strata-q8-vision-256k.sh) | Strata Q8, BF16 GPU vision, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-vision-128k.sh](scripts/start-qwen3.8-flash-next-strata-q6-vision-128k.sh) | Strata Q6, BF16 GPU vision, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-vision-256k.sh](scripts/start-qwen3.8-flash-next-strata-q6-vision-256k.sh) | Strata Q6, BF16 GPU vision, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-abliterated-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-abliterated-128k.sh) | Strata, abliterated Q8_0 (windowsxp811203), MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-abliterated-vision-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-abliterated-vision-128k.sh) | Strata, abliterated Q8_0, BF16 GPU vision, MTP, 128K context; foreground |
| `start-qwen3.6-27b-sglang-bf16-256k.sh` | Qwen3.6-27B · SGLang, official image, BF16, NEXTN speculation |
| `start-qwen3.8-27b-sglang-bf16-256k.sh` | Qwen3.8-27B · SGLang, official image, BF16, NEXTN speculation |
| [start-deepseek-v4.1-flash-llamacpp-ud-q3kxl-128k.sh](scripts/start-deepseek-v4.1-flash-llamacpp-ud-q3kxl-128k.sh) | DeepSeek-V4.1-Flash-1017 · llama.cpp PR #30817, Unsloth UD-Q3_K_XL; background, `HOST=0.0.0.0` for LAN (**checkpoint not on disk**; download it again to run) |
| [start-deepseek-v4.1-flash-ik-mxfp4-128k.sh](scripts/start-deepseek-v4.1-flash-ik-mxfp4-128k.sh) | DeepSeek-V4.1-Flash-1017 · ik_llama.cpp, original MXFP4 experts; background, `HOST=0.0.0.0` for LAN (**checkpoint not on disk**; download it again to run) |
| [start-deepseek-v4.1-flash-xolotl-xq-v2-128k.sh](scripts/start-deepseek-v4.1-flash-xolotl-xq-v2-128k.sh) | DeepSeek-V4.1-Flash-1017 · Xolotl engine, IndrasMirror XQ v2, exact routing; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-xolotl-xq-v2-vision-128k.sh](scripts/start-deepseek-v4.1-flash-xolotl-xq-v2-vision-128k.sh) | DeepSeek-V4.1-Flash-1017 · Xolotl engine, IndrasMirror XQ v2, exact routing, image input; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-128k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-128k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, 128K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-256k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-256k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, 256K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-512k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-512k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, 512K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-128k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-128k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, image input, 128K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-256k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-256k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, image input, 256K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-512k.sh](scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-512k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, Unsloth UD-Q8_K_XL, DSpark, image input, 512K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ist-iq2xs-128k.sh](scripts/start-deepseek-v4.1-flash-strata-ist-iq2xs-128k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, ISTA-DASLab IQ2_XS 2.25 bpw, DSpark, 128K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ist-iq2xs-256k.sh](scripts/start-deepseek-v4.1-flash-strata-ist-iq2xs-256k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, ISTA-DASLab IQ2_XS 2.25 bpw, DSpark, 256K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ist-iq2xs-vision-128k.sh](scripts/start-deepseek-v4.1-flash-strata-ist-iq2xs-vision-128k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, ISTA-DASLab IQ2_XS 2.25 bpw, DSpark, image input, 128K; background, `HOST=0.0.0.0` for LAN |
| [start-deepseek-v4.1-flash-strata-ist-iq2xs-vision-256k.sh](scripts/start-deepseek-v4.1-flash-strata-ist-iq2xs-vision-256k.sh) | DeepSeek-V4.1-Flash-1017 · Strata DeepSeek engine, ISTA-DASLab IQ2_XS 2.25 bpw, DSpark, image input, 256K; background, `HOST=0.0.0.0` for LAN |

```bash
scripts/start.sh                     # the default profile, on 0.0.0.0 (LAN)
scripts/start.sh --host 127.0.0.1    # the same, local only
scripts/status.sh
scripts/stop.sh
```

### Start the default model at boot

The workstation uses the user service
[strata-server.service](common/systemd/strata-server.service), which calls
`scripts/start.sh --host 0.0.0.0` (since 2026-10-21 also the script's own default). It starts Q8 vision 128K on port 8090,
accessible locally and from the LAN, with a 4,096-token image limit.
On this workstation the LAN API URL is `http://192.168.1.101:8090/v1`.
User lingering starts the service at boot without
requiring an interactive login. The former `ik-llama-server.service` is disabled.

To install on a checkout at `~/development/rtxpro6000-llm-toolkit`, after
preparing Strata and the vision encoder:

```bash
mkdir -p ~/.config/systemd/user
ln -s "$PWD/common/systemd/strata-server.service" ~/.config/systemd/user/strata-server.service
systemctl --user disable --now ik-llama-server.service
scripts/stop.sh
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now strata-server.service
```

If the checkout is elsewhere, adjust the unit's `WorkingDirectory` and
`ExecStart`. Manage the service and inspect startup logs with:

```bash
systemctl --user status strata-server.service
journalctl --user -u strata-server.service -f
systemctl --user restart strata-server.service
```

Before switching to another profile, stop the service with
`systemctl --user stop strata-server.service`, then run the desired launcher.
Use `systemctl --user disable --now strata-server.service` to turn off boot
startup. The unit restarts on failure; an explicit service stop stays stopped.

The `start-*.sh` scripts call the profile launcher and refuse to start while
another registered profile is running. Docker and SGLang launchers set
`MODEL_DIR` to the checkpoint under `models/`. Launcher variables can be
overridden from the environment, e.g.
`CHUNKED=8192 scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-256k.sh`.

Run Strata after preparing the engine and model pack. Its shell wrappers pass
`--port`, `--model`, `--context`, `--prefill` and other options to the Python
launcher. Direct launches default to localhost port 8090; `--host 0.0.0.0`
enables LAN access. The boot service uses this option. Context is 131,072 tokens, or 262,144 tokens
for the `-256k` wrappers. `scripts/status.sh` and `scripts/stop.sh` also manage
Strata started directly through its Python launchers; runtime state lives in
`logs/strata-server.json`.

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

`bench/compare_decode.py` measures sequential decode on prose, code and Slovak
prompts. `bench/prefill.py` measures prefill (cold and prefix-cached), `bench/evalplus_*`
scores code ability (HumanEval+ / MBPP+) in a sandbox, and
`bench/language_samples.py` compares non-English output between profiles blind.
See [bench/README.md](bench/README.md).

## Layout

```
.
├── README.md
├── RESULTS.md                    the models compared: recommendations, one table per model
├── scripts/                      start-<profile>-<context>k.sh, start.sh, stop.sh, status.sh, UI scripts
│   └── _lib.sh                   the profile registry (PROFILES) shared by the scripts
├── common/                       stop-docker.sh, shared by every Docker profile
├── qwen3.8-flash-next/           one directory per model
│   ├── README.md
│   ├── docs/                     model.md, setup, troubleshooting, profiles/<profile>.md
│   ├── quant_info.py
│   ├── sglang/<variant>/         launcher + stop per profile
│   ├── vllm/<variant>/
│   ├── exllamav3/<variant>/
│   └── strata/                  Q8/Q6 Python launchers, historical 0.1.39 patches, comparison
├── qwen3.6-27b/                  README.md, docs/profiles/, sglang/bf16/
├── qwen3.8-27b/                  README.md, docs/profiles/, sglang/bf16/
├── deepseek-v4.1-flash/          README.md, native/ (launcher, stop, expert mlock helper), xolotl/ (XQ launcher, vision variant, encoder patch), strata/ (UD-Q8_K_XL launcher)
├── bench/                        decode, prefill, EvalPlus, language samples — any engine
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
   checkpoint directory under `models/`, description, default context in tokens.
3. Add `scripts/start-<id>-<context>k.sh`: source `_lib.sh` and call
   `start_profile <id>`. If the ID already ends with the context suffix,
   use it once. `profile_field <id> script` resolves the wrapper filename.
4. Document it in `<model>/docs/profiles/<engine>-<variant>.md`, add a row to the
   model's README, its detailed measurements to the model's README or `docs/`, and a
   row to its table in `RESULTS.md`.

## Hardware

- **NVIDIA RTX PRO 6000 Blackwell Workstation**, 96 GB, SM120 (compute capability
  12.0). The Server and Max-Q editions have the same memory and architecture.
- 256 GB RAM: 4 × 64 GB non-ECC DDR5 at 6400 MT/s CL42 (244 GiB usable). The
  DeepSeek-V4.1-Flash profiles need 195–225 GB of it for experts (llama.cpp UD-Q3_K_XL
  locks 195 GB, ik_llama.cpp MXFP4 217 GB, Strata UD-Q8_K_XL sizes a 224 GB RAM store;
  Xolotl XQ v2 pins a 72 GiB arena and reads the 203 GB Engram from the SSD; Strata with the
  ISTA-DASLab 2.25 bpw set uses about 140 GB). With 64 GB
  of RAM none of them is usable. The Qwen3.8-Flash-Next profiles need less (see
  "Host RAM" in [RESULTS.md](RESULTS.md); several run with 64 GB).
- Checkpoints on NVMe; DeepSeek reads its Engram table from the SSD for every token
- An AMD Radeon AI PRO R9700 32 GB in the same machine. It is not used: CUDA and
  ROCm cannot share one tensor-parallel group. Its ROCm install does affect native
  (non-Docker) builds — see the Qwen3.8-Flash-Next troubleshooting notes.
- Driver 610.57.04, kernel 7.0.0-31, Docker 29.5.3, CUDA 13.3 and 13.4 toolkits

## License

Apache License 2.0 — see [LICENSE](LICENSE).

`qwen3.8-flash-next/sglang/build-local-image/` is a vendored copy of the
yepapa-nest recipe, also Apache-2.0, with its own `LICENSE` and the changes listed
in its `UPSTREAM.md`. Software fetched at build time — SGLang and vLLM images, the
pennyroyal fork, NIXL, Strata — and the model checkpoints are not part of this repository
and keep their own licenses.
