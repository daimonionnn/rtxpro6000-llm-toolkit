# rtxpro6000-llm-toolkit

Launch profiles, scripts and measurements for running large language models on a
**single NVIDIA RTX PRO 6000 Blackwell 96 GB**, with SGLang, vLLM, ExLlamaV3 or
Strata.

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

Each model directory has a README with its checkpoints and profiles, and one
document per profile in `docs/profiles/<profile>.md`; the Flash-Next one also has
setup, troubleshooting, the model's architecture and open TODOs. The eight Strata
profiles share [their setup and comparison](qwen3.8-flash-next/strata/README.md).

## Results

**[RESULTS.md](RESULTS.md)** puts every profile of every model side by side: speed,
context and memory, HumanEval+ / MBPP+, a blind Slovak check, and recommendations.
In short:

- **Most context and fastest prefill:** `qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal`
  (524K window); `-sglang-nvfp4-ram` for plain Docker.
- **Best non-English output:** `qwen3.8-flash-next-vllm-awq-w4a16-g32`.
- **No refusals:** `qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored` — no measurable
  loss on code, a few points below g32 in Slovak; dealignai's abliterated NVFP4 lost
  code ability.
- **On code** the Flash-Next configurations covered by the September benchmarks
  are within noise of each other. Strata has not had a new EvalPlus run.
- **Dense Qwen3.6-27B and Qwen3.8-27B at BF16** trail Flash-Next on code and
  clearly in Slovak; 3.8 is no measurable step over 3.6.
- **Q8 weights at interactive speed:** `strata-q8` measured 101.7 tok/s on prose,
  103.5 on code and 75.1 in Slovak with MTP. `strata-q6` is faster for generation
  and uses fewer bits. Q8 has shorter latency at 4–120K in the initial
  prefill sweep; Q6 is slightly faster at ~255K. See the tables below.
- **The scored Q8 quality reference:** ik_llama.cpp's older Q8 run narrowly led
  its blind Slovak comparison. Engine builds and serving scripts live in the
  separate public [ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit)
  repository; the newer Strata configurations have not been independently graded.

### Strata Q8 / Q6 profiles

Measured 2026-10-06 with MTP, thinking off and one request at a time. Decode
figures are medians of three 512-token responses to short prompts. All times
below are for fresh prompt prefixes and include warmup and weight reads.

| Profile | Maximum context | Decode: prose / code / Slovak | TTFT: ~4K / ~32K / ~120K | Pinned expert RAM |
|---|---:|---|---|---:|
| [`strata-q8`](qwen3.8-flash-next/strata/README.md#run) | 131,072 | 101.7 / 103.5 / 75.1 tok/s | 4.83 / 12.46 / 21.67 s | 34.83 GiB |
| [`strata-q6`](qwen3.8-flash-next/strata/README.md#strata-q6_kq8_0) | 131,072 | 117.1 / 142.5 / 81.3 tok/s | 7.27 / 23.31 / 27.41 s | 16.04 GiB |
| [`strata-q8-256k`](qwen3.8-flash-next/strata/README.md#256k-profiles) | 262,144 | 89.4 / 104.9 / 73.1 tok/s | 5.37 / 13.77 / 22.56 s | 36.75 GiB |
| [`strata-q6-256k`](qwen3.8-flash-next/strata/README.md#256k-profiles) | 262,144 | 119.5 / 144.4 / 93.2 tok/s | 7.30 / 23.74 / 28.86 s | 17.95 GiB |

The 256K profiles were also tested near their maximum capacity:

| Profile | Fresh TTFT, ~255K | Effective prefill | TG after ~120K | TG after ~255K |
|---|---:|---:|---:|---:|
| `strata-q8-256k` | 46.58 s | 5,471 tok/s | 127.4 tok/s | 117.8 tok/s |
| `strata-q6-256k` | 44.00 s | 5,791 tok/s | 139.9 tok/s | 153.1 tok/s |

Prefill uses 254,834 / 254,832 actual prompt tokens (Q8 / Q6); its rate is
prompt tokens divided by TTFT, including HTTP and tokenization. TG uses three
512-token prose responses after a ~120K or ~255K word corpus: first a fresh
prefix, then two cached repetitions, excluding TTFT. These prompts and warmed
cache differ from the short tests, so the rates do not isolate context overhead.
Both fit without OOM; after the tests Q8/Q6 had 873/939 MiB free GPU memory.
Repeated ~255K prefixes answer in about 0.51 s. Fresh prefix does not mean a
cold OS file cache; the prefill sweep also includes initial kernel/weight warmup.

All Strata profiles use INT8 KV and automatic 8,192-token prefill chunks.
Pinned expert RAM excludes the PLE table's OS file cache and other allocations. Q6 mixes Q6_K
gate/up with Q8_0 down, averaging 7.21 bits per routed weight including scales;
its PLE table stays Q8_0. Q8 has the quantized MMQ prefill path; Q6 currently
dequantizes to FP16 before its prefill matrix products. Strata's adaptive expert
cache warms across requests, so generation speed depends on prior workload.

The Strata profiles require local compatibility patches. Setup, numerical
validation and the small Slovak sample check are in [Strata documentation](qwen3.8-flash-next/strata/README.md);
the full comparison with ik_llama.cpp is in
[COMPARISON.md](qwen3.8-flash-next/strata/COMPARISON.md).

### Strata vision profiles

Q8 and Q6 also have parallel vision profiles at 128K and 256K. They use the
existing BF16 `mmproj-Qwen3.8-Flash-Next-BF16.gguf` through a GPU image encoder;
images are accepted as OpenAI `image_url` parts. By default each image uses up
to 4,096 context tokens to preserve more detail in screenshots. Override with
`--vision-tokens 1024` for faster image processing. INT8 KV, MTP and the context
capacities stay the same.

| Profile | Context | Launcher |
|---|---:|---|
| `strata-q8-vision` | 131,072 | [Q8 vision 128K](scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh) |
| `strata-q6-vision` | 131,072 | [Q6 vision 128K](scripts/start-qwen3.8-flash-next-strata-q6-vision-128k.sh) |
| `strata-q8-vision-256k` | 262,144 | [Q8 vision 256K](scripts/start-qwen3.8-flash-next-strata-q8-vision-256k.sh) |
| `strata-q6-vision-256k` | 262,144 | [Q6 vision 256K](scripts/start-qwen3.8-flash-next-strata-q6-vision-256k.sh) |

The encoder loads and warms up before the language engine sizes its expert
cache. It uses VRAM, so fewer experts fit and the text-only benchmark figures
above do not apply to these profiles. [Build, API example and validation](qwen3.8-flash-next/strata/README.md#vision-profiles).

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
| [start-qwen3.8-flash-next-strata-q8-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-128k.sh) | Strata Q8_0, native engine + Q8 PLE patch, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-128k.sh](scripts/start-qwen3.8-flash-next-strata-q6-128k.sh) | Strata Q6_K/Q8_0, additional Q6 expert patch, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-256k.sh](scripts/start-qwen3.8-flash-next-strata-q8-256k.sh) | Strata Q8_0, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-256k.sh](scripts/start-qwen3.8-flash-next-strata-q6-256k.sh) | Strata Q6_K/Q8_0, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-vision-128k.sh](scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh) | Strata Q8, BF16 GPU vision, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q8-vision-256k.sh](scripts/start-qwen3.8-flash-next-strata-q8-vision-256k.sh) | Strata Q8, BF16 GPU vision, MTP, 256K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-vision-128k.sh](scripts/start-qwen3.8-flash-next-strata-q6-vision-128k.sh) | Strata Q6, BF16 GPU vision, MTP, 128K context; foreground |
| [start-qwen3.8-flash-next-strata-q6-vision-256k.sh](scripts/start-qwen3.8-flash-next-strata-q6-vision-256k.sh) | Strata Q6, BF16 GPU vision, MTP, 256K context; foreground |
| `start-qwen3.6-27b-sglang-bf16-256k.sh` | Qwen3.6-27B · SGLang, official image, BF16, NEXTN speculation |
| `start-qwen3.8-27b-sglang-bf16-256k.sh` | Qwen3.8-27B · SGLang, official image, BF16, NEXTN speculation |

```bash
scripts/start.sh                     # the default profile
scripts/status.sh
scripts/stop.sh
```

### Start the default model at boot

The workstation uses the user service
[strata-server.service](common/systemd/strata-server.service), which calls
`scripts/start.sh --host 0.0.0.0`. It starts Q8 vision 128K on port 8090,
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
├── RESULTS.md                    all profiles side by side, benchmarks, recommendations
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
│   └── strata/                  Q8/Q6 Python launchers, compatibility patches, comparison
├── qwen3.6-27b/                  README.md, docs/profiles/, sglang/bf16/
├── qwen3.8-27b/                  README.md, docs/profiles/, sglang/bf16/
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
pennyroyal fork, NIXL, Strata — and the model checkpoints are not part of this repository
and keep their own licenses.
