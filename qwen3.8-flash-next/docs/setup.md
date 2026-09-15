# Setup

Start to finish on a fresh machine. Roughly 40 minutes, most of it the download.

## Prerequisites

| Requirement | Why |
|---|---|
| Blackwell GPU, 96 GB, SM120 | NVFP4 is Blackwell-only. This will not run on Ada or Hopper. |
| NVIDIA driver, kernel module matching userspace | Verified with `nvidia-smi` |
| Docker with the NVIDIA container runtime | `docker run --rm --gpus '"device=0"' ubuntu:24.04 nvidia-smi -L` |
| `io_uring` enabled | `sysctl kernel.io_uring_disabled` must report `0`; kernel 5.10+ |
| ~150 GB free on an **NVMe** filesystem | The PLE table is read continuously during decode |
| `git`, `rsync`, `curl`, `python3` | Used by `build.sh` |

The checkpoint must sit on NVMe with a real Linux filesystem. An NTFS mount via
ntfs3 is a bad fit for `O_DIRECT` io_uring reads, and a spinning disk is out of
the question.

### Driver sanity check

Both numbers must match. If they do not, `nvidia-smi` fails with
`Driver/library version mismatch` and nothing CUDA works:

```bash
cat /proc/driver/nvidia/version              # loaded kernel module
ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1   # userspace library
```

A mismatch usually means the driver was updated without a reboot. The `nvidia`
module cannot be unloaded while the display stack holds it, so reboot.

## 1. Checkpoint

```bash
uv tool install "huggingface_hub[cli]"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242a \
  --local-dir models/Qwen3.8-Flash-Next-NVFP4
```

**Pin the revision.** Day-0 uploads get amended without announcement.

For the `vllm-awq-w4a16` profile, the AWQ checkpoint as well (168 GiB, 6 shards
plus `model_mtp.safetensors`):

```bash
hf download wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 \
  --revision 0939125b929543a783ce700c90e36dd1a575c00c \
  --local-dir models/Qwen3.8-Flash-Next-AWQ-W4A16
```

For the `vllm-awq-w4a16-g32` profile (175 GiB, 38 shards):

```bash
hf download cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 \
  --revision d39638a0e740fccb3e24ae0ea5cab34c15371ae6 \
  --local-dir models/Qwen3.8-Flash-Next-AWQ-INT4-g32
```

For the `exllamav3-exl3-5.05bpw` profile, one branch of the EXL3 repository (115 GiB):

```bash
hf download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 7cef615f7fd3681295b68848018876dfabc336c7 \
  --local-dir models/Qwen3.8-Flash-Next-EXL3-5.05bpw
```

The revision is the head of branch `5.05bpw_h6_ng6`; the other branches hold the
2.05–6.05 bpw quants.

For the `vllm-fp8-offload` profile, Qwen's FP8 checkpoint (173 GiB, 131 shards):

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --local-dir models/Qwen3.8-Flash-Next-FP8
```

The NVFP4 checkpoint is 126 GiB, 206 shards. Verify nothing is missing before building:

```bash
python3 - <<'PY'
import json, os
d = "models/Qwen3.8-Flash-Next-NVFP4"
j = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need, have = set(j["weight_map"].values()), set(os.listdir(d))
print("shards:", len(need), "| missing:", sorted(need - have))
PY
```

## 2. Build the image

```bash
git clone https://github.com/yepapa-nest/qwen38-flashnext-rtx6000.git \
  qwen3.8-flash-next/sglang/build-local-image
cd qwen3.8-flash-next/sglang/build-local-image
./build.sh
```

`build.sh` pins an SGLang source tree, cherry-picks three upstream PRs, applies
one local patch, and runs `docker build`. It takes a while: the blobless clone of
SGLang is large, and it competes for bandwidth if the checkpoint is still
downloading.

**Apply the Dockerfile changes described in
[upstream-fixes.md](upstream-fixes.md) before building** — without them the build
fails at the last step with `ModuleNotFoundError: ..._storage`.

Result: image `sglang-flashnext-sm120:local`, about 37 GB.

## 3. Serve

```bash
cd qwen3.8-flash-next/sglang/nvfp4-nvme
MODEL_DIR=/abs/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-nvme.sh
```

`serve-nvfp4-nvme.sh` accepts these overrides:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | *(required)* | Absolute path to the checkpoint |
| `PORT` | `8090` | Bound to `127.0.0.1` only |
| `CTX` | `262144` | Native context window |
| `MEMFRAC` | `0.95` | `--mem-fraction-static` |
| `MAMBA_SLOTS` | `25` | Recurrent-state slots; the real concurrency limit (~5) |
| `CHUNKED` | `8192` | Chunked prefill size |
| `WEIGHT_QUANT` | `modelopt_fp4` | `--quantization`; the launch aborts if the checkpoint is not NVFP4 |
| `FP4_GEMM_BACKEND` | `flashinfer_cudnn` | FP4 matrix-multiply kernel |
| `KV_DTYPE` | `fp8_e4m3` | KV cache precision; `auto` means BF16 and roughly half the tokens |
| `EXTRA_ARGS` | *(empty)* | Extra launch flags, e.g. `--mamba-ssm-dtype bfloat16` |

Before launching, `serve-nvfp4-nvme.sh` runs `quant_info.py`, which reads the checkpoint and
prints what precision each part runs at:

```
Qwen3.8-Flash-Next — Qwen3.8-Flash-Next-NVFP4  (modelopt 0.46.0)
  routed MoE experts    NVFP4 W4A4, group 16, FP8 E4M3 block scales
                        --quantization modelopt_fp4, FP4 GEMM flashinfer_cudnn
  attention, router,    BF16 — not quantized (13 exclude patterns)
  shared experts, MTP,  self_attn, linear_attn, mlp.gate, shared_expert,
  vision, lm_head       hyper_connection, mtp, visual, embed_tokens, lm_head
  PLE n-gram table      FP8 E4M3, 47.7 GiB, streamed from NVMe (io_uring)
  KV cache              fp8_e4m3
```

The checkpoint is **not uniformly 4-bit**: only the routed MoE experts are NVFP4.
It can also be run on its own — `python3 qwen3.8-flash-next/quant_info.py MODEL_DIR`. On a running
container the same information is in its labels:
`docker inspect flashnext --format '{{json .Config.Labels}}'` (containers started
before the labels were added do not have them).

The script waits for `The server is fired up` and prints the container log if the
container exits first.

To see at any time which profile is running, from which directory, on which port
and how to stop it:

```bash
scripts/status.sh
```

For daily use, `scripts/` has a start script per profile, one `stop.sh` for
whichever is running, and start/stop scripts for the chat UI — see the README.

Ready when:

```bash
curl -s http://127.0.0.1:8090/v1/models
```

## 4. Stop

```bash
./stop.sh          # stop and wait until the VRAM is released
./stop.sh --rm     # also remove the container
```

`serve-nvfp4-nvme.sh` uses `--restart unless-stopped`: if the server is running when the
machine shuts down, Docker starts it again at boot and it takes ~93 GB of VRAM
without anyone asking. A container stopped by hand stays stopped across reboots.
`NAME` and `TIMEOUT` (graceful shutdown, default 60 s) can be overridden.

## 5. The RAM launchers

The steps above build the image for `sglang-nvfp4-nvme`, which `sglang-nvfp4-ram` also uses.
The results of all four launchers are compared in
[ple-ram-experiment.md](ple-ram-experiment.md). All the SGLang RAM profiles need
**at least ~65 GB of free host RAM** for the pinned 47.7 GiB PLE table.

### `sglang-nvfp4-ram` — same image

Nothing extra to install:

```bash
cd qwen3.8-flash-next/sglang/nvfp4-ram
MODEL_DIR=/abs/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
```

It accepts the same variables as the NVMe launcher, with different defaults:
`MAXRUN=4`, `MAMBA_SLOTS=12`, `CHUNKED=4096`, `MEMFRAC=0.96`, `MAX_TOTAL_TOKENS`
unset, plus `MAMBA_SSM_DTYPE=bfloat16`.

### `sglang-nvfp4-ram-official` — official image

```bash
docker pull lmsysorg/sglang:dev-qwen38-next-local      # 33 GB
cd qwen3.8-flash-next/sglang/nvfp4-ram-official
MAXRUN=4 MAMBA_SLOTS=12 KV_DTYPE=auto \
MODEL_DIR=/abs/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
```

Without overrides it reproduces the published cookbook cell (16 requests,
~77K KV tokens). **Keep `KV_DTYPE=auto`**: with `fp8_e4m3` the image crashes on
the first long prompt. Stop it with `./stop.sh`.

`scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official.sh` sets those three
variables. The same launcher also serves dealignai's abliterated NVFP4 checkpoint,
whose layout is identical:

```bash
scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official-abliterated.sh
```

Download command and results: [sglang-nvfp4-ram-official-abliterated.md](profiles/sglang-nvfp4-ram-official-abliterated.md).

### `sglang-nvfp4-ram-pennyroyal` — jpezzulli fork, native

Needs on the host: CUDA 13.3 at `/usr/local/cuda-13.3`, `gcc-15`/`g++-15`, Rust,
`uv`, and an unlimited memlock limit (`ulimit -l` → `unlimited`). Then:

```bash
qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/build.sh        # clones the fork to qwen3.8-flash-next/sglang/pennyroyal-fork and builds its venv
cd qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal
MODEL_DIR=/abs/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
./stop.sh
```

The build is the fork's `BUILD.md` with two differences: `CUDA_HOME` points at
13.3 (the host default is 13.4), and `wheel_stub` is installed first, without
which `cuda-tile` fails to build. The launcher is the fork's
`serve-flash-next.sh` minus HiCache/NIXL, and puts `shim/` first on `PATH` so
TileLang does not target the AMD GPU — see
[troubleshooting.md](troubleshooting.md).

**HiCache with NIXL persistence** (optional) keeps prefixes across restarts —
a 220K-token prompt comes back in 1.5 s instead of 17.7 s. It needs two host
packages and a build:

```bash
sudo apt install meson libaio-dev
qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/build-nixl.sh      # into qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/nixl, no root needed
cd qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal
HICACHE=1 MODEL_DIR=/abs/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
```

Cache files go to `qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/nixl-storage/` (`NIXL_STORAGE_BASE`
overrides), and the host-RAM tier is `HICACHE_SIZE_GB` (default 32). **Read the
header of `nixl-posix-local.toml` before first use:** its eviction watermarks are
percentages of the whole filesystem and were set for this machine's disk at 88%.

The first start compiles kernels into `qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/cache/`; later starts
reuse it. The launcher refuses to start while another process holds the GPU, so
stop any Docker profile first. It runs as a background process with its PID in
`qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/server.pid` and does not restart after a reboot.

## 6. The vLLM profiles

```bash
docker pull vllm/vllm-openai:qwen38-flash-next      # 19.8 GB
scripts/start-qwen3.8-flash-next-vllm-awq-w4a16.sh
```

Needs the AWQ checkpoint and at least 100 GiB of free host RAM for the BF16 PLE
table. First start is about three minutes. See [vllm-awq-w4a16.md](profiles/vllm-awq-w4a16.md) for the
launcher options, measurements, and why it must run with
`--distributed-executor-backend mp` on a single GPU.

```bash
scripts/start-qwen3.8-flash-next-vllm-awq-w4a16-g32.sh
```

Same image and settings as `vllm-awq-w4a16`, group-32 checkpoint. See
[vllm-awq-w4a16-g32.md](profiles/vllm-awq-w4a16-g32.md).

```bash
scripts/start-qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored.sh
```

Same image, leoncca's uncensored AWQ g32 checkpoint with an FP8 PLE table, so ~60
GiB of free host RAM is enough. The launcher patches one function of the image's
PLE layer and serves a hard-linked view of the checkpoint without the unindexed
tensors that stop vLLM; the first start takes 10–20 minutes. See
[vllm-awq-w4a16-g32-uncensored.md](profiles/vllm-awq-w4a16-g32-uncensored.md).

```bash
scripts/start-qwen3.8-flash-next-vllm-fp8-offload.sh
```

Same image, FP8 checkpoint. Keeps 50 GiB of routed experts in pinned RAM next to
the PLE table, so it needs ~106 GiB of free host RAM; first start is about eight
minutes. See [vllm-fp8-offload.md](profiles/vllm-fp8-offload.md).

## 7. The ExLlamaV3 profile

```bash
docker pull ghcr.io/theroyallab/tabbyapi@sha256:a0befeadd9b4609e5a39334aa587b5bd8da33f4eeb6c68c482f2d1751fad79d3
scripts/start-qwen3.8-flash-next-exllamav3-exl3-5.05bpw.sh
```

TabbyAPI with ExLlamaV3 1.5.0 and the EXL3 5.05 bpw checkpoint. Needs ~50 GiB of
free host RAM for the n-gram table; loads in under a minute, and the first request
compiles kernels for ~45 s. See [exllamav3-exl3-5.05bpw.md](profiles/exllamav3-exl3-5.05bpw.md).

## Re-measuring after a config change

```bash
python3 bench/prefill.py                                   # 4K / 32K / 128K, cold and prefix-cached
python3 bench/language_samples.py collect <profile>         # Slovak answers for the blind comparison
```

See [bench/README.md](../../bench/README.md) for both scripts.

## Notes on the launch flags

The defaults in `serve-nvfp4-nvme.sh` are the upstream author's measured choices, not
preferences. Two are worth knowing:

- `--cuda-graph-backend-decode breakable` is what lets CUDA graphs run together
  with NEXTN speculation. Full and `tc_piecewise` graphs are incompatible with
  the PLE device-to-host copy during capture.
- `--speculative-num-draft-tokens 4` is an architectural cap from the QSA
  compression ratio. Raising it does not help.
