# DeepSeek-V4.1-Flash-1017 on one RTX PRO 6000

DeepSeek-V4.1-Flash is a 552B-parameter MoE (384 routed experts, 8B active for input
and 16B for output) with a separate Engram n-gram table of about 190B parameters.
The 1017 checkpoint is the October update and scores clearly higher than the
original release on DeepSeek's coding and agent benchmarks. Neither the weights nor
the Engram fit in 96 GB of VRAM: the profiles here keep the shared weights and the
experts of the last layers on the GPU, the experts of the first layers in RAM, and
read the Engram from the SSD.

| Profile | Engine | Experts | Decode | Slovak blind score |
|---|---|---|---:|---:|
| `deepseek-v4.1-flash-llamacpp-ud-q3kxl` | llama.cpp PR #30817 | Unsloth UD-Q3_K_XL | **12.9–13.0 tok/s** | **83 / 84.5** |
| `deepseek-v4.1-flash-ik-mxfp4` | ik_llama.cpp `main` | original MXFP4 | not yet measured¹ | 75 / 79.5 |
| `deepseek-v4.1-flash-xolotl-xq-v2` | Xolotl (`ds41_generate`) | IndrasMirror XQ v2, PTQ1_0 / Q2_0 / Q3_K | **60–68 tok/s** (Slovak ~40) | 78² |

¹ Its first speed runs were in the desktop's power-saver mode and were discarded;
expect it to be slower than UD-Q3_K_XL in proportion to its larger experts.

² A separate sheet with one grader (UD-Q3_K_XL 87, MXFP4 79.5 on it); see
[Xolotl XQ v2](#xolotl-xq-v2).

The llama.cpp profiles are much slower than the Qwen3.8 Strata profiles (~140 tok/s)
and write the best Slovak measured on this machine; see [results](#results). The Xolotl
profile is about five times faster than UD-Q3_K_XL with clearly weaker Slovak.

```bash
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-llamacpp-ud-q3kxl-128k.sh
scripts/status.sh
scripts/stop.sh
```

The launcher runs the server in the background and returns once it answers and its
RAM experts are locked (about 4–6 minutes). `HOST`, `PORT`, `CTX`, `NCMOE`, `CPUSET`
and `LOCK_EXPERTS` override the defaults ([serve.sh](native/serve.sh)).

## Hardware budget

| | UD-Q3_K_XL | MXFP4 |
|---|---:|---:|
| Routed experts, all 40 layers | 271.6 GB | 288.8 GB |
| Other weights | 10.7 GB | 10.3 GB |
| Engram table (SSD, read on demand) | 202.8 GB (MXFP8) | 208.9 GB (Q8_0) |
| Expert layers in RAM (`--n-cpu-moe`) | 29 | 30 |
| Experts locked in RAM | 195.4 GB | 216.6 GB |
| VRAM in use | 89.7 GiB | 89.2 GiB |

The machine has 244 GiB of RAM. Stop the Qwen server first; nothing else large may
run beside these profiles.

### Why the experts are locked

Both engines map the GGUF with mmap, so the RAM experts live in the page cache, not
in the process's "used" memory (tools show the RAM as free). The Engram is read from
the same files at random positions, a few rows per token, and over a few hundred
tokens touches tens of GB. The kernel's LRU cannot tell the two apart: it evicted
expert pages to cache Engram rows, and the server re-read them from the SSD at
250–700 MB/s. [mlock_experts.py](native/mlock_experts.py) maps exactly the byte
ranges of the RAM experts and `mlock`s them; page-cache pages are shared between
mappings of a file, so the server's reads hit locked pages and only the Engram stays
evictable. Disk reads during decode fell to 0–35 MB/s. It needs an unlimited
`ulimit -l` (set here by `/etc/security/limits.d/30-rocm-memlock.conf`) and runs as
the user. Locking pushes other programs to swap the first time.

### Why the server is pinned to the P-cores

Unpinned, the 8 decode threads ran mostly on E-cores (4.7 GHz) while the P-cores
idled. `taskset -c 0-7` keeps them on the P-cores.

### Where the time goes

Each generated token reads 10 experts per layer. With 29–30 expert layers in RAM that
is about 5.1 GB (UD-Q3_K_XL) to 5.6 GB (MXFP4) from dual-channel DDR5-6400 per token,
so decode is bounded at roughly 13–15 tok/s by memory bandwidth alone; the profiles reach 60–70 % of that.
MXFP4 reads about 10 % more per token than UD-Q3_K_XL. Prefill streams
the RAM experts over PCIe to the GPU once per 2,048-token batch.

## Setup

### Engines

Build both against CUDA 13.3 **with `CUDAToolkit_ROOT` set**. Without it CMake links
the system `libcudart.so.12` while nvcc 13.3 compiles the kernels; `cudaDeviceProp`
then has a different layout, the engines read nonsense device limits
(`smpbo = 4294967297`) and abort in the first kernel that uses them
(llama.cpp: `SOFT_MAX failed: invalid argument`; ik_llama.cpp:
`mmq_x_best=0 ... fatal error`). `ldd .../llama-server | grep cudart` must show
`libcudart.so.13`.

```bash
# llama.cpp with the DeepSeek-V4.1 PR (open as of 2026-10-21)
git clone https://github.com/ggml-org/llama.cpp ~/development/llama.cpp-ds41
git -C ~/development/llama.cpp-ds41 fetch origin pull/30817/head:deepseek41
git -C ~/development/llama.cpp-ds41 checkout deepseek41
cmake -S ~/development/llama.cpp-ds41 -B ~/development/llama.cpp-ds41/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build ~/development/llama.cpp-ds41/build --target llama-server -j 16

# ik_llama.cpp main (deepseek41 is merged; tested at 15e7ecac)
git clone https://github.com/ikawrakow/ik_llama.cpp ~/development/ik_llama-ds41
cmake -S ~/development/ik_llama-ds41 -B ~/development/ik_llama-ds41/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_BUILD_RPATH=/usr/local/cuda-13.3/lib64 -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF
cmake --build ~/development/ik_llama-ds41/build --target llama-server -j 16
```

### Checkpoints

The two engines need different GGUF metadata for V4.1's Engram: Unsloth's files load
only in llama.cpp (ik_llama.cpp lacks `engram.layer_ids`, `head_count`, `primes`, …),
and ik-format files only in ik_llama.cpp. Verify every file against the Hugging Face
LFS sha256 after downloading; see the [RAM note](#ram-and-corrupted-downloads).

```bash
# 485 GB: 7 expert shards + the MXFP8 Engram shard
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-Q3_K_XL/*" \
  --local-dir models/DeepSeek-V4.1-Flash-1017-GGUF
# 508 GB: MXFP4 experts as released, other weights and the Engram in Q8_0
hf download mrfakename/DeepSeek-V4.1-Flash-1017-GGUF --include "Q8_0/*" \
  --local-dir models/DeepSeek-V4.1-Flash-1017-ik-GGUF
```

Give each pattern its own `--include`: `--include a b` takes only `a` as a pattern and
`b` as a file name, and as soon as any file name is given `hf download` ignores
`--include` altogether and fetches just the named files.

## Results

Measured 2026-10-22/23, single stream, thinking off, 131,072-token context,
experts locked, server pinned to P-cores, no other GPU or heavy CPU load.

The desktop's power profile matters: the first runs were in GNOME's power-saver
mode. Switching to `performance` (`powerprofilesctl set performance`; the P-cores
then run at 5.0–5.5 GHz) raised UD-Q3_K_XL decode by 25 % and prefill by 22–36 %.

| UD-Q3_K_XL | Performance | Power saver |
|---|---:|---:|
| Decode, 300 tokens (Slovak / English story / Python) | **13.0 / 12.9 / 13.0 tok/s** | 10.1 / 10.4 / 10.3 tok/s |
| Prefill, short prompts (14–38 tokens) | 18–26 tok/s | 6–16 tok/s |
| Prefill, cold, ~3.8K / ~30K tokens | **376 / 443 tok/s (10.1 / 68.5 s)** | 277 / 364 tok/s (13.7 / 83.3 s) |

The MXFP4 profile has not been timed in performance mode yet.

At 13 tok/s UD-Q3_K_XL decode is close to the memory-bandwidth bound estimated
above.

Slovak: the ten prompts of `bench/language_samples.py`, one greedy answer each,
shuffled into one sheet (`logs/language-samples/sk/blind-20261023-051943.md`) with the
best Qwen3.8 configurations, two LLM graders who saw only the sheet:

| Configuration | Grader 7 | Grader 8 | Weighted errors | Mean rank |
|---|---:|---:|---:|---:|
| **DeepSeek V4.1 1017 UD-Q3_K_XL** | **83** | **84.5** | 19 / 18 | 1.5 / 1.5 |
| DeepSeek V4.1 1017 MXFP4 | 75 | 79.5 | 18 / 18 | 1.9 / 1.5 |
| Qwen3.8 Q8_0, ik_llama.cpp | 67 | 70.5 | 38 / 37 | 3.0 / 3.1 |
| Qwen3.8 AWQ g32, vLLM | 67 | 68 | 44 / 46 | 3.5 / 3.7 |
| Qwen3.8 abliterated Q8_0, Strata 0.1.44.1 | 63 | 65 | 41 / 45 | 3.8 / 3.8 |

Both DeepSeek variants made about half the weighted errors of every Qwen3.8
configuration, mostly on idioms, the grammar correction and noun inflection. The
UD-Q3_K_XL lead over MXFP4 comes almost entirely from one prompt (inflection, 9 against
5) and is within the noise of one answer per prompt.

### DSpark

Speculative decoding did not help. Unsloth's 1017 drafter does not load in
ik_llama.cpp (missing `dflash.attention.indexer.head_count`); pokey2025's ik-format
drafter is built for the original V4.1 and accepted only 15–44 % of its drafts,
which made MXFP4 decode slower than without it. Even a matching drafter gains little here:
each drafted token routes to different experts, so verifying five tokens reads more
expert bytes from RAM, not fewer.

## Known issues

- **llama.cpp PR #30817 is not merged.** Its discussion reports corrupted output with
  `--lazy-mode on` (a race in the Engram row upload) and with KV prompt-prefix reuse.
  The UD-Q3_K_XL profile uses lazy mode; check important output and prefer the MXFP4
  profile when exact text matters.
- **The local UD-Q3_K_XL Engram shard (00008) does not match its sha256** after two
  downloads. The seven expert shards match. A few flipped bits in a 203 GB Engram
  table affect at most some n-gram rows; replace the shard if it matters.
- `--fit` does not work for this architecture in the PR: it tries to place every
  expert on the GPU. The profile sets `--n-cpu-moe` explicitly.
- ik_llama.cpp copies the RAM experts into pinned memory by default (247 GiB, killed by
  the OOM killer); the profile sets `GGML_CUDA_NO_PINNED=1`.
- mrfakename's `Q3_K_M` quantizes every tensor, including attention, the
  hyper-connection mixers and the Engram, to Q3_K; it is not used here.
- antirez's DwarfStar (`ds4`) Q2 was built but its 366 GB download failed the
  checksum twice and was not tested.

### RAM and corrupted downloads

All five downloads of single files above 180 GB (188, 203 twice and 366 GB twice)
arrived with a changed sha256, as did one 38.6 GB shard, which differed from the
original in a single bit. That pattern points at
memory, not the downloader: the four 64 GB non-ECC modules (two kits rated
DDR5-6400 for two modules each) ran at 6667 MT/s. Two files still failed after the
change to 6400 MT/s. Run memtest86+ and consider a lower speed for four modules;
silent bit flips also reach weights served from the page cache.

## Xolotl XQ v2

[Xolotl](https://github.com/Indras-Mirror/Xolotl) is a Strata fork with its own
DeepSeek-V4.1 engine. Its XQ quant stores the dense weights in Q8_0 (re-quantized to
Q6_K at load) and every routed expert as PTQ1_0, Q2_0 or Q3_K, chosen per expert from a
calibration run: 152.2 GB in all. On this card the engine keeps 9,633 of the 15,360
experts in VRAM and the rest in a 72 GiB pinned RAM arena, so nothing is read from the
model file during decode. Speculative decode uses the MTP sidecar (DSpark).

```bash
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-xolotl-xq-v2-128k.sh   # ready in ~1 minute
```

[serve.sh](xolotl/serve.sh) writes the server config to
`logs/deepseek-v4.1-flash-xolotl-xq-v2-config.json`; `CTX`, `ARENA_GIB`, `ROUTE_BIAS`,
`DSPARK` and `CPUSET` override its defaults. The config allows the chat UI's origin
(`cors_origins`, port 5173) and clamps oversized `max_tokens` (`fit_max_tokens`). The profile shares `native/stop.sh` and the
pid file with the llama.cpp profiles.

Setup:

```bash
git clone https://github.com/Indras-Mirror/Xolotl ~/development/xolotl
# build per its README for sm_120 with -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 (see Engines)
cd ~/development/xolotl && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# models/V41-XQ-1017: the IndrasMirror repository (tokenizer/, the MTP sidecar, v2/ in 15 parts)
bash models/V41-XQ-1017/v2/reassemble.sh models/V41-XQ-1017/v2   # joins and sha256-checks
# Engram: shards 47 and 48 + index of the original checkpoint, sha256-verified
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors \
  model-00048-of-00048.safetensors model.safetensors.index.json config.json \
  --local-dir models/DeepSeek-V4.1-Flash-1017-orig
```

The engine follows the checkpoint's `model.safetensors.index.json`, so `--engram` points
at the original directory; Xolotl's `extract_engram.py` copy (another 203 GB) is only
needed on slow filesystems such as exFAT.

XQ v2 has the refusal direction removed from its dense weights.

| Routing | Prose / code / Slovak decode (512 tokens, runs 2–3) | Slovak, one grader |
|---|---:|---:|
| **exact** (`ROUTE_BIAS="0 0"`, default) | **60–62 / 60–68 / 40–42 tok/s** | **78** |
| Xolotl's fast routing (`ROUTE_BIAS="0.5 0.3"`) | 83–86 / 94–97 / 56–61 tok/s | 72.5 |

The fast routing favours experts already in VRAM; on Slovak it cost case endings and
looped on the grammar correction, so exact routing is the default here. Its errors even
with exact routing ("hlavné mesta", "na brehu Orava", Czech "úsměv", missed
"jablka" -> "jablká") put it below both llama.cpp profiles on the same sheet
(`logs/language-samples/sk/blind-20261023-155939*.md`: UD-Q3_K_XL 87, MXFP4 79.5; the
exact-routing re-run in `blind-20261023-160420.md` was not fully blind, as the greedy
UD-Q3_K_XL answers were identical).

### Prefill and agent use

From an agent session on 2026-10-23 (68 tools, thinking on, 3K–65K-token prompts;
engine log `serve: prompt ...` lines):

| Prompt | Time | Speed |
|---|---:|---:|
| 12,992 tokens, cold | 35.7 s | 364 tok/s |
| 28,243 tokens, cold | 42.3 s | 668 tok/s |
| 47,828 tokens, cold | 61.0 s | **784 tok/s** |
| next agent turn, 0.3–5K new tokens on a cached 15K–65K prefix | **7–16 s** | |
| decode during the session, 50–65K context | | 36–52 tok/s |

A long cold prompt reads about 75 % faster than UD-Q3_K_XL (443 tok/s at ~30K). Each
agent turn costs at least 7–11 s even for a few hundred new tokens, because each batch
sweeps the experts of every layer; that time adds to every tool call.

Agents tend to send `max_tokens` 65536 on every request. Once the prompt passed ~65K
tokens, prompt + `max_tokens` exceeded the 131,072-token context and the server answered
400 ("requests are never truncated") until the agent compacted its history. The config
now sets `"fit_max_tokens": true`, which shortens `max_tokens` to the room left instead.

## TODO

- [ ] Build a standard (no refusal edit) XQ quant from the original checkpoint with
      Xolotl's `tools/v41` (budget ~230 GiB, mostly Q3_K experts, 128 calibration
      sequences including Slovak text) and compare its Slovak with UD-Q3_K_XL.
- [ ] Put experts on the second GPU, a Radeon AI PRO R9700 (32 GB, PCIe 4.0 x4, normally
      powered off). Build llama.cpp PR #30817 with CUDA and Vulkan together and move the
      experts of about four RAM layers to it, for example
      `-ot "blk\.(2[5-8])\.ffn_.*_exps=Vulkan0"`, keeping everything V4.1-specific
      (attention, indexer, Engram) on the RTX PRO 6000. Expected: ~14 % fewer expert bytes
      read from RAM per token, decode ~13 -> 14.5–15 tok/s; only activations cross the x4
      link during decode. Unknowns: Vulkan MoE kernel speed on RDNA4 for these quant types,
      per-layer cross-device synchronisation, and the PCIe AER errors seen from that slot.
      A DSpark drafter on the R9700 would not help: drafting was not the bottleneck, and the
      drafter needs the target's hidden states every step. Strata and Xolotl only split
      across GPUs of one vendor.
- [ ] Time the MXFP4 profile in the `performance` power profile (the power-saver runs
      were discarded), with the same prompts as UD-Q3_K_XL.
- [ ] Replace the UD-Q3_K_XL Engram shard (00008), whose sha256 does not match after two
      downloads, and verify it.
- [ ] Decide on antirez's `ds4` Q2 (341 GB, checksum failed twice, never run): download it
      a third time and test it, or delete it to make room (for example for Xolotl).
- [ ] Run memtest86+ overnight and settle the RAM speed (4 × 64 GB non-ECC, now
      6000 MT/s); then re-time the profiles at the final speed and record them. Until then
      sha256-verify every large download.
