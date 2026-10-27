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
| `deepseek-v4.1-flash-xolotl-xq-v2-vision` | the same + image input | | the same | |
| `deepseek-v4.1-flash-strata-ud-q8kxl` | Strata `strata-dsv41` 0.2.7 | Unsloth UD-Q8_K_XL (original MXFP4) | ~29 tok/s with DSpark (Slovak, English, code)³ | not yet graded |
| `deepseek-v4.1-flash-strata-ud-q8kxl-256k`, `-512k` | the same, 256K / 512K context | | ~19–22 tok/s at ~200K context | |
| `deepseek-v4.1-flash-strata-ud-q8kxl-vision`, `-vision-256k`, `-vision-512k` | the same + image input (engine's own tower) | | the same | |

¹ Its first speed runs were in the desktop's power-saver mode and were discarded;
expect it to be slower than UD-Q3_K_XL in proportion to its larger experts.

² A separate sheet with one grader (UD-Q3_K_XL 87, MXFP4 79.5 on it); see
[Xolotl XQ v2](#xolotl-xq-v2).

³ Warm, with the engine's learned hot-expert profile; the first requests of the very
first start ran at ~10 tok/s. See [Strata UD-Q8_K_XL](#strata-ud-q8_k_xl).

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

### Image input

`deepseek-v4.1-flash-xolotl-xq-v2-vision` is the same server with pictures: Xolotl's
`tools/v41/vision/ds41_vision.py` encodes them with DeepSeek's vision tower, which the
MTP sidecar carries in Q8_0, and DeepSeek's reference `vision.py` / `image_processor.py`
from the original checkpoint's `inference/` folder.

```bash
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-xolotl-xq-v2-vision-128k.sh
VISION_GPU=1 scripts/start-deepseek-v4.1-flash-xolotl-xq-v2-vision-128k.sh   # encoder on the GPU
```

It needs torch in Xolotl's venv and a fix in Xolotl itself: since its Windows commit
`38ab2166`, `ds41_vision.py` defines `_pread` twice and the encoder stops at start
(`TypeError: _pread() missing 1 required positional argument`).
[ds41-vision-pread.patch](xolotl/ds41-vision-pread.patch) renames the first one.

```bash
uv pip install --python ~/development/xolotl/.venv/bin/python "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu130
git -C ~/development/xolotl apply "$PWD/deepseek-v4.1-flash/xolotl/ds41-vision-pread.patch"
```

Measured 2026-10-27 with the two image fixtures of the Qwen3.8 vision profiles
(`bench/vision_smoke.py`, 1,536×1,024 PNGs with a code and three colored shapes;
1,013 prompt tokens): every answer named the code, colors, shapes and order correctly.

| Encoder | New picture | Same picture again | Experts in VRAM | Decode, Slovak / English / Python¹ |
|---|---:|---:|---:|---:|
| CPU (default, 8 P-cores, f32) | 27–29 s | 10 s | 9,635 | 37.0 / 44.7 / 60.1 tok/s |
| GPU (`VISION_GPU=1`) | 11–12 s | 10 s | 9,264 | 37.1 / 43.7 / 54.2 tok/s |

¹ 300-token answers, thinking off, mean of rounds 2–3 (the same prompts as the Strata
profile; not comparable with the 512-token table above).

The server caches encoded pictures by hash, so a repeated picture skips the encoder; the
1,013-token prompt itself takes ~9 s. The CPU encoder adds ~18 s per new picture and
leaves decode as it is; the GPU encoder takes ~1–2 s but costs 371 VRAM experts. The CPU
encoder is the default.

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

## Strata UD-Q8_K_XL

[Strata](https://github.com/Niko1221/Strata) 0.2.7 ships its own DeepSeek-V4.1 engine,
`strata-dsv41` (ported from Project Maya's `maya-ds4`), behind the same API server as
the Qwen3.8 Strata profiles. It runs Unsloth's UD-Q8_K_XL as released: the experts in
their original MXFP4, the dense weights in BF16 and the Engram on the SSD, 508.4 GB in
nine shards. On this card it keeps 4,323 of the 15,360 experts in VRAM (82.1 GB, plus
15.5 GB of dense weights) and computes the other 11,037 (207.8 GB) on the CPU.

```bash
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-128k.sh   # ready in ~2 minutes
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-256k.sh   # same, 256K context
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-512k.sh   # same, 512K context
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-128k.sh   # 128K with image input
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-256k.sh   # 256K with image input
HOST=0.0.0.0 scripts/start-deepseek-v4.1-flash-strata-ud-q8kxl-vision-512k.sh   # 512K with image input
```

The three profiles differ only in `--max-context`. The engine's compressed KV cache is
small: 0.12 / 0.24 / 0.49 GB of VRAM at 128K / 256K / 512K, which costs 50 to 65 VRAM
experts (of ~4,300) at 512K.

[serve.sh](strata/serve.sh) writes the server config to
`logs/deepseek-v4.1-flash-strata-ud-q8kxl-config.json`; `CTX`, `MMAP_EXPERTS`, `DSPARK`,
`CPU_THREADS` and `CPUSET` override its defaults. Like the Xolotl profile it allows the
chat UI's origin, clamps oversized `max_tokens` and shares `native/stop.sh` and the pid
file with the other DeepSeek profiles.

The figures below up to [Image input](#image-input-strata-027) were measured on 0.2.6;
see [Strata 0.2.7](#strata-027) for the upgrade. The launcher passes `--experts 6`: 0.2.7
drops to five routed experts a word on cards of 16 GiB or less, never here, but the
flag keeps full quality if that rule changes.

### Memory

The engine fills a RAM store with the CPU experts it reads, sized to the free RAM:
224 GB of the 244 GiB here, leaving about 20–35 GB for everything else. Close other
large programs before starting it.

The profile passes `--mmap-experts on`. With the engine's default (`auto`) it copies all
208 GB of CPU experts into its own memory at load. That load evicted the whole page
cache at once, and Ubuntu's `systemd-oomd` killed the terminal or the VS Code window
that started it (memory pressure above 50 % for 20 s in the user session; three times
on 2026-10-27). Two changes prevent that:

- every DeepSeek launcher runs its server in its own systemd scope
  (`systemd-run --user --scope`), so a kill can no longer take the terminal with it;
- on this machine the pressure kill is off for the user session
  (`/etc/systemd/system/user@1000.service.d/90-no-oomd-pressure.conf` with
  `[Service] ManagedOOMMemoryPressure=auto`; the kernel's OOM killer still applies).

### Hot experts

The engine records which experts are routed most and saves the profile into the pack
(`packs/dsv41-ud-q8kxl/expert-profile.bin`). The first start began with an empty profile:
decode was ~10 tok/s with 53 % of the expert work on the CPU, rising to 20–27 tok/s within
eight requests. Later starts read the hottest 162–189 GB into the RAM store in ~25–30 s and
are fast from the first request.

Measured 2026-10-27, DDR5-6400 CL42, `performance` power profile, thinking off. Each
setting got the same sequence: a fresh start with the learned profile, three rounds of
300-token answers, then `bench/prefill.py 4096 32768` (cold prompts):

| CPU threads | Slovak story | English story | Python | Prefill 3.8K | Prefill 30K |
|---|---:|---:|---:|---:|---:|
| 22 (engine default) | 26.3–29.4 tok/s | 24.6–26.6 tok/s | 17.8–21.5 tok/s | 335 tok/s (11.3 s) | 1,050 tok/s (28.9 s) |
| 8 (`CPU_THREADS=8`) | 26.9–30.5 tok/s | 25.9–26.5 tok/s | 18.8–20.4 tok/s | 337 tok/s (11.3 s) | 1,050 tok/s (28.8 s) |

Fewer P-cores (`CPUSET` and `CPU_THREADS` together), same sequence, with the CPU package
power from RAPL averaged over the decode rounds and over the prefill runs:

| P-cores / threads | Slovak | English | Python | Prefill 3.8K / 30K | CPU power, decode / prefill |
|---|---:|---:|---:|---:|---:|
| 8 / 22 (default) | 28.1 tok/s | 26.3 tok/s | 19.6 tok/s | 337 / 1,039 tok/s | 186 W / 47 W |
| 4 / 4 (`CPUSET=0-3 CPU_THREADS=4`) | 26.4 (−6 %) | 24.0 (−9 %) | 17.3 (−12 %) | 338 / 1,114 tok/s | 66 W / 40 W |
| 2 / 2 (`CPUSET=0-1 CPU_THREADS=2`) | 21.6 (−23 %) | 19.3 (−27 %) | 12.9 (−34 %) | 332 / 1,020 tok/s | 45 W / 34 W |

(Averages of three rounds.) 8 and 22 threads on eight P-cores decode alike. Prefill runs
on the GPU and does not depend on the CPU cores. Four P-cores cost 6–12 % of decode for
about a third of the CPU power; two cost 23–34 %. Short prompts (29–37 tokens) took
0.5–0.9 s on eight cores, 0.6–1.0 s on four and 1.0–1.8 s on two. What ran
before matters more than the threads: a 30K prompt right after a start read at
1,170–1,590 tok/s (it was 1,050 after the decode rounds), and decode right after the
prefill runs fell to 15–23 tok/s, as each workload shifts which experts stay in VRAM.
Treat single runs as ±20 %.

With `--dspark auto` the engine does not load the DSpark drafter: it expects the verify
rounds to read too many expert bytes when most experts are on the CPU. Forced on
(same sequence, 8 P-cores / 22 threads) it pays off anyway, so the profile passes
`--dspark on` (`DSPARK=auto` restores the engine's choice):

| | Slovak | English | Python | Prefill 3.8K / 30K |
|---|---:|---:|---:|---:|
| `DSPARK=auto` (off) | 28.1 tok/s | 26.3 tok/s | 19.6 tok/s | 337 / 1,039 tok/s |
| `DSPARK=on` (default) | 28.9 (+3 %) | 28.7 (+9 %) | 28.9 (+47 %) | 333 / 1,043 tok/s |

The drafter takes ~0.8 GB more for dense weights and leaves 4,279 experts in VRAM
instead of 4,323. It keeps 75–95 % of the drafts per position on code and stories, so
code gains most. Slovak quality is not graded yet.

Long prompts on the 512K profile (cold, `bench/prefill.py` 121K and 454K, then a
300-token English summary of ~197K tokens of source code, cold and from the engine's
conversation cache). The two settings ran back to back in the same order:

| | Prefill 121K | Prefill 197K | Prefill 454K | Decode at 197K, cold / cached |
|---|---:|---:|---:|---:|
| `DSPARK=on` (default) | 1,336 tok/s (91 s) | 1,080 tok/s (182 s) | 1,410 tok/s (322 s) | 19.0 / 21.8 tok/s |
| `DSPARK=auto` (off) | 1,354 tok/s (89 s) | 1,107 tok/s (178 s) | 1,428 tok/s (318 s) | 17.7 / 17.5 tok/s |

The drafter leaves prefill alone and adds 7–25 % to decode at long context. A first run
right after the profiles were created read 454K tokens at 2,154 tok/s and 197K at
1,426 tok/s; two later runs of either setting did not repeat it, and the cause is not
known. The conversation cache brings a repeated 454K prompt back in 0.25 s. Decode at
~200K context is 25–35 % below short prompts.

### Image input (Strata 0.2.7)

`strata-dsv41` has its own vision tower: DeepSeek's BF16 weights (0.97 GB) stay in pinned
RAM and stream to the GPU a block at a time, so the VRAM cost is small. Since 0.2.7 the
server decodes a picture, scales it to sides that are multiples of 42 (at most 1,024 image
tokens) and hands the engine raw RGB. The pack carries the tower; `--vision on` (the
`-vision` profile) turns it on, the text profiles pass `--vision off`. There is no CPU
encoder.

Measured 2026-10-27 with the two image fixtures (`bench/vision_smoke.py`, 975 prompt
tokens, 76 answer tokens): every answer named the code, colors, shapes and order
correctly, in 12.8–14.1 s per request. The tower leaves 4,216 experts in VRAM against
4,237 for `--vision off`, and decode is the same as in the text profile (below). The
256K and 512K vision profiles keep 4,208 and 4,192 experts in VRAM and read fixture B
correctly in 15.3 and 16.1 s.

### Strata 0.2.7

0.2.7 (DeepSeek engine 0.1.52) reads prompts with a different chunk rule and no longer
pre-fills the RAM store after a prompt. Two fresh starts each, the same sequence as above
(three rounds of 300-token answers, then `bench/prefill.py`), DSpark on:

| | Slovak, rounds 1 / 2 / 3 | English | Python | Prefill 3.8K / 30K |
|---|---:|---:|---:|---:|
| 0.2.6 | 27.7 / 29.4 / 29.8 | 25.3 / 31.2 / 29.5 | 26.3 / 32.0 / 28.5 | 333 / 1,043 tok/s |
| 0.2.7 | 19.8–20.7 / 26.5–26.9 / 29.5–31.3 | 20.8–22.7 / 25.4–25.8 / 28.2–28.8 | 27.0–27.9 / 27.0–30.2 / 28.3–33.8 | 331 / 984–987 tok/s |
| 0.2.7, `DS4_DEFAULTS=026` | 20.4 / 27.8 / 29.8 | 21.8 / 26.3 / 28.7 | 25.4 / 28.2 / 30.5 | 326 / 1,011 tok/s |
| 0.2.7, vision on | 20.6 / 26.9 / 29.2 | 20.9 / 25.5 / 26.2 | 26.7 / 31.9 / 31.9 | 330 / 965 tok/s |

Warm, 0.2.7 matches 0.2.6. Its first answers after a start are about 25 % slower, and
30K prompts about 6 %. `DS4_DEFAULTS=026` (every DeepSeek default back to 0.2.6) did not
change the first round, so the cause is not one of those defaults; the profiles keep
0.2.7's defaults.

### Setup

`~/development/Strata` is the same checkout as the Qwen3.8 Strata profiles (tag `v0.2.7`).
The vision tower comes from shard 1 of DeepSeek's own checkpoint (0.97 GB, only the tower);
`vision_pack.py` turns it into `vision.bin`, reads the three image-marker rows (30 KB) from
shard 2 with an HTTP range request, and `pack_gguf.py` takes `../vision` by itself.

```bash
cmake -S ~/development/Strata -B ~/development/Strata/build-dsv41 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.3
cmake --build ~/development/Strata/build-dsv41 -j 16
# 508 GB, nine shards; sha256-verify them (see the RAM note)
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-Q8_K_XL/*" \
  --local-dir models/DeepSeek-V4.1-Flash-1017-GGUF
# the vision tower: shard 1 + config.json of the original checkpoint, sha256-checked by vision_pack.py
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00001-of-00048.safetensors config.json \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir models/DeepSeek-V4.1-Flash-1017-GGUF/vision
cd ~/development/Strata && .venv/bin/python tools/dsv41/vision_pack.py \
  --dir ~/development/rtxpro6000-llm-toolkit/models/DeepSeek-V4.1-Flash-1017-GGUF/vision
.venv/bin/python tools/dsv41/pack_gguf.py \
  --model ~/development/rtxpro6000-llm-toolkit/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL \
  --out packs/dsv41-ud-q8kxl
```

Re-packing loses the learned `expert-profile.bin`; copy it from the old pack to start fast.

The pack (8.5 MB) records the shard folder; `serve.sh` refuses a pack made from another one.

## TODO

- [ ] Strata UD-Q8_K_XL: blind Slovak grading against UD-Q3_K_XL.

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
