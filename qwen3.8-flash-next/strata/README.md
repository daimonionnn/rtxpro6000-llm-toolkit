# Strata with Q8 and Q6 weights

Experimental native setup for the RTX PRO 6000 Blackwell 96 GB. It reuses the
local lmstudio-community Q8_0 checkpoint and compares it with the published
Q6_K/Q8_0 checkpoint.
The engine checkout lives at `../Strata`, outside this toolkit. All ten text/vision profiles
are in the toolkit registry. Launchers run in the foreground; Ctrl+C or
`scripts/stop.sh` stops them. Use `scripts/status.sh` from another terminal.

| Profile | Launcher | Maximum context | Prefill chunk | Decode: prose / code / Slovak |
|---|---|---:|---:|---|
| `strata-q8` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-128k.sh) | 131,072 | 8,192 (auto) | 101.7 / 103.5 / 75.1 tok/s |
| `strata-q6` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q6-128k.sh) | 131,072 | 8,192 (auto) | 117.1 / 142.5 / 81.3 tok/s |
| `strata-q8-256k` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-256k.sh) | 262,144 | 8,192 (auto) | 89.4 / 104.9 / 73.1 tok/s |
| `strata-q6-256k` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q6-256k.sh) | 262,144 | 8,192 (auto) | 119.5 / 144.4 / 93.2 tok/s |
| `strata-q8-vision` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh) | 131,072 | 8,192 (auto) | not benchmarked (GPU vision) |
| `strata-q8-vision-256k` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-vision-256k.sh) | 262,144 | 8,192 (auto) | not benchmarked (GPU vision) |
| `strata-q6-vision` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q6-vision-128k.sh) | 131,072 | 8,192 (auto) | not benchmarked (GPU vision) |
| `strata-q6-vision-256k` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q6-vision-256k.sh) | 262,144 | 8,192 (auto) | not benchmarked (GPU vision) |

Decode above is the median of three 512-token responses to short prompts,
with MTP and thinking off,
measured 2026-10-06. Cache history affects the rates. Full prompt timings and
comparison methodology are in [COMPARISON.md](COMPARISON.md).

## Source and compatibility

- Strata 0.2.7 (Qwen engine 0.1.52), commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`
  (tag `v0.2.7`), unpatched, built from source. It reads Q8_0 PLE tables and contains
  Q6_K expert kernels upstream.
- The launchers set `STRATA_ROUTE_TAIL_SKIP=0` unless the caller sets it. Strata
  0.1.42+ (still in 0.2.7) otherwise skips low-ranked experts missing from VRAM on CUDA,
  which changes answers, most on non-English text ([NON-ENGLISH.md](NON-ENGLISH.md)).
- Its pinned ggml dependency: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- CUDA 13.3, compiled for SM120.
- Checkpoint: `lmstudio-community/Qwen3.8-Flash-Next-GGUF`, Q8_0, six local shards.

Strata 0.1.41 (`fb58e0db`) ran unpatched from 2026-10-09; the 2026-10-14 accuracy
tests below used it. On 2026-10-14 the engine moved to 0.1.44.1 and all packs were
rebuilt with its `iq_pack.py`; the MTP runtime files were kept (the MTP tools did
not change). The Q6 profiles have not been started on 0.1.41 or later.

On 2026-10-27 the engine moved to 0.2.6. Its `iq_pack.py` produced byte-identical
`q8_0` and `q8_0-abliterated` packs; `mtp_pack.py` and `mtp_rt.py` did not change, so
the MTP runtime files were kept. The older builds and packs were deleted; the
previously running commits are tagged `toolkit-0.1.41-running` and
`toolkit-0.1.44.1-running` in `../Strata` and can be rebuilt with the commands below.

On the same day the checkout moved on to 0.2.7, which adds pictures to the DeepSeek
engine (the DeepSeek profiles run from the same checkout). `iq_pack.py` and the MTP
tools are unchanged between 0.2.6 and 0.2.7, so the packs and MTP runtime files were
kept. The 0.2.6 commit is tagged `toolkit-0.2.6-running` in `../Strata`.

### Upgrade to 0.2.7, speed A/B, 2026-10-27

The same run as the 0.2.6 A/B below (`strata-q8-vision`, fresh start, default
launcher settings, DDR5-6400 CL42), three hours later:

| | 0.2.6 | 0.2.7 |
|---|---:|---:|
| Decode prose / code / Slovak, tok/s | 166 / 187 / 119 | 159 / 176 / 122 |
| Cold prefill 32K / 120K, tok/s | 6,550 / 6,542 | 6,659 / 6,691 |
| Load time | 50 s | 44 s |

Both read the two vision fixtures exactly. The decode differences are within the
spread of earlier runs (157–176 tok/s on code with one setting). 0.2.7 has a
"Qwen: 0.2.6 behaviour" switch in Strata's Terminal for troubleshooting.
Raw results: `logs/strata-comparison/strata-q8-vision-0.2.7-fresh*`.

### Upgrade to 0.2.6, speed A/B, 2026-10-27

`strata-q8-vision`, default launcher settings, one fresh start per engine on the
same day and RAM setting (DDR5-6400 CL42; earlier figures on this page used
6667 MT/s), `bench/compare_decode.py` medians and the second `bench/prefill.py`
sweep:

| | 0.1.44.1 | 0.2.6 |
|---|---:|---:|
| Decode prose / code / Slovak, tok/s | 139 / 159 / 111 | **166 / 187 / 119** |
| Cold prefill 32K / 120K, tok/s | 6,063 / 6,213 | **6,550 / 6,542** |
| Repeated 120K prompt, TTFT | **0.08 s** | 0.8 s |
| Load time | 55 s | 50 s |

Both read the two vision fixtures exactly and got the same 17,062-slot GPU expert
cache. Repeated prompts are slower because 0.2.6 parks each finished conversation
in a host-RAM conversation cache (four by default, sized from free memory) and
restores it on the next request: about 0.1 s at 4K and 0.8–1.1 s at 120K. In
exchange several conversations stay cached at once, which helps several agents
sharing the server. Upstream reports Flash-Next answers byte-identical only for
0.1.45; Slovak and accuracy checks on 0.2.6 are open in the
[TODO](../README.md#strata-026-qwen-engine-0150-quality-checks).
Raw results: `logs/strata-comparison/strata-q8-vision-0.{1.44.1,2.6}-fresh*`.

### Historical patches (Strata 0.1.39)

The measurements dated 2026-10-06 in this directory, [COMPARISON.md](COMPARISON.md)
and [VISION.md](VISION.md) used Strata 0.1.39 (`6f32ec070f23ced9f50e704d854d775da52591ab`)
with two local patches. They are kept only to reproduce those runs; 0.1.41 needs
neither.

- [q8-ple.patch](q8-ple.patch) added Q8_0 PLE table reads. Strata 0.1.39 accepted Q8
  experts but rejected the Q8 PLE table. The patch preserved its 170-byte rows and
  decoded them through Strata's existing Q8_0 reference dequantizer. It increased
  all PLE row buffers from 160 to 170 bytes.
- [q6-experts.patch](q6-experts.patch) added Q6_K experts; see [Q6_K](#strata-q6_kq8_0).

With the 0.1.39 patch, the PLE test compared real rows, including rows crossing
a disk page, and a 16-row batch against ggml. Both mmap and direct I/O passed with zero float
difference. The existing synthetic Q8_0/Q8_0 expert test also passed: GPU
dequantization agreed bit for bit with ggml; GPU and CPU expert outputs had
relative difference `1.09e-7` on that fixture.

`iq_pack.py --compat-bf16` leaves the experts, embeddings, native projections
and PLE table in their source formats. It converts 459 small tensors to the
engine's expected format, 96 exactly and 363 with rounding. The rounded
tensors occupy 1.22 GiB, and their largest absolute conversion error is 0.0144.
The conversion manifest is `../Strata/packs/q8_0/conversions.json`. This is a
comparison of usable serving configurations; the two engines are not
numerically identical.

## Preparation

Run from this toolkit's root. The model is already installed under LM Studio;
replace the model path if necessary. Strata's MTP preparation downloads about
5 GB of raw tensors and creates about 1.6 GB of runtime files. It checks the
raw tensors against the upstream pinned hashes.

```bash
git clone https://github.com/Niko1221/Strata.git ../Strata
git -C ../Strata checkout v0.2.7
python3 -m venv ../Strata/.venv
../Strata/.venv/bin/pip install -r ../Strata/requirements.txt
cmake -S ../Strata -B ../Strata/build \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ../Strata/build --target strata -j 12
```

Use the absolute path to the dependency's Python modules:

```bash
export STRATA_GGUF_PY="$(realpath ../Strata/build/_deps/strata_llamacpp-src/gguf-py)"
STRATA_PYTHON="$(realpath ../Strata/.venv/bin/python)"
STRATA_MODEL="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
"$STRATA_PYTHON" ../Strata/tools/iq_pack.py \
  --gguf "$STRATA_MODEL" --out ../Strata/packs/q8_0 --compat-bf16
"$STRATA_PYTHON" ../Strata/tools/mtp_fetch.py fetch --out ../Strata/mtp
"$STRATA_PYTHON" ../Strata/tools/mtp_fetch.py verify --out ../Strata/mtp
"$STRATA_PYTHON" ../Strata/tools/mtp_pack.py \
  --src ../Strata/mtp --experts q2_0 --out ../Strata/mtp/mtp-q2_0.gguf
"$STRATA_PYTHON" ../Strata/tools/mtp_rt.py \
  --gguf ../Strata/mtp/mtp-q2_0.gguf --out ../Strata/mtp/rt
cp ../Strata/data/draft_vocab.bin ../Strata/mtp/rt/draft_vocab.bin
```

The 2-bit file here is only the speculative draft layer. The target model's
routed experts remain Q8_0; the target verifies draft proposals.

## Run

Stop the other GPU server first. The launcher runs in the foreground; Ctrl+C
stops it. It binds only to localhost and exposes an OpenAI-compatible API.

```bash
scripts/start-qwen3.8-flash-next-strata-q8-128k.sh
# For an isolated benchmark port:
scripts/start-qwen3.8-flash-next-strata-q8-128k.sh --port 8093
```

Defaults: 131,072-token context, INT8 KV cache, MTP verify window 4 with minimum
draft probability 0.5, automatic prefill chunks, 135 GiB maximum resident expert
budget, mmap PLE reads, 1,536 MiB VRAM reserve. After loading this leaves about
1,002 MiB free, with 17,414 experts (84.70 GiB) in the GPU cache and 34.83 GiB of
remaining experts pinned in RAM. Adaptive cache swaps can warm the GPU cache as
requests run, so performance depends on prior workload.

All Q8/Q6 launchers, including vision and 256K variants, set
`"fit_max_tokens": true`. If a client's requested output limit exceeds the space
left in the context, the API reduces that limit instead of returning HTTP 400.
It preserves the prompt; a prompt leaving no room for an answer is still rejected.
Generation can end at the reduced limit, so agents still need context compaction
for long conversations. Restart the server to apply changes to this setting.

`--dry-run` prints the configuration. `--model`, `--strata-dir`, `--context`,
`--resident-gib`, `--prefill`, `--vram-reserve-mib`, `--pool-workers` and
`--pool-affinity` override the defaults.
The engine log is `logs/strata-q8-engine.log`.

### CPU expert pool

The engine computes some of the experts missing from VRAM on a CPU pool. Its
workers spin while a request runs, so each one keeps a core fully busy even when it
has little to do. The engine's default is one worker on every physical core but the
host's: on this Core Ultra 7 270K Plus (8 P-cores, 16 E-cores) that is 23 workers,
including all E-cores. The launchers pass `--pool-affinity p-cores` (on a hybrid
CPU the pool stays on the P-cores; ignored on CPUs without E-cores) and, since
2026-10-19, `--pool-workers 2`, which matched seven workers within a few percent at
about half the CPU power (measurements below). `--pool-workers 0` lets the engine
choose (seven P-cores with `p-cores`), and `--pool-affinity all --pool-workers 0`
restores the engine default of every physical core.

Measured 2026-10-15, `strata-q8-abliterated-vision`, Strata 0.1.44.1, fresh start
per setting, `bench/compare_decode.py` medians (prose / code / Slovak) and
`bench/prefill.py` cold prefixes. CPU load is the engine's average over the decode
benchmark:

| Pool | Decode tok/s | Prefill 1K / 4K / 32K tok/s | CPU load in decode |
|---|---|---|---:|
| 23 workers, all cores (engine default), run 1 / 2 | 106 / 121 / 84 · 106 / 111 / 78 | 1,219–1,252 / 2,448–2,473 / 5,230–5,262 | 22.3 cores |
| 8 workers (7 P + 1 E), run 1 / 2 | 152 / 162 / 122 · 150 / 169 / 125 | 1,280–1,325 / 2,392–2,477 / 5,240–5,268 | 8.9 cores |
| 7 workers, P-cores only | 144 / 176 / 123 | 1,334 / 2,433 / 5,334 | 7.9 cores |
| `--pool-affinity p-cores` (7) | 146 / 172 / 126 | 1,238 / 2,410 / 5,279 | 8.0 cores |
| 4 workers (P-cores) | 141 / 158 / 116 | 1,264 / 2,457 / 5,221 | 5.1 cores |

CPU package power, 2026-10-16, `turbostat` (1 s samples) during the same benchmarks,
one fresh start per setting. The idle server drew 14–15 W:

| Pool | Decode tok/s | Prefill 1K / 4K / 32K tok/s | CPU load in decode | Package power in decode | in prefill |
|---|---|---|---:|---:|---:|
| 1 worker (P-core) | 137 / 147 / 114 | 1,273 / 2,427 / 5,335 | 2.4 cores | 38 W | 35 W |
| 2 workers (P-cores) | 138 / 158 / 120 | 1,310 / 2,423 / 5,305 | 3.3 cores | 45 W | 36 W |
| `--pool-affinity p-cores` (7) | 147 / 157 / 120 | 1,301 / 2,478 / 5,281 | 7.9 cores | 85 W | 40 W |
| 23 workers, all cores | 108 / 120 / 85 | 1,230 / 2,480 / 5,296 | 22.3 cores | 134 W | 39 W |

Two P-core workers decode within a few percent of seven (code and Slovak equal, prose
6% lower; the three seven-worker runs spread 157–176 tok/s on code) at about half the
package power. One worker loses 5–7%. Prefill is the same in every setting.

Keeping the pool off the E-cores raised decode by about 40% and freed 14–15 cores;
prefill did not change, since the GPU computes it. A likely cause is that each CPU
phase waits for its slowest worker, and E-cores run at 4.7 GHz against 5.4–5.5 GHz;
23 spinning cores may also lower the P-cores' turbo. Neither was profiled.

## 256K profiles

After the same preparation, use either registered launcher:

```bash
scripts/start-qwen3.8-flash-next-strata-q8-256k.sh
# Or, after preparing the Q6 pack:
scripts/start-qwen3.8-flash-next-strata-q6-256k.sh
```

These configure `--max-context 262144`, within the model's native context;
no RoPE extension is needed. The context budget includes prompt and generation.
KV remains INT8, automatic prefill uses 8,192-token chunks, and MTP uses the
same draft and verification policy as the 128K variants. The larger KV allocation
reduces the automatic GPU expert cache, increasing the pinned RAM complement.

The Python entry points are [run_q8_256k.py](run_q8_256k.py) and
[run_q6_256k.py](run_q6_256k.py). Options pass through the shell wrappers:

```bash
scripts/start-qwen3.8-flash-next-strata-q8-256k.sh --dry-run
scripts/start-qwen3.8-flash-next-strata-q8-256k.sh --port 8097
# From another terminal:
scripts/status.sh
scripts/stop.sh
```

Runtime state is `logs/strata-server.json`. PID start times protect against
stale files and PID reuse. `stop.sh` terminates the API process and its native
engine, waiting for both to exit. Logs are `logs/strata-q8-256k-engine.log` and
`logs/strata-q6-256k-engine.log`; generated configs are under
`logs/strata-comparison/`. All ten launchers run in the foreground and bind
only to localhost. Direct Python launches are also tracked by status/stop.

The launchers set `/proc/self/coredump_filter` to 0, which the API process and
engine inherit. On 2026-10-09 the engine stalled in a ~69K prefill, and Strata's
watchdog (issue #29) stopped it. The kernel then hung dumping the crashed
engine's ~213 GB of host memory. The process stayed in state D inside
`vfs_coredump` holding 94 GB of VRAM. Neither `stop.sh` nor SIGKILL could end it,
and only a reboot frees the GPU in that state. With the filter at 0, a crash writes only the
ELF headers and exits at once.

Measured 2026-10-06, after the short decode test and prefill sweep:

| Profile | GPU expert slots | GPU expert cache | Pinned expert RAM | GPU used / free after tests | Fresh TTFT, ~255K | TG after ~255K |
|---|---:|---:|---:|---|---:|---:|
| `strata-q8-256k` | 17,021 | 82.79 GiB | 36.75 GiB | 96,416 / 873 MiB | 46.58 s | 117.8 tok/s |
| `strata-q6-256k` | 20,224 | 83.42 GiB | 17.95 GiB | 96,350 / 939 MiB | 44.00 s | 153.1 tok/s |

Pinned RAM counts experts only, excluding PLE's OS file cache and other process
allocations. Both profiles completed the long requests without OOM. TG is the
median of three 512-token prose responses after a ~255K corpus, excluding TTFT;
first prefix fresh, two repeats cached. Prefill rates including tokenization
and HTTP were 5,471 / 5,791 tok/s. The cache was already warmed by earlier tests;
fresh prefixes do not imply cold filesystem caches. Q8 wins the initial 4–120K
prefill sweep here, but Q6 is slightly faster at ~255K. The different expert
prefill kernels do not establish a universal speed ranking.

Measure sequentially against a running 256K profile:

```bash
python3 bench/compare_decode.py strata-q8-256k --base http://127.0.0.1:8097
BASE=http://127.0.0.1:8097 MODEL=qwen3.8-flash-next-strata-q8-256k \
  python3 bench/prefill.py 4096 32768 124000 262000
python3 bench/long_context.py strata-q8-256k-long --base http://127.0.0.1:8097
```

The approximate 262K target produces about 255K actual prompt tokens, leaving
space for 512 output tokens. Long-context TG is reported separately from the
short-context prose/code/Slovak medians in [COMPARISON.md](COMPARISON.md).

## Vision profiles

The pinned engine already supports images through its `--vision` M-RoPE path
and the server's optional `vision` configuration. These profiles add the GPU
encoder while keeping the corresponding Q8/Q6 target, INT8 KV, MTP, context,
prefill and 1,536 MiB reserve defaults. The encoder is the original BF16
`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, shared by Q8 and Q6; target quantization
and vision encoder quantization are separate.

The implementation and configuration follow [Strata's pinned vision documentation](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/DETAILS.md#images-vision).

### Build the CUDA image encoder

Run from the toolkit root, after the main engine preparation above. This uses
exactly the main engine's pinned llama.cpp checkout, not a separate version:

```bash
cmake -S ../Strata/tools/vision -B ../Strata/build-vision -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$PWD/../Strata/build/_deps/strata_llamacpp-src" \
  -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF
cmake --build ../Strata/build-vision --target strata-vision -j 12
```

The executable is `../Strata/build-vision/bin/strata-vision`. The encoder uses
FP16/BF16 attention; disabling additional quantized FA variants reduces build
work. CMake resolves SM120 to SM120a for this pinned ggml CUDA backend.

The default mmproj path is the existing LM Studio file:

```text
~/.lmstudio/models/lmstudio-community/Qwen3.8-Flash-Next-GGUF/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

If absent, obtain the matching BF16 mmproj from the
[lmstudio-community checkpoint](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF).
The locally installed file is 907,542,592 bytes. Use `--mmproj /path/to/mmproj.gguf`
to change its location; `--vision-exe` can point to another build of the helper.

### Start and stop

Choose one of the parallel launchers:

```bash
scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh
scripts/start-qwen3.8-flash-next-strata-q6-vision-128k.sh
scripts/start-qwen3.8-flash-next-strata-q8-vision-256k.sh
scripts/start-qwen3.8-flash-next-strata-q6-vision-256k.sh
```

Each runs in the foreground on localhost port 8090 by default; `--port` changes
it. Use `--host 0.0.0.0` for LAN access; the workstation's boot service already
uses this option, with LAN API URL `http://192.168.1.101:8090/v1`.
The API model ID is the full profile ID, e.g.
`qwen3.8-flash-next-strata-q8-vision` at 128K or
`qwen3.8-flash-next-strata-q8-vision-256k` at 256K.

GPU is the default encoder device. `--vision-device cpu` uses the same helper on
the CPU, without its GPU allocations. `--vision-tokens` defaults to 4,096 tokens
per image to preserve more detail in screenshots. This is an image representation
budget, not an OCR character limit or the text output `max_tokens` setting.
More image tokens take more encoder memory and processing time; the GPU encoder
warms up before the language engine sizes its expert cache. Images, text and
generation share the profile's 131,072 / 262,144 context budget.

Restart a vision profile to apply the new default. For faster photo processing,
use `--vision-tokens 1024`; the same override works on all four launchers:

```bash
scripts/start-qwen3.8-flash-next-strata-q8-vision-128k.sh --vision-tokens 4096
```

The recorded [vision measurements](VISION.md) used the previous 1,024-token
limit; their image latency and memory figures do not characterize the new default.
[Q8 text throughput with vision at 4,096](VISION.md#q8-text-throughput-vision-4096-vs-text-only)
now compares TG, prefill and expert-cache memory against text-only Q8.
For very long screenshots, send readable crops as separate images; increasing
the token budget cannot recover detail absent from the original image.

The server loads the encoder and warms it at the image-token limit before
starting the language engine, so its automatic expert cache accounts for the
encoder's VRAM. The expert cache is smaller than in the text-only profiles;
those decode benchmark figures do not describe vision-enabled configurations.

`/health` reports `images: true`. `scripts/status.sh` shows the encoder device,
image-token limit and mmproj path; `scripts/stop.sh` tracks and stops the API,
native engine and vision helper. Log files use `strata-q8-vision-engine.log`,
`strata-q8-vision-256k-engine.log` and the analogous Q6 names under `logs/`.

### Send an image

OpenAI-compatible chat content can contain a data URL:

```python
import base64
import json
import urllib.request

image = base64.b64encode(open("photo.png", "rb").read()).decode()
body = {
    "model": "qwen3.8-flash-next-strata-q8-vision",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_effort": "none",
}
request = urllib.request.Request(
    "http://127.0.0.1:8090/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

The reusable [vision smoke harness](../../bench/vision_smoke.py) saves image
responses, usage and latency:

```bash
python3 bench/vision_smoke.py q8-vision-check image.png \
  --base http://127.0.0.1:8090 --prompt "Describe this image."
```

### Validation on this machine

All four Q8/Q6 128K/256K vision variants correctly read two different four-digit
codes and identify three colored shapes in order, including an identical-image
repeat. Text-only arithmetic on each vision profile also works. The encoder
uses 1,742 MiB on the GPU here; the automatic expert cache shrinks accordingly.
Start/status/stop checks confirm both native workers exit. Exact latency, memory,
fixtures and local evidence are in [VISION.md](VISION.md). This is a smoke check,
not a scored vision or long-context benchmark.

## Abliterated Q8_0

Two profiles serve
[windowsxp811203/Qwen3.8-Flash-Next-Abliterated-GGUF](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated-GGUF),
a refusal-removed Qwen3.8-Flash-Next, as a single 188 GB Q8_0 file:

| Profile | Launcher | Context |
|---|---|---:|
| `strata-q8-abliterated` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-abliterated-128k.sh) | 131,072 |
| `strata-q8-abliterated-vision` | [shell wrapper](../../scripts/start-qwen3.8-flash-next-strata-q8-abliterated-vision-128k.sh) | 131,072 |

The upstream card reports, on its BF16 parent: AdvBench refusals 99.42% → 0.96%,
MMLU −1.6 points (paired), GSM8K unchanged. The quantized files were not
benchmarked upstream, and Slovak quality has not been measured here. The license
is the Qwen Community License 1.0, not Apache-2.0.

The file has the same 1,224 tensors, types and shapes as the lmstudio-community
Q8_0, the same tokenizer and chat template, and the corrected sparse-attention
`compress_ratios` (`[0,0,0,4,…]`; uploads before 2026-09-17 had zeros). It ships
no MTP head; the profiles reuse `../Strata/mtp/rt`, built from the original
model. Vision reuses the original BF16 mmproj; `strata-vision` opens the text
model vocabulary-only, and the vocabularies are identical.

Download the pinned revision and check the hash. Large downloads can arrive
corrupted without an error: on 2026-10-13 two 1 GiB ranges differed, and only
a full `sha256sum` revealed it. Re-fetching those byte ranges fixed the file.

```bash
hf download windowsxp811203/Qwen3.8-Flash-Next-Abliterated-GGUF \
  Qwen3.8-Flash-Next-Abliterated-Q8_0.gguf \
  --revision c3365c410baa29bdd3d7cc8cbc2bf9bee0de2f3a \
  --local-dir models/Qwen3.8-Flash-Next-Abliterated-Q8_0-GGUF
sha256sum models/Qwen3.8-Flash-Next-Abliterated-Q8_0-GGUF/Qwen3.8-Flash-Next-Abliterated-Q8_0.gguf
# e4b92e69943ebe8aafbd82296bdb5fcc0d44d9f344c98b77a30e86c81e3eb3d2
"$STRATA_PYTHON" ../Strata/tools/iq_pack.py \
  --gguf "$PWD/models/Qwen3.8-Flash-Next-Abliterated-Q8_0-GGUF/Qwen3.8-Flash-Next-Abliterated-Q8_0.gguf" \
  --out ../Strata/packs/q8_0-abliterated --compat-bf16
```

The pack has its own `dense.bin`; it differs from the original Q8_0 pack.

Optionally, replace the 363 small tensors that `--compat-bf16` rounds from Q8_0 with
the original BF16 bytes of the same checkpoint. Only the header and those tensors
(1.22 GiB) are downloaded from the BF16 GGUF; the tool refuses any tensor whose values
differ by more than Q8_0-sized error, and keeps the previous `dense.bin` as
`dense.bin.before-bf16`. Tested on 2026-10-14, it did not measurably change Slovak
or GSM8K results ([NON-ENGLISH.md](NON-ENGLISH.md)), so the running pack is the
plain `iq_pack.py` output. Re-running `iq_pack.py` undoes the patch.

```bash
../Strata/.venv/bin/python qwen3.8-flash-next/strata/bf16_dense_patch.py \
  --pack ../Strata/packs/q8_0-abliterated \
  --source https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated-GGUF/resolve/c3365c410baa29bdd3d7cc8cbc2bf9bee0de2f3a/Qwen3.8-Flash-Next-Abliterated-BF16.gguf
```

Smoke test, 2026-10-13, `strata-q8-abliterated-vision`, Strata 0.1.41 build,
default launcher settings. It started in 50 s. Arithmetic, a thinking-mode
question and short creative prompts were answered correctly. AdvBench's first
prompt was not refused. Both vision fixtures were read exactly (codes 8426 and
2957, all colors, shapes and order). `bench/compare_decode.py`, thinking off,
3 × 512 tokens, medians: 100.8 prose / 117.2 code / 72.8 Slovak tok/s. The
first repeat ran on a cold expert cache (67.8 / 81.0 / 61.2). These are smoke
figures, not a comparison under the conditions in [COMPARISON.md](COMPARISON.md).

### Full test against the original Q8_0, 2026-10-14

Both profiles ran as vision 128K on the same Strata 0.1.41 build, one after the
other, without other API traffic. Accuracy runs are greedy with thinking off;
the Slovak sheet also included ik_llama.cpp Q8_0 and both AWQ g32 checkpoints
from [Qwen3.8 results](../docs/results.md#non-english-slovak-blind-check).

| Test | Abliterated Q8_0 | Original Q8_0 | Paired difference |
|---|---:|---:|---|
| EvalPlus plus tests, of 542 | 455 | 457 | 5 vs 7 tasks, p = 0.77 |
| GSM8K, seeded 300 | 290 (96.7%) | 291 (97.0%) | 0 vs 1, p = 1.0 |
| MMLU-Pro, seeded 600 | 481 (80.2%) | 471 (78.5%) | 19 vs 9, p = 0.09 |
| Slovak blind, grader 1 / 2, of 100 | 62 / 64 | 64 / 66 | within grader noise |
| Decode prose / code / Slovak | 101.4 / 125.2 / 81.3 tok/s | 89.3 / 106.5 / 71.5 tok/s | single runs, adaptive cache |

No loss from abliteration was measured. MMLU-Pro answers hit the 4,096-token
limit 37 and 38 times, mostly in engineering, math and physics: long non-thinking
reasoning, not loops, equally for both. The Slovak sheet ranked ik_llama.cpp Q8_0
first (74 / 72), then AWQ g32 (69 / 76) and the uncensored AWQ g32 (70 / 68);
both Strata variants were last, mainly on the story and inflection prompts. With
identical Q8_0 weights that points at the engine rather than the abliteration —
possibly its BF16-rounded small tensors — but one greedy answer per prompt cannot
establish that. Candidate causes and how to test them: [NON-ENGLISH.md](NON-ENGLISH.md).
After the upgrade to Strata 0.1.44.1 (tail skip off), a second blind check scored the
abliterated profile 67 / 70 against 64 / 63 on 0.1.41 and 72 / 73 for ik_llama.cpp.
Decode on 0.1.44.1: 105.5 / 120.6 / 80.3 tok/s (prose / code / Slovak); both vision
fixtures read exactly.

Abliterated prefill, `bench/prefill.py`, second sweep, fresh prefixes:
4,010 / 31,881 / 120,592 tokens in 0.04 s cached; cold 5.56 s at 32K
(5,736 tok/s) and 20.06 s at 120K (6,011 tok/s). `bench/long_context.py` at
120,632 tokens: TTFT 20.0 s, then 97.9 tok/s; repeated prefix 0.08 s TTFT and
133–134 tok/s.

Raw results: `logs/evalplus/strata-q8-{abliterated,orig-0141}/`,
`logs/qa-accuracy/`, `logs/language-samples/sk/blind-20261014-035124*.md`,
`logs/strata-comparison/strata-q8-{abliterated-vision,vision-0141}-full.json`.

## Comparison

```bash
python3 bench/compare_decode.py strata-q8 --base http://127.0.0.1:8093
BASE=http://127.0.0.1:8093 MODEL=qwen3.8-flash-next-strata-q8 \
  python3 bench/prefill.py 4096 32768 124000
python3 bench/language_samples.py collect strata-q8 \
  --base http://127.0.0.1:8093 --model qwen3.8-flash-next-strata-q8
```

The decode script uses three fixed prompts, 512 generated tokens, three
repetitions and greedy sampling with thinking disabled. It saves complete
responses, usage and client timings under `logs/strata-comparison/`.
Client decode rate excludes time to the first text chunk. Speculative engines
may stream several tokens in a chunk, making that rate approximate. Compare
the same method across configurations and retain engine timings separately.

## Q6_K

The published five-shard Q6_K checkpoint lives in
`models/Qwen3.8-Flash-Next-Q6_K-GGUF/`. Gate/up experts are Q6_K; down experts
and the PLE table are Q8_0. Routed weights average 7.21 bits including block
scales. Q8 and Q6 variants at each context size use the same KV and MTP settings.

Download the pinned checkpoint with Hugging Face's `hf` CLI (156.13 GiB):

```bash
hf download lmstudio-community/Qwen3.8-Flash-Next-GGUF \
  --revision 158fc825df3eaa6c22d3c57a5927a5adf1c7cda7 \
  --include 'Qwen3.8-Flash-Next-Q6_K-*.gguf' \
  --local-dir models/Qwen3.8-Flash-Next-Q6_K-GGUF
cp qwen3.8-flash-next/strata/Q6_SHA256SUMS \
  models/Qwen3.8-Flash-Next-Q6_K-GGUF/SHA256SUMS
(cd models/Qwen3.8-Flash-Next-Q6_K-GGUF && sha256sum -c SHA256SUMS)
```

### Strata Q6_K/Q8_0

Strata 0.1.41 and later contain Q6_K expert kernels upstream, so the current build needs
no patch; the Q6 profiles have not yet been started on it. Build the pack and
start the profile:

```bash
export STRATA_GGUF_PY="$(realpath ../Strata/build/_deps/strata_llamacpp-src/gguf-py)"
../Strata/.venv/bin/python ../Strata/tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Q6_K-GGUF/Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf \
  --out ../Strata/packs/q6_k --compat-bf16
scripts/start-qwen3.8-flash-next-strata-q6-128k.sh --port 8096
```

The rest of this section describes the 2026-10-06 run on Strata 0.1.39, which
rejected Q6_K routed experts. [q6-experts.patch](q6-experts.patch), applied after
the Q8 PLE patch, added Q6_K gate/up and output-head dispatch, row byte
accounting, embedding and FP32/FP16 prompt dequantization, using the pinned
llama.cpp signed-scale/DP4A contract. Q6_K down was unsupported: this model's
640-wide down rows cannot contain whole 256-value Q6_K blocks. It was checked with
`native_expert_parity --synthetic q6_K/q8_0 q8_0/q8_0` and on real layers
0, 1, 16, 33 and 47.

The Q6 and Q8 synthetic expert checks and the five real Q6 layers passed.
GPU dequantization agrees bit for bit with ggml, including FP16 conversion
and gathered embedding rows. GPU expert relative error against the float
reference is 1.08–1.24% on the real layers (quantized activations).

The Q6 pack converts 460 small tensors: 96 exactly and 364 with BF16 rounding
(1.27 GiB, maximum absolute error 0.0144). Routed experts, embedding, head
and PLE table bytes remain unchanged. `packs/q6_k/conversions.json` records
each conversion. The upstream startup log has a hardcoded "Q5_K head" label;
the head loaded from this checkpoint is Q6_K (ggml type 14).

The launcher keeps the same context, MTP, KV, prefill and reserve settings as
Strata Q8. Q6 holds 20,687 experts in 85.33 GiB GPU cache, with 16.04 GiB of
remaining experts pinned in RAM.

Q8 supported the quantized MMQ expert prefill path. Q6_K had no MMQ support in
that build: it dequantized expert matrices to FP16 before the batched matrix
products. The Q6 patch supplied decode and dequantization support, not Q6 MMQ
prefill; whether later versions change this has not been checked. This is why smaller weights need not give faster prefill;
filesystem reads and warmup also contribute to the measured time.

In a separate terminal, run the measurements sequentially:

```bash
python3 bench/compare_decode.py strata-q6 --base http://127.0.0.1:8096
BASE=http://127.0.0.1:8096 MODEL=qwen3.8-flash-next-strata-q6 \
  python3 bench/prefill.py 4096 32768 124000
python3 bench/language_samples.py collect strata-q6 \
  --base http://127.0.0.1:8096 --model qwen3.8-flash-next-strata-q6
```

## ik_llama.cpp reference runs

The reference engine and its launchers belong to the separate public
[ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit) repository.
Follow that repository's build and serving instructions to run ik_llama.cpp;
use this toolkit's API benchmark scripts to compare it with Strata.

The measured Q8 and Q6 references use a 131,072-token context, Q8_0 KV,
`-b 2048 -ub 2048 -t 8 -tb 24`, and the Q4_K_M MTP draft with
`mtp:n_max=3,p_min=0.75`. Q8 uses `-ncmoe 19`, Q6 uses `-ncmoe 13`.
The complete configuration and results are in [COMPARISON.md](COMPARISON.md).
After the long-context Q6 test, only 249 MiB of VRAM remained;
`-ncmoe 14` leaves more headroom, but the reported Q6 measurements use 13.

Once the external ik server is running, benchmark its API from this repository:

```bash
python3 bench/compare_decode.py ik-q6-mtp --base http://127.0.0.1:8095
```

Set `BASE` and `MODEL` when using `bench/prefill.py` with an external server.
Only one GPU server should run at a time; complete each measurement before
starting the next one.
