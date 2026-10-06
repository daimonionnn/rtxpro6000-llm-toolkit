# Strata with Q8 and Q6 weights

Experimental native setup for the RTX PRO 6000 Blackwell 96 GB. It reuses the
local lmstudio-community Q8_0 checkpoint and compares it with the published
Q6_K/Q8_0 checkpoint.
The engine checkout lives at `../Strata`, outside this toolkit. All eight text/vision profiles
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

- Strata 0.1.39, commit `6f32ec070f23ced9f50e704d854d775da52591ab`.
- Its pinned ggml dependency: llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- CUDA 13.3, compiled for SM120.
- Checkpoint: `lmstudio-community/Qwen3.8-Flash-Next-GGUF`, Q8_0, six local shards.
- `q8-ple.patch` adds Q8_0 PLE table reads. Stock Strata accepts Q8 experts but
  rejects the Q8 PLE table. The patch preserves its 170-byte rows and decodes
  them through Strata's existing Q8_0 reference dequantizer. It increases all
  PLE row buffers from 160 to 170 bytes.

The PLE test compares real rows, including rows crossing a disk page, and a
16-row batch against ggml. Both mmap and direct I/O pass with zero float
difference. The existing synthetic Q8_0/Q8_0 expert test also passes: GPU
dequantization agrees bit for bit with ggml; GPU and CPU expert outputs have
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
git -C ../Strata checkout 6f32ec070f23ced9f50e704d854d775da52591ab
git -C ../Strata apply "$PWD/qwen3.8-flash-next/strata/q8-ple.patch"
python3 -m venv ../Strata/.venv
../Strata/.venv/bin/pip install -r ../Strata/requirements.txt
cmake -S ../Strata -B ../Strata/build \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ../Strata/build --target strata ple_q8_parity native_expert_parity -j 12
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
`--resident-gib`, `--prefill` and `--vram-reserve-mib` override the defaults.
The engine log is `logs/strata-q8-engine.log`.

## 256K profiles

After the same preparation, use either registered launcher:

```bash
scripts/start-qwen3.8-flash-next-strata-q8-256k.sh
# Or, after preparing the Q6 patch and pack:
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
`logs/strata-comparison/`. All eight launchers run in the foreground and bind
only to localhost. Direct Python launches are also tracked by status/stop.

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
it. The API model ID is the full profile ID, e.g.
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

Stock Strata at this commit rejects Q6_K routed experts. Apply
[q6-experts.patch](q6-experts.patch) **after** the Q8 PLE patch from the build
instructions above. It adds Q6_K gate/up and output-head dispatch, row byte
accounting, embedding and FP32/FP16 prompt dequantization. It uses the pinned
llama.cpp signed-scale/DP4A contract. Q6_K down remains unsupported: this
model's 640-wide down rows cannot contain whole 256-value Q6_K blocks.

```bash
git -C ../Strata apply "$PWD/qwen3.8-flash-next/strata/q6-experts.patch"
cmake --build ../Strata/build --target strata native_expert_parity -j 12
../Strata/build/native_expert_parity --synthetic q6_K/q8_0 q8_0/q8_0
../Strata/build/native_expert_parity \
  models/Qwen3.8-Flash-Next-Q6_K-GGUF/Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf \
  0 1 16 33 47
export STRATA_GGUF_PY="$(realpath ../Strata/build/_deps/strata_llamacpp-src/gguf-py)"
../Strata/.venv/bin/python ../Strata/tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Q6_K-GGUF/Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf \
  --out ../Strata/packs/q6_k --compat-bf16
scripts/start-qwen3.8-flash-next-strata-q6-128k.sh --port 8096
```

The Q6 and Q8 synthetic expert checks and the five real Q6 layers pass.
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

Q8 supports the quantized MMQ expert prefill path. Q6_K does not yet have MMQ
support in this build: it dequantizes expert matrices to FP16 before the batched
matrix products. The Q6 patch supplies decode and dequantization support, not
Q6 MMQ prefill. This is why smaller weights need not give faster prefill;
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
