# Strata vision validation

Measured 2026-10-06 on the RTX PRO 6000 Blackwell 96 GB, Intel Core Ultra 7
270K Plus, 244 GiB RAM, CUDA 13.3. Pinned Strata and llama.cpp commits are the
same as the text setup. Main Q8/Q6 GGUF bytes and compatibility patches are
unchanged; the additional encoder uses the original BF16 mmproj on the GPU.

All four profiles were started separately, without another GPU server. Vision
warmup completed at 1,024 image tokens before loading the language engine.
Settings measured: INT8 KV, MTP verify window 4 / min probability 0.5, prefill auto 8,192,
135 GiB resident budget, 1,536 MiB reserve, GPU vision limit 1,024 tokens/image.

Launchers now default to 4,096 image tokens for screenshot detail. The results
below retain the original 1,024-token setting; use `--vision-tokens 1024` to
reproduce them. Text throughput with the higher limit is measured separately
below; image quality and encoding latency at 4,096 remain unmeasured.

## Q8 text throughput: vision 4,096 vs text-only

Measured 2026-10-06, same hardware and engine as above, 128K context. Both
profiles start freshly and use identical Q8 weights, INT8 KV, MTP 4 / 0.5,
prefill auto 8,192, resident budget 135 GiB, reserve 1,536 MiB and
`fit_max_tokens: true`. Text-only runs first, vision second; no OS weight-cache
flush is performed. These are text requests with no images. Native logs contain
exactly the 22 expected benchmark requests per profile, without interleaved
agent traffic. The original vision configuration is restored and health checked.

TG is the median of three 512-token responses per prompt, temperature 0,
seed 1234, thinking off. Rates use SSE client timings and exclude TTFT.

| TG workload | Q8 text-only | Q8 vision 4,096 | Vision change |
|---|---:|---:|---:|
| English prose | 91.1 tok/s | 94.4 tok/s | +3.7% |
| Python code | 103.8 tok/s | 108.6 tok/s | +4.6% |
| Slovak | 72.2 tok/s | 75.8 tok/s | +4.9% |

Prefill follows TG: two sweeps of new random-nonce prefixes from the same
seeded word corpus, each followed by an identical-prefix cached request.
Every fresh-prefix request has zero native reused tokens. The table below
uses the second sweep with a warmed engine but a fresh prefix. Effective
throughput is actual prompt tokens / time to first token, including HTTP and
tokenization; it is not isolated native kernel throughput.

| Actual prompt size | Q8 text-only TTFT / tok/s | Q8 vision 4,096 TTFT / tok/s |
|---|---:|---:|
| ~4,010 tokens | 1.357 s / 2,955 | 1.386 s / 2,892 |
| ~31,882 tokens | 5.897 s / 5,406 | 5.921 s / 5,384 |
| ~120,593 tokens | 21.246 s / 5,676 | 21.683 s / 5,562 |

First-sweep fresh-prefix TTFT is 2.809 / 9.692 / 22.682 s for text-only
versus 1.479 / 6.123 / 21.497 s for vision. Startup warmup and sequential
weight-cache ordering affect those figures. Second-sweep cached-prefix TTFT
is 0.046 / 0.103 / 0.248 s versus 0.050 / 0.096 / 0.257 s; these requests
read only seven new tokens, so their apparent full-prompt tok/s is not prefill.

In this comparison warmed prefill differs by 0.4–2.1%; TG is slightly higher
with vision. Three samples and an adaptive cache do not establish a speed
benefit from enabling vision. Within-profile TG also improves over repetitions;
different cache contents, generated text and run order can affect the comparison.

The 4,096-token encoder uses 2,212 MiB (2.16 GiB), versus the earlier live
1,024-token snapshot of 1,758 MiB: +454 MiB (0.44 GiB). Compared with text-only,
vision reduces expert slots from 17,414 to 16,968, GPU expert cache from
84.70 to 82.53 GiB, and increases pinned expert RAM from 34.83 to 37.00 GiB.
After benchmarking, total compute-process VRAM is 96,300 MiB text-only versus
96,288 MiB vision: automatic expert-cache sizing consumes the available space.

Raw evidence (ignored by git):

- `logs/strata-vision/q8-text-comparison-20261006/`: orchestration, original and
  restored configurations, both prefill sweeps, native logs, memory and summary.
- `logs/strata-comparison/q8-{text,vision}-4096-comparison-20261006.json`:
  all TG samples, full responses, usage and client timings.

## Image checks

Two committed 1,536×1,024 PNG fixtures have high-contrast text and colored shapes:

| Image | Code | Objects, left to right |
|---|---|---|
| [vision-a.png](../../bench/fixtures/vision-a.png) | 8426 | red square, blue circle, green triangle |
| [vision-b.png](../../bench/fixtures/vision-b.png) | 2957 | blue triangle, green circle, red square |

The question asks for the code, colors, shapes and order as JSON, without giving
any expected image contents. Each profile receives A, repeats A, then receives
B through the OpenAI API as base64 `image_url` content. Thinking is off,
temperature 0, max output 256. Every response is parsed and checked against the
fixture, including all colors and shapes in order. A separate text-only
arithmetic request on each vision profile correctly returns 42.

All 12 image requests pass. All use 1,084 actual input tokens and generate 95
output tokens; A-repeat reports 1,077 reused prefix tokens, while the different
image B reports zero reused tokens. This also checks that a different picture
with the same text and dimensions produces the corresponding new answer.

| Profile | A, first request | A, repeated | B, different picture | OCR / shapes / order |
|---|---:|---:|---:|---|
| `q8-vision-128k` | 3.301 s | 1.219 s | 2.227 s | all correct |
| `q8-vision-256k` | 3.350 s | 1.287 s | 2.200 s | all correct |
| `q6-vision-128k` | 2.811 s | 0.698 s | 1.514 s | all correct |
| `q6-vision-256k` | 2.257 s | 0.798 s | 1.549 s | all correct |

These are single-request end-to-end latencies, not isolated encoder times or a
vision quality score. They include encoding/cache reuse, prompt processing and
generation, plus initial kernel/expert warmup. Expert caches adapt across
requests. These simple OCR/shapes do not measure natural-photo understanding,
small-text accuracy, grounding, long multimodal context or code quality.

## GPU and expert memory

The resident GPU encoder process uses 1,742 MiB as reported by nvidia-smi.
Automatic expert-cache sizing adapts to it; the memory snapshot below is after
three image requests and the text smoke check.

| Profile | GPU expert slots | GPU expert cache | Pinned expert RAM | Total GPU used | GPU free |
|---|---:|---:|---:|---:|---:|
| `q8-vision-128k` | 17,062 | 82.99 GiB | 36.55 GiB | 96,373 MiB | 916 MiB |
| `q8-vision-256k` | 16,669 | 81.07 GiB | 38.46 GiB | 96,381 MiB | 908 MiB |
| `q6-vision-128k` | 20,273 | 83.62 GiB | 17.75 GiB | 96,315 MiB | 974 MiB |
| `q6-vision-256k` | 19,809 | 81.71 GiB | 19.66 GiB | 96,321 MiB | 968 MiB |

Pinned RAM counts only the expert complement, excluding PLE's OS file cache and
other allocations. All requests finish without CUDA errors or OOM. The context
capacity is configured at 131,072 / 262,144 tokens, but this check exercises only
1,084-token multimodal prompts; the existing long-context text measurements
were made on the text-only profiles.

## Lifecycle and reproducibility

For each profile, `/health` reports `loaded: true`, `images: true` and the
expected capacity. `scripts/status.sh` shows GPU vision, the 1,024-token limit
and mmproj path. A second registered profile is rejected. The port preflight
now matches HTTPServer's reuse policy: TIME_WAIT after closing old client
connections permits a restart, while a live listener is still rejected; this
was verified with real localhost sockets. `scripts/stop.sh`
terminates the API, language engine and encoder; nvidia-smi reports no compute
process afterwards. The previously running text Q8 128K server is restored
with its exact original config and API process arguments on port 8090.

To reproduce a request on a running vision profile:

```bash
python3 bench/vision_smoke.py q8-vision-a bench/fixtures/vision-a.png \
  --base http://127.0.0.1:8090
python3 bench/vision_smoke.py q8-vision-b bench/fixtures/vision-b.png \
  --base http://127.0.0.1:8090
```

Local raw evidence under `logs/strata-vision/` (ignored by git):

- `summary-20261006.json`: health, usage, latencies and memory for all variants.
- `{q8,q6}-vision-{128k,256k}-{a,a-repeat,b}-20261006.json`: full responses.
- `{q8,q6}-vision-{128k,256k}-{engine,server}-20261006.log`: startup, encoder
  warmup and native generation timings.
- `{q8,q6}-vision-{128k,256k}-{status,memory}-20261006.txt`: lifecycle and VRAM.
- `build-vision-20261006.log`: native CUDA encoder build.
- `port-reuse-regression-20261006.txt`: TIME_WAIT and live-listener regression.
- `original-{state,config,api-argv}-20261006.json`,
  `restored-original-health-20261006.json`: restoration checks.

[Build, launchers and API usage](README.md#vision-profiles).
