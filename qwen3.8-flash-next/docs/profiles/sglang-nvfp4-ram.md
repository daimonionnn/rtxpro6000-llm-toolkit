# SGLang, NVFP4, PLE table in RAM

Profile `sglang-nvfp4-ram`, directory `qwen3.8-flash-next/sglang/nvfp4-ram/`,
started with `scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram.sh`.

The same local image and RadixArk NVFP4 checkpoint as
[`sglang-nvfp4-nvme`](sglang-nvfp4-nvme.md), with the PLE table in pinned host RAM
instead of on NVMe. Holding the table in RAM costs 1.83 GB of VRAM; this profile
wins it back from the mamba state cache, and ends up with twice the KV pool and
much faster prefill. How that was found, including the failed first attempt:
[ple-ram-experiment.md](../ple-ram-experiment.md). Measured 2026-09-12.

## Settings that differ from `sglang-nvfp4-nvme`

| Setting | `sglang-nvfp4-nvme` | `sglang-nvfp4-ram` |
|---|---|---|
| PLE table | NVMe (`SGLANG_QWEN4_PLE_NVME_*`) | pinned host RAM (SGLang's default path) |
| `--mamba-radix-cache-strategy` | `extra_buffer` | `extra_buffer_lazy` |
| `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` | unset | `1` |
| State slots per request | 5 | 3 |
| `--mamba-ssm-dtype` | model default (FP32) | `bfloat16` |
| `--max-running-requests` / `--max-mamba-cache-size` | 8 (capped to 5) / 25 | 4 / 12 |
| `--chunked-prefill-size` | 8192 | 4096 |
| `--mem-fraction-static` | 0.95 | 0.96 |
| `--max-total-tokens` | 393216 | unset (engine sizes the pool) |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `expandable_segments:True` |
| Docker | — | `--ulimit memlock=-1` |

KV stays FP8 with the local chunked-prefill patch; NEXTN speculation and breakable
CUDA graphs are unchanged. The slot arithmetic comes from
`kv_cache_configurator.py`: a base ratio of 3, minus 1 with `SKIP_DECODE_LOCK`,
plus 2 for `extra_buffer` or 1 for `extra_buffer_lazy` under the overlap scheduler.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 498,624 tokens, FP8 | 6.2 GiB | VRAM |
| Mamba cache — 12 slots, BF16 state | 1.8 GiB | VRAM |
| Free after CUDA graph capture | 4.6 GiB | VRAM |
| **VRAM in use** | **92.1–92.9 GiB** of 95.6 GiB | |
| **Host RAM** | **~65 GB** | |
| Disk besides the checkpoint | same Docker image as `sglang-nvfp4-nvme` | |

**Pinned memory does not show in the process's `VmLck` or `VmRSS`** (16 kB and
2.1 GB): `cudaHostAlloc` pins through the driver, not `mlock`. The table is visibly
in RAM through host `used`/`shared` rising by ~65 GB at load, and the absence of a
`Qwen4 PLE NVMe table` line in the log.

## Measurements

| | `sglang-nvfp4-nvme` | `sglang-nvfp4-ram` |
|---|---|---|
| **KV cache** | 231,936 tokens | **498,624 tokens** (2.15×) |
| Mamba cache | 5.34 GB | 1.79 GB |
| Concurrency | 5 | 4 |
| Decode, warm runs | 214–222 tok/s | **236–249 tok/s** |
| Cold prefill 4K | 7,314 tok/s · TTFT 0.55 s | **12,911 tok/s · 0.31 s** |
| Cold prefill 32K | 7,320 tok/s · 4.37 s | **11,010 tok/s · 2.90 s** |
| Cold prefill 128K | 7,076 tok/s · 18.0 s | **11,892 tok/s · 10.7 s** |
| Prefix-cached 128K | 0.745 s | 0.466 s |
| Needle, 57K–220K × 3 depths | not run | **12/12** |

Decode is three warm runs after a warmup, so treat the ~10% gain as a range.
**Prefill gains the most**: a prefill step gathers a PLE row for every prompt token,
which on NVMe is a direct SSD read each. The first 4K request after startup
measured 4,091 tok/s (warmup); repeats gave 12,811–12,911.

The needle test inserts a 10-character code at 10/50/90% depth into 57K, 115K,
176K and 220K-token prompts and asks for it back, flushing the prefix cache first.
It covers the FP8 KV cache, chunked prefill across dozens of chunks, and the RAM
PLE path together.

### Code benchmarks

HumanEval 0.970 / HumanEval+ 0.957, MBPP 0.929 / MBPP+ 0.794 — 457 of 542 plus tests,
the same as `sglang-nvfp4-nvme` despite the BF16 SSM state
([RESULTS.md](../../../RESULTS.md#code-humaneval-and-mbpp)).

Slovak blind check: 64 of 100, fourth of seven ([RESULTS.md](../../../RESULTS.md#non-english-slovak-blind-check)).
