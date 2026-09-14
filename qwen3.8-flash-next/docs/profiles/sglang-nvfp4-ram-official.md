# SGLang official image, NVFP4, PLE table in RAM

Profile `sglang-nvfp4-ram-official`, directory
`qwen3.8-flash-next/sglang/nvfp4-ram-official/`, started with
`scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-official.sh`.

The only Flash-Next profile with no local patches or builds: the official image
`lmsysorg/sglang:dev-qwen38-next-local` (commit `4ccff141db`, pulled 2026-09-12,
33 GB) running SGLang's verified cookbook cell for one RTX PRO 6000 with the
RadixArk NVFP4 checkpoint, retuned for a single agent. Measured 2026-09-12.

## Configuration

It shares [`sglang-nvfp4-ram`](sglang-nvfp4-ram.md)'s memory settings
(`extra_buffer_lazy`, `SKIP_DECODE_LOCK`, BF16 SSM state, chunked prefill 4096,
`expandable_segments`, `memlock=-1`) and differs in: the stock image,
`flashinfer_cutlass` for both FP4 GEMM and the MoE runner, the default CUDA graph
backend instead of `breakable`, and **BF16 KV cache**.

The launcher's own defaults reproduce the published cookbook cell (16 requests,
48 mamba slots). The start script sets `MAXRUN=4 MAMBA_SLOTS=12 KV_DTYPE=auto` for
one agent:

| | 2a: cookbook as published | 2b: 4 requests, FP8 KV | **2c: 4 requests, BF16 KV** (start script) |
|---|---|---|---|
| Concurrency / mamba slots | 16 / 48 | 4 / 12 | 4 / 12 |
| KV cache | 76,864 (BF16) | 498,624 (FP8) | **256,832 (BF16)** |
| Mamba cache | 6.34 GB | 1.79 GB | 1.79 GB |
| Free VRAM after graphs | 4.10 GB | 4.68 GB | 4.64 GB |
| Needle, 57K–220K × 3 | not run | **crash on first prompt** | **12/12** |
| Decode, warm | 248–263 tok/s | — | **258–260 tok/s** |
| Cold prefill 4K / 32K / 128K | 13,217 / 10,828 / — tok/s | — | **13,377 / 12,997 / 12,217** |
| TTFT 4K / 32K / 128K | 0.30 / 2.94 / — s | — | 0.30 / 2.45 / 10.4 s |

- **2a reproduces the published numbers** — the cookbook states ~78K KV tokens and
  4.2 GB free; this card gave 76,864 and 4.10 GB.
- **2b: FP8 KV crashes this image.** The pool sizes to the same 498,624 tokens as
  `sglang-nvfp4-ram`, but the first 57K-token prompt killed the scheduler with
  `AssertionError: Unsupported rhs dtype fp8e4nv` — the chunked-prefill bug the
  local image's `0001-qsa-fp8-kv-dequant-on-read.patch` fixes (upstream #36644,
  unmerged). With `--restart unless-stopped` the container then crash-loops. Keep
  `KV_DTYPE=auto`.
- **2c is the usable configuration.** 256,832 BF16 tokens cover nearly the whole
  262,144-token window for one request.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 256,832 tokens, **BF16** | 6.4 GiB | VRAM |
| Mamba cache — 12 slots, BF16 state | 1.8 GiB | VRAM |
| Free after CUDA graph capture | 4.6 GiB | VRAM |
| **VRAM in use** | not recorded; same budget as `sglang-nvfp4-ram` | |
| **Host RAM** | **~65 GB** | |
| Disk besides the checkpoint | Docker image 33 GB | |

## Against `sglang-nvfp4-ram`

| | `sglang-nvfp4-ram` (local image, FP8 KV) | `sglang-nvfp4-ram-official` (BF16 KV) |
|---|---|---|
| KV cache | **498,624** | 256,832 |
| Decode, warm | 236–249 tok/s | **258–260 tok/s** |
| TTFT 128K cold | 10.7 s | **10.4 s** |
| Needle to 220K | 12/12 | 12/12 |
| Local patches / build | yes | **none** |

For a single agent the window caps a request at 262,144 tokens either way, so the
larger FP8 pool mostly buys prefix-cache retention across turns. This profile
trades that for full-precision KV, no local build and ~5% faster decode — within
the spread of three warm runs, so suggestive rather than established.

## Code benchmarks

HumanEval 0.976 / HumanEval+ 0.951, MBPP 0.929 / MBPP+ 0.799 — 458 of 542 plus tests,
within noise of the FP8-KV profiles ([RESULTS.md](../../../RESULTS.md#code-humaneval-and-mbpp)).
