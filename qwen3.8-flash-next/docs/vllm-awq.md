# vLLM with the AWQ W4A16 checkpoint

Profile `vllm-awq-w4a16`, directory `qwen3.8-flash-next/vllm/awq-w4a16/`, started with
`scripts/start-qwen3.8-flash-next-vllm-awq-w4a16.sh`.

It runs a different checkpoint from the SGLang profiles —
[wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
@ `0939125b` — on the vLLM preview image `vllm/vllm-openai:qwen38-flash-next`
(vLLM `0.1.dev20073+g8e685d198`). Measured 2026-09-13 on the same RTX PRO 6000.

## Checkpoint

Read from the safetensors headers by `qwen3.8-flash-next/quant_info.py`:

| Component | Parameters | Precision | Bits / param | On disk |
|---|---|---|---|---|
| Routed experts | 120.8B | INT4 weight-only, group 128, symmetric | 4.13 | 58.0 GiB |
| Attention, router, shared expert, MTP, vision, embeddings | 8.0B | BF16 | 16.00 | 14.9 GiB |
| PLE n-gram table | 51.2B | **BF16** | 16.00 | 95.4 GiB |
| **Model total** | **180.0B** | | **8.03** | **168.3 GiB** |

Compared with the NVFP4 checkpoint the SGLang profiles use:

- **Experts get fewer bits** (4.13 vs 4.50) with coarser groups (128 vs 16) and
  integer rather than floating-point values — but **activations stay BF16**, where
  NVFP4 quantizes them to 4 bits too.
- **The PLE table is BF16**, twice the size of the NVFP4 checkpoint's FP8 table.
- Everything else is the same BF16 weights.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights (experts + BF16 part) | 72.9 GiB on disk; **69.1 GiB** as loaded by vLLM | **VRAM** |
| PLE table | 95.4 GiB | **RAM**, held by vLLM's PLE offload worker (RSS 97.0 GiB) |
| **Model total** | **168.3 GiB (180.8 GB)** | 72.9 GiB VRAM + 95.4 GiB RAM |
| KV cache — 605,187 tokens, BF16 | 14.7 GiB | VRAM |
| CUDA graphs (piecewise + full) | 0.2 GiB | VRAM |
| **VRAM in use** | **87.1–90.5 GiB** of 95.6 GiB | |
| **Host RAM** | **~115 GB** | |
| Disk besides the checkpoint | Docker image `vllm/vllm-openai:qwen38-flash-next` 19.8 GB | |

vLLM reports a maximum concurrency of 2.31 full 262,144-token requests for that pool.
The BF16 KV pool is more than twice the 256,832 tokens `sglang-nvfp4-ram-official`
gets with BF16 KV, since this checkpoint leaves more VRAM free and vLLM sizes the
pool differently.

## Measurements

Same prompts and scripts as the SGLang profiles ([benchmarks.md](benchmarks.md),
[comparison.md](comparison.md)). MTP speculative decoding is **off** (`SPEC_TOKENS=0`).

### Decode

| | tok/s |
|---|---|
| LRU-cache prompt, warm runs | 102.1 / 102.6 / 103.4 (first run 68.0, warmup) |

| Context behind the request | TTFT (cold) | Decode |
|---|---|---|
| 36 tokens | 0.07 s | 105.8–106.6 tok/s |
| 29K | 2.62 s | 104.0–110.1 tok/s |
| 116K | 11.1 s | 102.0–107.0 tok/s |
| 221K | 22.8–23.0 s | 100.6–100.7 tok/s |

Decode is flat across context, as with SGLang — the property comes from the model's
architecture — but at **less than half the SGLang NVFP4 profiles' 230–260 tok/s**.
The likely causes are the missing speculative decoding (SGLang runs NEXTN with four
draft tokens) and W4A16 Marlin kernels, which dequantize weights on every step,
against NVFP4's native Blackwell FP4 path.

### Prefill

| Prompt | Cold TTFT | Cold tok/s | Prefix-cached TTFT |
|---|---|---|---|
| 4K | 0.36 s | 11,156 | 0.37 s |
| 32K | 3.28 s | 9,721 | 0.74 s |
| 128K | 12.1 s | 10,573 | 0.65 s |

The first 4K request after startup took 1.28 s (warmup). Cold prefill is within
~10–20% of the SGLang RAM profiles.

### Long-context retrieval

Needle test (a 10-character code at 10/50/90% depth in 57K, 115K, 176K and
220K-token prompts): **12/12**.

## Setup notes

### `--distributed-executor-backend mp` is required on one GPU

Without it startup never finishes. The model loads, CUDA graphs are captured, and
then nothing: no `/health`, no further log lines, `VLLM::EngineCore` spinning a
core, and no PLE worker process in the container.

In this image the PLE offload worker is spawned and awaited only by the
multiprocess executor (`spawn_ple_offload()` and `wait_ple_offload_ready()` in
`v1/executor/multiproc_executor.py`). With a single GPU vLLM defaults to running
the model in-process (the uniproc executor), which has no PLE offload code at all.
The published recipes all use two or more GPUs, so they never hit it. The launcher
passes `--distributed-executor-backend mp`; the log then shows a `PleOffloadWorker`
process and `Worker ready - 1 PleOffloadLayer(s)`.

### Other differences from the SGLang profiles

- **No prefix-cache flush endpoint.** SGLang's `POST /flush_cache` does not exist in
  vLLM (and `/reset_prefix_cache` only in dev mode). `bench/prefill.py` tolerates
  both; its per-run nonce keeps cold runs cold regardless.
- **CUDA graphs work with PLE offload** in this image — both piecewise and full
  graphs were captured — despite the upstream PLE offload PR listing them as
  incompatible.
- **Thinking is on by default** here too; send `chat_template_kwargs:
  {"enable_thinking": false}` for plain answers.
- The log prints harmless `min_frames` / `max_frames` docstring errors from the
  transformers Qwen3-VL video processor at startup.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `CTX` | 262144 | `--max-model-len` |
| `MAX_SEQS` | 4 | `--max-num-seqs`, concurrent requests |
| `GPU_UTIL` | 0.92 | `--gpu-memory-utilization` |
| `KV_DTYPE` | `auto` (BF16) | `fp8` roughly doubles the KV pool; not tested here |
| `SPEC_TOKENS` | 0 | >0 enables MTP speculative decoding with that many draft tokens; the vLLM recipe notes it can *reduce* throughput; not tested here |
| `EXTRA_ARGS` | — | appended to `vllm serve` |

The launcher refuses to start unless the checkpoint is W4A16 and at least 100 GiB
of host RAM is available.
