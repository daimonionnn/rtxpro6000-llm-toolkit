# vLLM, uncensored AWQ INT4 group 32, with a PLE patch

Profile `vllm-awq-w4a16-g32-uncensored`, directory
`qwen3.8-flash-next/vllm/awq-w4a16-g32-uncensored/`, started with
`scripts/start-qwen3.8-flash-next-vllm-awq-w4a16-g32-uncensored.sh`.

## Checkpoint

[leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32)
@ `fa561462`, quantized from
[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored),
an abliterated (refusal-removed) build of Qwen3.8-Flash-Next.

| Component | Parameters | Precision | Bits / param | On disk |
|---|---|---|---|---|
| Routed experts | 121.6B | INT4 AWQ, AutoAWQ GEMM layout, group 32, zero points | 4.65 | 65.8 GiB |
| Attention, linear attention, router, shared expert, MTP, vision, embeddings | 8.1B | BF16 | 16.00 | 15.1 GiB |
| PLE n-gram table | 51.2B | FP8, the official table with its global scale | 8.00 | 47.7 GiB |
| **Model total** | | | **6.11** | **128.6 GiB** |

It also carries calibrated FP8 KV scales, which this image cannot use: its QSA
attention requires a BF16 KV cache. The optional FP8 MTP draft (`mtp-fp8/`, 5.3 GB)
is not needed and was not downloaded.

> **Refusals are removed** in the source model. Output is the operator's
> responsibility.

## Why it needs a patch and a filtered view

The checkpoint's author states that generic AWQ support is not enough to load it.
On the vLLM preview image two things break:

1. **FP8 PLE table next to AWQ.** vLLM selects its FP8 PLE embedding method only
   when the whole checkpoint is FP8 (`Fp8Config`). Here the table would be created
   in BF16 and the FP8 bytes copied in without their ~0.0002 scale — silently wrong
   embeddings. `patches/0001-fp8-ple-with-non-fp8-quantization.patch` also selects
   the FP8 method when `text_config.ple_embedding_dtype` is `float8_e4m3fn`. The
   launcher applies it to the image's own `ple_layer.py` at every start (failing if
   it no longer applies) and mounts the result read-only.
2. **Unlisted tensors in reused shard files.** The PLE table comes as complete shard
   files of the official FP8 checkpoint; two of them also hold 1,056 tensors the
   index does not list (1.0 GiB: FP8 experts and duplicate PLE buffers). vLLM reads
   every tensor in a file and stops at the first one:
   `has no parameter 'w2_weight' for checkpoint weight '…experts.0.down_proj.weight'`.
   `index_filter.py` builds `models/Qwen3.8-Flash-Next-Uncensored-AWQ-g32.indexed/`
   — every file hard-linked, the two files rewritten with only listed tensors
   (~1.5 GiB extra) — and that directory is served. The download itself is
   untouched and still verifies against its `SHA256SUMS`.

## Measurements

2026-09-15, same machine.

| | |
|---|---|
| Model weights as loaded | 76.1 GiB |
| KV cache | **339,153 tokens**, BF16 (8.2 GiB) |
| VRAM in use | 87.4 GiB |
| Decode, LRU-cache prompt with code | 110.6 / 109.0 / 109.8 tok/s |
| HumanEval / HumanEval+ | **0.988 / 0.970** (g32 baseline: 0.970 / 0.951) |
| MBPP / MBPP+ | 0.942 / 0.799 (g32 baseline: 0.937 / 0.796) |
| Plus tests passed, of 542 | **461**, the most of any profile (g32: 457) |
| Tool calls, Slovak and English sanity answers | correct |

Task by task against `vllm-awq-w4a16-g32` it solved 10 plus tests the original
missed and missed 6 it solved (sign test p = 0.45): the abliteration cost no
measurable code ability. No crash during the MBPP+ run at concurrency 4.

Slovak blind check: fourth of nine on score (70 of 100) with the best mean rank and
the most first places; second of four against the dense 27B models (70, original g32
77). No clear cost, though it scored below the original g32 in both runs
([RESULTS.md](../../../RESULTS.md#non-english-slovak-blind-check)).

## Caveats

- **One crash under the first concurrent load after startup.** Four parallel
  requests at the start of a benchmark made the PLE offload connector fail with
  `queue.Full` and the engine died; the container restarted itself (restart
  policy) within ~5 minutes. Afterwards, four concurrent requests and a full
  HumanEval run at concurrency 4 went through without a failure. Not reproduced;
  it may be a race in the FP8 PLE offload path, which no other profile exercises
  under concurrency.
- The patch targets the pinned preview image. A different image may need it
  refreshed — the launcher refuses to start if it does not apply.
- Same-mother BF16 validation was never published for this checkpoint.
