# vLLM with the AWQ INT4 group-32 checkpoint

Profile `vllm-awq-w4a16-g32`, directory `qwen3.8-flash-next/vllm/awq-w4a16-g32/`,
started with `scripts/start-qwen3.8-flash-next-vllm-awq-w4a16-g32.sh`.

It runs
[cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
@ `d39638a0` on the same vLLM preview image and with the same launcher settings as
`vllm-awq-w4a16` ([vllm-awq-w4a16.md](vllm-awq-w4a16.md)). Measured 2026-09-14.

## Checkpoint

| Component | Parameters | Precision | Bits / param | On disk |
|---|---|---|---|---|
| Routed experts | 120.8B | INT4 weight-only, **group 32, asymmetric**, MSE observer | 4.63 | 65.0 GiB |
| Attention, linear attention, router, shared expert, MTP, vision, embeddings | 8.0B | BF16 | 16.00 | 14.9 GiB |
| PLE n-gram table | 51.2B | BF16 | 16.00 | 95.4 GiB |
| **Model total** | **180.0B** | | **8.37** | **175.3 GiB** |

Against `vllm-awq-w4a16`'s checkpoint (group 128, symmetric, 4.13 bits) the
experts get four times as many scales plus zero points. Calibration used STEM and
agentic data in ten languages (English, Chinese, Hindi, Arabic, Russian, Japanese,
Korean, Dutch, French, Spanish), per its model card. Everything outside the routed
experts is the same BF16. vLLM runs it with the Marlin WNA16 MoE kernels.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights as loaded | **76.2 GiB** (vs 69.1 GiB for group 128) | VRAM |
| PLE table | 95.4 GiB | RAM, vLLM's PLE offload worker |
| KV cache — **315,039 tokens**, BF16 | 7.7 GiB | VRAM |

The extra 7 GiB of weights halve the KV pool of `vllm-awq-w4a16` (605,187
tokens), which still holds one full 262,144-token request.

## Measurements

| | `vllm-awq-w4a16-g32` | `vllm-awq-w4a16` |
|---|---|---|
| Decode, LRU-cache prompt, warm | **109.5 / 109.7 / 110.1 tok/s** | 102.1 / 102.6 / 103.4 |
| TTFT 4K cold / cached | 1.83 s¹ / 0.78 s | 0.36 s / 0.37 s |
| TTFT 32K cold / cached | 3.36 s / 0.73 s | 3.28 s / 0.74 s |
| TTFT 128K cold / cached | 12.1 s / 0.66 s | 12.1 s / 0.65 s |
| Needle | 3/3 at 170K (10/50/90% depth) | 12/12 to 220K |

¹ First request after startup; includes warmup (`vllm-awq-w4a16`'s first request
took 1.28 s).

The finer groups cost nothing measurable in speed: prefill is identical and decode
is ~7% faster, likely from the Marlin kernel configuration for group 32 rather than
from the checkpoint itself (not investigated).

## Quality

- Code: HumanEval 0.970 / HumanEval+ 0.951, MBPP 0.937 / MBPP+ 0.796 — within noise of
  every other Flash-Next quantization.
- Slovak blind check: first of nine (75 of 100) and first of seven in the run before;
  in a four-way run level with FP8 (70 vs 71).

Both in [RESULTS.md](../../../RESULTS.md).
