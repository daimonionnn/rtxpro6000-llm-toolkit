# vLLM with the official FP8 checkpoint, experts partly in RAM

Profile `vllm-fp8-offload`, directory `qwen3.8-flash-next/vllm/fp8-offload/`, started
with `scripts/start-qwen3.8-flash-next-vllm-fp8-offload.sh`.

It runs Qwen's own
[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
@ `236dfdf2` on the vLLM preview image `vllm/vllm-openai:qwen38-flash-next`
(vLLM `0.1.dev20073+g8e685d198`) — the highest-precision checkpoint in this
toolkit, at a large cost in speed. Measured 2026-09-14 on the same RTX PRO 6000.

No single-GPU recipe for this checkpoint had been published. The published FP8
setups use two or more GPUs; this profile combines two offload mechanisms the
preview image already has.

## Checkpoint

Read from the safetensors headers by `qwen3.8-flash-next/quant_info.py`:

| Component | Parameters | Precision | Bits / param | On disk |
|---|---|---|---|---|
| Routed experts | 120.8B | FP8 W8A8, 128×128 blocks, dynamic activation scales | 8.00 | 112.5 GiB |
| Attention, linear attention, router, shared expert, MTP, vision, embeddings | 8.0B | BF16 (MTP routed experts FP8) | 13.48 | 12.6 GiB |
| PLE n-gram table | 51.2B | FP8 | 8.00 | 47.7 GiB |
| **Model total** | **180.0B** | | **8.24** | **172.8 GiB** |

Only the routed experts and the PLE table are 8-bit; everything the 4-bit
checkpoints keep in BF16 is BF16 here too.

## Where it lives

The routed experts alone are larger than the GPU, so part of them stays in host
RAM:

| | Size | Where |
|---|---|---|
| BF16 weights + 62 GiB of routed experts | **74.1 GiB** as loaded | **VRAM** |
| Routed experts, `OFFLOAD_GIB` | **50 GiB** | **pinned RAM**, read by the GPU in place (vLLM UVA offloader) |
| PLE table | 47.7 GiB | **RAM**, vLLM's PLE offload worker (RSS 46.5 GiB) |
| KV cache — 313,483 tokens, BF16 | 7.6 GiB | VRAM |
| **VRAM in use** | **~88 GiB** of 95.6 GiB | |
| **Host RAM** | **~131 GB** (engine worker RSS 57.9 GiB) | |

The launcher passes:

```
--offload-backend uva --cpu-offload-gb 50 --cpu-offload-params experts
```

`--cpu-offload-params experts` restricts offloading to the routed-expert tensors.
Without it vLLM offloads parameters in declaration order, dense weights first,
which is slower. With the UVA backend the GPU reads pinned host memory directly,
so a forward pass moves only the expert weights it touches — but it moves them
on every pass.

`OFFLOAD_GIB` trades speed for KV cache. 60 GiB left a 703,199-token pool (17.1
GiB); 50 GiB leaves 313,483, the smallest offload that still holds one full
262,144-token request. Below that the context window has to shrink.

## Measurements

Same prompts as the other profiles, thinking off, temperature 0. MTP speculative
decoding off.

| | `OFFLOAD_GIB=60` | `OFFLOAD_GIB=50` (default) |
|---|---|---|
| Decode, 250 tokens | 13.5 tok/s | **15.6 tok/s** |
| TTFT, 38-token prompt | 1.56 s | 1.32 s |
| TTFT, 8K prompt (cold) | 14.0 s (570 tok/s) | 11.9 s (679 tok/s) |
| TTFT, 69K prompt (cold) | — | 86.3 s (798 tok/s) |
| TTFT, same 69K prompt (prefix-cached) | — | 10.1 s |
| Decode behind 69K context | — | 16.1–16.6 tok/s |

Retrieval of a key from the 8K and 69K prompts was correct.

**Both speeds are bound by PCIe, not the GPU.** During decode with
`OFFLOAD_GIB=60` the link ran at Gen5 x16 with ~23.4 GB/s flowing host-to-GPU
(`nvidia-smi dmon -s t`) while the GPU drew 140–170 W of its 450 W limit: about
1.7 GB of expert weights per token. Consequences:

- Every forward pass pays for the experts it touches, so even a 38-token prompt
  takes over a second, and a prefix-cache hit on 69K tokens still takes 10 s
  (the uncached tail and the linear-attention state are recomputed through the
  experts).
- Offloading less is the only setting that helps; decode rose in proportion
  when the offload fell from 60 to 50 GiB.
- `--moe-backend marlin`, a faster GPU or more `--max-num-batched-tokens` would
  not change the transfer volume and were not tried.

For comparison on the same machine: `vllm-awq-w4a16` decodes at 102 tok/s and
prefills ~10,000 tok/s. A Q8_0 GGUF of this model under ik_llama.cpp, with the
routed experts of 17 layers in host RAM, decodes at 38 tok/s and prefills
1,440–1,830 tok/s (measured through its API with the benchmarks here, see
[RESULTS.md](../../../RESULTS.md)). That setup also computes the host-resident experts on the GPU:
without `-rtr`, llama.cpp and ik_llama offload those ops and copy the weights
across PCIe. Both engines therefore pay for expert weights in transit; ik_llama
gets ~2.3x the decode from a comparable amount of expert weight in RAM. The cause
was not measured. The mechanisms differ: vLLM's UVA offloader lets the GPU
kernels read pinned memory in place, while ik_llama copies the selected experts
to GPU buffers before running the kernel.

## Quality

- Slovak blind check: first of four in its first run (71 of 100, against AWQ g32
  70, g128 62 and EXL3 61), third of four in a second run (69, against
  ik_llama.cpp Q8_0 73, AWQ g32 70 and the uncensored g32 68). Its answers still
  carry the model's own errors — „vereta“, the missed „jablká“.
- Code benchmarks: not run. ik_llama.cpp Q8_0 already covers 8-bit weights — 454 of
  542, within noise of every profile with the original weights — and a run here would take 1–2 hours at
  this speed.

Both in [RESULTS.md](../../../RESULTS.md).

## When to use it

Rarely. **For 8-bit weights on one card, ik_llama.cpp with a Q8_0 GGUF is the
better choice** — more than twice the decode and prefill on the same machine. This
profile documents what vLLM can do with the official FP8 checkpoint and can serve
as a vLLM-side quality reference. For an interactive agent it is slow: a turn with
a long conversation behind it waits for the cached prefix (~10 s at 69K) plus
~800 tok/s for anything new, then generates at ~16 tok/s.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `OFFLOAD_GIB` | 50 | GiB of routed experts in pinned RAM (`--cpu-offload-gb`) |
| `CTX` | 262144 | `--max-model-len`; lower it to offload less |
| `MAX_SEQS` | 4 | `--max-num-seqs` |
| `BATCH_TOKENS` | 16384 | `--max-num-batched-tokens`, the prefill chunk size |
| `GPU_UTIL` | 0.92 | `--gpu-memory-utilization` |
| `KV_DTYPE` | `auto` (BF16) | this image's QSA attention requires BF16 KV |
| `MOE_BACKEND` | `auto` | Triton for block FP8 on SM120; `marlin` runs FP8 as W8A16 |
| `SPEC_TOKENS` | 0 | MTP draft tokens; drafts make each pass touch more experts — not tested |
| `EXTRA_ARGS` | — | appended to `vllm serve` |

The launcher refuses to start unless the checkpoint is FP8 and host RAM covers
the PLE table plus `OFFLOAD_GIB` plus 8 GiB. `--distributed-executor-backend mp`
is required on one GPU, as for `vllm-awq-w4a16` ([vllm-awq-w4a16.md](vllm-awq-w4a16.md)).
