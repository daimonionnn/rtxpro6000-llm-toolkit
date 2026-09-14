# ExLlamaV3 (TabbyAPI) with the EXL3 5.05 bpw checkpoint

Profile `exllamav3-exl3-5.05bpw`, directory `qwen3.8-flash-next/exllamav3/exl3-5.05bpw/`,
started with `scripts/start-qwen3.8-flash-next-exllamav3-exl3-5.05bpw.sh`.

It runs
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
branch `5.05bpw_h6_ng6` @ `7cef615f`, on TabbyAPI
`ghcr.io/theroyallab/tabbyapi:cu13` pinned by digest (`sha256:a0befead…`, built
2026-09-13, ExLlamaV3 1.5.0, torch 2.11 cu130). Measured 2026-09-14.

## Checkpoint

EXL3 is ExLlamaV3's trellis quantization. Packed tensors do not reveal parameter
counts, so `quant_info.py` reports sizes only:

| Component | Precision | On disk |
|---|---|---|
| Routed experts | EXL3 5.05 bpw | 70.8 GiB |
| Attention, router, shared expert, MTP, vision, embeddings | EXL3 (head 6 bpw, MTP 5, vision 6) and BF16 norms | 7.5 GiB |
| n-gram (PLE) table | 6 bpw | 36.4 GiB |
| **Model total** | 5.47 bits per parameter overall | **114.6 GiB** |

Unlike the vLLM checkpoints, attention, linear attention and the shared experts are
quantized too, at the same 5.05 bpw.

**Quality, published:** in the quantizer's own KL-divergence chart (in-domain text,
same trace for every format) 5.05 bpw scores **0.0040**, against 0.0067 for EXL3
4.05 bpw, 0.0100 for NVFP4 W4A16 and 0.0241 for NVFP4 W4A4; perplexity 1.3848
against 1.3842 for BF16. No multilingual measurement exists.

## Where it lives

| | Size | Where |
|---|---|---|
| Weights | ~78 GiB | VRAM |
| KV cache — 262,144 tokens, FP16 | the rest of the card | VRAM |
| **VRAM in use** | **92.0–93.4 GiB** | |
| n-gram table (`ngram_ram: true`) | ~36 GiB | host RAM (container 42.8 GiB) |

Load time: **47 s**, against 5–6 minutes for the vLLM profiles.

## Measurements

MTP drafting on (`draft_mode: mtp`), thinking off, temperature 0.

### Decode

| Prompt | Context behind it | tok/s |
|---|---|---|
| LRU cache explanation with code | 40 tokens | **215.5 / 215.6** |
| Slovak prose, ~300 words | 45 tokens | **119.0 / 119.0** |
| Slovak prose | 195K | 108.4 |
| Python code | 195K | 211.7 |

MTP drafts are accepted far more often on code than on prose, hence the gap. Decode
barely falls with context — unlike the 61 tok/s at 32K reported with ExLlamaV3 1.4.6
on the same card.

The first request after a start took 48 s to its first token while kernels
compiled; the cache is mounted from `exl3-5.05bpw/cache/` and survives restarts.

### Prefill

| Prompt | Cold TTFT | Cold tok/s | Prefix-cached TTFT |
|---|---|---|---|
| 4K | 0.68 s | 5,910 | 0.11 s |
| 32K | 4.83 s | 6,604 | 0.19 s |
| 128K | 19.4 s | 6,564 | 0.43 s |

Second run after startup; the first run was slower at 4K and 32K (1.12 s, 8.67 s)
while kernels warmed. Cold prefill is ~1.6x slower than the vLLM profiles; cached
prefixes come back faster.

### Long context and concurrency

- Needle: 3/3 at ~170K tokens (10/50/90% depth).
- Four concurrent ~32K-token streaming requests all completed (34 s each). A
  benchmark with ExLlamaV3 1.4.6 reported most such requests failing
  (`Request disconnected`); not reproduced with this image.
- Tool calls (`tool_format: qwen3_coder`) and `reasoning_content` work with thinking
  on and off.

### Quality

- Code: HumanEval 0.976 / HumanEval+ 0.957, MBPP 0.937 / MBPP+ 0.802 — the most plus
  tests passed (460 of 542), within noise of the other quantizations.
- Slovak blind check: 61 of 100, last of four, despite the lowest published KL
  divergence.

Both in [RESULTS.md](../../../RESULTS.md).

## Configuration

`serve-exl3-5.05bpw.sh` fills `config.template.yml` into `config.generated.yml` and
mounts it as `/app/config.yml`. The settings that matter for this model:

| Key | Value | Why |
|---|---|---|
| `ngram_ram` | `true` | keeps the n-gram table in RAM instead of reading it from disk per token |
| `draft_model.draft_mode` | `mtp` | uses the model's MTP head; ~+60% single-user decode in published runs |
| `cache_mode` | `FP16` | full-precision KV; `8,8` halves it at a mean KL of ~0.007 at 128K (measured on 4.05 bpw) |
| `tool_format` | `qwen3_coder` | Qwen3.8 tool-call format |
| `reasoning` + tokens | `true`, `<think>` / `</think>` | splits `reasoning_content` from `content` |
| `sampling.override_preset` | `qwen38_flash_next` | Qwen's temperature 1.0 / top_k 20 / top_p 0.95 when a client sends none |

| Variable | Default | Meaning |
|---|---|---|
| `CTX` | 262144 | `max_seq_len` and `cache_size` (a multiple of 256) |
| `CACHE_MODE` | `FP16` | or `k_bits,v_bits`, e.g. `8,8` |
| `MAX_SEQS` | 4 | `max_batch_size` |
| `CHUNK` | 4096 | prefill chunk size |
| `DRAFT_MODE` | `mtp` | `disabled` turns drafting off |
| `EXTRA_ENV` | — | extra `-e VAR=value` arguments for `docker run` |

## Differences from the vLLM and SGLang profiles

- **`usage` is `null` in non-streaming responses.** Streaming responses carry it.
  Clients that count tokens from non-streaming replies get nothing.
- **`/v1/models` has no `max_model_len`.** The context window is on `/v1/model`
  (`parameters.max_seq_len`); `scripts/status.sh` reads it there. Clients that probe
  only `/v1/models` fall back to their own default.
- **Sampling from a client wins** (`force: false`). If a client sends settings that
  make the model's reasoning drift into mixed-language gibberish — reported with
  unset sampling — set `force: true` in `sampler-qwen38-flash-next.yml`.
- **Older `cu13` images lack Python headers**, and Triton then fails on the first
  Gated DeltaNet kernel with `Python.h: No such file or directory`. The pinned image
  has them.
