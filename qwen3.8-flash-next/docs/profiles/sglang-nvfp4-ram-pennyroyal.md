# SGLang pennyroyal fork, NVFP4, PLE table in RAM, 524K context

Profile `sglang-nvfp4-ram-pennyroyal`, directory
`qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/`, started with
`scripts/start-qwen3.8-flash-next-sglang-nvfp4-ram-pennyroyal.sh`, or
`…-pennyroyal-hicache.sh` for prefix persistence across restarts.

[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
("Pennyroyal"), tag `pennyroyal-v2.5.0` (commit `2c675da096`): a personal SGLang
fork tuned for exactly this card, with the RadixArk NVFP4 checkpoint. Built
natively per its `BUILD.md` into `qwen3.8-flash-next/sglang/pennyroyal-fork/`
(Python 3.12.13 venv, CUDA 13.3, GCC 15, torch 2.13.0+cu130) by `build.sh`; NIXL for
HiCache by `build-nixl.sh`. Setup: [setup.md](../setup.md); build traps:
[troubleshooting.md](../troubleshooting.md). Measured 2026-09-12 and 2026-09-13.

## What the fork adds

- **`--gdn-mtp-cache-mode none`** — not in upstream SGLang. MTP verification
  normally keeps an intermediate SSM state per draft position (1.05 GB at these
  settings); `none` drops that buffer and re-runs the recurrence from the committed
  state over the accepted draft prefix. The log confirms
  `intermediate_ssm_state_cache size: 0.00GB`.
- **`--mem-fraction-static 0.981`** so automatic KV sizing uses the freed memory.
- **YaRN factor 2** via `--json-model-override-args`, for a 524,288-token window.
  It is static: every prompt, short ones included, gets rescaled positions.
- FP8 KV with its own chunked-prefill handling.
- HiCache with NIXL POSIX persistence (below), opt-in with `HICACHE=1`.

## Where it lives

| | Size | Where |
|---|---|---|
| Model weights | 78.2 GiB | **VRAM** |
| PLE table | 47.7 GiB | **RAM**, pinned |
| **Model total** | **125.9 GiB** | 78.2 GiB VRAM + 47.7 GiB RAM |
| KV cache — 831,872 tokens, FP8 | 10.3 GiB | VRAM |
| Mamba cache — 24 slots, BF16 state, no intermediate buffer | 1.4 GiB | VRAM |
| Free after CUDA graph capture | 3.8–3.9 GiB | VRAM |
| **VRAM in use** | **91.4–93.1 GiB** of 95.6 GiB | |
| **Host RAM** | **~65 GB**, **~97 GB** with HiCache (32 GB host tier) | |
| Disk besides the checkpoint | fork + venv 11 GB, kernel cache 0.2 GB, NIXL 14 MB; HiCache files grow until the disk reaches 92% | |

## Measurements

| | |
|---|---|
| KV cache | **831,872 tokens** FP8 (fork's README: 824,384) |
| Context window | 524,288 |
| Decode, warm runs | 234.6 / 249.1 / 253.8 tok/s |
| Cold prefill 4K / 32K / 128K | 14,992 / 12,174 / 13,171 tok/s |
| TTFT 4K / 32K / 128K | 0.27 / 2.62 / 9.68 s |
| TTFT 492K | 57.3 s — about 8,600 tok/s (fork's README: 8,773 at 490K) |
| Needle 57K / 115K / 220K / 352K / 492K × 3 depths | **15/15** |
| Scheduler crashes | 0 |

The first 4K request after startup measured 894 tok/s while kernels finished
compiling; the repeat gave 14,992.

### With HiCache on

HiCache did not measurably change decode or cold prefill.

**Decode stays flat as context grows** — 500 generated tokens after a cold prompt
of each length, two runs each:

| Context | TTFT | Decode |
|---|---|---|
| 36 tokens | 0.08 s | 228 tok/s |
| 7.3K | 0.52 s | 244–247 tok/s |
| 29K | 2.13 s | 231–243 tok/s |
| 116K | 8.7 s | 235–239 tok/s |
| 221K | 18.1 s | 235–236 tok/s |

**Prefill, cold and prefix-cached** (`bench/prefill.py`):

| Prompt | Cold TTFT | Cold tok/s | Cached TTFT |
|---|---|---|---|
| 4K | 0.29 s | 13,857–13,943 | 0.10 s |
| 16K | 1.14 s | 13,983–14,024 | 0.33 s |
| 32K | 2.36 s | 13,506 | 0.16 s |
| 64K | 4.77 s | 13,368 | 0.27 s |
| 128K | 9.89 s | 12,888 | 0.45 s |
| 255K | 21.5 s | 11,834 | 0.83 s |

## HiCache with NIXL persistence

`HICACHE=1` adds the fork's hierarchical cache: a 32 GB host-RAM tier with
write-through to NIXL POSIX files (io_uring, `O_DIRECT`), in a namespace directory
derived from the whole configuration by the fork's `derive_namespace.py`. The GPU
KV pool is unchanged.

Tested with two needle prompts (fixed seeds, byte-identical each time), sent cold,
again in the same process, and after a full stop and relaunch into the same
namespace:

| | 57,678 tokens | 220,011 tokens |
|---|---|---|
| Cold | TTFT 4.34 s · PASS | TTFT 17.71 s · PASS |
| Same process (GPU radix) | 0.28 s · PASS · 57,664 cached | 1.01 s · PASS · 219,968 cached |
| **After restart (NIXL)** | **0.53 s · PASS** · 64 recomputed | **1.51 s · PASS** · 64 recomputed |

After the restart the log shows `HiCache prefetch success … matched=0
loaded=219968`: nothing on the GPU, everything loaded from storage, and both
needles still answered correctly. Those two prompts wrote 5.6 GB in 13,053 files.

- **Cleaner watermarks are whole-filesystem percentages.** The fork's sample values
  (evict above 54.6%, stop at 53.0%) would evict continuously on a disk that is
  already fuller; `nixl-posix-local.toml` uses 92 / 90. Anything else filling the
  disk past 92% also triggers eviction.
- **Benchmarking with HiCache needs fresh prompts.** `/flush_cache` clears the GPU
  and host tiers but not the NIXL files, so a repeated benchmark with fixed prompts
  restores its "cold" runs from disk; `bench/prefill.py` adds a per-run nonce.
- Harmless in the log: `POSIX path-mode open failed: nixl::FileFd("/nonexistent-nixl-probe")`
  is NIXL probing which registration mode works; `hybrid pool mamba is not
  OS-page-aligned. Falling back to bounce buffers` means the mamba state goes
  through a bounce buffer rather than zero-copy.

## Caveats

- **YaRN on every prompt.** A static rope-scaling factor can change quality at
  short context. On code it did not: HumanEval 0.963 / HumanEval+ 0.951, MBPP 0.918 /
  MBPP+ 0.794, 456 of 542 plus tests — within noise of every other profile
  ([RESULTS.md](../../../RESULTS.md#code-humaneval-and-mbpp)). Other tasks are not
  measured.
- **Thinking is on by default** in the fork's recipe (`reasoning_effort: medium`,
  its own chat template `froggeric-v22.5.jinja`); `enable_thinking: false` is
  respected.
- **HiCache: cancel requests at the protocol level**, not with a terminal signal;
  cutting a request off mid-write is not covered by its persistence guarantees.
- **A single maintainer's fork.** Updates follow its tags rather than upstream.
- **Native, not Docker**: no restart policy, so it does not come back after a
  reboot. `scripts/stop.sh` signals its whole process group.
- **Absolute paths**: moving the toolkit directory requires rebuilding the venv and
  NIXL ([troubleshooting.md](../troubleshooting.md)).
- `torchcodec` logs `libavutil.so.56: cannot open shared object file` at startup —
  video decoding only; text serving is unaffected.
