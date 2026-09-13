# Keeping the PLE table in RAM instead of NVMe

**Solved, three ways.** The first attempt (at the bottom of this document) failed:
pinned-host PLE cost 1.83 GB of VRAM and the KV cache collapsed. On 2026-09-12
three approaches found on the web were tested, all with the table in RAM, all
passing a needle-in-a-haystack test. Every one of them beats the NVMe baseline
on context and prefill.

| | NVMe baseline | Variant 1 | Variant 2c | Variant 3 |
|---|---|---|---|---|
| What | our image, `serve-nvfp4-nvme.sh` | our image, memory flags | official image, cookbook recipe | jpezzulli fork, native |
| PLE table | NVMe (io_uring) | pinned RAM | pinned RAM | pinned RAM |
| **KV cache** | 231,936 | 498,624 (FP8) | 256,832 (BF16) | **831,872 (FP8)** |
| **Context window** | 262,144 | 262,144 | 262,144 | **524,288** (YaRN ×2) |
| Decode, warm | 214–222 tok/s | 236–249 | 258–260 | 235–254 |
| TTFT 4K / 32K / 128K cold | 0.55 / 4.37 / 18.0 s | 0.31 / 2.90 / 10.7 s | 0.30 / 2.45 / 10.4 s | **0.27 / 2.62 / 9.68 s** |
| Needle | not run | 12/12 to 220K | 12/12 to 220K | **15/15 to 492K** |
| Concurrency | 5 | 4 | 4 | 4 |
| Host RAM added | ~0 | ~65 GB | ~65 GB | ~65 GB |
| Local patches / build | yes | yes | **none** | fork + native build + host shim |

Decode differences between the RAM variants are inside the spread of three warm
runs and should not be read as a ranking. Context, KV size, prefill and needle
results are robust.

---

## Variant 1: same image, mamba cache shrunk

Same image, same checkpoint, PLE in pinned host RAM. The 1.83 GB the RAM path
costs is won back from the mamba state cache, which was taking 5.34 GB:

| Setting | NVMe baseline | Variant 1 |
|---|---|---|
| `--mamba-radix-cache-strategy` | `extra_buffer` | `extra_buffer_lazy` |
| `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` | unset | `1` |
| State slots per request | 5 | 3 |
| `--mamba-ssm-dtype` | model default | `bfloat16` |
| `--max-running-requests` / `--max-mamba-cache-size` | 8 (capped to 5) / 25 | 4 / 12 |
| `--chunked-prefill-size` | 8192 | 4096 |
| `--mem-fraction-static` | 0.95 | 0.96 |
| `--max-total-tokens` | 393216 | unset (engine sizes the pool) |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `expandable_segments:True` |
| Docker | — | `--ulimit memlock=-1` |

KV stays FP8 with the local chunked-prefill patch; NEXTN speculation and breakable
CUDA graphs are unchanged.

The slot arithmetic comes from `kv_cache_configurator.py`: a base ratio of 3, minus
1 with `SKIP_DECODE_LOCK`, plus 2 for `extra_buffer` or 1 for `extra_buffer_lazy`
under the overlap scheduler.

### Results

| | NVMe baseline | Variant 1 (RAM) |
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
| VRAM in use | 91.3 GiB | 92.1–92.9 GiB |
| Host RAM | ~0 | +67 GB used (+64 GB shared) |

Same prompts, settings and scripts as [benchmarks.md](benchmarks.md). Decode is
three warm runs after a warmup, so treat the ~10% gain as a range.

**Prefill gains the most** — a prefill step gathers PLE rows for every prompt
token, so reading them from the SSD cost far more there than in decode. The first
4K request after startup measured 4,091 tok/s; repeats gave 12,811–12,911, so that
first figure was warmup.

**Pinned memory does not show in the process's `VmLck` or `VmRSS`** (16 kB and
2.1 GB). `cudaHostAlloc` pins through the driver, not `mlock`. The evidence that
the table is in RAM is indirect: no `Qwen4 PLE NVMe table` line in the log, and
host `used`/`shared` jumping by ~65 GB at load.

The needle test inserts a 10-character code at 10/50/90% depth into 57K, 115K,
176K and 220K-token prompts and asks for it back, flushing the prefix cache first.
It covers the FP8 KV cache, chunked prefill across dozens of chunks, and the RAM
PLE path together.

---

## Variant 2: official image, cookbook recipe

`lmsysorg/sglang:dev-qwen38-next-local` (commit `4ccff141db`, pulled 2026-09-12,
33 GB) with SGLang's verified cookbook cell for 1× RTX PRO 6000, NVFP4 RadixArk,
low latency. Launcher: `sglang/v2-official-image/serve-nvfp4-ram.sh`,
whose defaults reproduce the published cell; `MAXRUN`, `MAMBA_SLOTS` and
`KV_DTYPE` tune it.

It shares Variant 1's memory settings (`extra_buffer_lazy`, `SKIP_DECODE_LOCK`,
bf16 SSM, chunked prefill 4096, `expandable_segments`, `memlock=-1`) and differs
in: stock image with no local patches, `flashinfer_cutlass` for both FP4 GEMM and
the MoE runner, and the default CUDA graph backend instead of `breakable`.

| | 2a: recipe as published | 2b: 4 req, FP8 KV | 2c: 4 req, BF16 KV |
|---|---|---|---|
| Concurrency / mamba slots | 16 / 48 | 4 / 12 | 4 / 12 |
| KV cache | 76,864 (BF16) | 498,624 (FP8) | **256,832 (BF16)** |
| Mamba cache | 6.34 GB | 1.79 GB | 1.79 GB |
| Free VRAM after graphs | 4.10 GB | 4.68 GB | 4.64 GB |
| Needle, 57K–220K × 3 | not run | **crash on first prompt** | **12/12** |
| Decode, warm | 248–263 tok/s | — | **258–260 tok/s** |
| Cold prefill 4K / 32K / 128K | 13,217 / 10,828 / — tok/s | — | **13,377 / 12,997 / 12,217** |
| TTFT 4K / 32K / 128K | 0.30 / 2.94 / — s | — | 0.30 / 2.45 / 10.4 s |

**2a reproduces the published numbers** — the cookbook states ~78k KV tokens and
4.2 GB free; this card gave 76,864 and 4.10 GB. The image works as documented.

**2b shows the FP8-KV bug is still there.** The KV pool sizes to 498,624 tokens,
identical to Variant 1, so the official image has the same memory ceiling. But
the first 57K-token prompt killed the scheduler:

```
AssertionError: Unsupported rhs dtype fp8e4nv
triton.compiler.errors.CompilationError: at 79:42
```

That is the chunked-prefill crash the local `0001-qsa-fp8-kv-dequant-on-read.patch`
exists for; upstream's fix (#36644) was unmerged on 2026-09-12. With
`--restart unless-stopped` the container then crash-loops. The cookbook leaving
KV at BF16 is presumably why.

**2c is the usable configuration of this image.** BF16 KV halves the pool
relative to FP8, but 256,832 tokens still covers nearly the whole 262,144-token
window for one request, and decode and prefill are the fastest measured so far.

### Variant 1 vs 2c

| | Variant 1 (our image, FP8 KV) | Variant 2c (official, BF16 KV) |
|---|---|---|
| KV cache | **498,624** | 256,832 |
| Decode, warm | 236–249 tok/s | **258–260 tok/s** |
| TTFT 128K cold | 10.7 s | **10.4 s** |
| Needle to 220K | 12/12 | 12/12 |
| Local patches / build | yes | **none** |

For a single agent the window caps a request at 262,144 tokens either way, so
Variant 1's larger pool mostly buys prefix-cache retention across turns and room
for concurrent requests. Variant 2c trades that for ~5% faster decode and no
local build. The decode difference is within the spread of three warm runs; it is
suggestive, not established.

---

## Variant 3: jpezzulli/sglang-rtxpro6000 fork

[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
("Pennyroyal"), tag `pennyroyal-v2.5.0` (commit `2c675da096`): a personal SGLang
fork tuned for exactly this card. Built natively per its `BUILD.md` into
`sglang/pennyroyal-fork/` (Python 3.12.13 venv, CUDA 13.3, GCC 15, torch
2.13.0+cu130). Launcher: `sglang/v3-pennyroyal/serve-nvfp4-ram.sh` — the fork's
`configs/pennyroyal/serve-flash-next.sh` (native NEXTN, no FR-Spec) with HiCache/NIXL
removed, since NIXL was not installed. HiCache persists prefix state to host RAM
and disk; it does not change the GPU KV pool.

What it has that Variants 1 and 2 cannot:

- **`--gdn-mtp-cache-mode none`** — not in upstream SGLang. MTP verify normally
  keeps an intermediate SSM state per draft position (1.05 GB at these settings);
  `none` drops that buffer and re-runs the recurrence from the committed state
  over the accepted draft prefix instead. The log confirms
  `intermediate_ssm_state_cache size: 0.00GB`.
- **`--mem-fraction-static 0.981`** so automatic KV sizing uses the freed memory.
  The fork's README is candid that the estimator does not account for `none` mode
  and the higher fraction compensates.
- **YaRN factor 2** via `--json-model-override-args`, for a 524,288-token window.
- FP8 KV with its own chunked-prefill handling.

### Results

| | |
|---|---|
| KV cache | **831,872 tokens** FP8 (fork's README: 824,384) |
| Context window | 524,288 |
| Mamba cache | 1.39 GB, 24 slots, intermediate SSM 0.00 GB |
| Free VRAM after graphs | 3.78 GiB (VRAM in use 93.1 GiB) |
| Decode, warm runs | 234.6 / 249.1 / 253.8 tok/s |
| Cold prefill 4K / 32K / 128K | 14,992 / 12,174 / 13,171 tok/s |
| TTFT 4K / 32K / 128K | 0.27 / 2.62 / 9.68 s |
| TTFT 492K | 57.3 s — about 8,600 tok/s (fork's README: 8,773 at 490K) |
| Needle 57K / 115K / 220K / 352K / 492K × 3 depths | **15/15** |
| Scheduler crashes | 0 |

The first 4K request after startup measured 894 tok/s while kernels finished
compiling; the repeat gave 14,992.

### Performance with HiCache on (2026-09-13)

Measured on the running `HICACHE=1` server. HiCache did not measurably change
decode or cold prefill against the runs above.

**Decode stays flat as context grows.** 500 generated tokens after a cold prompt
of each length, two runs each (`enable_thinking: false`, temperature 0):

| Context | TTFT | Decode |
|---|---|---|
| 36 tokens | 0.08 s | 228 tok/s |
| 7.3K | 0.52 s | 244–247 tok/s |
| 29K | 2.13 s | 231–243 tok/s |
| 116K | 8.7 s | 235–239 tok/s |
| 221K | 18.1 s | 235–236 tok/s |

That is the architecture showing: three of every four layers are linear
attention with a fixed-size recurrent state, and the full-attention layers use
QSA sparse attention with a 2,048-token indexer budget, so the per-token cost of
decode barely depends on how much context sits behind it.

**Prefill, cold and prefix-cached** (`bench/prefill.py`):

| Prompt | Cold TTFT | Cold tok/s | Cached TTFT |
|---|---|---|---|
| 4K | 0.29 s | 13,857–13,943 | 0.10 s |
| 16K | 1.14 s | 13,983–14,024 | 0.33 s |
| 32K | 2.36 s | 13,506 | 0.16 s |
| 64K | 4.77 s | 13,368 | 0.27 s |
| 128K | 9.89 s | 12,888 | 0.45 s |
| 255K | 21.5 s | 11,834 | 0.83 s |

Cold prefill slows only gently with length — about 12% from 4K to 255K.

**Benchmarking pitfall with HiCache.** The first run of this table showed 4K at
28,709 tok/s. `/flush_cache` clears the GPU and host tiers but not the NIXL files,
and `bench/prefill.py` generated its prompts from a fixed seed, so a repeated run
restored its "cold" prompts from disk. The script now adds a random per-run nonce.
It also retries `/flush_cache`, which returns HTTP 400 for a moment after a
request finishes while HiCache is still writing it through.

### HiCache with NIXL persistence

`HICACHE=1 ./serve-nvfp4-ram.sh` adds the fork's hierarchical cache: a 32 GB
host-RAM tier with write-through to NIXL POSIX files (io_uring, `O_DIRECT`), in a
namespace directory derived from the whole configuration by the fork's
`derive_namespace.py`. NIXL is built by `sglang/v3-pennyroyal/build-nixl.sh` into
`sglang/v3-pennyroyal/nixl`. The GPU KV pool is unchanged at 831,872 tokens; host RAM
in use rises by the 32 GB tier.

Tested 2026-09-13 with two needle prompts (code at 50% depth, fixed seeds so the
prompt is byte-identical each time), sent three times: cold, again in the same
process, and after a full `stop.sh` + relaunch into the same namespace:

| | 57,678 tokens | 220,011 tokens |
|---|---|---|
| Cold | TTFT 4.34 s · PASS | TTFT 17.71 s · PASS |
| Same process (GPU radix) | 0.28 s · PASS · 57,664 cached | 1.01 s · PASS · 219,968 cached |
| **After restart (NIXL)** | **0.53 s · PASS** · 64 recomputed | **1.51 s · PASS** · 64 recomputed |

The log after the restart confirms the source:

```
HiCache prefetch success req=… completed=219968 matched=0 loaded=219968 occupied=0
```

`matched=0` — nothing on the GPU in the fresh process — and `loaded=219968` from
storage. Both needles were still answered correctly, so the silent context loss
upstream HiCache showed after a restore (see troubleshooting) did not occur here.
Those two prompts wrote 5.6 GB in 13,053 files.

**The cleaner watermarks had to change.** The fork's `nixl-posix.toml` evicts when
the *whole filesystem* passes 54.6% and stops at 53.0%. This root filesystem was
already at 88%, so the cleaner would have evicted continuously and nothing would
persist. `sglang/v3-pennyroyal/nixl-posix-local.toml` uses 92 / 90, about 140 GB of
headroom; the startup log confirms `HiCacheL3Cleaner started: … high=92.0% low=90.0%`.
Anything else filling the disk past 92% also triggers eviction.

Harmless noise in the log: `POSIX path-mode open failed: nixl::FileFd("/nonexistent-nixl-probe")`
is NIXL deliberately probing which registration mode works, and
`hybrid pool mamba is not OS-page-aligned. Falling back to bounce buffers` means
the mamba state is copied through a bounce buffer rather than zero-copy.

### Things that broke first, all specific to this machine

**1. `cuda-tile` needs `wheel_stub`.** The fork's install command uses
`--no-build-isolation`, so uv will not fetch build dependencies, and `cuda-tile`
builds through NVIDIA's `wheel_stub`:

```
ModuleNotFoundError: No module named 'wheel_stub'
```

`sglang/v3-pennyroyal/build.sh` adds `wheel_stub` to the bootstrap packages.

**NIXL's build assumes an activated venv and `pip`.** Following BUILD.md from a
script that calls the venv's Python directly failed three ways in turn:
`pip install .` injects its build isolation through a `sitecustomize.py` on
`PYTHONPATH`, which leaked into the nested `uv build` NIXL runs for its
`nixl-meta` wheel and broke its interpreter (`No module named '__future__'`);
`./contrib/tomlutil.py`'s shebang picked the system Python, which lacks
`tomlkit`; and `meson` found no `pybind11-config` on `PATH`.
`sglang/v3-pennyroyal/build-nixl.sh` uses `uv pip install`, puts the venv's `bin`
first on `PATH`, and installs `pybind11`.

**2. TileLang compiled kernels for the AMD card.** The first launch allocated
memory correctly and then died:

```
tilelang/rocm/op/gemm/gemm_mfma.py ... compute_warp_partition
tvm.error.InternalError: Check failed: (N % kNPerWarp == 0) is false:
N must be divisible by 16, but got 8
```

The host has ROCm installed for the Radeon AI PRO R9700, including
`/usr/bin/hipcc`. TileLang decides whether ROCm is present by running
`which hipcc`, and its ROCm detector runs before CUDA's, so it chose the `hip`
target. The Docker variants never hit this because the container has no
`hipcc`. TileLang has no environment variable to force a target, so the
launcher puts `sglang/v3-pennyroyal/shim/` first on `PATH`: a `which` that answers
"not found" for `hipcc` only and defers to `/usr/bin/which` otherwise. Verified:
without it `auto_detect_target()` returns `hip`; with it,
`{"kind":"cuda", ..., "arch":"sm_120a"}`. Nothing outside that one server process
is affected.

### Caveats

- **YaRN is applied to every prompt, not only long ones.** A static rope-scaling
  factor can change quality at short context. That was not measured here — the
  needle test checks retrieval, not reasoning or code quality. Worth an eval
  (e.g. HumanEval+) against Variant 1 before relying on it for a coding agent.
- **The recipe defaults thinking on** (`enable_thinking: true`,
  `reasoning_effort: medium`) and uses its own pinned chat template
  (`froggeric-v22.5.jinja`). Sending `enable_thinking: false` is respected —
  checked: content `'OK'`, empty reasoning, `finish_reason: stop`.
- **HiCache: do not abort a request with a terminal signal.** The fork's RUN.md
  asks for protocol-level cancellation; cutting a request off mid-write is not
  covered by its persistence guarantees.
- **A single maintainer's fork**, 70 stars. Updates follow its tags rather than
  upstream SGLang.
- **Native, not Docker**: no restart policy, so it does not come back after a
  reboot; stopped with `sglang/v3-pennyroyal/stop.sh`, which signals the whole
  process group.
- `torchcodec` logs `libavutil.so.56: cannot open shared object file` at startup.
  That is video decoding (it wants FFmpeg 4's libavutil 56); text serving is
  unaffected.

---

## The first attempt (failed)

What follows is the original experiment, kept because it explains why the naive
switch does not work.

### The question

Of the model's 176B parameters, 51B are an N-gram (PLE) lookup table — 47.68 GiB
in this fp8 checkpoint. Upstream streams it off NVMe with io_uring so the host
does not need 100+ GB free. On a machine with 244 GB of RAM that trade looks
backwards, so it is worth checking.

It is more backwards than it first appears: the io_uring backend opens the files
with **`O_DIRECT`** (`qwen4_ple_nvme.py:348`), which bypasses the page cache
entirely. Having 226 GB sitting in `buff/cache` does not help — every gather goes
to the SSD for real.

### The path exists

SGLang has a first-class host-RAM implementation, not a workaround.
`Qwen4ExpPinnedHostEmbedding` in `qwen4_exp.py:766`:

> *"PLE table read directly from pinned host memory. The table stays in its
> checkpoint storage dtype (fp8 with a per-tensor weight_scale for fp8
> checkpoints, bf16 otherwise); gathers emit bf16."*

Its three preconditions are all met by this checkpoint: an unquantized embedding
table, dtype `float8_e4m3fn` (bf16 and fp8 are accepted), and no added vocabulary
rows.

`ple_offload_embedding=True` is **already set** — SGLang enables it on its own.
It is simply overridden, because `qwen4_exp.py:495` checks
`SGLANG_QWEN4_PLE_NVME_PATH` first. So the entire switch is *not setting that
variable*.

The measurements below were taken with a `PLE_MODE=ram` switch that the launcher
had at the time. It has since been removed. To reproduce, drop the four
`SGLANG_QWEN4_PLE_NVME_*` environment variables from the `docker run` in
`serve-nvfp4-nvme.sh` — and rename the copy, because it no longer streams from
NVMe.

### It is faster

Same prompt and settings as [benchmarks.md](benchmarks.md):

| Mode | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| NVMe | 200.5 | 221.6 | 214.1 |
| RAM | 196.3 | 236.2 | 240.1 |

Warm runs land roughly 5–15% ahead. Treat the range, not a point estimate, as the
result — three samples with this much spread do not support a tighter claim. It
is still more than the upstream author's "under 2% of the decode budget", which
is consistent with the `O_DIRECT` finding above.

### It does not fit

The pinned host path uses **82.45 GB of VRAM against 80.62 GB** for the NVMe
path. That 1.83 GB comes straight out of the KV cache, which only had about
2.9 GB to begin with.

| Configuration | Result |
|---|---|
| RAM, mamba 25 | `ValueError` — will not start |
| RAM, mamba 25, `--mem-fraction-static 0.97` | `ValueError` |
| RAM, mamba 25, `--mamba-ssm-dtype bfloat16` | `ValueError` |
| RAM, mamba 25, `--chunked-prefill-size 2048` | `ValueError` |
| RAM, mamba 10 | starts, but KV collapses to **2,560** tokens (from 231,936) |

The error:

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.95. Raise --mem-fraction-static above 0.923
(minimum viable = 1 - available/pre = 0.9224).
```

### Why the obvious knobs do not help

**`--mem-fraction-static` is inert here.** From
`kv_cache_configurator.py:1827`, the slack is

```python
slack_gb = pre_model_load_memory * (1 - mem_fraction_static)
if self.mambaish_config is not None and self.post_capture_kv_active:
    slack_gb = max(slack_gb, pre_capture_activation_reserve_mb(...) / 1024)
```

For a mamba model the fixed floor is the larger term, so raising the fraction
changes nothing — the engine suggests the identical 0.923 at both 0.95 and 0.97.
The suggestion in the error message is misleading for this model.

**Shrinking the mamba cache backfires — on its own.** Cutting slots from 25 to 10
freed memory but also dropped `max_running_requests` to 2, and the KV pool ended
up far smaller rather than larger. *In hindsight this was the right lever used
badly:* Variant 1 shrinks the same cache, but by lowering the slots needed per
request (`extra_buffer_lazy`, `SKIP_DECODE_LOCK`) and halving the state
(`bfloat16`) rather than by starving concurrency, together with
`expandable_segments`.

**Reducing chunked prefill was not enough** either, despite
`pre_capture_activation_reserve_mb` deriving its activation tokens from
`max(chunked_prefill_size, 2048)` — the function has other terms that dominate.

### The one thing that would free enough — wrong

This attempt concluded that only disabling NEXTN speculation (4.37 GB of draft
weights) would free enough, which is a bad trade at roughly 3× decode speed.
That was wrong: Variant 1 keeps NEXTN and fits with room to spare. One unexplained
difference: Variant 1's draft-model load line reports `mem usage=0.51 GB`, where
every earlier run reported 4.37 GB. The cause was not investigated.

### Superseded

The "when to revisit" conditions this attempt listed no longer apply — see
Variant 1 at the top of this document.
