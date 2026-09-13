# Changes relative to upstream

The [upstream repo](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000) is
sound work — it documents exactly why each of its changes is required and is
honest about the model's limits. Everything below is on top of commit `2ef81d5`.

## The base image tag moved

`Dockerfile` used `FROM lmsysorg/sglang:qwen38flashnext`, a floating tag. The
upstream README describes the image pushed **2026-08-26T12:30Z**. That tag now
resolves to an image built **2026-09-03T11:36:48Z**.

The rebuild ships `_grpc`, `_multimodal` and `_server` as prebuilt extensions but
**not `_storage`**, and it sets `ENV SGLANG_RUST_BUILD_MODE=never`. The loader
therefore refuses to invoke Cargo and the build dies on the last `RUN`:

```
ModuleNotFoundError: sglang.srt.rust_extensions._storage is not bundled
or cached, and Rust extension build mode is 'never'
```

This is the same trap the upstream author warns about for checkpoints ("day-0
uploads get silently amended") but did not apply to their own base image.

### `_storage` is not optional

The Dockerfile comment describes the prebuild as a startup optimisation. It is
not. `python/sglang/srt/models/qwen4_ple_nvme.py:324` pulls `IoUringReader` out
of it:

```python
IoUringReader = load_rust_extension(
    "sglang.srt.rust_extensions._storage"
).IoUringReader
```

That is the NVMe PLE streaming reader — the reason a 176B model fits on one card
at all. Without `_storage` there is no server, not merely a slower first start.

### The fix

Two changes in `qwen3.8-flash-next/sglang/build-local-image/Dockerfile`:

1. **Pin the base image by digest** so the tag cannot move again:

   ```dockerfile
   FROM lmsysorg/sglang:qwen38flashnext@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521
   ```

2. **Override the build mode for the prebuild step only:**

   ```dockerfile
   RUN cd /sgl-workspace/sglang \
    && SGLANG_RUST_BUILD_MODE=auto python3 -c "..."
   ```

Cargo 1.98 is already in the image and the `rust/sglang-storage` crate comes in
with the source overlay, so the crate compiles in about 9 seconds. The result is
cached under `/root/.cache/sglang/rust_extensions`, which the runtime finds even
with the image's `mode=never`, because `never` still permits bundled and cached
extensions — it only forbids invoking Cargo.

## Layout: the clone builds the image, launchers live per profile

The upstream repo is cloned as `qwen3.8-flash-next/sglang/build-local-image/` and is only used to
build the Docker image (`build.sh`, `Dockerfile`, `patches/`,
`seccomp-iouring.json`). The launchers that were added or renamed here moved out
of it, one directory per profile: `qwen3.8-flash-next/sglang/nvfp4-nvme/serve-nvfp4-nvme.sh` and
`qwen3.8-flash-next/sglang/nvfp4-ram/serve-nvfp4-ram.sh` both use that image. `quant_info.py` and the
Docker stop script are shared from `common/`.

## `serve.sh` renamed to `serve-nvfp4-nvme.sh`

The file name states the two things this launcher is fixed to, and both are
enforced rather than just labels:

- **`nvfp4`** — it passes `--quantization modelopt_fp4` and refuses to start a
  checkpoint that is not NVFP4 (via `quant_info.py`).
- **`nvme`** — the PLE table is always streamed from NVMe with io_uring. There is
  no switch to hold it in RAM.

The upstream `README.md` and `BENCHMARKS.md` inside the clone still say `serve.sh`;
they are left as the author wrote them.

## The RAM launcher is a separate file

For a while `serve-nvfp4-nvme.sh` had a `PLE_MODE=nvme|ram` switch. It was
removed when a plain switch to RAM turned out not to fit: the pinned-host path
costs 1.83 GB more VRAM and the KV cache collapsed to 2,560 tokens.

The RAM path was later made to fit by shrinking the mamba state cache
(`extra_buffer_lazy`, `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`, bf16 SSM state, fewer
slots), and then gave more KV cache than NVMe. Those settings differ from the NVMe
launcher in about a dozen places, so it lives in its own file,
`serve-nvfp4-ram.sh`, rather than behind a switch. That keeps the `nvme` in the
other file's name true. See [ple-ram-experiment.md](ple-ram-experiment.md).

## An `EXTRA_ARGS` passthrough in `serve-nvfp4-nvme.sh`

Appends arbitrary flags to the launch command, so options can be tried without
editing the script:

```bash
EXTRA_ARGS="--mamba-ssm-dtype bfloat16" ./serve-nvfp4-nvme.sh
```

## What needed no intervention

Steps 1–3 of `build.sh` ran clean: cherry-picks of PRs **#36556** (SM120/SM121
sparse decode), **#36749** (BCG buffers sized by token rows) and **#36750**
(`max_thinking_tokens` on the OpenAI endpoint), plus the local
`0001-qsa-fp8-kv-dequant-on-read.patch`.

## Upstream state, as of 2026-09-12

None of the PRs are merged into `main`:

| PR | State | Base branch |
|---|---|---|
| #36497 model support | closed, not merged | `main` |
| #36567 NVMe PLE streaming | open | `qwen4-main-squashed` |
| #36556 SM120/SM121 sparse decode | open | `qwen4-main-squashed` |
| #36749 BCG buffers by token rows | closed, not merged | `qwen4-main-squashed` |
| #36750 `max_thinking_tokens` | closed, not merged | `main` |
| #36644 proper FP8 KV fix in QSA | open | `qwen4-main-squashed` |

`qwen4_exp.py` is now in `main`, so model support landed, but
`qwen4_ple_nvme.py` exists only in the PR. A stock SGLang release will not work
on this card: without #36556 the server dies at startup with an `MLIRError` out
of the QSA kernel.

**If the cherry-picks break later:** #36556 is already in the
`qwen4-main-squashed` branch (updated 2026-09-07, newer than the repo's pin), so
re-pin `build.sh` to that tree and drop the #36556 cherry-pick.

**When #36644 merges:** switch to it and delete the local patch 4. Upstream
solves the same crash properly, with per-layer KV descale and a FlashAttention
fallback; the local patch assumes a direct-cast e4m3 store with scale 1.0.
