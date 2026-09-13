# Troubleshooting

## Thinking is on by default, and `content` can come back empty

The most likely thing to bite an agent integration. With a small `max_tokens` the
entire budget is spent on `reasoning_content`, and `content` is an empty string
with `finish_reason: "length"`:

```
no kwargs,   max_tokens=10  ->  content=''    reasoning='We need to respond...'  finish_reason='length'
enable_thinking=False       ->  content='OK'  reasoning=''                       finish_reason='stop'
no kwargs,   max_tokens=800 ->  content='OK'  reasoning='We need to respond...'  finish_reason='stop'
```

Either pin the mode per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

or give a generous `max_tokens` and handle an empty `content`. Upstream hit the
same thing — their eval harness runs "through a proxy that pins the thinking mode
and coerces null content".

Thinking is selected per request via `chat_template_kwargs.enable_thinking` and
`reasoning_effort`. It buys about 2 points of HumanEval+ upstream, at 22% of
coding throughput and 3.6× the TTFT — worth it for hard single-shot problems, not
for volume.

## `nvidia-smi`: Driver/library version mismatch

The loaded kernel module and the userspace library disagree, usually after a
driver update without a reboot. Everything CUDA fails, including `llama.cpp` in
LM Studio (`ggml_cuda_init: failed to initialize CUDA: system has unsupported
display driver / cuda driver combination`) and anything in Docker.

```bash
cat /proc/driver/nvidia/version                      # loaded module
ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1    # userspace
dkms status | grep nvidia                            # what will load next boot
```

`rmmod` will not work — the module is held by the display stack (refcount in the
dozens). Reboot. Afterwards, config-only leftovers from an older driver can be
cleared with `sudo apt purge '~c^nvidia' '~c^libnvidia'`.

## Build fails: `_storage is not bundled or cached`

```
ModuleNotFoundError: sglang.srt.rust_extensions._storage is not bundled
or cached, and Rust extension build mode is 'never'
```

The base image tag moved and the current image lacks `_storage`. See
[upstream-fixes.md](upstream-fixes.md) — the fix is `SGLANG_RUST_BUILD_MODE=auto`
on the prebuild `RUN`, plus pinning the image by digest.

## Server dies at startup with an `MLIRError`

The QSA kernel has no SM120 path. PR #36556 supplies it; make sure that
cherry-pick actually applied (`build.sh` step 2 prints `ok #36556`).

## First run dies with an io_uring error

Docker's default seccomp profile blocks `io_uring_setup`, `io_uring_enter` and
`io_uring_register`, so the PLE reader cannot start. `serve-nvfp4-nvme.sh` passes
`--security-opt seccomp=seccomp-iouring.json`, which is the default profile plus
those three syscalls. If you launch the container by hand, carry that flag.

The engine raises a clear error for this case:

```
io_uring is blocked; allow io_uring_setup, io_uring_enter,
and io_uring_register in the container seccomp profile
```

## `ValueError: Loaded weights leave no GPU memory for the KV cache`

Something is using more VRAM than the KV budget tolerates. The suggestion in the
message — raise `--mem-fraction-static` — **usually does not help for this model**:
the slack is `max(pre × (1 - fraction), mamba floor)` and the floor is the larger
term. The lever that works is the mamba state cache: `extra_buffer_lazy`,
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, `--mamba-ssm-dtype bfloat16`, and fewer
slots. See [ple-ram-experiment.md](ple-ram-experiment.md).

## Long prompts crash on the second chunk

```
AssertionError: Unsupported rhs dtype fp8e4nv
```

**On the official `dev-qwen38-next-local` image (`sglang-nvfp4-ram-official`)** this kills the
scheduler on the first long prompt whenever `--kv-cache-dtype fp8_e4m3` is set,
and `--restart unless-stopped` then crash-loops the container. That image has no
fix; run it with `KV_DTYPE=auto`. Our image carries the local patch below, and
the fork (`sglang-nvfp4-ram-pennyroyal`) handles it itself.

With `--kv-cache-dtype fp8_e4m3`, any prompt long enough to be chunked dies on
the second chunk. `patches/0001-qsa-fp8-kv-dequant-on-read.patch` handles it by
casting the selected KV rows back to the query dtype. Dropping
`--kv-cache-dtype fp8_e4m3` also avoids it, at the cost of roughly half the KV
cache.

## `Using FP8 KV cache but no scaling factors provided`

Expected with this checkpoint. The log line reads:

```
Defaulting to scaling factors of 1.0. This may lead to less accurate results!
```

That is exactly the assumption the local patch 4 is built on (a direct-cast e4m3
store, scale 1.0), so the two are consistent. Upstream #36644 does it properly
with per-layer KV descale — switch when it merges.

## Do not enable HiCache on the upstream-based images

The hierarchical KV cache attaches and reopens 2.4× faster, but after a restore a
needle-in-haystack probe answers "there is no code in the context" — silent
context loss, most likely because the QSA indexer's side cache is not tiered
along with KV. Off by default; leave it off in the NVMe launcher and Profiles 1
and 2.

The `sglang-nvfp4-ram-pennyroyal` fork fixes this. Tested here: 57K- and 220K-token needle prompts
restored from NIXL after a full restart and still answered correctly. Use
`HICACHE=1` with that launcher only.

## `sglang-nvfp4-ram-pennyroyal` dies with `N must be divisible by 16, but got 8`

```
tilelang/rocm/op/gemm/gemm_mfma.py ... compute_warp_partition
tvm.error.InternalError: Check failed: (N % kNPerWarp == 0) is false:
N must be divisible by 16, but got 8
```

The path says it: TileLang is compiling kernels for **ROCm**. This machine has
ROCm installed for the Radeon AI PRO R9700, including `/usr/bin/hipcc`, and
TileLang detects ROCm with `which hipcc` before it checks CUDA. It happens only
natively — Docker containers have no `hipcc`.

`qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/serve-nvfp4-ram.sh` fixes it by putting `qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/shim/`
first on `PATH`; the `which` there reports `hipcc` as not found and defers to
`/usr/bin/which` for everything else. If you launch the fork any other way, carry
that `PATH`. To check what TileLang will pick:

```bash
PATH="$PWD/sglang/nvfp4-ram-pennyroyal/shim:$PATH" qwen3.8-flash-next/sglang/pennyroyal-fork/.venv/bin/python \
  -c "from tilelang.backend.target import auto_detect_target; print(auto_detect_target())"
```

It should print a `cuda` target with `"arch":"sm_120a"`, not `hip`.

## `sglang-nvfp4-ram-pennyroyal` build fails: `No module named 'wheel_stub'`

`cuda-tile` builds through NVIDIA's `wheel_stub`, and the fork installs with
`--no-build-isolation`, so uv does not fetch it. `qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/build.sh`
installs `wheel_stub` into the venv first.

## NIXL build fails

`qwen3.8-flash-next/sglang/nvfp4-ram-pennyroyal/build-nixl.sh` already handles these; they appear if NIXL is
built by following the fork's BUILD.md from a script without an activated venv:

- `No module named '__future__'` inside `nixl-meta` — `pip install .`'s build
  isolation leaks through `PYTHONPATH` into NIXL's nested `uv build`. Use
  `uv pip install`.
- `No module named 'tomlkit'` from `contrib/tomlutil.py` — its shebang runs the
  system `python3`. Run it with the venv's Python.
- `Dependency "pybind11" not found` from `meson setup` — put the venv's `bin`
  first on `PATH` and install `pybind11` into it.

Host packages it needs: `meson` and `libaio-dev`.

## `sglang-nvfp4-ram-pennyroyal` breaks after the toolkit directory moves

The fork's venv, the NIXL build and the JIT caches under
`nvfp4-ram-pennyroyal/cache/` all record absolute paths. After moving or renaming the
toolkit:

- the launcher fails with `No module named 'sglang'`, and the venv's scripts point
  at an interpreter that no longer exists;
- if a symlink with the old name points at the new location, the server still
  starts, but the cached kernels keep writing to the old relative layout — a stray
  `sglang/…/cache/` directory appears in the toolkit root;
- HiCache starts a new, empty namespace under `nixl-storage/` (the namespace hash
  includes the paths); the old one is unused and can be deleted.

Rebuild from the new location, with the server stopped:

```bash
cd qwen3.8-flash-next/sglang
rm -rf pennyroyal-fork/.venv nvfp4-ram-pennyroyal/cache \
       nvfp4-ram-pennyroyal/nixl nvfp4-ram-pennyroyal/nixl-src/build-posix
nvfp4-ram-pennyroyal/build.sh && nvfp4-ram-pennyroyal/build-nixl.sh
```

Both builds reuse the uv cache and take about 3 minutes together; the first server
start afterwards recompiles its kernels. The Docker profiles are unaffected — their
mounts are resolved each time they start.

## HiCache never keeps anything

Check the cleaner watermarks in `nixl-posix-local.toml` against `df -h /`. They
are percentages of the **whole filesystem**. If the disk is fuller than the high
watermark, the cleaner evicts everything on every pass (every 30 s). The fork's
sample values, 54.6 / 53.0, only suit a mostly empty dedicated volume.

## A RAM profile fails to pin the PLE table

The 47.7 GiB table is page-locked host memory. Docker needs
`--ulimit memlock=-1` (the Docker RAM launchers pass it). A native process
inherits the limit of the shell that started it: interactive logins here have
`unlimited` from `/etc/security/limits.conf`, but systemd user services default
to 8 MB, so starting `sglang-nvfp4-ram-pennyroyal` from a service would fail. Its launcher checks
`ulimit -l` and refuses to start otherwise.

Pinned memory does not show in a process's `VmLck` or `VmRSS`; look at host
`used`/`shared` in `free -g` instead, which rise by about 65 GB.

## vLLM hangs at startup after capturing CUDA graphs

The log stops after `Graph capturing finished` and `Free memory on device …`,
`/health` never answers, `VLLM::EngineCore` spins one core, and there is no
`PleOffloadWorker` process in the container.

With `VLLM_PLE_CPU_OFFLOAD=1` on a single GPU, vLLM runs the model in-process
(uniproc executor), but only the multiprocess executor spawns and waits for the PLE
offload worker. Start with `--distributed-executor-backend mp`, as the
`vllm-awq-w4a16` launcher does. Details in [vllm-awq.md](vllm-awq.md).

## Benchmarks fail with HTTP 404 on `/flush_cache`

`/flush_cache` is SGLang's. vLLM has no equivalent outside its dev mode.
`bench/prefill.py` skips the flush when the endpoint is missing; its per-run nonce
keeps cold measurements cold anyway.

## Other traps carried over from upstream

- **Do not `git clean` the base image's tree.** It removes untracked build
  artifacts that ship with the image (`_grpc` / `_server` / `_multimodal` `.so`).
- **Pin the checkpoint revision.** Day-0 uploads get amended hours later without
  an announcement.
- **Speculative draft depth is capped at 4** by the QSA compression ratio. This
  is architectural; raising `--speculative-num-draft-tokens` does not help.
