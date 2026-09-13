# Upstream

This directory is a vendored copy of
[yepapa-nest/qwen38-flashnext-rtx6000](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000)
at commit `2ef81d5`, licensed Apache-2.0 (see `LICENSE`). It is used only to build
the Docker image `sglang-flashnext-sm120:local` for the NVMe and RAM SGLang profiles.

Changes made here:

- `Dockerfile` — base image pinned by digest, and `SGLANG_RUST_BUILD_MODE=auto` for
  the `_storage` prebuild. See `docs/upstream-fixes.md`.
- `serve.sh` — moved out of this directory. The launchers now live in the per-profile
  directories under `sglang/`, and the shared `quant_info.py` and Docker stop script
  are in `sglang/common/`.

`README.md` and `BENCHMARKS.md` are the upstream author's and still refer to
`serve.sh`. `.build/`, created by `build.sh`, is not tracked.
