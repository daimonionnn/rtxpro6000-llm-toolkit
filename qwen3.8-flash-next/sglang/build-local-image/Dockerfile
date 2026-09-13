# Qwen3.8-Flash-Next on a single RTX PRO 6000 Blackwell (SM120).
#
# Base is the official day-0 image. It does NOT contain the NVMe PLE path:
# the image was pushed 2026-08-26T12:30Z, before PR #36567 was opened
# (verified in the container: qwen4_exp.py present, qwen4_ple_nvme.py absent).
# So we overlay a source tree instead of pip-installing on top.
#
# Two rules learned the hard way:
#   - No git/network during the build. Fetch on the host (build.sh does), copy in.
#   - Never `git clean` the base tree: it deletes untracked build artifacts that
#     ship with the image (_grpc/_server/_multimodal .so files).
# Pinned by digest. The `qwen38flashnext` tag is floating and has already moved:
# README describes the 2026-08-26T12:30Z push, but the tag now resolves to an
# image built 2026-09-03T11:36:48Z. That rebuild ships _grpc/_multimodal/_server
# but NOT _storage, which is why the prebuild below needs a build mode override.
FROM lmsysorg/sglang:qwen38flashnext@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521

COPY overlay/python/ /sgl-workspace/sglang/python/
COPY overlay/rust/   /sgl-workspace/sglang/rust/

# Fail the build, not the first request, if the NVMe module did not make it in.
RUN python3 -c "import sglang.srt.models.qwen4_ple_nvme; print('nvme module import OK')"

# Prebuild the Rust _storage extension. Without this the first server start
# pays a cargo build, which is slow and can fail inside a read-only rootfs.
RUN cd /sgl-workspace/sglang \
 && SGLANG_RUST_BUILD_MODE=auto python3 -c "from sglang.srt.rust_extensions import load_rust_extension; \
m = load_rust_extension('sglang.srt.rust_extensions._storage'); print('rust ext OK:', m)"
