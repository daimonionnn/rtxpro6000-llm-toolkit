#!/usr/bin/env bash
# Native build of jpezzulli/sglang-rtxpro6000 @ pennyroyal-v2.5.0, per its BUILD.md,
# except CUDA_HOME points at the 13.3 toolkit (the host default is 13.4).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d pennyroyal-fork/.git ] || git clone --branch pennyroyal-v2.5.0 --single-branch \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal-fork
cd pennyroyal-fork
echo "commit: $(git rev-parse --short=10 HEAD)"
uv python install 3.12.13
[ -d .venv ] || uv venv --python 3.12.13 .venv
source .venv/bin/activate
uv pip install pip "setuptools>=61.0" "setuptools-rust>=1.10" "setuptools-scm>=8.0" wheel build wheel_stub  # wheel_stub: cuda-tile builds through it, and --no-build-isolation will not fetch it
export CUDA_HOME=/usr/local/cuda-13.3 CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export PATH="/usr/local/cuda-13.3/bin:$HOME/.cargo/bin:$PATH"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export TORCH_CUDA_ARCH_LIST=12.0
export PENNY_BUILD_JOBS=8 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_BUILD_JOBS=8
export FLASHINFER_NINJA_JOBS=8 FLASHINFER_NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --no-build-isolation -e python
echo "BUILD DONE"
.venv/bin/python -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
