#!/usr/bin/env bash
# Build NIXL with the POSIX plugin for the Pennyroyal fork's HiCache persistence.
#
#   ./build-nixl.sh
#
# Follows the NIXL section of the fork's BUILD.md: the same pinned commit, release
# build, SM120, POSIX plugin with io_uring. Differences:
#   - installs into ./nixl instead of /opt/nvidia/nvda_nixl, so no root is needed;
#     serve-nvfp4-ram.sh adds ./nixl/lib64 to LD_LIBRARY_PATH
#   - uses the fork's venv Python and CUDA 13.3, like ./build.sh
#   - `uv pip install` instead of `python -m pip install`. pip isolates builds by
#     injecting a sitecustomize.py through PYTHONPATH; NIXL's meson build shells out
#     to `uv build` for its nixl-meta wheel, that child inherits PYTHONPATH, and its
#     interpreter loses the standard library ("No module named '__future__'").
#
# Host packages it needs (apt): meson, libaio-dev. liburing comes from NIXL's
# meson wrap when the system has none.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../pennyroyal-fork/.venv/bin/python"
SRC="$HERE/nixl-src"
NIXL_PREFIX="$HERE/nixl"
NIXL_COMMIT=aecbc3846d92c34c7507a58d776e1fda50ff4fba

[ -x "$PY" ] || { echo "no $PY — run ./build.sh first" >&2; exit 2; }
for c in meson ninja cmake pkg-config; do
  command -v "$c" >/dev/null || { echo "missing $c (apt install meson ninja-build cmake pkg-config)" >&2; exit 2; }
done
[ -f /usr/include/libaio.h ] || { echo "missing libaio headers (apt install libaio-dev)" >&2; exit 2; }

# The fork's BUILD.md runs this inside an activated venv, and meson relies on it:
# pybind11-config and the Python it builds bindings for are looked up on PATH.
VENV="$(cd "$(dirname "$PY")/.." && pwd)"
export VIRTUAL_ENV="$VENV"
export CUDA_HOME=/usr/local/cuda-13.3 PATH="$VENV/bin:/usr/local/cuda-13.3/bin:$PATH"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
JOBS="${JOBS:-8}"

[ -d "$SRC/.git" ] || git clone https://github.com/ai-dynamo/nixl.git "$SRC"
cd "$SRC"
git fetch --quiet origin "$NIXL_COMMIT" 2>/dev/null || true
git checkout --quiet "$NIXL_COMMIT"
echo "nixl @ $(git rev-parse --short=10 HEAD)"

rm -rf .mesonpy-*          # leftovers from an interrupted build
uv pip install --python "$PY" tomlkit pybind11
uv pip install --python "$PY" .
"$PY" contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
meson setup build-posix --buildtype=release --reconfigure \
  --prefix="$NIXL_PREFIX" --libdir=lib64 \
  -Denable_plugins=POSIX -Dnixl_cuda_arch_list=120
ninja -C build-posix -j "$JOBS" install
uv pip install --python "$PY" build-posix/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl

echo "NIXL DONE -> $NIXL_PREFIX"
LD_LIBRARY_PATH="$NIXL_PREFIX/lib64:$LD_LIBRARY_PATH" "$PY" -c "import nixl; print('nixl python import OK', getattr(nixl, '__version__', ''))"
ls "$NIXL_PREFIX/lib64/plugins" 2>/dev/null || find "$NIXL_PREFIX" -name "*POSIX*" | head
