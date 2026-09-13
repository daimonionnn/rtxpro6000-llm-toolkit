#!/usr/bin/env bash
# Profile sglang-nvfp4-ram-pennyroyal: Qwen3.8-Flash-Next on the jpezzulli/sglang-rtxpro6000 fork
# ("Pennyroyal", tag pennyroyal-v2.5.0), PLE table in RAM. Native, not Docker.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
#   ./stop.sh
#
# Build the environment first with ./build.sh. The checkout lives in
# ../pennyroyal-fork and is left untouched, as the fork's update procedure
# requires a clean tree.
#
# This is the fork's configs/pennyroyal/serve-flash-next.sh (native NEXTN, no
# FR-Spec). HiCache/NIXL is off by default and enabled with HICACHE=1:
#
#   HICACHE=1 MODEL_DIR=... ./serve-nvfp4-ram.sh
#
# HiCache keeps evicted prefix state in a 32 GB host-RAM tier and persists it
# through NIXL to files, so a prefix survives a server restart. It does not change
# the GPU KV pool. Needs NIXL built into ./nixl (./build-nixl.sh). Watermarks for
# this machine's disk are in nixl-posix-local.toml — read its header.
#
# What the fork adds over the other SGLang profiles:
#   --gdn-mtp-cache-mode none   drops the ~1.05 GB intermediate SSM buffer used by
#                               MTP verify; recomputes the state after verify instead
#   --mem-fraction-static 0.981 lets automatic KV sizing use the freed memory
#   YaRN factor 2               524,288-token context window
#   FP8 KV cache                with the fork's own chunked-prefill handling
#
# The server runs as a background process; its PID is kept in ./server.pid.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLKIT="$(cd "$HERE/../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../pennyroyal-fork" && pwd)}"
SGLANG_EXE="$REPO_ROOT/.venv/bin/sglang"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"
PORT="${PORT:-8090}"
CACHE_BASE="${CACHE_BASE:-$HERE/cache}"
LOG="${LOG:-$TOOLKIT/logs/pennyroyal-serve.log}"
PIDFILE="$HERE/server.pid"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"

# ── Fork recipe values ──────────────────────────────────────────────────────
CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"
PAGE_SIZE=64
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
CHUNKED="${CHUNKED:-4096}"
MEMFRAC="${MEMFRAC:-0.981}"
MAXRUN="${MAXRUN:-4}"
MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
GDN_MTP_CACHE_MODE="${GDN_MTP_CACHE_MODE:-none}"

# ── HiCache / NIXL (opt-in) ─────────────────────────────────────────────────
HICACHE="${HICACHE:-0}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-32}"
NIXL_PREFIX="${NIXL_PREFIX:-$HERE/nixl}"
NIXL_STORAGE_BASE="${NIXL_STORAGE_BASE:-$HERE/nixl-storage}"
NIXL_CONFIG="${NIXL_CONFIG:-$HERE/nixl-posix-local.toml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── Preflight ───────────────────────────────────────────────────────────────
[ -x "$SGLANG_EXE" ] || { echo "no $SGLANG_EXE — run ./build.sh first" >&2; exit 2; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (PID $(cat "$PIDFILE")); ./stop.sh first" >&2; exit 2
fi

busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$busy" ]; then
  echo "the GPU is in use — stop the other server first:" >&2
  echo "$busy" | sed 's/^/  /' >&2
  echo "  (stop whichever profile holds it with scripts/stop.sh)" >&2
  exit 2
fi

# The 47.7 GiB PLE table is pinned. Docker needed --ulimit memlock=-1; a native
# process inherits the shell's limit, and systemd user services default to 8 MB.
if [ "$(ulimit -l)" != "unlimited" ]; then
  echo "memlock limit is $(ulimit -l) KB, not unlimited; pinning the PLE table will fail." >&2
  echo "  run from a login shell where 'ulimit -l' is unlimited, or raise it in /etc/security/limits.conf" >&2
  exit 2
fi

if [ "$HICACHE" = 1 ]; then
  [ -d "$NIXL_PREFIX/lib64" ] || { echo "HICACHE=1 but no NIXL at $NIXL_PREFIX — run ./build-nixl.sh" >&2; exit 2; }
  [ -r "$NIXL_CONFIG" ] || { echo "NIXL config missing: $NIXL_CONFIG" >&2; exit 2; }
fi

python3 "$HERE/../../quant_info.py" "$MODEL_DIR" \
  --weight-quant modelopt_fp4 --kv-dtype "$KV_DTYPE" --fp4-gemm-backend auto --ple-mode ram || exit 2
echo "fork: $(git -C "$REPO_ROOT" describe --tags 2>/dev/null) ($(git -C "$REPO_ROOT" rev-parse --short=10 HEAD))"
echo "context $CONTEXT_LENGTH, max-running-requests $MAXRUN, mamba slots $MAMBA_SLOTS, gdn-mtp-cache-mode $GDN_MTP_CACHE_MODE"
echo "HiCache: $([ "$HICACHE" = 1 ] && echo "on, ${HICACHE_SIZE_GB} GB host tier, NIXL -> $NIXL_STORAGE_BASE" || echo off)"

source "$REPO_ROOT/configs/pennyroyal/chat-template.sh"

# ── Environment (from the fork's launcher) ──────────────────────────────────
mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit} "$(dirname "$LOG")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# shim/ first: hides ROCm's hipcc so TileLang targets CUDA, not the host's AMD GPU.
export CUDA_HOME CUDACXX="$CUDA_HOME/bin/nvcc" PATH="$HERE/shim:$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_BUILD_JOBS=8
export FLASHINFER_NINJA_JOBS=8 FLASHINFER_NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch" TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton" CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE="$MAMBA_CONV_DTYPE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export NUMPY_MADVISE_HUGEPAGE=0 SGLANG_MM_PREPROCESS_DEVICE=cpu

HICACHE_ARGS=()
if [ "$HICACHE" = 1 ]; then
  export LD_LIBRARY_PATH="$NIXL_PREFIX/lib64:$LD_LIBRARY_PATH"
  mkdir -p "$NIXL_STORAGE_BASE"
fi

TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'

if [ "$HICACHE" = 1 ]; then
  # The fork's helper picks a cache directory keyed to everything that changes the
  # stored representation (source, checkpoint, template, dtypes, page size, ...),
  # so an incompatible config never reads another's prefixes. Hashing the
  # checkpoint's identity takes a little while on first use.
  echo "deriving NIXL namespace..."
  NIXL_STORAGE="$("$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/pennyroyal/derive_namespace.py" \
    --base-root "$NIXL_STORAGE_BASE" \
    --slug "qwen3_8_flash_next_${CONTEXT_LENGTH}_nextn_$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)" \
    --git-repo "$REPO_ROOT" \
    --model "target=$MODEL_DIR" \
    --field "chat_template_sha256=$CHAT_TEMPLATE_SHA" \
    --field "online_mxfp8=${SGLANG_SM120_ONLINE_MXFP8:-false}" \
    --field "image_processor_backend=pil" \
    --field "mm_preprocess_device=$SGLANG_MM_PREPROCESS_DEVICE" \
    --field "context_length=$CONTEXT_LENGTH" \
    --field "tp_size=1" \
    --field "page_size=$PAGE_SIZE" \
    --field "compute_dtype=bfloat16" \
    --field "target_kv_dtype=$KV_DTYPE" \
    --field "speculative_algorithm=NEXTN" \
    --field "speculative_num_steps=3" \
    --field "speculative_eagle_topk=1" \
    --field "speculative_num_draft_tokens=4" \
    --field "speculative_draft_quantization=unquant" \
    --field "gdn_mtp_cache_mode=$GDN_MTP_CACHE_MODE" \
    --field "hicache_io_backend=kernel" \
    --field "hicache_mem_layout=page_first" \
    --field "mamba_ssm_dtype=$MAMBA_SSM_DTYPE" \
    --field "mamba_conv_dtype=$MAMBA_CONV_DTYPE" \
    --field "max_mamba_cache_size=$MAMBA_SLOTS" \
    --field "mamba_radix_cache_strategy=extra_buffer" \
    --field "mamba_track_interval=64" \
    --field "linear_attn_decode_backend=flashinfer" \
    --field "linear_attn_prefill_backend=flashinfer" \
    --field "ple_offload_embedding=true" \
    --field "qsa_compressed_hicache=true" \
    --field "chunked_prefill_size=$CHUNKED" \
    --field "target_model_overrides=$TARGET_OVERRIDES" \
    --field "torch_version=$("$REPO_ROOT/.venv/bin/python" -c 'import torch; print(torch.__version__)')" \
    --field "cuda_arch=12.0")"
  export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$NIXL_STORAGE"
  echo "NIXL namespace: $NIXL_STORAGE"
  HICACHE_ARGS=(
    --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB" --hicache-host-memory-mode cache
    --hicache-write-policy write_through --hicache-io-backend kernel
    --hicache-mem-layout page_first --hicache-storage-backend nixl
    --hicache-storage-prefetch-policy timeout
    --hicache-storage-backend-extra-config "@$NIXL_CONFIG"
  )
fi

# setsid: own process group, so stop.sh can take down the scheduler and
# detokenizer children with the launcher.
: > "$LOG"
setsid nohup "$SGLANG_EXE" serve \
  --warmups=structured_output \
  --model-path "$MODEL_DIR" \
  --load-format safetensors \
  --served-model-name Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port "$PORT" --tp 1 \
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static "$MEMFRAC" \
  --context-length "$CONTEXT_LENGTH" --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size "$PAGE_SIZE" --max-running-requests "$MAXRUN" --sleep-on-idle \
  --chunked-prefill-size "$CHUNKED" \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --max-mamba-cache-size "$MAMBA_SLOTS" --gdn-mtp-cache-mode "$GDN_MTP_CACHE_MODE" \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval 64 \
  "${HICACHE_ARGS[@]}" \
  --ple-offload-embedding --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" --image-processor-backend pil \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant --watchdog-timeout 1800 \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started PID $(cat "$PIDFILE"), log: $LOG"

# The first start compiles kernels and can take a long time; later starts reuse
# $CACHE_BASE.
echo "waiting for the server (first start compiles kernels — allow up to 45 min)"
for _ in $(seq 1 270); do
  if grep -q "The server is fired up" "$LOG" 2>/dev/null; then
    echo "ready on http://127.0.0.1:${PORT}  ·  Pennyroyal fork, PLE FP8 in pinned RAM, KV $KV_DTYPE, HiCache $([ "$HICACHE" = 1 ] && echo on || echo off)"; exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited:"; tail -40 "$LOG"; rm -f "$PIDFILE"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; tail -40 "$LOG"; exit 1
