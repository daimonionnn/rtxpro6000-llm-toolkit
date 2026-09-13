#!/usr/bin/env bash
# Variant 1: serve Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell (96 GB), PLE table in RAM.
#
# Uses the Docker image built by ../build-local-image/build.sh.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
#
# Same image and checkpoint as ../v0-nvme/serve-nvfp4-nvme.sh, but the 47.7 GiB N-gram (PLE)
# table is held in pinned host memory (Qwen4ExpPinnedHostEmbedding) instead of
# being streamed from NVMe.
#
# That path costs ~1.83 GB more VRAM, which on its own collapses the KV cache
# (docs/ple-ram-experiment.md). This launcher wins it back from the mamba state
# cache, using the settings of SGLang's verified RTX PRO 6000 cookbook recipe:
#
#   extra_buffer_lazy + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK   5 -> 3 state slots/request
#   --mamba-ssm-dtype bfloat16                             SSM state halved
#   MAXRUN=4 with 12 slots                                 sized for one agent loop
#   expandable_segments                                    less allocator fragmentation
#   --chunked-prefill-size 4096                            smaller activation reserve
#
# Pinned memory needs --ulimit memlock=-1 in Docker and >= 64 GB free host RAM.
#
# ── Quantization: what runs at which precision ──────────────────────────────
# Checkpoint RadixArk/Qwen3.8-Flash-Next-NVFP4 (modelopt 0.46.0). It is NOT
# uniformly 4-bit:
#
#   routed MoE experts   NVFP4 W4A4 — 4-bit float weights and activations, FP8
#                        E4M3 scale per block of 16. The bulk of the parameters.
#   everything else      BF16 — attention, linear attention, router, shared
#                        experts, hyper-connections, MTP draft, vision, lm_head
#                        (the checkpoint's exclude_modules).
#   PLE n-gram table     FP8 E4M3, 47.7 GiB — pinned in host RAM.
#   KV cache             fp8_e4m3 — a launch choice (KV_DTYPE), not the checkpoint's.
#
# quant_info.py reads this from MODEL_DIR at launch, prints it, and aborts the
# launch if the checkpoint is not NVFP4.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-sglang-flashnext-sm120:local}"
NAME="${NAME:-flashnext}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"
SECCOMP="${SECCOMP:-$HERE/../build-local-image/seccomp-iouring.json}"

# ── Tuning (measured on this card; see BENCHMARKS.md) ───────────────────────
CTX="${CTX:-262144}"           # native window
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"   # empty = let the engine size the KV pool
MEMFRAC="${MEMFRAC:-0.96}"
MAXRUN="${MAXRUN:-4}"
MAMBA_SLOTS="${MAMBA_SLOTS:-12}"   # 3 state slots per request with extra_buffer_lazy -> 4 requests
CHUNKED="${CHUNKED:-4096}"

# ── Quantization knobs ──────────────────────────────────────────────────────
WEIGHT_QUANT="${WEIGHT_QUANT:-modelopt_fp4}"           # --quantization; must match the checkpoint
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cudnn}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"                       # 'auto' = BF16 KV, roughly half the tokens

# ── PLE table: pinned host RAM ──────────────────────────────────────────────
# No SGLANG_QWEN4_PLE_NVME_* variables: server_args already enables
# ple_offload_embedding, and leaving the NVMe path unset is what selects it.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

# Extra launch flags, e.g. EXTRA_ARGS="--mamba-ssm-dtype bfloat16"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --enable-strict-thinking is only needed if you send per-request
# max_thinking_tokens (PR #36750). It requires xgrammar, which is already the
# default grammar backend, and does not change tool-call grammars.
STRICT_ARG=""
[ "${STRICT_THINKING:-1}" = "1" ] && STRICT_ARG="--enable-strict-thinking"

python3 "$HERE/../common/quant_info.py" "$MODEL_DIR" \
  --weight-quant "$WEIGHT_QUANT" --kv-dtype "$KV_DTYPE" \
  --fp4-gemm-backend "$FP4_GEMM_BACKEND" --ple-mode ram || exit 2

TOKEN_CAP_ARGS=""
[ -n "$MAX_TOTAL_TOKENS" ] && TOKEN_CAP_ARGS="--max-total-tokens $MAX_TOTAL_TOKENS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# --security-opt seccomp: Docker's default profile blocks io_uring_setup /
# io_uring_enter / io_uring_register. Without this profile the NVMe PLE reader
# cannot start. This is the single most likely reason a first run fails.
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --security-opt label=disable \
  --security-opt seccomp="$SECCOMP" \
  --ulimit memlock=-1 \
  --label "flashnext.variant=v1-ram" \
  --label "flashnext.quant.experts=NVFP4-W4A4" \
  --label "flashnext.quant.rest=BF16" \
  --label "flashnext.quant.kv_cache=$KV_DTYPE" \
  --label "flashnext.quant.ple=FP8_E4M3/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  "$IMAGE" \
  python -m sglang.launch_server \
    --model-path /model \
    --served-model-name Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8000 \
    --quantization "$WEIGHT_QUANT" \
    --fp4-gemm-backend "$FP4_GEMM_BACKEND" \
    --kv-cache-dtype "$KV_DTYPE" \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
    --mamba-track-interval 64 \
    --chunked-prefill-size "$CHUNKED" \
    --max-running-requests "$MAXRUN" \
    --context-length "$CTX" \
    $TOKEN_CAP_ARGS \
    --max-mamba-cache-size "$MAMBA_SLOTS" \
    --mem-fraction-static "$MEMFRAC" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --reasoning-parser auto --tool-call-parser auto \
    $STRICT_ARG \
    --enable-metrics \
    $EXTRA_ARGS

echo "waiting for the server (cold start ~3 min, ~1.5 min with a warm page cache)"
for _ in $(seq 1 60); do
  if docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; then
    echo "ready on http://127.0.0.1:${PORT}  ·  NVFP4 experts, BF16 rest, KV $KV_DTYPE, PLE FP8 in pinned RAM"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 40 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 40 "$NAME"; exit 1
