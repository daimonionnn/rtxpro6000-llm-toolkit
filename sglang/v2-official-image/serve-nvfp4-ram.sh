#!/usr/bin/env bash
# Variant 2: Qwen3.8-Flash-Next on the official SGLang image, PLE table in RAM.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 ./serve-nvfp4-ram.sh
#
# Runs SGLang's verified cookbook recipe for 1x RTX PRO 6000 Blackwell, NVFP4
# (RadixArk export), low-latency cell, on lmsysorg/sglang:dev-qwen38-next-local —
# the qwen4-main-squashed build (4ccff141db) the recipe is verified on. No local
# build, no patches.
#
# Defaults reproduce the published cell exactly. Override to tune for one agent:
#
#   MAXRUN=4 MAMBA_SLOTS=12 ./serve-nvfp4-ram.sh
#
# Differences from ../v1-ram/serve-nvfp4-ram.sh (variant 1):
#   - stock image, so no local FP8-KV chunked-prefill patch. The recipe leaves KV
#     at the checkpoint default (BF16); KV_DTYPE=fp8_e4m3 is untested here and
#     upstream's proper fix (#36644) was still unmerged on 2026-09-12.
#   - flashinfer_cutlass for both the FP4 GEMM and the MoE runner.
#   - default CUDA graph backend rather than `breakable`.
#   - no io_uring seccomp profile; nothing reads from NVMe.
#
# Pinned memory needs --ulimit memlock=-1 and >= 64 GB free host RAM.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-lmsysorg/sglang:dev-qwen38-next-local}"
NAME="${NAME:-flashnext}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

# ── Published cookbook values ───────────────────────────────────────────────
CTX="${CTX:-262144}"
MEMFRAC="${MEMFRAC:-0.96}"
MAXRUN="${MAXRUN:-16}"
MAMBA_SLOTS="${MAMBA_SLOTS:-48}"     # 3 state slots per request with extra_buffer_lazy
CHUNKED="${CHUNKED:-4096}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

# ── Quantization ────────────────────────────────────────────────────────────
WEIGHT_QUANT="${WEIGHT_QUANT:-modelopt_fp4}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cutlass}"
KV_DTYPE="${KV_DTYPE:-auto}"          # recipe default: checkpoint's own (BF16)

EXTRA_ARGS="${EXTRA_ARGS:-}"

python3 "$HERE/../common/quant_info.py" "$MODEL_DIR" \
  --weight-quant "$WEIGHT_QUANT" --kv-dtype "$KV_DTYPE" \
  --fp4-gemm-backend "$FP4_GEMM_BACKEND" --ple-mode ram || exit 2
echo "image: $IMAGE   max-running-requests $MAXRUN, mamba slots $MAMBA_SLOTS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --ulimit memlock=-1 \
  --label "flashnext.variant=v2-official-image" \
  --label "flashnext.quant.experts=NVFP4-W4A4" \
  --label "flashnext.quant.rest=BF16" \
  --label "flashnext.quant.kv_cache=$KV_DTYPE" \
  --label "flashnext.quant.ple=FP8_E4M3/ram" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8000 \
    --tp 1 \
    --quantization "$WEIGHT_QUANT" \
    --fp4-gemm-backend "$FP4_GEMM_BACKEND" \
    --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype "$KV_DTYPE" \
    --page-size 64 \
    --mamba-track-interval 64 \
    --chunked-prefill-size "$CHUNKED" \
    --context-length "$CTX" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-running-requests "$MAXRUN" \
    --max-mamba-cache-size "$MAMBA_SLOTS" \
    --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
    --ple-offload-embedding \
    --reasoning-parser qwen3 \
    --mem-fraction-static "$MEMFRAC" \
    --enable-metrics \
    $EXTRA_ARGS

echo "waiting for the server (cold start ~3-4 min; first start on a new image is slower)"
for _ in $(seq 1 90); do
  if docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; then
    echo "ready on http://127.0.0.1:${PORT}  ·  official image, PLE FP8 in pinned RAM, KV $KV_DTYPE"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 40 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 40 "$NAME"; exit 1
