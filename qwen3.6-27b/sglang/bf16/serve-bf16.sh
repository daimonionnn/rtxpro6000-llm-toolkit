#!/usr/bin/env bash
# Profile sglang-bf16: Qwen3.6-27B, original BF16 weights, on SGLang, one RTX PRO 6000
# Blackwell (96 GB).
#
#   MODEL_DIR=/models/Qwen3.6-27B ./serve-bf16.sh
#   ./stop.sh
#
# Checkpoint Qwen/Qwen3.6-27B: a dense 27B hybrid (qwen3_5 architecture — Gated
# DeltaNet linear attention with full attention every 4th layer), 262K context,
# vision encoder, one MTP layer. ~52 GiB of BF16 weights, entirely on the GPU; the
# rest of the card holds the KV cache and the linear-attention state.
#
# Image lmsysorg/sglang:dev-qwen38-next-local — the official image the Flash-Next
# official profile uses; its SGLang has the qwen3_5 model and MTP code.
#
# Settings follow a published single-card BF16 27B setup (Qwen3.8-27B on an RTX PRO
# 6000): NEXTN speculation from the MTP head, BF16 SSM state, a CUDA-graph batch
# cap. KV cache stays BF16: FP8 KV needs calibrated scales that a BF16 checkpoint
# does not have, and without them output is corrupted.
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:dev-qwen38-next-local}"
NAME="${NAME:-rtxpro6000-llm}"
PORT="${PORT:-8090}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"

CTX="${CTX:-262144}"
MAXRUN="${MAXRUN:-4}"                 # concurrent requests; sized for one agent
MEMFRAC="${MEMFRAC:-0.88}"            # 0.94 left too little headroom in the reference setup
SPEC_STEPS="${SPEC_STEPS:-3}"         # NEXTN draft steps; 3 was the reference's optimum for BF16
EXTRA_ARGS="${EXTRA_ARGS:-}"

[ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in $MODEL_DIR" >&2; exit 2; }
python3 - "$MODEL_DIR" <<'PY' || exit 2
import json, os, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.json")))
t = c.get("text_config", c)
if c.get("model_type") != "qwen3_5" or c.get("quantization_config"):
    sys.exit(f"expected an unquantized qwen3_5 checkpoint, got model_type={c.get('model_type')} "
             f"quantization={bool(c.get('quantization_config'))}")
size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
print(f"\nQwen3.6-27B — {os.path.basename(os.path.normpath(d))}  ·  engine sglang")
print(f"  weights  BF16, {t['num_hidden_layers']} layers (full attention every "
      f"{t.get('full_attention_interval')}), {size / 2**30:.1f} GiB  VRAM")
print(f"  KV cache BF16, context {t['max_position_embeddings']:,}\n")
PY

echo "image: $IMAGE   context $CTX, max-running-requests $MAXRUN, mem-fraction $MEMFRAC, NEXTN steps $SPEC_STEPS"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0"' \
  --ipc host --shm-size 32g \
  --label "rtxpro6000-llm.profile=qwen3.6-27b-sglang-bf16" \
  --label "rtxpro6000-llm.quant.weights=BF16" \
  --label "rtxpro6000-llm.quant.kv_cache=bfloat16" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_SANITIZE_NAN_LOGITS=True \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name Qwen3.6-27B \
    --host 0.0.0.0 --port 8000 \
    --tp 1 \
    --context-length "$CTX" \
    --mem-fraction-static "$MEMFRAC" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $((SPEC_STEPS + 1)) \
    --speculative-attention-mode decode \
    --cuda-graph-max-bs 8 \
    --max-running-requests "$MAXRUN" \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-metrics \
    $EXTRA_ARGS

echo "waiting for the server (cold start ~2-4 min)"
for _ in $(seq 1 120); do
  if docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; then
    echo "ready on http://127.0.0.1:${PORT}  ·  Qwen3.6-27B BF16, SGLang, NEXTN $SPEC_STEPS"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container exited:"; docker logs --tail 40 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log lines:"; docker logs --tail 40 "$NAME"; exit 1
