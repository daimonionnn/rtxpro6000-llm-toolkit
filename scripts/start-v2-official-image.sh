#!/usr/bin/env bash
# Start Variant 2: Docker, official lmsysorg image, PLE table in RAM.
# MODEL_DIR defaults to models/Qwen3.8-Flash-Next-NVFP4; every launcher variable can be overridden.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_idle
# Tuned for one agent (the launcher itself defaults to the published 16-request cookbook cell).
# Keep BF16 KV: FP8 KV crashes this image on long prompts.
export MAXRUN="${MAXRUN:-4}" MAMBA_SLOTS="${MAMBA_SLOTS:-12}" KV_DTYPE="${KV_DTYPE:-auto}"
exec "$ROOT/sglang/v2-official-image/serve-nvfp4-ram.sh"
