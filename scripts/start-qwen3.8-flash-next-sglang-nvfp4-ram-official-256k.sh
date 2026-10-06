#!/usr/bin/env bash
# Start SGLang, official lmsysorg image, NVFP4, PLE table in RAM.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
# Tuned for one agent (the launcher itself defaults to the published 16-request cookbook cell).
# Keep BF16 KV: FP8 KV crashes this image on long prompts.
export MAXRUN="${MAXRUN:-4}" MAMBA_SLOTS="${MAMBA_SLOTS:-12}" KV_DTYPE="${KV_DTYPE:-auto}"
start_profile qwen3.8-flash-next-sglang-nvfp4-ram-official
