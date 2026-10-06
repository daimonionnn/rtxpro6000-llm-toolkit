#!/usr/bin/env bash
# Start vLLM, official image, AWQ W4A16 group 32 (cyankiwi), PLE table in RAM.
# MODEL_DIR defaults to the profile's checkpoint under models/; launcher variables pass through.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

start_profile qwen3.8-flash-next-vllm-awq-w4a16-g32
