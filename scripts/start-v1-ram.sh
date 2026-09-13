#!/usr/bin/env bash
# Start Variant 1: Docker, local image, PLE table in RAM.
# MODEL_DIR defaults to models/Qwen3.8-Flash-Next-NVFP4; every launcher variable can be overridden.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_idle
exec "$ROOT/sglang/v1-ram/serve-nvfp4-ram.sh"
