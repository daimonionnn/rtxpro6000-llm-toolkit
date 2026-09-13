#!/usr/bin/env bash
# Start Variant 3: native jpezzulli fork, PLE table in RAM, 524K context.
# MODEL_DIR defaults to models/Qwen3.8-Flash-Next-NVFP4; every launcher variable can be overridden.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_idle
exec "$ROOT/sglang/v3-pennyroyal/serve-nvfp4-ram.sh"
