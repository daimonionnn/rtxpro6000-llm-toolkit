#!/usr/bin/env bash
# Start the NVMe baseline: Docker, local image, PLE table streamed from NVMe.
# MODEL_DIR defaults to models/Qwen3.8-Flash-Next-NVFP4; every launcher variable can be overridden.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_idle
exec "$ROOT/sglang/v0-nvme/serve-nvfp4-nvme.sh"
