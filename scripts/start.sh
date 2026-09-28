#!/usr/bin/env bash
# Start the default profile: whatever DEFAULT_PROFILE below names.
# Shorthand for scripts/start-<default profile>.sh; launcher variables pass through.
set -euo pipefail

DEFAULT_PROFILE=qwen3.8-flash-next-vllm-awq-w4a16-g32

exec "$(dirname "$0")/start-${DEFAULT_PROFILE}.sh" "$@"
