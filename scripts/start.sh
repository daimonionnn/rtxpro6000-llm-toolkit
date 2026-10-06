#!/usr/bin/env bash
# Start the default profile: whatever DEFAULT_PROFILE below names.
# Shorthand for the default profile's context-labelled script; variables pass through.
set -euo pipefail

DEFAULT_PROFILE=qwen3.8-flash-next-strata-q8-vision

source "$(dirname "$0")/_lib.sh"
exec "$ROOT/scripts/$(profile_field "$DEFAULT_PROFILE" script)" "$@"
