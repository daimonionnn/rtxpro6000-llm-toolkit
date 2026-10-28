#!/usr/bin/env bash
# Start the default profile: whatever DEFAULT_PROFILE below names.
# Shorthand for the default profile's context-labelled script; variables pass through.
# It binds to every interface (--host 0.0.0.0) so LAN clients can reach the API;
# pass --host 127.0.0.1 to keep it local.
set -euo pipefail

DEFAULT_PROFILE=qwen3.8-flash-next-strata-q8-vision-256k

source "$(dirname "$0")/_lib.sh"
args=("$@")
case " $* " in
  *" --host "*|*" --host="*) ;;
  *) args=(--host 0.0.0.0 "${args[@]}") ;;
esac
exec "$ROOT/scripts/$(profile_field "$DEFAULT_PROFILE" script)" "${args[@]}"
