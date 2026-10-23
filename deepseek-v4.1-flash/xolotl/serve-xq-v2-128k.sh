#!/usr/bin/env bash
# Thin wrapper: see serve.sh.
exec "$(dirname "$0")/serve.sh" xq-v2 "$@"
