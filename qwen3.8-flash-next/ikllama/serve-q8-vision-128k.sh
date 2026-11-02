#!/usr/bin/env bash
# Thin wrapper: see serve.sh.
CTX="${CTX:-131072}" exec "$(dirname "$0")/serve.sh" q8-vision "$@"
