#!/usr/bin/env bash
# Thin wrapper: see serve.sh.
CTX=262144 exec "$(dirname "$0")/serve.sh" ist-iq2xs-vision "$@"
