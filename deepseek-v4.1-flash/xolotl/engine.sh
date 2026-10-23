#!/usr/bin/env bash
# engine.sh - the DeepSeek-V4.1 engine (Xolotl's ds41_generate) as Xolotl's API server starts it
# (`<exe> --serve <args>`). Same engine settings as Xolotl's tools/v41/serve/ds41-engine*.sh
# (DSpark speculative decode, prompt-lookup drafts, Q6_K dense re-quant, chunk prestage), without
# their machine-specific GPU lock, ComfyUI check and RAM cap.
#
# stdout is the engine's line protocol: nothing else may print to it.
set -u
XOLOTL_DIR="${XOLOTL_DIR:-$HOME/development/xolotl}"
cd "$XOLOTL_DIR"
export GGML_NO_BACKTRACE=1
# 4 KiB pages pin the arena far faster than transparent huge pages
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# turns of <= 96 new prompt tokens run as exact passes, longer ones as the exact batch
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark speculative decode (the output distribution stays the trunk's); serve.sh sets the sidecar path,
# STRATA_SERVE_DSPARK= (empty) turns it off
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$XOLOTL_DIR/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-4096}"
# dense half (attention, shared expert, head; Q8_0 in the file) re-quantized to Q6_K at load: more VRAM
# for expert slots. STRATA_DENSE_REQUANT= (empty) keeps Q8_0.
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-1.5}"
[ -n "${STRATA_PROMPT_BATCH:-}" ] && set -- "$@" --prompt-batch "$STRATA_PROMPT_BATCH"
[ -z "${STRATA_SERVE_DSPARK:-}" ] && unset STRATA_SERVE_DSPARK
exec "${STRATA_BUILD_DIR:-$XOLOTL_DIR/build-xolotl}/ds41_generate" "$@"
