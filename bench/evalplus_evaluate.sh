#!/usr/bin/env bash
# Score the solutions bench/evalplus_codegen.py generated, inside the official
# EvalPlus image with no network: model-written code never runs on the host.
#
#   bench/evalplus_evaluate.sh LABEL [humaneval|mbpp ...]
#
# Prints base / plus pass@1 per dataset and leaves EvalPlus's per-task result
# file next to the samples in logs/evalplus/<LABEL>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:?usage: $0 LABEL [humaneval|mbpp ...]}"; shift
DATASETS=("${@:-humaneval mbpp}")
IMAGE="${EVALPLUS_IMAGE:-ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740}"
DIR="$ROOT/logs/evalplus/$LABEL"
# EvalPlus caches ground-truth outputs next to the datasets, so the cache must be
# writable: a copy under logs/, used as the container user's home.
CACHE="$ROOT/logs/evalplus/.cache"
SRC_CACHE="${EVALPLUS_CACHE:-$HOME/.cache/evalplus}"   # datasets, downloaded by the codegen step
mkdir -p "$CACHE"
for f in HumanEvalPlus-v0.1.10.jsonl MbppPlus-v0.2.0.jsonl; do
  [ -f "$CACHE/$f" ] || cp "$SRC_CACHE/$f" "$CACHE/$f"
done

for ds in ${DATASETS[*]}; do
  [ -f "$DIR/$ds.jsonl" ] || { echo "no $DIR/$ds.jsonl — run bench/evalplus_codegen.py first" >&2; exit 2; }
  echo "=== $LABEL · $ds"
  docker run --rm --network none --memory 16g --pids-limit 4096 \
    --user "$(id -u):$(id -g)" -e HOME=/eval \
    -v "$DIR:/app/samples" -v "$CACHE:/eval/.cache/evalplus" \
    -e HUMANEVAL_OVERRIDE_PATH=/eval/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl \
    -e MBPP_OVERRIDE_PATH=/eval/.cache/evalplus/MbppPlus-v0.2.0.jsonl \
    --entrypoint evalplus.evaluate "$IMAGE" \
    --dataset "$ds" --samples "/app/samples/$ds.jsonl" --i-just-wanna-run 2>&1 \
    | grep -E "pass@1|^(humaneval|mbpp)|base|Error|error" || true
done
