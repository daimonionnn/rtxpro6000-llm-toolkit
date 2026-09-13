#!/usr/bin/env bash
# Build the serving image for Qwen3.8-Flash-Next on a single RTX PRO 6000 Blackwell.
#
# The image is the day-0 official image with a source overlay: the model-support
# tree plus four upstream PRs and one local patch. Everything is pinned — no
# floating refs — so two people running this get the same bytes.
#
# Usage:  ./build.sh            (writes image $IMAGE, default below)
#         SRC=/path ./build.sh  (reuse an existing checkout)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

IMAGE="${IMAGE:-sglang-flashnext-sm120:local}"
WORK="${WORK:-$HERE/.build}"
SRC="${SRC:-$WORK/sglang-src}"

# ── Pinned coordinates ───────────────────────────────────────────────────────
# Source tree: head of PR #36567 (NVMe PLE streaming), which is stacked on
# #36497 (model support). Pinned by commit, not by branch name.
PLE_REMOTE="https://github.com/jzinno/sglang.git"
PLE_COMMIT="d4477bd298aef3edae611eb7b2e533d5526e324b"
UPSTREAM="https://github.com/sgl-project/sglang.git"

# Cherry-picked upstream PRs, applied in this order.
#   36556 — SM120/SM121 sparse decode paths. Without it the server dies at
#           startup on this card with an MLIRError out of the QSA kernel.
#   36749 — BCG buffers sized by token rows. Without it you cannot combine
#           --cuda-graph-backend-decode breakable with NEXTN speculation:
#           verify reads 1/num_draft_tokens of the logits and indexes
#           accept_index past the end (a device-side assert that looks like a
#           hang). This is what makes graph+spec work at all here.
#   36750 — max_thinking_tokens on the OpenAI chat endpoint. Optional: only
#           needed if you want per-request thinking budgets over the OpenAI
#           protocol. Requires --enable-strict-thinking at launch.
PRS="${PRS:-36556 36749 36750}"

echo "[1/4] source tree @ ${PLE_COMMIT:0:12}"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --filter=blob:none "$PLE_REMOTE" "$SRC"
fi
git -C "$SRC" fetch --quiet "$PLE_REMOTE" "$PLE_COMMIT"
git -C "$SRC" checkout --quiet --detach "$PLE_COMMIT"

echo "[2/4] cherry-picking upstream PRs: $PRS"
git -C "$SRC" remote get-url upstream >/dev/null 2>&1 || \
  git -C "$SRC" remote add upstream "$UPSTREAM"
# Ask the API which commits each PR actually contains. Deriving the range with
# `merge-base HEAD pr-N` looks equivalent and is not: these PRs target branches
# (qwen4-main-squashed, main) that this pinned tree is not descended from, so
# the merge base sits far enough back that the range also picks up the target
# branch's own tip. Cherry-picking that fails, which is exactly what happened.
for pr in $PRS; do
  git -C "$SRC" fetch --quiet upstream "pull/$pr/head:pr-$pr"
  shas=$(curl -fsSL "https://api.github.com/repos/sgl-project/sglang/pulls/$pr/commits" \
         | python3 -c 'import json,sys; print(" ".join(c["sha"] for c in json.load(sys.stdin)))') || {
    echo "  !! could not list commits for PR #$pr (GitHub API unreachable or rate-limited)."
    echo "     Set PRS= to skip, or apply the PR by hand."
    exit 1
  }
  for sha in $shas; do
    if ! git -C "$SRC" cherry-pick --keep-redundant-commits -x "$sha" >/dev/null 2>&1; then
      git -C "$SRC" cherry-pick --abort >/dev/null 2>&1 || true
      echo "  !! PR #$pr commit ${sha:0:12} did not apply cleanly — it may already be"
      echo "     merged into the pinned tree, or upstream may have rebased it."
      echo "     Inspect: git -C $SRC show $sha"
      exit 1
    fi
  done
  echo "  ok #$pr ($(echo $shas | wc -w) commit(s))"
done

echo "[3/4] local patches"
# A failed apply here used to print its error and carry on, which produced an
# image that looked fine and died on the first prompt long enough to be chunked.
# Stop instead.
for p in "$HERE"/patches/*.patch; do
  [ -e "$p" ] || continue
  if ! git -C "$SRC" apply --3way "$p"; then
    echo "  !! $(basename "$p") did not apply. If upstream has since fixed this"
    echo "     (see the patch header), drop the file; otherwise rebase it."
    exit 1
  fi
  echo "  ok $(basename "$p")"
done

echo "[4/4] docker build -> $IMAGE"
mkdir -p "$WORK/docker/overlay"
rsync -a --delete --exclude='__pycache__' "$SRC/python/" "$WORK/docker/overlay/python/"
rsync -a --delete "$SRC/rust/" "$WORK/docker/overlay/rust/"
cp "$HERE/Dockerfile" "$WORK/docker/Dockerfile"
docker build -t "$IMAGE" "$WORK/docker"
echo "done: $IMAGE"
