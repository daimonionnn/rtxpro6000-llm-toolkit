# Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell

Serving Qwen's 176B-parameter Qwen4 preview (6B active) on a **single 96 GB
workstation card**, with a 233K-token KV cache and CUDA graphs on top of NEXTN
speculative decoding.

The official SGLang recipes for this model target H200 / B200 / B300 / GB300 /
MI350X / MI355X. This repository is the missing single-card cell: what to build,
what to launch, what it actually does, and where it falls short.

> **Status:** four of the five changes below are open upstream PRs, not
> inventions of this repo. As they merge, the build gets shorter. The intent is
> for this repository to become unnecessary.

---

## Why this is not just "run the official image"

**The 51B N-gram embedding table.** Of the 176B parameters, 51B are an N-gram
(PLE) lookup table. In BF16 that is ~95 GB of host RAM in the PLE offload
worker, which is why the published recipes assume a host with 100+ GB free.
Lookup addresses are known in advance, so the table does not have to be resident
at all: this setup reads it **straight off NVMe with io_uring** as tokens need
it. Measured gather latency is **0.7–2.7 ms**, under 2% of the decode budget.
That is what leaves the GPU free for weights (78.2 GB) and a real KV cache.

**CUDA graphs with speculation.** Every community report on this card ran eager.
The breakable CUDA-graph backend sized its replay buffers by request count, but
a speculative-verify body emits `requests × num_draft_tokens` rows — verify then
read a quarter of the logits and indexed `accept_index` past the end of the
tensor, which surfaces as a hang rather than an exception. Fixing the row unit
took single-stream decode from **57.6 to 167 tok/s**.

---

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation, 96 GB (SM120) |
| Host RAM | 60 GB is enough — the PLE table never lands there |
| Storage | NVMe SSD holding the checkpoint; the PLE tensor is read from it continuously |
| Host | Linux with io_uring available (kernel 5.10+; tested on 6.8) |
| Checkpoint | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) — pinned at revision `7b719225242a` |
| Base image | `lmsysorg/sglang:qwen38flashnext` (day-0) |

NVFP4 is Blackwell-only. This will not run on Ada or Hopper.

---

## Quick start

```bash
git clone https://github.com/yepapa-nest/qwen38-flashnext-rtx6000.git
cd qwen38-flashnext-rtx6000

# 1. Get the checkpoint (pin the revision — day-0 uploads get silently amended)
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242a --local-dir /models/Qwen3.8-Flash-Next-NVFP4

# 2. Build (fetches the pinned tree + PRs, then docker build)
./build.sh

# 3. Serve
MODEL_DIR=/models/Qwen3.8-Flash-Next-NVFP4 ./serve.sh
```

Cold start is about 3 minutes, ~1.5 with a warm page cache.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Explain NVMe io_uring gather in two sentences."}]
}'
```

---

## What is applied, and why

`build.sh` pins a source tree and replays changes onto it. Each one is here
because the server does not work on this card without it — none are preferences.

| # | Change | Without it |
| --- | --- | --- |
| base | [#36497](https://github.com/sgl-project/sglang/pull/36497) model support + [#36567](https://github.com/sgl-project/sglang/pull/36567) NVMe PLE streaming (tree pinned at `d4477bd298`) | No model, or no NVMe path — the day-0 image predates #36567 |
| 1 | [#36556](https://github.com/sgl-project/sglang/pull/36556) SM120/SM121 sparse decode | Server dies at startup with an `MLIRError` from the QSA kernel |
| 2 | [#36749](https://github.com/sgl-project/sglang/pull/36749) BCG buffers sized by token rows | Cannot use `breakable` graphs with NEXTN — device-side assert that presents as a hang. 57.6 → 167 tok/s |
| 3 | [#36750](https://github.com/sgl-project/sglang/pull/36750) `max_thinking_tokens` on the OpenAI endpoint | *Optional.* Per-request thinking budgets sent over the OpenAI protocol are silently dropped |
| 4 | `patches/0001-qsa-fp8-kv-dequant-on-read.patch` | With `--kv-cache-dtype fp8_e4m3`, any prompt long enough to be chunked dies on the 2nd chunk: `Unsupported rhs dtype fp8e4nv` |

**On patch 4:** it casts the selected KV rows back to the query dtype, assuming
the direct-cast e4m3 store (scale 1.0). Upstream
[#36644](https://github.com/sgl-project/sglang/pull/36644) solves the same crash
properly, with per-layer KV descale and a FlashAttention fallback, and was
reproduced independently by two people on this card. **Prefer #36644 once it
merges**, and check your checkpoint's KV scales before relying on the shortcut.
Alternatively drop `--kv-cache-dtype fp8_e4m3` from `serve.sh` and you do not
need patch 4 at all — you lose roughly half the KV cache.

---

## Traps

**Docker's default seccomp profile blocks io_uring.** `io_uring_setup`,
`io_uring_enter` and `io_uring_register` are all denied, so the PLE reader
cannot start. `serve.sh` passes `--security-opt seccomp=seccomp-iouring.json`,
which is the default profile plus those three syscalls. If your first run dies
early with an io_uring error, this is why.

**The day-0 image does not contain the NVMe path.** It was pushed
2026-08-26T12:30Z; PR #36567 was opened later the same day. Verified inside the
container: `qwen4_exp.py` present, `qwen4_ple_nvme.py` absent. Hence the source
overlay rather than a pip install on top.

**Do not `git clean` the base image's tree.** It removes untracked build
artifacts that ship with the image (`_grpc` / `_server` / `_multimodal` `.so`).

**Speculative draft depth is capped at 4** by the QSA compression ratio — this
is architectural, not a tuning choice. Higher `--speculative-num-draft-tokens`
does not help.

**Pin the checkpoint revision.** Day-0 uploads get amended hours later without
an announcement.

---

## Known limits

Honest ones, measured:

- **Concurrency tops out around 5 streams.** The mamba recurrent-state slots and
  the KV pool compete for the same budget, so this behaves differently from a
  dense model: aggregate throughput is flat from c4 (408 tok/s) to c16 (391).
  If you need many parallel users, a dense 27B on the same card does 1,100+.
- **Prefix cache is small.** 233K tokens against ~1.8M for a dense 27B at the
  same memory. Long shared prefixes re-prefill more often.
- **Hierarchical KV cache (HiCache) is not safe here yet.** It attaches and
  reopening is 2.4× faster, but after restore a needle-in-haystack probe answers
  "there is no code in the context" — silent context loss, most likely because
  the QSA indexer's side cache is not tiered along with KV. Off by default.
- **Tool calling: 6/7** on our battery, in both thinking modes — it prefers
  sequential calls over emitting two in parallel. Argument schemas and enums are
  respected, and Korean free-text arguments are preserved.

---

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the full table — both thinking modes,
a same-card comparison against a dense Qwen3.8-27B in both of its modes, and
what the numbers do *not* say.

Short version, comparing like with like (both non-thinking): it wins **all
five** single-stream workloads and both accuracy scores against a dense 27B —
HumanEval+ 0.939/0.921 vs 0.915/0.890, coding throughput 180.9 vs 78.1 tok/s.
Turning thinking on buys about 2 points of HumanEval+ (0.957/0.927) and costs
22% of coding throughput plus 3.6× the TTFT. A *thinking* 27B is still the more
accurate coder (0.982/0.939), and for many concurrent users the dense 27B wins
by roughly 3×. What you get here is a 176B model's breadth on hardware that
should not fit one.

---

## License

The patches and scripts here are Apache 2.0, matching SGLang. The model
checkpoint has its own license — see the model card.
