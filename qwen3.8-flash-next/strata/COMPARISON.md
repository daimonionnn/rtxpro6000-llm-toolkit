# Strata quantization comparison

Measured on 2026-10-06: RTX PRO 6000 Blackwell Workstation 96 GB, Intel Core Ultra
7 270K Plus (24 physical cores), 244 GiB host RAM. One request at a time on private
localhost ports. Neither model downloads nor builds ran during these measurements.
Earlier runs on the application port and an overlapping prefill run are excluded.

The ik_llama.cpp reference engine, build and serving scripts belong to the
separate public [ik-llama-toolkit](https://github.com/daimonionnn/ik-llama-toolkit)
repository. This toolkit contains the Strata launchers and the API measurement
harness used for this comparison.

## All four configurations

| Runtime / target | English prose | Python code | Slovak | Cold TTFT, ~120K |
|---|---:|---:|---:|---:|
| ik_llama.cpp Q8_0 + MTP | 50.9 tok/s | 55.9 tok/s | 46.8 tok/s | 94.729 s |
| Strata Q8_0 + MTP | 101.7 tok/s | 103.5 tok/s | 75.1 tok/s | 21.666 s |
| ik_llama.cpp Q6_K/Q8_0 + MTP | 60.9 tok/s | 68.5 tok/s | 60.0 tok/s | 86.993 s |
| Strata Q6_K/Q8_0 + MTP | 117.1 tok/s | 142.5 tok/s | 81.3 tok/s | 27.411 s |

All use a 131,072-token context. Decode columns are medians of three short-context
512-token responses. Each server starts fresh, receives the same arithmetic warmup,
then alternates the three prompts for three rounds. Strata's expert cache adapts
across these rounds; the median is not a fixed warm-cache speed. TTFT is a fresh
prefix, not a controlled cold OS file cache: it includes weight reads and prefill
warmup. This is a comparison of usable configurations, not isolated kernel speed.

Strata Q6 is faster for generation here, but Q8 processes the first long prompts
faster. At ~120K, both Strata configurations beat both ik configurations.
The language check below cannot establish that Q6 preserves Q8 quality.

## Q8_0: Strata versus ik_llama.cpp

Both use the same six lmstudio-community Q8_0 GGUF shards, a 131,072-token
context, greedy sampling and thinking disabled. Both use MTP, but with different
draft weights and policies. Strata also converts small projections to BF16 and
uses a different INT8 KV encoding. This compares complete serving configurations.

| Workload | ik_llama.cpp, tok/s | Strata, tok/s | Strata / ik |
|---|---:|---:|---:|
| English prose | 50.9 | 101.7 | 2.00x |
| Python code and explanation | 55.9 | 103.5 | 1.85x |
| Slovak explanation | 46.8 | 75.1 | 1.60x |

Medians of three 512-token responses after a short warmup. Client decode timing
excludes the first text chunk; token batches make the rate approximate. Individual
Strata samples: 78.0/102.3/101.7 for prose, 90.3/103.5/121.1 for code and
66.9/75.1/80.6 for Slovak. Its adaptive expert cache warms across requests.

### Prompt processing

TTFT includes HTTP and tokenization and ends at the first nonempty text chunk.
The scripts generate approximately equal prompts with fresh nonces.

| Size | ik tokens | ik cold TTFT | Strata tokens | Strata cold TTFT | ik repeated TTFT | Strata repeated TTFT |
|---|---:|---:|---:|---:|---:|---:|
| ~4K | 4,009 | 2.901 s | 4,011 | 4.827 s | 0.121 s | 0.067 s |
| ~32K | 31,881 | 20.442 s | 31,882 | 12.460 s | 0.132 s | 0.098 s |
| ~120K | 120,592 | 94.729 s | 120,593 | 21.666 s | 0.244 s | 0.263 s |

The 4K cold request is the first batched prefill after startup and includes
prefill warmup. The repeated request resends the identical prompt and benefits
from the prefix cache. At ~120K, Strata's cold TTFT is 4.37x shorter; with a
cached prefix both answer in about a quarter second. Generation throughput at
long context was not measured in these 128K runs; the 256K test below adds it.

### Configuration

- ik_llama.cpp: commit in `logs/strata-comparison/environment.json`;
  `-ngl 99 -ncmoe 19 -c 131072 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048
  -t 8 -tb 24 -thp --parallel 1`, Q4_K_M MTP head, `mtp:n_max=3,p_min=0.75`.
  Main CUDA-host weight buffer 98.61 GiB; 95,202 MiB GPU memory after language collection.
- Strata 0.1.39 `6f32ec070f23ced9f50e704d854d775da52591ab` plus
  `q8-ple.patch`: MTP Q2_0 draft, verify window 4, minimum draft probability
  0.5; INT8 KV; automatic 8,192-token prefill; 1,536 MiB VRAM reserve;
  17,414 experts in 84.70 GiB GPU cache and 34.83 GiB pinned RAM complement;
  PLE via mmap. 96,420 MiB GPU memory after language collection.

Strata needs the local patch to accept this Q8 PLE table. PLE rows agree exactly
with ggml in mmap and direct I/O tests. The synthetic Q8 expert test passes.
The 363 rounded small-tensor conversions are described in [README.md](README.md).

### Slovak samples

Both completed arithmetic warmup correctly and generated coherent prose,
code and Slovak explanations. Both made grammatical errors in the ten fixed
Slovak prompts. For example, both misused dative in a sentence like "na rozvoj
mestu" and locative in "v centre meste". Strata's declension answer repeatedly
tried to correct itself and reached 800 tokens; ik's answer ended after 116.
Both produced the same correct numeral-agreement sentences and natural translation.

This is a small qualitative check, not a quality score or proof of equivalence.
No new HumanEval+/MBPP+ run was performed. A shuffled sheet is saved as
`logs/language-samples/sk/blind-20261006-095746.md`, with an adjacent key.

### Raw evidence

- `logs/strata-comparison/ik-q8-mtp-clean-20261006.json`
- `logs/strata-comparison/strata-q8-20261006.json`
- `logs/strata-comparison/ik-q8-clean-prefill-20261006.log`
- `logs/strata-comparison/strata-q8-prefill-20261006.log`
- `logs/strata-comparison/strata-q8-engine-20261006.log`
- `logs/language-samples/sk/{ik-q8-mtp-20261006,strata-q8-20261006}.json`

## Q6_K/Q8_0 checkpoint

Checkpoint: lmstudio-community Q6_K, revision
`158fc825df3eaa6c22d3c57a5927a5adf1c7cda7`, five shards, 156.13 GiB.
Its PLE shard contains the same Q8_0 table as the local Q8 model, with only
`split.count` different in its 192-byte header. Reusing the table and replacing
that header produced SHA-256
`8aeb4c8e478809a91b2094f61ba276265d10425d036a3a9c5c9959ea650858d4`,
exactly the published Q6 shard hash, saving a 50.66 GiB download. No
requantization was performed.

Expert gate/up tensors are Q6_K; expert down tensors and PLE are Q8_0.
Routed experts therefore average about 7.21 bits per weight, including scales.
The stock-Strata preflight rejects Q6_K/Q8_0 expert pairs.

### Step 1: ik_llama.cpp Q6_K

All five checkpoint hashes match the published SHA-256 values in
[Q6_SHA256SUMS](Q6_SHA256SUMS). Same context, KV and Q4_K_M MTP settings as
ik Q8, with `-ncmoe 13`. Main CUDA-host buffer 78.60 GiB and main CUDA
buffer 77.51 GiB. The Q8 setup uses 19 expert layers in RAM; Q6 uses 13.

| Workload | ik Q8, tok/s | ik Q6, tok/s | Q6 / Q8 |
|---|---:|---:|---:|
| English prose | 50.9 | 60.9 | 1.20x |
| Python code and explanation | 55.9 | 68.5 | 1.23x |
| Slovak explanation | 46.8 | 60.0 | 1.28x |

| Size | Prompt tokens | Q6 cold TTFT | Q6 repeated TTFT |
|---|---:|---:|---:|
| ~4K | 4,010 | 2.585 s | 0.103 s |
| ~32K | 31,881 | 18.692 s | 0.119 s |
| ~120K | 120,592 | 86.993 s | 0.234 s |

Evidence: `logs/strata-comparison/ik-q6-mtp-20261006.json`,
`ik-q6-prefill-20261006.log`, `ik-q6-server-20261006.log` and
`q6-sha256-verified.log` in that directory.

### Step 2: Strata Q6_K/Q8_0

Local [q6-experts.patch](q6-experts.patch), applied after the Q8 PLE patch,
adds Q6_K gate/up, head dispatch and GPU dequantization. The random Q6_K/Q8_0
expert test, Q8 regression test and real expert tests at layers 0, 1, 16, 33
and 47 pass. GPU FP32 dequantization, FP16 conversion and embedding gathers
match ggml bit for bit. Real-layer GPU expert error against the float reference
is 1.08–1.24%, including activation quantization; CPU/GPU arithmetic differs.
Invalid Q6_K down at width 640 is still rejected.

Same Strata context, MTP, KV, prefill and VRAM reserve as Q8. Cache: 20,687
experts, 85.33 GiB on GPU and 16.04 GiB pinned RAM complement. GPU memory
was 96,348 MiB after language collection. The pack rounds 364 small tensors
to BF16 (1.27 GiB, max absolute error 0.0144); 96 other conversions are exact.
Main expert, embedding, head and PLE quantization is unchanged.

| Workload | Strata Q8, tok/s | Strata Q6, tok/s | Q6 / Q8 |
|---|---:|---:|---:|
| English prose | 101.7 | 117.1 | 1.15x |
| Python code and explanation | 103.5 | 142.5 | 1.38x |
| Slovak explanation | 75.1 | 81.3 | 1.08x |

Individual Q6 rates: 80.4/117.1/142.7 prose, 106.5/142.5/145.2 code,
75.1/81.3/91.7 Slovak. Cache history and draft acceptance affect generation
speed; these figures do not promise a gain for every prompt.

| Size | Prompt tokens | Strata Q6 cold TTFT | Strata Q6 repeated TTFT |
|---|---:|---:|---:|
| ~4K | 4,005 | 7.270 s | 0.054 s |
| ~32K | 31,877 | 23.311 s | 0.106 s |
| ~120K | 120,587 | 27.411 s | 0.263 s |

Q6's first prefill is slower than Q8 here, despite the smaller target weights.
Source inspection explains a concrete implementation difference: Q8_0 is covered
by `src/prefill/moe_mmq.cu`'s quantized matrix path, but Q6_K is not. MMQ was
compiled into this build and enabled by default. With Q6_K gate/up, the planner
therefore selects the FP16 fallback for the layer: `prefill.cpp` dequantizes both
gate/up and down matrices into FP16 scratch before its FP16 GEMMs. The Q6 patch
adds decode and dequantization support, not Q6_K MMQ prefill.

This makes the Q6 prefill path substantially different from Q8; smaller GGUF
weights alone do not guarantee lower prefill latency. The measurement also
includes filesystem reads and warmup; we have not profiled the time attributable
to each part. Both repeated ~120K prompts answer in 0.263 s.

### Q6 Slovak samples

Both Q6 servers return the same correct numeral-agreement sentences and the
same natural translation as both Q8 servers. All four answer the arithmetic
warmup correctly. Strata Q6's declension response ends after 120 tokens instead
of Q8's 800-token correction loop, but still misuses dative and locative.
ik Q6 fixes the dative sentence; its sentence labelled locative uses a
grammatical genitive instead, so it still does not meet the requested case.

Ten fixed prompts per configuration are a qualitative check, not a scored
quality benchmark. No new EvalPlus or independently graded comparison was run.
A four-way shuffled sheet is at
`logs/language-samples/sk/blind-20261006-103642.md`, with a separate key.

### Q6 evidence

- `logs/strata-comparison/q6-download-manifest.json`, `q6-layout.json`,
  `q6-ple-reuse.json`, `q6-sha256-verified.log`
- `logs/strata-comparison/ik-q6-mtp-20261006.json`, `ik-q6-prefill-20261006.log`,
  `ik-q6-server-20261006.log`, `ik-q6-memory-20261006.txt`
- `logs/strata-comparison/strata-q6-20261006.json`, `strata-q6-prefill-20261006.log`,
  `strata-q6-engine-20261006.log`, `strata-q6-memory-20261006.txt`
- `logs/strata-comparison/q6-strata-{synthetic-parity,real-parity,invalid-down}.log`
- `logs/language-samples/sk/{ik-q6-mtp-20261006,strata-q6-20261006}.json`

Run commands and both reusable patches are in [README.md](README.md).
The original ik Q8 configuration is retained; these experiments do not change
the toolkit's default profile.

## 256K context profiles

Measured later on 2026-10-06, sequentially on private ports 8097 (Q8) and 8098
(Q6), with no other GPU server or benchmark running. Both use the same pinned
Strata commit and both compatibility patches, native `--max-context 262144`,
INT8 KV, MTP verify window 4 / minimum probability 0.5, prefill auto (8,192),
resident budget 135 GiB and VRAM reserve 1,536 MiB. No RoPE extension is used.
The registry contains both 128K and 256K variants; start/status/stop manage them.

### Short-context TG

The same arithmetic warmup, three fixed prompts and three alternating rounds
as the 128K comparison, 512 tokens per response, greedy and thinking off.
Capacity is 262,144 tokens; these inputs themselves are only 44–74 tokens.

| Profile | Prose | Code | Slovak |
|---|---:|---:|---:|
| `strata-q8-256k` | 89.4 tok/s | 104.9 tok/s | 73.1 tok/s |
| `strata-q6-256k` | 119.5 tok/s | 144.4 tok/s | 93.2 tok/s |

Individual Q8 rates: 75.7/89.4/92.4 prose, 81.8/104.9/109.2 code,
60.0/75.4/73.1 Slovak. Q6: 81.6/119.5/132.0 prose,
97.4/144.4/166.8 code, 70.1/93.2/102.5 Slovak. Cache history affects these
medians; the 128K versus 256K runs do not isolate KV allocation overhead.

### Prefill sweep

Run after the short TG test, fresh nonce for each size; identical prefix
repeated once. TTFT ends at the first nonempty text chunk and includes HTTP,
tokenization, weight reads and initial batched-prefill warmup. The nonce prevents
prefix reuse, but does not clear expert caches or the OS filesystem cache.

| Size | Q8 tokens | Q8 fresh TTFT | Q8 effective tok/s | Q8 repeated TTFT | Q6 tokens | Q6 fresh TTFT | Q6 effective tok/s | Q6 repeated TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~4K | 4,011 | 5.365 s | 747.6 | 0.061 s | 4,009 | 7.295 s | 549.6 | 0.064 s |
| ~32K | 31,882 | 13.768 s | 2,315.7 | 0.140 s | 31,880 | 23.740 s | 1,342.9 | 0.095 s |
| ~120K | 120,593 | 22.558 s | 5,346.0 | 0.267 s | 120,591 | 28.863 s | 4,178.0 | 0.263 s |
| ~255K | 254,834 | 46.581 s | 5,470.8 | 0.506 s | 254,832 | 44.004 s | 5,791.2 | 0.506 s |

Effective prefill is prompt tokens / TTFT. Repeated TTFT measures prefix reuse,
not processing every token again; its prompt-tokens/TTFT ratio is not prefill
throughput. Q8 has shorter latency in the first 4–120K sweep, but Q6 is slightly
faster at ~255K. Q6 still uses the FP16 expert fallback, and Q8 the MMQ path;
this implementation difference alone does not predict latency for every request.

### TG after long prompts

After finishing the sweep, `bench/long_context.py` sends a fresh word corpus
followed by the prose instruction, then repeats the identical prompt twice.
All responses reach the 512-token budget, contain text and no reasoning output.
Rates below exclude TTFT and use the same approximate client SSE timing as the
short tests. Speculation can deliver several tokens per chunk; engine log rates
agree to rounding. These are different prompts with already warmed expert/OS
caches, so comparing long TG against short TG does not isolate context overhead.

| Profile | Actual prompt tokens | Fresh TTFT | Repeated TTFT, runs 2 / 3 | TG runs 1 / 2 / 3 | Median TG |
|---|---:|---:|---|---|---:|
| Q8, ~120K | 120,632 | 21.481 s | 0.264 / 0.280 s | 92.1 / 127.4 / 130.0 tok/s | 127.4 tok/s |
| Q6, ~120K | 120,631 | 20.413 s | 0.259 / 0.268 s | 139.2 / 139.9 / 149.9 tok/s | 139.9 tok/s |
| Q8, ~255K | 254,876 | 46.192 s | 0.512 / 0.504 s | 117.8 / 113.0 / 119.3 tok/s | 117.8 tok/s |
| Q6, ~255K | 254,875 | 44.218 s | 0.505 / 0.498 s | 153.1 / 145.3 / 166.2 tok/s | 153.1 tok/s |

The second fresh ~120K prefill is already quicker for Q6 (20.413 versus 21.481
s), unlike the initial sweep. Prefix coldness, kernel warmup, converted expert
scratch, expert residency and filesystem cache are separate conditions. This
experiment did not time each component or control a cold OS cache. It measures
the sequential workflow, not a universal Q8/Q6 prefill ranking. It also does not
score long-context retrieval or quality.

### Memory and integration

| Profile | GPU expert slots | GPU expert cache | Pinned expert complement | GPU used after all tests | GPU free after all tests |
|---|---:|---:|---:|---:|---:|
| Q8, 128K reference | 17,414 | 84.70 GiB | 34.83 GiB | 96,420 MiB | not recorded |
| Q8, 256K | 17,021 | 82.79 GiB | 36.75 GiB | 96,416 MiB | 873 MiB |
| Q6, 128K reference | 20,687 | 85.33 GiB | 16.04 GiB | 96,348 MiB | not recorded |
| Q6, 256K | 20,224 | 83.42 GiB | 17.95 GiB | 96,350 MiB | 939 MiB |

The larger KV allocation reduces automatic expert slots; VRAM usage stays near
the card's capacity. Pinned expert RAM excludes PLE's OS file cache and other
allocations. Both 256K profiles completed all short, prefill and long requests
without OOM. Measurements exercise ~255K prompts plus 512 output tokens within
the configured 262,144-token capacity.

Live start/status/stop checks passed for both 256K variants, including refusal
to start another registered profile and release of the native GPU process on
stop. All four shell wrappers produce the expected configs; a stale PID/start
record does not cause stop to signal an unrelated live process. After testing,
the original ik Q8 server is restored on port 8090 with its saved arguments.

### 256K raw evidence

Under `logs/strata-comparison/` (local runtime output, ignored by git):

- `strata-{q8,q6}-256k-20261006.json`: short decode, full responses and usage.
- `strata-{q8,q6}-256k-prefill-20261006.log`: fresh/repeated sweep.
- `strata-{q8,q6}-256k-long-20261006.json`: long TG, responses and usage.
- `strata-{q8,q6}-256k-engine-20261006.log`: expert cache and engine timings.
- `strata-{q8,q6}-256k-{memory,status}-20261006.txt`: VRAM snapshots and registry status.
- `strata-256k-summary-20261006.json`: aggregated medians and sweep values.
- `strata-q8-256k-8097.json`, `strata-q6-256k-8098.json`: exact server configs.

Reusable commands and all four launchers: [README.md](README.md).
