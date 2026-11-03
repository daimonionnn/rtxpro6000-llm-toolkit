#!/usr/bin/env python3
"""Perplexity of runs against a teacher_forcing.py reference, on the positions where the target
is in every file's top 64, so that the reference's capped log-probabilities do not bias it.

    ~/development/Strata/.venv/bin/python bench/teacher_forcing_ppl.py REF.tsv RUN.tsv [...]
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser(os.environ.get("STRATA_DIR", "~/development/Strata")), "tools"))
import quant_quality as qq  # noqa: E402

runs = [qq.read_run(p) for p in sys.argv[1:]]
names = [os.path.basename(p) for p in sys.argv[1:]]
for seg in list(runs[0]) + ["all"]:
    segs = list(runs[0]) if seg == "all" else [seg]
    nll, n, out = [0.0] * len(runs), 0, 0
    for s in segs:
        for pos, row in runs[0][s].items():
            got = [r[s].get(pos) for r in runs]
            if any(g is None for g in got):
                continue
            if any(g[0] not in dict(g[3]) for g in got):
                out += 1
                continue
            n += 1
            for i, g in enumerate(got):
                nll[i] -= g[1]
    print(f"{seg:13s} {n:5d} pos ({out} with the target outside a top 64)  "
          + "  ".join(f"{nm[:34]} {math.exp(x / max(n, 1)):.4f}" for nm, x in zip(names, nll)))
