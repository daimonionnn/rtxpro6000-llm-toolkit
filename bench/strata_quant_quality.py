#!/usr/bin/env python3
"""Strata's tools/quant_quality.py with the K/V format as an option.

    ~/development/Strata/.venv/bin/python bench/strata_quant_quality.py [--kv int8] run --config CFG.json --out RUN.tsv ...

quant_quality.py pins the yardstick to FP16 K/V; the served profiles use INT8. Everything else
(subcommands, arguments, fixed settings) is the original tool's.
"""
import os
import sys

STRATA = os.environ.get("STRATA_DIR", os.path.expanduser("~/development/Strata"))
sys.path.insert(0, os.path.join(STRATA, "tools"))
import quant_quality as qq  # noqa: E402

if len(sys.argv) > 2 and sys.argv[1] == "--kv":
    kv = sys.argv[2]
    del sys.argv[1:3]
    i = qq.FIXED.index("--kv")
    qq.FIXED[i + 1] = kv
sys.exit(qq.main())
