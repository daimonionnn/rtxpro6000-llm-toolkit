#!/usr/bin/env python3
"""Run Strata Q6_K/Q8_0 after applying both patches and preparing packs/q6_k.

Foreground server on localhost. Stop the other GPU server first.
"""
from run_q8_128k import main

if __name__ == "__main__":
    main(quant="q6", description=__doc__)
