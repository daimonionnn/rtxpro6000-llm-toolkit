#!/usr/bin/env python3
"""Run patched Strata Q6_K/Q8_0 with a 262,144-token context on localhost."""
from run_q8_128k import main

if __name__ == "__main__":
    main(quant="q6", context=262144, description=__doc__)
