#!/usr/bin/env python3
"""Run Strata Q6 with GPU vision and a 262144-token context."""
from run_q8_128k import main

if __name__ == "__main__":
    main(quant="q6", context=262144, vision=True, description=__doc__)
