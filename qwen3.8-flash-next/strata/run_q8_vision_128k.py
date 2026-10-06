#!/usr/bin/env python3
"""Run Strata Q8 with GPU vision and a 131072-token context."""
from run_q8_128k import main

if __name__ == "__main__":
    main(quant="q8", context=131072, vision=True, description=__doc__)
