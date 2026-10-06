#!/usr/bin/env python3
"""Run Strata Q8_0 with a 262,144-token context on localhost."""
from run_q8_128k import main

if __name__ == "__main__":
    main(context=262144, description=__doc__)
