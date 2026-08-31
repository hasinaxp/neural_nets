#!/usr/bin/env python
"""Legacy entry point -- see ``scripts/download_data.py dpo``."""

import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.data.download_dpo import main

if __name__ == "__main__":
    sys.exit(main())
