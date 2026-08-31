#!/usr/bin/env python
"""Legacy entry point -- see ``scripts/download_data.py sft``."""

import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.data.download_sft import main

if __name__ == "__main__":
    sys.exit(main())
