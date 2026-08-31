"""Makes ``src/nanollm`` importable without installing the package.

``pip install -e .`` is the supported path; this exists so the repo's legacy
top-level modules (``simple_transformer``, ``config``, ``tokenizer``) keep
working when run straight from a clone.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
