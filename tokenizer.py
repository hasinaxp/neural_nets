"""Legacy import shim -- the tokenizer now lives in ``nanollm.tokenizer``.

    from tokenizer import Tokenizer          # still works
    from nanollm.tokenizer import Tokenizer  # preferred
"""

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.tokenizer import Tokenizer  # noqa: F401

__all__ = ["Tokenizer"]
