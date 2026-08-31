"""Legacy import shim -- this module now lives in ``nanollm.data.pretrain_legacy``.

    from pretrain_dataset import *   # still works
    from nanollm.data.pretrain_legacy import *   # preferred
"""

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from nanollm.data.pretrain_legacy import *  # noqa: F401,F403
from nanollm.data.pretrain_legacy import __dict__ as _mod_dict  # noqa: F401

# Re-export private helpers and module constants too: the old modules were
# imported for their internals in a few places.
globals().update({k: v for k, v in _mod_dict.items() if not k.startswith("__")})
