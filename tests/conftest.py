"""Test session setup: the tests run on the CPU with two threads (reproducible and fast on small problems).

TWISTED_BILAYER_FORCE_CPU must be set before the package's `_shared.engine.backend` is imported, since it picks
the device at import. GPU runs are production runs, not tests.
"""

import os

os.environ.setdefault("TWISTED_BILAYER_FORCE_CPU", "1")

import torch  # noqa: E402

torch.set_num_threads(2)
