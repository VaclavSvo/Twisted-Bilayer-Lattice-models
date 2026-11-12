"""Device and dtype selection, grid-size hygiene.

Everything numerical runs on PyTorch. The device is chosen once per process
(`DEVICE`); `TWISTED_BILAYER_FORCE_CPU=1` forces the CPU even when CUDA is present, so
`twisted-bilayer run gpu_check` can build a CPU reference on a machine that has a GPU.

Precision policy (a consumer GPU with fp64 at 1/64 of the fp32 rate):
  * complex128 on the supercell (<~1e6 points): superfluid fractions come from
    energy differences O(theta^2) and residuals ~1e-8, which float32 cannot
    carry. Even at the fp64 rate a 352^2 x 2-component step is well under a
    millisecond.
  * complex64 for anything large (full trapped cloud, thin-z 3D sweeps).
The coupling scan relaxes in complex64 and polishes in complex128 (two
engines, `dtype_relax` / `dtype_polish` of its scenario row).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # annotations only
    import numpy as np

_FORCE_CPU = os.environ.get("TWISTED_BILAYER_FORCE_CPU", "").strip().lower() not in ("", "0", "false", "no")
HAS_GPU = torch.cuda.is_available() and not _FORCE_CPU
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")

COMPLEX_DTYPES = {
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
REAL_OF = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def complex_dtype(name_or_dtype) -> torch.dtype:
    if isinstance(name_or_dtype, torch.dtype):
        return name_or_dtype
    return COMPLEX_DTYPES[str(name_or_dtype)]


def describe_device() -> str:
    if HAS_GPU:
        p = torch.cuda.get_device_properties(DEVICE)
        return f"CUDA {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}, torch {torch.__version__}"
    return f"CPU ({torch.get_num_threads()} threads), torch {torch.__version__}"


def to_numpy(a) -> np.ndarray:
    return a.detach().cpu().numpy()


def synchronize() -> None:
    if HAS_GPU:
        torch.cuda.synchronize()


def _largest_prime_factor(n: int) -> int:
    largest, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            largest, n = d, n // d
        d += 1
    return n if n > 1 else largest


def check_fft_grid_size(N: int, max_prime: int = 13) -> str | None:
    """cuFFT/pocketfft are fast for N whose prime factors are all small and
    fall onto Bluestein otherwise. Returns a warning string (or None); the
    physics is unaffected, only the speed."""
    factor = _largest_prime_factor(N)
    if factor <= max_prime:
        return None
    suggestion = N
    while _largest_prime_factor(suggestion) > max_prime:
        suggestion += 1
    return (
        f"[perf warning] N={N} has largest prime factor {factor}; FFTs will be "
        f"slower than at a nearby size with prime factors <= {max_prime}, e.g. {suggestion}"
    )
