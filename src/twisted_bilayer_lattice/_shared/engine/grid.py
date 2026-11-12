"""Periodic FFT grid in d = 1, 2 or 3 dimensions.

Conventions:

  * dx_i = L_i / N_i exactly, points x_j = (j - N_i//2) dx_i, j = 0..N_i-1.
    The FFT's periodic box is then exactly L_i, which is what a lattice
    potential needs: the commensurate supercell must be *exactly* periodic
    on the box or the potential has a seam. The origin is a grid point (for
    even N), so the AA stacking point sits on the grid.
  * Field layout is (B, C, *N): leading batch axis, component axis, then the
    d spatial axes. Spatial axes are always the LAST d axes, so every
    reduction and FFT uses dim=self.sdims.
  * Per-axis k vectors are 1-D tensors; broadcastable (…,N_i,…) views are
    built by `axis_view`. Nothing dense is stored on the grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .backend import DEVICE


class Grid:
    def __init__(
        self,
        N: Sequence[int],
        L: Sequence[float],
        device: torch.device | None = None,
        real_dtype: torch.dtype = torch.float64,
    ):
        N = tuple(int(n) for n in N)
        L = tuple(float(l) for l in L)
        if len(N) != len(L) or not 1 <= len(N) <= 3:
            raise ValueError(f"Grid: N and L must have equal length 1..3, got {N}, {L}")
        if any(n < 2 for n in N):
            raise ValueError(f"Grid: need N >= 2 per axis, got {N}")
        self.N, self.L = N, L
        self.d = len(N)
        self.device = DEVICE if device is None else device
        self.real_dtype = real_dtype
        self.dx = tuple(l / n for l, n in zip(L, N))
        self.dV = math.prod(self.dx)
        self.volume = math.prod(L)
        self.numel = math.prod(N)
        self.sdims = tuple(range(-self.d, 0))  # spatial axes of a (B,C,*N) field
        # host-side float64 master copies; device copies in the working dtype
        self.x64 = [torch.arange(n, dtype=torch.float64) * dx - (n // 2) * dx for n, dx in zip(N, self.dx)]
        self.k64 = [2.0 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64) for n, dx in zip(N, self.dx)]
        self.x = [t.to(self.device, real_dtype) for t in self.x64]
        self.k = [t.to(self.device, real_dtype) for t in self.k64]
        self.k_sq_max = float(sum(float((t * t).max()) for t in self.k64))
        self.k_nyquist = tuple(math.pi / dx for dx in self.dx)

    # -- views ---------------------------------------------------------------
    def axis_view(self, v: torch.Tensor, axis: int, lead: int = 2) -> torch.Tensor:
        """Reshape a 1-D per-axis tensor (N_i,) so it broadcasts against a
        field with `lead` leading (batch, component) axes."""
        shape = [1] * (lead + self.d)
        shape[lead + axis] = self.N[axis]
        return v.reshape(shape)

    def coord(self, axis: int, lead: int = 2) -> torch.Tensor:
        return self.axis_view(self.x[axis], axis, lead)

    def k_shifted(self, axis: int, q=None) -> torch.Tensor:
        """k_i + q for a twisted boundary condition psi(x + L e_i) = e^{i theta}
        psi(x): with psi = e^{i q x} phi, q = theta/L_i, the kinetic operator
        on the periodic phi has k -> k + q. `q` may be a scalar or a tensor
        broadcastable against (B, 1, *1): then the shifted k is batched."""
        kv = self.axis_view(self.k[axis], axis)
        if q is None:
            return kv
        if not torch.is_tensor(q):
            if float(q) == 0.0:
                return kv
            return kv + float(q)
        return kv + q.to(kv.dtype)

    # -- transforms and reductions ---------------------------------------
    def fft(self, psi: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(psi, dim=self.sdims)

    def ifft(self, psi_k: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(psi_k, dim=self.sdims)

    def integrate(self, f: torch.Tensor) -> torch.Tensor:
        """Integral over space, leading axes kept."""
        return f.sum(dim=self.sdims) * self.dV

    def to(self, real_dtype: torch.dtype) -> Grid:
        if real_dtype == self.real_dtype:
            return self
        return Grid(self.N, self.L, self.device, real_dtype)

    def describe(self) -> str:
        pts = " x ".join(str(n) for n in self.N)
        box = " x ".join(f"{l:.4g}" for l in self.L)
        dxs = ", ".join(f"{dx:.4g}" for dx in self.dx)
        return f"{self.d}D grid {pts} = {self.numel:,} points, box {box} (a), dx = ({dxs}) a"
