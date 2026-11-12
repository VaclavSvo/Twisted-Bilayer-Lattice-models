"""Potentials: the twisted square bilayer of Meng et al., harmonic traps, and
1D stand-ins. Units a = 1, E_R = 1.

The bilayer follows Supplement Eq. (5) of the paper exactly:

    V_1 = V_0 [ sin^2(k x cos(th/2) - k y sin(th/2)) + sin^2(k y cos(th/2) + k x sin(th/2)) ]
    V_2 = V_0 [ sin^2(k x cos(th/2) + k y sin(th/2)) + sin^2(k y cos(th/2) - k x sin(th/2)) ]

with k = 2 pi / lambda = pi / a. Both lattices have a minimum at the origin,
so the origin is an AA stacking point. For a commensurate angle
tan(theta/2) = n/m the pattern is exactly periodic on the axis-aligned box of
side sqrt(m^2 + n^2) a (the supercell of the paper's band calculation,
"sqrt(c) a x sqrt(c) a"); for (m + n) odd that box is one primitive supercell
= 2 L_M on a side, holding 2 x 2 moire cells. The experiment's 5.21 deg is
approximated by (m, n) = (22, 1): theta = 5.2051 deg, L = sqrt(485) = 22.02 a.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .backend import _largest_prime_factor
from .grid import Grid
from .units import M_LATTICE


@dataclass(frozen=True)
class MoireGeometry:
    m: int
    n: int
    alpha: float  # theta / 2
    theta: float  # twist angle (rad)
    L_box: float  # exactly periodic box side, units of a
    L_moire: float  # moire period a / (2 sin(theta/2))
    primitive: bool  # True if the box is one primitive supercell ((m+n) odd)

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta)

    def grid_size(self, points_per_a: float = 16.0, max_prime: int = 13) -> int:
        """Smallest even FFT-friendly N with at least points_per_a points per a."""
        return fft_friendly_size(int(math.ceil(self.L_box * points_per_a)), max_prime)

    def describe(self) -> str:
        return (
            f"tan(theta/2) = {self.n}/{self.m}: theta = {self.theta_deg:.4f} deg, "
            f"box L = sqrt({self.m**2 + self.n**2}) a = {self.L_box:.4f} a "
            f"= {self.L_box / self.L_moire:.3g} L_M (L_M = {self.L_moire:.4f} a)"
            f"{'' if self.primitive else ' [2 primitive cells: (m+n) even]'}"
        )


def moire_geometry(m: int = 22, n: int = 1) -> MoireGeometry:
    alpha = math.atan2(n, m)
    c = m * m + n * n
    return MoireGeometry(
        m=m,
        n=n,
        alpha=alpha,
        theta=2.0 * alpha,
        L_box=math.sqrt(c),
        L_moire=1.0 / (2.0 * math.sin(alpha)),
        primitive=(m + n) % 2 == 1,
    )


def fft_friendly_size(n_min: int, max_prime: int = 13) -> int:
    n = n_min + (n_min % 2)
    while _largest_prime_factor(n) > max_prime:
        n += 2
    return n


# -- builders ----------------------------------------------------------------


def _xy(grid: Grid, lead: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = grid.coord(0, lead)
    y = grid.coord(1, lead) if grid.d >= 2 else None
    return x, y


def square_lattice(grid: Grid, V0: float, alpha: float = 0.0, lead: int = 2) -> torch.Tensor:
    """One square lattice of depth V0 rotated by -alpha (its potential is
    U(R(+alpha) r) with U = V0[sin^2(pi x) + sin^2(pi y)]). In 1D: V0 sin^2(pi x).
    Returns a dense real tensor of shape (1,)*lead + N."""
    x, y = _xy(grid, lead)
    pi = math.pi
    if y is None:
        return V0 * torch.sin(pi * x) ** 2 * torch.ones((1,) * lead + grid.N, dtype=x.dtype, device=x.device)
    c, s = math.cos(alpha), math.sin(alpha)
    u = x * c - y * s
    v = y * c + x * s
    V = V0 * (torch.sin(pi * u) ** 2 + torch.sin(pi * v) ** 2)
    return V.expand((1,) * lead + grid.N)


def bilayer_potential(
    grid: Grid, V0: float, geometry: MoireGeometry | None = None, theta: float | None = None
) -> torch.Tensor:
    """Both lattices, stacked on the component axis: shape (1, 2, *N).
    Lattice 1 rotated by -theta/2, lattice 2 by +theta/2 (paper Eq. 5)."""
    if geometry is not None:
        alpha = geometry.alpha
    elif theta is not None:
        alpha = 0.5 * theta
    else:
        alpha = 0.0
    V1 = square_lattice(grid, V0, +alpha, lead=2)
    V2 = square_lattice(grid, V0, -alpha, lead=2)
    return torch.cat([V1, V2], dim=1)


def harmonic_trap(grid: Grid, omega: Sequence[float], m: float = M_LATTICE, lead: int = 2) -> torch.Tensor:
    """(1/2) m sum_i omega_i^2 x_i^2 over the grid's axes (omega_i = 0 skips an
    axis). Shape (1,)*lead + N."""
    V = None
    for ax, w in enumerate(omega[: grid.d]):
        if w == 0.0:
            continue
        term = 0.5 * m * w * w * grid.coord(ax, lead) ** 2
        V = term if V is None else V + term
    if V is None:
        return torch.zeros((1,) * lead + grid.N, dtype=grid.real_dtype, device=grid.device)
    return V.expand((1,) * lead + grid.N)


def lattice_1d(grid: Grid, V0: float, period: float = 1.0, lead: int = 2) -> torch.Tensor:
    """V0 sin^2(pi x / period) along the first axis."""
    x = grid.coord(0, lead)
    return (V0 * torch.sin(math.pi * x / period) ** 2).expand((1,) * lead + grid.N)


def toy_bilayer_1d(grid: Grid, V0: float, n_sites: int, V0_2: float | None = None) -> torch.Tensor:
    """A 1D stand-in for the bilayer: lattice 1 with period a on a
    box of n_sites sites, lattice 2 with period L/(n_sites - 1) so the two beat
    once per box. Shape (1, 2, *N)."""
    L = grid.L[0]
    V1 = lattice_1d(grid, V0, period=1.0)
    V2 = lattice_1d(grid, V0 if V0_2 is None else V0_2, period=L / (n_sites - 1))
    return torch.cat([V1, V2], dim=1)
