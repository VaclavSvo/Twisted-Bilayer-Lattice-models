"""The paper's own lattice model (Meng et al., Nature 615, 231 (2023), Supplement section IV,
Eqs. 12-13) as a one-layer `model.LatticeModel`, variant "paper":

    H = -t sum_<ij> b_i^dag b_j + (U/2) sum_i n_i (n_i - 1) + sum_i (M_i - mu) n_i,

    M_i = M_R [sin^2(pi (i_x cos theta + i_y sin theta)) + sin^2(pi (i_y cos theta - i_x sin theta))],

on the c = m^2 + n^2 sites of lattice 1 of the commensurate supercell (`supercell_sites`), with
nearest-neighbour bonds only, the paper's t and U (`parameters.paper_parameters`: the deep-lattice
hopping and the 3D-cubic-lattice U) and mu/U = 1. M_i is the other lattice's potential, V2/V0 at the
sites of lattice 1: with (i_x, i_y) the integer coordinates along lattice 1's axes and theta the full
twist angle, cos theta = (m^2 - n^2)/c and sin theta = 2mn/c are rational, so M_i is exactly periodic on
the supercell, M = 0 at AA and 2 M_R where a lattice-1 site sits at a lattice-2 potential maximum (AB).

The paper does not say how M_R depends on Omega_R (its phase diagram, Fig. 5a / S4a, is in the
(V0, Omega_R) plane). M_R is therefore an explicit input here, and `M_R_from_Omega` offers three
documented readings, with rule="k0" as the default assumption:
  * "k0": the lattice model's own k = 0 coupling. In the two-layer model the band-bottom moire potential
    at a lattice-1 site is -Omega_R sum_j S_ij (`build_interlayer`, F_k0), deepest at AA (sum = 1 plus
    images) and shallowest at AB; matching M_i = M_R f_i (f = 0 at AA, 2 at AB) at those two points gives
    M_R = Omega_R (S_sum_AA - S_sum_AB) / 2  (about 0.35 Omega_R at 8 E_R with cutoff 2). It vanishes at
    Omega_R = 0 and is what the two-layer Hamiltonian contains at first order.
  * "omega": M_R = Omega_R, the naive reading (the paper compares Omega_R with U directly).
  * "eq8": M_R = V0 / 2, the linear term of the strong-coupling effective potential of the Supplement's
    Eq. 8 at a lattice-1 site, V_eff = V2/2 - V2^2/(8 Omega_R) + const (equivalently Eq. 9, the average
    lattice). Independent of Omega_R; the quadratic term is the Omega_R >> V0 expansion and is dropped
    (for V0 >= 4 E_R and Omega_R <= 1.2 E_R it would dominate and change sign).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice._shared import config

from .parameters import paper_parameters

M_R_RULES = ("k0", "omega", "eq8")


def paper_site_offsets(cell: lm.Supercell, M_R: float) -> np.ndarray:
    """M_i (c,) of Eq. 12 on the sites of lattice 1 of `cell`, using the exact rational cos theta =
    (m^2 - n^2)/c, sin theta = 2mn/c of the commensurate twist (theta = 2 atan(n/m))."""
    geo = cell.geometry
    m, n, c = geo.m, geo.n, cell.c
    ij = cell.layers[0].ij.astype(np.float64)
    cos_t, sin_t = (m * m - n * n) / c, 2.0 * m * n / c
    x = ij[:, 0] * cos_t + ij[:, 1] * sin_t
    y = ij[:, 1] * cos_t - ij[:, 0] * sin_t
    return float(M_R) * (np.sin(math.pi * x) ** 2 + np.sin(math.pi * y) ** 2)


def paper_site_offsets_formula(i_x, i_y, theta: float, M_R: float):
    """Eq. 12 as written, for a direct check: M_R [sin^2(pi (i_x cos theta + i_y sin theta)) +
    sin^2(pi (i_y cos theta - i_x sin theta))]."""
    i_x, i_y = np.asarray(i_x, dtype=np.float64), np.asarray(i_y, dtype=np.float64)
    a = math.pi * (i_x * math.cos(theta) + i_y * math.sin(theta))
    b = math.pi * (i_y * math.cos(theta) - i_x * math.sin(theta))
    return M_R * (np.sin(a) ** 2 + np.sin(b) ** 2)


def M_R_from_Omega(
    V0: float,
    Omega_R: float,
    rule: str = "k0",
    m: int = config.TWIST_M,
    n: int = config.TWIST_N,
    cutoff: float = 2.0,
    cell: lm.Supercell | None = None,
) -> float:
    """M_R (E_R) for the coupling Omega_R by one of the readings in the module docstring (an assumption
    in every case; "k0" is the default). "k0" uses the interlayer overlap sums of the (m, n) supercell
    with S cut at |d'| < cutoff (2: the converged lattice sum, `build_interlayer`)."""
    if rule == "k0":
        cell = lm.supercell_sites(m, n) if cell is None else cell
        _, diag, _ = lm.build_interlayer(cell.layers[0], cell.layers[1], V0, cell.geometry, cutoff)
        return 0.5 * float(Omega_R) * (diag["F_k0_AA"] - diag["F_k0_AB"])
    if rule == "omega":
        return float(Omega_R)
    if rule == "eq8":
        return 0.5 * float(V0)
    raise ValueError(f"rule must be one of {M_R_RULES}, got {rule!r}")


def paper_model(
    V0: float,
    M_R: float,
    m: int = config.TWIST_M,
    n: int = config.TWIST_N,
    mu_over_U: float = 1.0,
    a_s_m: float = config.A_S_M,
    cell: lm.Supercell | None = None,
) -> lm.LatticeModel:
    """The paper's one-layer model (module docstring) as a `LatticeModel` with n_layers = 1 on the c
    sites of lattice 1: bonds -t to the four nearest neighbours, onsite = M_i, U and t from
    `paper_parameters(V0, a_s_m)`, and the paper's mu/U (1 by default) carried as `mu_over_U`, which
    `gutzwiller(model=...)` uses when given neither mu_over_U nor filling. S is empty, Omega_R = 0 and
    V_pairs = None (the interlayer coupling is the site offset). `ground_state`, `superfluid_fraction`,
    `lattice_bands` and `gutzwiller` take it like the two-layer model; `filling` counts atoms per a^2
    on this one layer."""
    cell = lm.supercell_sites(m, n) if cell is None else cell
    c = cell.c
    p = paper_parameters(V0, a_s_m)
    t = p["t"]
    bonds = lm._hopping_bonds(cell.layers[0], [t], 0)
    return lm.LatticeModel(
        cell=cell,
        V0=float(V0),
        Omega_R=0.0,
        delta=0.0,
        J=np.array([t], dtype=np.float64),
        U=p["U"],
        S=sp.csr_matrix((c, c)),
        bonds=bonds,
        onsite=paper_site_offsets(cell, M_R),
        V_pairs=None,
        diagnostics={"M_R": float(M_R), "t": t, "U": p["U"], "U_over_t": p["U_over_t"], "a_s_m": float(a_s_m)},
        n_layers=1,
        g2d=None,
        g2d_source="paper",
        nu_z_hz=None,
        variant="paper",
        mu_over_U=float(mu_over_U),
    )
