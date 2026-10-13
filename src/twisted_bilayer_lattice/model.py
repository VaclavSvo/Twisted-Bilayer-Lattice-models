"""Model C: the two-layer tight-binding (Bose-Hubbard) model of the twisted
bilayer on the commensurate supercell (docs/1-theory.md §2.6, docs/2-methods.md
§12.2). Units a = 1, E_R = 1, hbar = 1, kappa = 1/pi^2.

    H = - sum_sigma sum_{i != i'} J(|R_i - R_i'|) b_i^dag b_i'
        + Omega_R sum_{ij} S(R_i - R'_j) (b_i^dag c_j + h.c.)
        + (U/2) sum_i n_i (n_i - 1) + sum_i (eps_i - mu) n_i,

eps_i = +delta/2 on layer 1 and -delta/2 on layer 2 (plus an optional site
potential, e.g. a trap). Omega_R is the off-diagonal matrix element, as in the
continuum engine.

Geometry is the continuum model's frame exactly (`bilayer.lattice`): lattice 1
(potential U(R(+alpha) r)) has its sites at R(-alpha)(i, j), lattice 2 at
R(+alpha)(i, j), tan(alpha) = n/m, AA at the origin, box [-L/2, L/2)^2 with
L = sqrt(m^2 + n^2). In integer coordinates X = L x, Y = L y every site is a
lattice point, (X, Y) = M_sigma (i, j) with M_1 = [[m, n], [-n, m]],
M_2 = [[m, -n], [n, m]], and the box translations are X -> X + c, Y -> Y + c
(c = m^2 + n^2): each layer has exactly c sites in the box, and all
bookkeeping (wrapping, neighbour lookup, minimum image) is exact integer
arithmetic.

Matrix elements (from `bilayer.bands`, exact lowest band of V0 sin^2(pi x)):
  * J(d) = hopping_1d(V0) along each layer's own axes only (the Wannier
    function is a product w(x')w(y'), so there is no diagonal hopping);
    J1 and J2 by default (d_max = 2), longer range on request.
  * S(Delta) = s(Delta_x') s(Delta_y'), Delta in lattice 1's axes, s = overlap_1d,
    set to 0 for |Delta_x'| or |Delta_y'| >= cutoff. cutoff = 1 leaves a cusp
    at AA in the k = 0 coupling that pins the moire pockets to the primary
    lattice; cutoff = 2 removes it and is recommended (`build_interlayer`).
    The separable form ignores the relative rotation of the two Wannier
    functions; frame="mid" (factorise along the bisector) is closer to the
    exact 2D overlap and mirror symmetric between the layers.
  * U = onsite_U(V0, g2d), the same in both layers. Optional
    (interlayer_interaction=True): the interlayer density-density term
    V_ij = g2d I2(Delta_x') I2(Delta_y'), I2(d) = int w^2(x) w^2(x - d) dx,
    V = U at AA -- the a12 = a11 contact interaction of the two spin states
    (see `build_model`).

Relative to the continuum model `bilayer.physics.BilayerGPE` (foundations note
S6.2) the lattice model keeps the lowest band of each layer (exact J1, J2), the
interlayer coupling projected on the two lowest bands, the on-site interaction,
the geometry, twists (Peierls phases on every bond) and -- beyond the GPE --
number quantisation (`gutzwiller`, the Mott corner). It drops higher bands
(valid for V0 >~ 8 E_R; at 4 E_R J2/J1 = -7 %: use the continuum); J(d >= 3)
by default (so the U -> 0 superfluid fraction is (J1 + 4 J2)/kappa, not the
exact band curvature; see `tb_superfluid_fraction`); the part of Omega_R that
couples a Wannier function to the other layer's higher bands (the second-order
shift ~ -Omega_R^2 / gap, largest at AB, is missing, so the moire potential
comes out too deep, more so at large Omega_R); inter-site interactions (by
default also the interlayer V_ij above: without it a layer-symmetric state has
half the continuum's mean-field interaction and no density dip at AA); and the
density between sites (no Leggett density suppression of f_s). Validation
numbers: docs/2-methods.md §12.2 and docs/3-results.md §10.

Mean-field ground state (`ground_state`): the discrete two-layer GPE (a product
of coherent states) minimised at fixed N = filling x c (filling = atoms per a^2,
both layers together, as in the continuum) with torch LBFGS (complex128,
autograd), Newton-polished to residual ~1e-15 and checked to be a minimum
(lowest constrained Hessian eigenvalue >= 0; saddles in the inter-pocket
phases are escaped).
Superfluid fraction (`superfluid_fraction`) by phase twist with the
continuum's definitions (`bilayer.observables`), by energy and by current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from .bands import bands_1d, hopping_1d, onsite_U, overlap_1d, wannier_1d
from .lattice import MoireGeometry, moire_geometry
from .observables import superfluid_fraction_energy
from .units import KAPPA

G2D_DEFAULT = 0.01576  # E_R a^2: g_3D/(sqrt(2 pi) l_z) at omega_z = 2 pi x 1 kHz (1-theory §2.3)


# -- sites ------------------------------------------------------------------------------


def _layer_matrix(m: int, n: int, layer: int) -> np.ndarray:
    """(X, Y) = M (i, j): integer box coordinates (L x position) of the site
    with integer lattice coordinates (i, j) of layer 0 (lattice 1, rotated by
    -alpha) or layer 1 (lattice 2, rotated by +alpha). Columns / L are the
    layer's axis unit vectors."""
    if layer == 0:
        return np.array([[m, n], [-n, m]], dtype=np.int64)
    return np.array([[m, -n], [n, m]], dtype=np.int64)


@dataclass(frozen=True)
class LayerSites:
    layer: int  # 0 = lattice 1 (sites R(-alpha) (i, j)), 1 = lattice 2
    ij: np.ndarray  # (c, 2) int64, coordinates along the layer's own axes
    XY: np.ndarray  # (c, 2) int64, L * position, wrapped into [-c/2, c/2)
    positions: np.ndarray  # (c, 2) float64, box frame, in [-L/2, L/2)^2
    axes: np.ndarray  # (c, 2, 2) float64: axes[s, mu] = unit vector of axis mu
    M: np.ndarray  # (2, 2) int64, XY = M ij
    lookup: np.ndarray  # (c, c) int64: site index of XY = (X mod c, Y mod c)
    c: int
    L: float

    def index(self, XY: np.ndarray) -> np.ndarray:
        """Site index of (any, unwrapped) integer box coordinates (..., 2);
        -1 if the point is not a site of this layer."""
        XY = np.asarray(XY, dtype=np.int64)
        return self.lookup[np.mod(XY[..., 0], self.c), np.mod(XY[..., 1], self.c)]


@dataclass(frozen=True)
class Supercell:
    geometry: MoireGeometry
    layers: tuple  # (LayerSites, LayerSites)
    c: int  # sites per layer = m^2 + n^2
    L: float

    @property
    def positions(self) -> np.ndarray:
        """(2, c, 2): site positions of both layers, box frame."""
        return np.stack([ly.positions for ly in self.layers])

    def origin_index(self, layer: int) -> int:
        return int(self.layers[layer].lookup[0, 0])


def supercell_sites(m: int = 22, n: int = 1) -> Supercell:
    """All sites of both layers inside the box [-L/2, L/2)^2, L = sqrt(m^2+n^2),
    in the continuum model's frame (lattice 1 at R(-alpha)(i, j), lattice 2 at
    R(+alpha)(i, j), AA at the origin). Each layer has exactly c = m^2 + n^2
    sites (the box is a superlattice of index c of each layer; for (m + n) odd
    it is also one primitive moire supercell). Verifies the count, that the
    origin is a site of both layers, and that the set is closed under the box
    translations (every site moved by +-L e_x, +-L e_y and every unit lattice
    step lands on a site: each is a permutation of the c sites)."""
    geo = moire_geometry(m, n)
    c = m * m + n * n
    L = math.sqrt(c)
    K = int(math.ceil(L / math.sqrt(2.0))) + 2
    g = np.arange(-K, K + 1)
    I, Jg = np.meshgrid(g, g, indexing="ij")
    ij_all = np.stack([I.ravel(), Jg.ravel()], axis=1)
    layers = []
    for layer in (0, 1):
        M = _layer_matrix(m, n, layer)
        XY = ij_all @ M.T
        inside = np.all((2 * XY >= -c) & (2 * XY < c), axis=1)
        ij = ij_all[inside]
        XY = XY[inside]
        order = np.lexsort((XY[:, 0], XY[:, 1]))  # rows of increasing y, then x
        ij, XY = ij[order], XY[order]
        if ij.shape[0] != c:
            raise RuntimeError(f"layer {layer}: {ij.shape[0]} sites in the box, expected {c}")
        lookup = -np.ones((c, c), dtype=np.int64)
        kx, ky = np.mod(XY[:, 0], c), np.mod(XY[:, 1], c)
        if np.any(lookup[kx, ky] >= 0):
            raise RuntimeError("two sites map to the same periodic image")
        lookup[kx, ky] = np.arange(c)
        if len(np.unique(kx * c + ky)) != c:
            raise RuntimeError("periodic images not unique")
        axes_vec = (M.T / L).astype(np.float64)  # axes_vec[mu] = column mu of M / L
        ly = LayerSites(
            layer=layer,
            ij=ij,
            XY=XY,
            positions=XY / L,
            axes=np.broadcast_to(axes_vec, (c, 2, 2)).copy(),
            M=M,
            lookup=lookup,
            c=c,
            L=L,
        )
        # closure: box translations (exact integer vectors of the layer) and unit steps
        for shift in ([c, 0], [-c, 0], [0, c], [0, -c]):
            v = np.linalg.solve(M.astype(float), shift)
            if not np.all(np.abs(v - np.rint(v)) < 1e-9):
                raise RuntimeError(f"box translation {shift} is not a lattice vector")
            idx = ly.index(XY + np.array(shift))
            if not np.array_equal(idx, np.arange(c)):
                raise RuntimeError("site set not closed under the box translation")
        for step in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            idx = ly.index((ij + np.array(step)) @ M.T)
            if np.any(idx < 0) or len(np.unique(idx)) != c:
                raise RuntimeError("unit lattice step is not a permutation of the sites")
        if lookup[0, 0] < 0:
            raise RuntimeError("the origin (AA) is not a site")
        layers.append(ly)
    return Supercell(geometry=geo, layers=tuple(layers), c=c, L=L)


# -- bonds --------------------------------------------------------------------------------


@dataclass
class Bonds:
    """Off-diagonal matrix elements <row|H|col> = amp, both directions listed,
    with the bond vector disp = R_col - R_row (box frame, the physical
    displacement = minimum image for every bond shorter than L/2)."""

    rows: np.ndarray
    cols: np.ndarray
    amp: np.ndarray  # float64 at zero twist
    disp: np.ndarray  # (n_bonds, 2) float64

    def __len__(self) -> int:
        return self.rows.size

    def amplitudes(self, theta=(0.0, 0.0), L: float = 1.0) -> np.ndarray:
        """amp * exp(i q . disp), q = theta / L: the Peierls phases of the
        twist psi = e^{i q.r} phi (the engine's convention, `BilayerGPE(twist=...)`)."""
        th = np.asarray(theta, dtype=np.float64)
        if not np.any(th):
            return self.amp.astype(np.float64)
        return self.amp * np.exp(1j * (self.disp @ th) / L)

    @staticmethod
    def concat(parts) -> Bonds:
        return Bonds(
            rows=np.concatenate([p.rows for p in parts]),
            cols=np.concatenate([p.cols for p in parts]),
            amp=np.concatenate([p.amp for p in parts]),
            disp=np.concatenate([p.disp for p in parts]),
        )

    def matrix(self, shape, theta=(0.0, 0.0), L: float = 1.0) -> sp.csr_matrix:
        return sp.coo_matrix((self.amplitudes(theta, L), (self.rows, self.cols)), shape=shape).tocsr()


def _hopping_bonds(sites: LayerSites, J_list, offset: int = 0) -> Bonds:
    """-J_d to the sites at +-d along each of the layer's two axes, d = 1..len(J_list)."""
    rows, cols, amp, disp = [], [], [], []
    src = np.arange(sites.c)
    for d, Jd in enumerate(np.atleast_1d(np.asarray(J_list, dtype=np.float64)), start=1):
        for step in ([d, 0], [-d, 0], [0, d], [0, -d]):
            dXY = sites.M @ np.array(step)
            dst = sites.index(sites.XY + dXY)
            if np.any(dst < 0):
                raise RuntimeError("hopping target is not a site")
            rows.append(src + offset)
            cols.append(dst + offset)
            amp.append(np.full(sites.c, -Jd))
            disp.append(np.broadcast_to(dXY / sites.L, (sites.c, 2)))
    return Bonds(
        rows=np.concatenate(rows),
        cols=np.concatenate(cols),
        amp=np.concatenate(amp),
        disp=np.concatenate(disp).astype(np.float64),
    )


def build_hopping(sites: LayerSites, J_list, geometry: MoireGeometry | None = None, theta=(0.0, 0.0)) -> sp.csr_matrix:
    """Intralayer hopping matrix (c x c, scipy CSR) of one layer: -J_d between
    sites d = 1, 2, ... apart along that layer's own axes (no diagonal hopping:
    separable Wannier functions), periodic on the supercell. With a twist theta
    (rad across the box) every element carries the Peierls phase
    exp(i theta . (R_j - R_i) / L) of its bond vector (complex matrix)."""
    if geometry is not None and abs(geometry.L_box - sites.L) > 1e-12:
        raise ValueError("geometry does not match the sites")
    b = _hopping_bonds(sites, J_list)
    return b.matrix((sites.c, sites.c), theta, sites.L)


def _s_exact(V0: float, d: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """overlap_1d on an array, in chunks (it builds a len(d) x n_q n_pw table)."""
    d = np.asarray(d, dtype=np.float64).ravel()
    out = np.empty_like(d)
    for k in range(0, d.size, chunk):
        out[k : k + chunk] = overlap_1d(V0, d[k : k + chunk])
    return out


def _frame_axes(sites1: LayerSites, frame: str) -> np.ndarray:
    """Rows = the axes along which S is factorised: lattice 1's ("layer1")
    or the bisector of the two lattices ("mid" = the box axes; mirror
    symmetric between the layers)."""
    if frame == "layer1":
        return sites1.axes[0]
    if frame == "mid":
        return np.eye(2)
    raise ValueError(f"frame must be 'layer1' or 'mid', got {frame!r}")


def _interlayer_pairs(sites1: LayerSites, sites2: LayerSites, window: float, frame: str = "layer1"):
    """All (lattice-1 site, lattice-2 site) pairs with |Delta_x'|, |Delta_y'| <
    window, Delta = R'_j - R_i (box frame, unwrapped neighbour = minimum image)
    and Delta' its components along the `frame` axes. Returns (i, j, Delta,
    Delta')."""
    c, L = sites1.c, sites1.L
    M2 = sites2.M.astype(np.float64)
    ij2f = sites1.XY @ np.linalg.inv(M2).T  # lattice-2 coordinates of R_i
    w = int(math.ceil(window)) + 1
    o = np.arange(-w, w + 1)
    O = np.stack(np.meshgrid(o, o, indexing="ij"), axis=-1).reshape(-1, 2)
    cand = np.floor(ij2f).astype(np.int64)[:, None, :] + O[None, :, :]  # (c, n_o, 2)
    XY2 = cand @ sites2.M.T
    delta = (XY2 - sites1.XY[:, None, :]) / L  # (c, n_o, 2)
    dprime = delta @ _frame_axes(sites1, frame).T  # components along the frame axes
    keep = np.all(np.abs(dprime) < window, axis=-1)
    i_idx = np.broadcast_to(np.arange(c)[:, None], keep.shape)[keep]
    j_idx = sites2.index(XY2[keep])
    if np.any(j_idx < 0):
        raise RuntimeError("interlayer partner is not a site")
    return i_idx, j_idx, delta[keep], dprime[keep]


def build_interlayer(
    sites1: LayerSites,
    sites2: LayerSites,
    V0: float,
    geometry: MoireGeometry | None = None,
    cutoff: float = 1.0,
    frame: str = "layer1",
):
    """Interlayer overlap matrix S (c x c, scipy CSR, rows = lattice-1 sites,
    columns = lattice-2 sites), S_ij = s(Delta_x') s(Delta_y') with Delta the
    displacement R'_j - R_i rotated into lattice 1's axes and s = 0 for
    |Delta_x'| or |Delta_y'| >= cutoff (s(1) = 0 exactly, so cutoff = 1 keeps
    every pair with an overlap above |s(1.25)| = 0.016 at V0 = 8).
    The Hamiltonian element is Omega_R S_ij. Partners are enumerated in the
    infinite lattice and folded into the box, so for bonds shorter than L/2
    Delta is the minimum image, and in small boxes periodic images are summed
    (each image is its own bond with its own Peierls phase).

    The separable form ignores the relative rotation of the two Wannier
    functions: at V0 = 8 the lattice-1 frame errs by up to 1.5e-3 against the
    exact 2D overlap (breaking the layer-exchange mirror symmetry at that
    level); frame="mid" (the bisector = box axes) errs by 1.2e-4, symmetrically.

    Cutoff: the k = 0 coupling at a lattice-1 site at local stacking shift u is
    F_i = sum_j S_ij = prod_axes sum_n s(n + u_axis). The partners at d = 1 -+ u
    give s(1 - u) + s(1 + u) = O(u^2); cutoff = 1 keeps only the inner one and
    leaves a cusp F ~ 1 + |s'(1)| |u| at AA (s'(1) = -0.153 at V0 = 8) that pins
    the pockets to the primary lattice. cutoff = 2 removes it (F within 1e-3 of
    the full lattice sum).

    Returns (S, diagnostics, bonds) where bonds = (i, j, Delta (n, 2) box
    frame, S value) per bond (images separate) and diagnostics holds (per
    lattice-1 site i)
      S_eff_one = max_j S_ij (one-neighbour estimate of the dressed depth: the
          lower dressed level is -Omega_R S_eff), S_eff_hyb = sqrt(sum_j S_ij^2)
          (i hybridised with all its partners), F_k0 = sum_j S_ij (coupling of
          a k = 0 envelope: the band-bottom moire potential is -Omega_R F_k0),
          and their max (AA, ~1) and min (AB) over sites; depth_one = 1 - min
          S_eff_one (1 - s(a/2)^2 = 0.92 at V0 = 8), depth_k0 =
          1 - min F_k0 (the band-bottom value, 1 - (2 s(a/2))^2 = 0.67 at
          V0 = 8); lowest_band_weight_min = min_i sum_j S_ij^2 (the fraction of
          a Wannier function in the other layer's lowest band; the rest couples
          to higher bands the model drops); S_dropped_max = the largest |s s|
          beyond the cutoff (window cutoff + 1)."""
    if geometry is not None and abs(geometry.L_box - sites1.L) > 1e-12:
        raise ValueError("geometry does not match the sites")
    c = sites1.c
    i, j, delta, dp = _interlayer_pairs(sites1, sites2, cutoff, frame)
    s = _s_exact(V0, dp.ravel()).reshape(dp.shape)
    vals = s[:, 0] * s[:, 1]
    S = sp.coo_matrix((vals, (i, j)), shape=(c, c)).tocsr()
    S_one = np.zeros(c)
    np.maximum.at(S_one, i, vals)
    Srow = np.asarray(S.sum(axis=1)).ravel()
    S_hyb = np.sqrt(np.asarray(S.multiply(S).sum(axis=1)).ravel())
    diag = {
        "S_eff_one": S_one,
        "S_eff_hyb": S_hyb,
        "F_k0": Srow,
        "S_eff_AA": float(S_one.max()),
        "S_eff_AB": float(S_one.min()),
        "S_eff_hyb_AB": float(S_hyb.min()),
        "F_k0_AA": float(Srow.max()),
        "F_k0_AB": float(Srow.min()),
        "depth_one": float(1.0 - S_one.min()),
        "depth_hyb": float(1.0 - S_hyb.min()),
        "depth_k0": float(1.0 - Srow.min()),
        "lowest_band_weight_min": float((S_hyb**2).min()),
        "AB_site": int(np.argmin(S_one)),
        "s_half_sq": float(overlap_1d(V0, 0.5) ** 2),
    }
    _, _, _, dp2 = _interlayer_pairs(sites1, sites2, cutoff + 1.0, frame)
    outer = np.any(np.abs(dp2) >= cutoff, axis=1)
    s2 = _s_exact(V0, dp2[outer].ravel()).reshape(-1, 2)
    diag["S_dropped_max"] = float(np.abs(s2[:, 0] * s2[:, 1]).max()) if s2.size else 0.0
    return S, diag, (i, j, delta, vals)


# -- the model ----------------------------------------------------------------------------


def onsite_energy(V0: float, n_q: int = 64) -> float:
    """On-site energy <w|H|w> of the 2D separable Wannier function, 2 x the
    BZ average of the exact 1D lowest band: the zero of the tight-binding
    energies (add it to compare absolute energies with the continuum)."""
    _, E = bands_1d(V0, n_q=n_q, n_bands=1)
    return 2.0 * float(E[:, 0].mean())


def density_overlap_1d(V0: float, d, n_x: int = 64, n_q: int = 64) -> np.ndarray:
    """I2(d) = int w(x)^2 w(x - d)^2 dx (units 1/a) of the 1D Wannier function,
    by the Fourier series of w^2 on the Born-von Karman ring (spectrally exact
    for any real d). I2(0) = I4_1d; the interlayer density-density element is
    V_ij = g2d I2(Delta_x') I2(Delta_y') (V = U at AA)."""
    x, w = wannier_1d(V0, n_x=n_x, n_q=n_q)
    dx = x[1] - x[0]
    F = np.fft.rfft(w * w) * dx
    k = 2.0 * math.pi * np.fft.rfftfreq(x.size, d=dx)
    wts = np.abs(F) ** 2
    wts[1:] *= 2.0
    if x.size % 2 == 0:
        wts[-1] /= 2.0
    d_arr = np.asarray(d, dtype=np.float64)
    out = (wts[None, :] * np.cos(np.outer(d_arr.ravel(), k))).sum(axis=1) / (x.size * dx)
    return out.reshape(d_arr.shape)


@dataclass
class LatticeModel:
    cell: Supercell
    V0: float
    Omega_R: float
    delta: float
    J: np.ndarray  # J(d), d = 1..len(J)
    U: float
    S: sp.csr_matrix  # (c, c) lattice 1 x lattice 2
    bonds: Bonds  # every off-diagonal element of H0, global indices (layer * c + site)
    onsite: np.ndarray  # (2c,) diagonal of H0
    V_pairs: tuple | None = None  # (i, j, V_ij): interlayer density-density, or None
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_sites(self) -> int:
        return 2 * self.cell.c

    @property
    def L(self) -> float:
        return self.cell.L

    @property
    def coupled(self) -> bool:
        """True if any interlayer bond is present (then only the total phase is a symmetry)."""
        return bool(np.any((self.bonds.rows < self.cell.c) != (self.bonds.cols < self.cell.c)))

    def hamiltonian(self, theta=(0.0, 0.0)) -> sp.csr_matrix:
        """Single-particle H0 (2c x 2c CSR), layer 1 first; real at zero twist."""
        n = self.n_sites
        return self.bonds.matrix((n, n), theta, self.L) + sp.diags(self.onsite)

    def torch_operator(self, theta=(0.0, 0.0)):
        """phi (..., 2c) complex128 -> H0 phi, differentiable (index_add)."""
        rows = torch.as_tensor(self.bonds.rows)
        cols = torch.as_tensor(self.bonds.cols)
        amp = torch.as_tensor(self.bonds.amplitudes(theta, self.L), dtype=torch.complex128)
        diag = torch.as_tensor(self.onsite, dtype=torch.float64)

        def apply(phi: torch.Tensor) -> torch.Tensor:
            out = diag * phi
            return out.index_add(-1, rows, amp * phi[..., cols])

        return apply

    def hartree_operator(self):
        """dens (2c,) real -> (2c,) interlayer Hartree shift sum_j V_ij n_j
        (zero if the model has no interlayer interaction)."""
        if self.V_pairs is None:
            return lambda dens: torch.zeros_like(dens)
        c = self.cell.c
        i = torch.as_tensor(self.V_pairs[0])
        j = torch.as_tensor(self.V_pairs[1]) + c
        v = torch.as_tensor(self.V_pairs[2], dtype=torch.float64)

        def apply(dens: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(dens)
            out = out.index_add(0, i, v * dens[j])
            return out.index_add(0, j, v * dens[i])

        return apply


def build_model(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    delta: float = 0.0,
    g2d: float = G2D_DEFAULT,
    J_list=None,
    d_max: int = 2,
    U: float | None = None,
    cutoff: float = 1.0,
    interlayer_interaction: bool = False,
    frame: str = "layer1",
    cell: Supercell | None = None,
) -> LatticeModel:
    """Assemble Model C. J_list overrides hopping_1d(V0, d_max) (J1, J2 by
    default); U overrides onsite_U(V0, g2d); frame picks the axes S is
    factorised along (`build_interlayer`).

    interlayer_interaction=True adds sum_ij V_ij n_i^(1) n_j^(2) with
    V_ij = g2d I2(Delta_x') I2(Delta_y') on the same pairs as S. The two
    "layers" are two spin states in the same pancake with a12 = a11 = a22 (the
    continuum interaction is g (|psi1|^2 + |psi2|^2)^2), so without it a
    co-located AA pair does not interact and the mean-field interaction energy
    of a layer-symmetric state is half the continuum's."""
    cell = supercell_sites(m, n) if cell is None else cell
    c = cell.c
    J = np.asarray(hopping_1d(V0, d_max) if J_list is None else J_list, dtype=np.float64)
    U = onsite_U(V0, g2d) if U is None else float(U)
    parts = [_hopping_bonds(cell.layers[0], J, 0), _hopping_bonds(cell.layers[1], J, c)]
    S, diag, (i, j, dlt, svals) = build_interlayer(cell.layers[0], cell.layers[1], V0, cell.geometry, cutoff, frame)
    if Omega_R != 0.0:
        vals = Omega_R * svals
        parts.append(Bonds(rows=i, cols=j + c, amp=vals, disp=dlt))
        parts.append(Bonds(rows=j + c, cols=i, amp=vals, disp=-dlt))
    onsite = np.concatenate([np.full(c, 0.5 * delta), np.full(c, -0.5 * delta)])
    V_pairs = None
    if interlayer_interaction:
        dp = dlt @ _frame_axes(cell.layers[0], frame).T
        I2 = density_overlap_1d(V0, dp.ravel()).reshape(dp.shape)
        V_pairs = (i, j, g2d * I2[:, 0] * I2[:, 1])
    return LatticeModel(
        cell=cell,
        V0=V0,
        Omega_R=float(Omega_R),
        delta=float(delta),
        J=J,
        U=U,
        S=S,
        bonds=Bonds.concat(parts),
        onsite=onsite,
        V_pairs=V_pairs,
        diagnostics=diag,
    )


# -- single particle ----------------------------------------------------------------------


@dataclass
class LatticeBands:
    energies: np.ndarray  # (n_bands,), relative to the on-site energy
    vectors: np.ndarray  # (n_bands, 2, c)
    weights: np.ndarray  # (n_bands, 2) layer weights
    offset: float  # on-site energy 2 eps_1D (add for absolute energies)
    band_bottom: float  # TB band bottom of one decoupled layer, -4 sum_d J_d


def lattice_bands(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    n_bands: int = 8,
    delta: float = 0.0,
    theta=(0.0, 0.0),
    model: LatticeModel | None = None,
    **model_kw,
) -> LatticeBands:
    """Lowest n_bands eigenpairs of the two-layer tight-binding Hamiltonian on
    the supercell at twist theta (default k = 0), dense (2c = 970 at m = 22)."""
    model = build_model(V0, Omega_R, m, n, delta, **model_kw) if model is None else model
    E, V = np.linalg.eigh(model.hamiltonian(theta).toarray())
    c = model.cell.c
    vec = V[:, :n_bands].T.reshape(n_bands, 2, c)
    return LatticeBands(
        energies=E[:n_bands],
        vectors=vec,
        weights=(np.abs(vec) ** 2).sum(axis=-1),
        offset=onsite_energy(V0),
        band_bottom=float(-4.0 * model.J.sum()),
    )


def compare_with_continuum(
    V0: float,
    Omega_R: float,
    m: int = 4,
    n: int = 1,
    n_bands: int = 8,
    points_per_a: float = 8.0,
    delta: float = 0.0,
    **model_kw,
) -> dict:
    """Lowest k = 0 levels of Model C against the continuum `moire_bands_q0`
    on the same supercell. The TB zero is the on-site energy, so the
    comparison is of (i) the level spacings E_k - E_0, (ii) the shift of the
    lowest level from the decoupled band bottom (TB: E_0 + 4 sum J_d;
    continuum: E_0 - 2 E_1D(q = 0), the exact 1D band bottom), and (iii)
    absolute levels after adding the on-site energy 2 eps_1D."""
    from .bands import moire_bands_q0

    tb = lattice_bands(V0, Omega_R, m, n, n_bands, delta, **model_kw)
    cont = moire_bands_q0(V0, Omega_R, m, n, points_per_a=points_per_a, n_bands=n_bands, delta=delta)
    q, E1 = bands_1d(V0, n_q=64, n_bands=1)
    bottom_cont = 2.0 * float(E1[np.argmin(np.abs(q)), 0])
    sp_tb = tb.energies - tb.energies[0]
    sp_c = cont.energies - cont.energies[0]
    return {
        "E_tb": tb.energies,
        "E_tb_abs": tb.energies + tb.offset,
        "E_cont": cont.energies,
        "spacing_tb": sp_tb,
        "spacing_cont": sp_c,
        "spacing_maxdiff": float(np.abs(sp_tb - sp_c).max()),
        "shift_tb": float(tb.energies[0] - tb.band_bottom),
        "shift_cont": float(cont.energies[0] - bottom_cont),
        "abs_maxdiff": float(np.abs(tb.energies + tb.offset - cont.energies).max()),
        "weights_tb": tb.weights,
        "weights_cont": cont.weights,
        "cont_residual": float(cont.residuals.max()),
    }


def tb_superfluid_fraction(J_list, theta: float, L: float, alpha: float = 0.0, kappa: float = KAPPA) -> tuple:
    """Exact non-interacting f_s^xx of one square layer with hoppings J(d)
    (axes rotated by alpha from x) at finite twist: band
    E(q) = -2 sum_d J_d [cos(d q cos a) + cos(d q sin a)], q = theta / L;
    returns (energy route [E(q) - E(0)] / (kappa q^2), current route
    E'(q) / (2 kappa q)). Both -> sum_d d^2 J_d / kappa as q -> 0:
    (J1 + 4 J2)/kappa = 0.2789 at V0 = 8 with J1, J2; 0.2808 with J3 as well;
    the exact band curvature (`bands.effective_mass_ratio`) is 0.2806."""
    q = theta / L
    ca, sa = math.cos(alpha), math.sin(alpha)
    J = np.asarray(J_list, dtype=np.float64)
    d = np.arange(1, J.size + 1)

    def E(k):
        return float(-2.0 * np.sum(J * (np.cos(d * k * ca) + np.cos(d * k * sa))))

    dE = float(2.0 * np.sum(J * d * (ca * np.sin(d * q * ca) + sa * np.sin(d * q * sa))))
    return (E(q) - E(0.0)) / (kappa * q * q), dE / (2.0 * kappa * q)


# -- discrete GPE (coherent-state mean field) ----------------------------------------------


@dataclass
class LatticeGroundState:
    phi: torch.Tensor  # (2, c) complex128, sum |phi|^2 = N
    density: np.ndarray  # (2, c) site occupations |phi_i|^2
    N: float
    E_per_N: float  # relative to the on-site energy
    mu: float
    residual: float  # ||H_mf phi - mu phi|| / ||phi||
    participation_ratio: float  # (sum n)^2 / (2c sum n^2) over the sites of both layers
    AA_AB_ratio: float
    layer_fractions: np.ndarray
    theta: tuple
    iterations: int  # energy evaluations (LBFGS) + Newton steps
    hessian_min: float = float("nan")  # lowest constrained Hessian eigenvalue (>= 0: local minimum)
    model: LatticeModel = field(repr=False, default=None)


class _Functional:
    """E[phi] = phi^dag H0 phi + (U/2) sum |phi|^4 + sum_ij V_ij |phi_i|^2 |phi_j|^2
    and H_mf phi = dE/dphi^*, torch complex128."""

    def __init__(self, model: LatticeModel, theta):
        self.model = model
        self.theta = theta
        self.H = model.torch_operator(theta)
        self.hartree = model.hartree_operator()
        self.U = model.U

    def energy(self, phi):
        dens = phi.real**2 + phi.imag**2
        E = (phi.conj() * self.H(phi)).sum().real + 0.5 * self.U * (dens * dens).sum()
        if self.model.V_pairs is not None:
            E = E + 0.5 * (dens * self.hartree(dens)).sum()
        return E

    def mf_apply(self, phi):
        dens = phi.real**2 + phi.imag**2
        return self.H(phi) + (self.U * dens + self.hartree(dens)) * phi

    def residual(self, phi):
        with torch.no_grad():
            g = self.mf_apply(phi)
            nrm2 = (phi.abs() ** 2).sum()
            mu = (phi.conj() * g).sum().real / nrm2
            return float((g - mu * phi).norm() / nrm2.sqrt()), float(mu)


def _initial_phi(model: LatticeModel, theta, N: float) -> torch.Tensor:
    """Lowest single-particle state scaled to N; for uncoupled layers the
    layer-balanced sum of each layer's lowest state (a lone-layer eigenvector
    is a stationary saddle of the interacting functional)."""
    H = model.hamiltonian(theta)
    c = model.cell.c

    def lowest(A):
        if A.shape[0] <= 64:
            return np.linalg.eigh(A.toarray())[1][:, 0]
        from scipy.sparse.linalg import eigsh

        v0 = np.ones(A.shape[0], dtype=A.dtype)
        return eigsh(A, k=1, which="SA", v0=v0, tol=1e-12)[1][:, 0]

    if model.coupled:
        v = lowest(H)
    else:
        v = np.concatenate([lowest(H[s, s]) for s in (slice(0, c), slice(c, 2 * c))])
    v = v * math.sqrt(N) / np.linalg.norm(v)
    return torch.as_tensor(v.astype(np.complex128))


def _linearisation(F: _Functional, phi: torch.Tensor):
    """Real-form Jacobian of F(phi) = H_mf phi - mu phi at fixed mu, i.e. half
    the Hessian of E - mu |phi|^2 in (a, b) = (Re phi, Im phi):
    Jm = [[Re A + Re B, -Im A + Im B], [Im A + Im B, Re A - Re B]],
    A = H0 + diag(2U|phi|^2 + hartree) - mu (+ V_ij phi_i phi_j^*),
    B = U diag(phi^2) (+ V_ij phi_i phi_j). Also returns F, the norm direction
    v1 = (a, b) and the gauge direction(s) i phi (one per independent U(1):
    two for uncoupled layers)."""
    model = F.model
    c = model.cell.c
    dens = phi.abs() ** 2
    hart = F.hartree(dens)
    _, mu = F.residual(phi)
    Fv = F.mf_apply(phi) - mu * phi
    A = torch.as_tensor(model.hamiltonian(F.theta).toarray(), dtype=torch.complex128)
    A = A + torch.diag((2.0 * F.U * dens + hart - mu).to(torch.complex128))
    B = torch.diag(F.U * phi * phi)
    if model.V_pairs is not None:
        i = torch.as_tensor(model.V_pairs[0])
        j = torch.as_tensor(model.V_pairs[1]) + c
        v = torch.as_tensor(model.V_pairs[2], dtype=torch.float64)
        A[i, j] += v * phi[i] * phi[j].conj()
        A[j, i] += v * phi[j] * phi[i].conj()
        B[i, j] += v * phi[i] * phi[j]
        B[j, i] += v * phi[j] * phi[i]
    Jm = torch.cat(
        [torch.cat([A.real + B.real, -A.imag + B.imag], 1), torch.cat([A.imag + B.imag, A.real - B.real], 1)], 0
    )
    v1 = torch.cat([phi.real, phi.imag])
    if model.coupled:
        gauge = [torch.cat([-phi.imag, phi.real])]
    else:
        m1 = torch.cat([torch.ones(c), torch.zeros(c)]).to(torch.float64)
        gauge = [torch.cat([-phi.imag * mk, phi.real * mk]) for mk in (m1, 1.0 - m1)]
    return Jm, Fv, v1, gauge, dens


def _newton_direction(F: _Functional, phi: torch.Tensor, N: float) -> torch.Tensor:
    """Newton direction for F(phi, mu) = 0, |phi|^2 = N: the Jacobian of
    `_linearisation` bordered by the norm and gauge directions."""
    n = F.model.n_sites
    Jm, Fv, v1, gauge, dens = _linearisation(F, phi)
    border = torch.stack([v1] + gauge, 1)  # (2n, 1 + n_g)
    k = border.shape[1]
    K = torch.zeros(2 * n + k, 2 * n + k, dtype=torch.float64)
    K[: 2 * n, : 2 * n] = Jm
    K[: 2 * n, 2 * n :] = border
    K[2 * n :, : 2 * n] = border.T
    rhs = torch.zeros(2 * n + k, dtype=torch.float64)
    rhs[: 2 * n] = -torch.cat([Fv.real, Fv.imag])
    rhs[2 * n] = -0.5 * (float(dens.sum()) - N)
    try:
        x = torch.linalg.solve(K, rhs)
        if not torch.isfinite(x).all():
            raise RuntimeError
    except Exception:
        x = torch.linalg.lstsq(K, rhs[:, None], driver="gelsd").solution[:, 0]
    return torch.complex(x[:n], x[n : 2 * n])


def _hessian_lowest(F: _Functional, phi: torch.Tensor):
    """Lowest eigenvalue of the constrained Hessian (Jm projected on the
    tangent space of |phi|^2 = N, gauge directions removed) and its direction
    as a complex vector. Negative = the stationary point is a saddle (e.g.
    the wrong relative phase between weakly linked pockets)."""
    n = F.model.n_sites
    Jm, _, v1, gauge, _ = _linearisation(F, phi)
    Q, _ = torch.linalg.qr(torch.stack([v1] + gauge, 1))
    P = torch.eye(2 * n, dtype=torch.float64) - Q @ Q.T
    lam, V = torch.linalg.eigh(P @ Jm @ P)
    big = float(Q.shape[1])  # skip the projected-out (zero) directions
    order = torch.argsort(lam)
    for k in order[: int(big) + 2]:
        x = V[:, k]
        if float((Q.T @ x).norm()) < 0.5:
            return float(lam[k]), torch.complex(x[:n], x[n:])
    k = order[0]
    return float(lam[k]), torch.complex(V[:n, k], V[n:, k])


def _minimise_gpe(
    model: LatticeModel,
    N: float,
    theta,
    phi0: torch.Tensor | None,
    tol: float,
    max_rounds: int,
    lbfgs_iter: int,
    history: int,
    newton_max_sites: int = 6000,
    stability_max_sites: int = 2000,
    max_escapes: int = 6,
):
    """LBFGS (strong Wolfe, autograd) on z in C^{2c}, phi = sqrt(N) z / |z|,
    while it makes progress (it stalls near residual 1e-8, where the energy
    ~ N can no longer resolve the remaining decrease in float64), then dense
    Newton steps on the stationarity equations to the residual target. Newton
    converges to the nearest stationary point, and a symmetric start can sit
    on a saddle of the inter-pocket phases, so the result is checked with the
    lowest constrained Hessian eigenvalue and, if negative, pushed downhill
    along that mode and relaxed again. Returns (phi, E, mu, residual,
    iterations, hessian_min)."""
    F = _Functional(model, theta)
    if phi0 is None:
        phi0 = _initial_phi(model, theta, N)
    phi0 = phi0.reshape(-1).to(torch.complex128)
    sN = math.sqrt(N)
    newton_ok = model.n_sites <= newton_max_sites
    n_eval = [0]

    def relax(start):
        start = start * sN / start.norm()
        z = torch.view_as_real(start.clone()).contiguous().requires_grad_(True)

        def make_opt():
            return torch.optim.LBFGS(
                [z],
                lr=1.0,
                max_iter=lbfgs_iter,
                history_size=history,
                tolerance_grad=1e-300,
                tolerance_change=1e-300,
                line_search_fn="strong_wolfe",
            )

        def phi_of(zz):
            zc = torch.view_as_complex(zz)
            return zc * (sN / zc.norm())

        opt = make_opt()

        def closure():
            opt.zero_grad()
            E = F.energy(phi_of(z))
            E.backward()
            n_eval[0] += 1
            return E

        res, _ = F.residual(phi_of(z).detach())
        stall = 0
        for _ in range(max_rounds):
            if res < tol or (newton_ok and res < 1e-6 and stall >= 1):
                break
            opt.step(closure)
            new, _ = F.residual(phi_of(z).detach())
            stall = stall + 1 if new > 0.5 * res else 0
            res = new
            if stall >= 3:
                if newton_ok and res < 1e-4:
                    break
                opt = make_opt()  # LBFGS lost its curvature model: restart
                stall = 0
        phi = phi_of(z).detach()
        if newton_ok:
            # full Newton steps, keeping the best iterate: with soft inter-pocket
            # modes the residual is not monotone (2e-8 -> 3e-6 -> 2e-10 -> 3e-16),
            # and backtracking would stall; the Hessian check below catches a
            # step onto a different (saddle) stationary point
            cur, r_start = phi, res
            for _ in range(12):
                if res < tol:
                    break
                cur = cur + _newton_direction(F, cur, N)
                cur = cur * sN / cur.norm()
                n_eval[0] += 1
                r_cur, _ = F.residual(cur)
                if not math.isfinite(r_cur) or r_cur > 1e3 * max(r_start, 1e-6):
                    break
                if r_cur < res:
                    phi, res = cur, r_cur
        return phi

    phi = relax(phi0)
    hmin = float("nan")
    if model.n_sites <= stability_max_sites:
        E = float(F.energy(phi))
        for _ in range(max_escapes + 1):
            hmin, d = _hessian_lowest(F, phi)
            if hmin > -1e-11:
                break
            moved = None
            for step in (0.1, 0.3, 1.0):  # downhill along the unstable mode
                cand = relax(phi + step * sN * d)
                E_c = float(F.energy(cand))
                if E_c < E - 1e-13 * abs(E):
                    moved = cand
                    E = E_c
                    break
            if moved is None:
                break
            phi = moved
    res, mu = F.residual(phi)
    E = float(F.energy(phi))
    return phi, E, mu, res, n_eval[0], hmin


def moire_points(cell: Supercell):
    """AA and AB points of the moire pattern in the box: AA at (k, l) L_M,
    AB at (k + 1/2, l + 1/2) L_M, k, l = 0 .. 2n - 1, L_M = L / 2n (the local
    shift between the lattices, -2 sin(alpha) R(90 deg) r, is integer at AA
    and (1/2, 1/2) at AB). Returns (AA (K, 2), AB (K, 2)), wrapped into the box."""
    n = cell.geometry.n
    LM = cell.L / (2 * n)
    k = np.arange(2 * n)
    K = np.stack(np.meshgrid(k, k, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float64)

    def wrap(P):
        return P - cell.L * np.floor(P / cell.L + 0.5)

    return wrap(K * LM), wrap((K + 0.5) * LM)


def local_density(cell: Supercell, density: np.ndarray, point, radius: float = 1.0) -> float:
    """Total density (atoms per a^2) near a point: for each layer the mean
    site occupation over its sites within `radius` (minimum image), summed
    over the two layers."""
    total = 0.0
    for layer in (0, 1):
        d = cell.layers[layer].positions - np.asarray(point)[None, :]
        d -= cell.L * np.round(d / cell.L)
        sel = np.hypot(d[:, 0], d[:, 1]) < radius
        if not sel.any():
            raise ValueError("no site within radius")
        total += float(density[layer][sel].mean())
    return total


def aa_ab_ratio(cell: Supercell, density: np.ndarray, radius: float = 1.0) -> float:
    """Mean local density at the AA points over that at the AB points
    (`moire_points`, `local_density` with a 1 a disc)."""
    AA, AB = moire_points(cell)
    nAA = np.mean([local_density(cell, density, p, radius) for p in AA])
    nAB = np.mean([local_density(cell, density, p, radius) for p in AB])
    return float(nAA / nAB)


def ground_state(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    filling: float = 1.0,
    g2d: float = G2D_DEFAULT,
    delta: float = 0.0,
    theta=(0.0, 0.0),
    phi0: torch.Tensor | None = None,
    model: LatticeModel | None = None,
    tol: float = 1e-11,
    max_rounds: int = 400,
    lbfgs_iter: int = 50,
    history: int = 50,
    **model_kw,
) -> LatticeGroundState:
    """Mean-field (coherent-state) ground state of Model C: the discrete
    two-layer GPE

        E[phi] = phi^dag H0(theta) phi + (U/2) sum_i |phi_i|^4,   sum |phi_i|^2 = N,

    N = filling x c (filling = atoms per a^2 summed over both layers, as in
    the continuum; c = sites of one layer), minimised with torch LBFGS
    (complex128, autograd) and polished by Newton to residual
    ||H_mf phi - mu phi|| / ||phi|| < tol, H_mf = H0 + U |phi|^2,
    mu = phi^dag H_mf phi / N. Energies are relative to the on-site energy
    (`onsite_energy`). model_kw go to `build_model` (e.g. J_list, d_max,
    interlayer_interaction)."""
    if model is None:
        model = build_model(V0, Omega_R, m, n, delta, g2d=g2d, **model_kw)
    cell = model.cell
    N = float(filling * cell.c)
    theta = (float(theta[0]), float(theta[1]))
    phi, E, mu, res, its, hmin = _minimise_gpe(model, N, theta, phi0, tol, max_rounds, lbfgs_iter, history)
    dens = (phi.abs() ** 2).reshape(2, cell.c).numpy()
    pr = float(dens.sum() ** 2 / (dens.size * (dens**2).sum()))
    return LatticeGroundState(
        phi=phi.reshape(2, cell.c),
        density=dens,
        N=N,
        E_per_N=E / N,
        mu=mu,
        residual=res,
        participation_ratio=pr,
        AA_AB_ratio=aa_ab_ratio(cell, dens),
        layer_fractions=dens.sum(axis=1) / N,
        theta=theta,
        iterations=its,
        hessian_min=hmin,
        model=model,
    )


def lattice_current(gs: LatticeGroundState, axis: int = 0) -> float:
    """Number current through a cut perpendicular to `axis`, averaged over the
    cuts: I = (1/L) sum_bonds Delta_axis I_{i->j}, with the bond current
    I_{i->j} = -2 Im(phi_i^* h_ij(theta) phi_j) (continuity of the discrete
    GPE; h_ij(theta) includes the Peierls phase, so this is the physical
    current of psi = e^{i q.r} phi). Intra- and interlayer bonds both count.
    Equals dE/dtheta for a stationary state."""
    model = gs.model
    phi = gs.phi.reshape(-1).numpy()
    b = model.bonds
    h = b.amplitudes(gs.theta, model.L)
    Iij = -2.0 * np.imag(np.conj(phi[b.rows]) * h * phi[b.cols])
    return float(0.5 * np.sum(b.disp[:, axis] * Iij) / model.L)


def superfluid_fraction(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    filling: float = 1.0,
    g2d: float = G2D_DEFAULT,
    delta: float = 0.0,
    theta: float = 0.3,
    axis: int = 0,
    gs0: LatticeGroundState | None = None,
    tol: float = 1e-11,
    **model_kw,
) -> dict:
    """f_s along `axis` by phase twist (theta across the box, the continuum
    sweep's default 0.3; Peierls phases on every intra- and interlayer bond),
    with the continuum's definitions (`bilayer.observables`, foundations
    note S4.1-4.2):
        energy:  f_s = (E(theta) - E(0)) L^2 / (N kappa theta^2),
        current: f_s = L^2 I / (2 kappa N theta).
    Returns {"fs_energy", "fs_current", "E0", "E_theta", "I", "gs0", "gs_theta"}."""
    if gs0 is None:
        gs0 = ground_state(V0, Omega_R, m, n, filling, g2d, delta, tol=tol, **model_kw)
    tw = [0.0, 0.0]
    tw[axis] = theta
    gst = ground_state(V0, Omega_R, m, n, filling, g2d, delta, theta=tuple(tw), phi0=gs0.phi, model=gs0.model, tol=tol)
    L, N = gs0.model.L, gs0.N
    E0, Et = gs0.E_per_N * N, gst.E_per_N * N
    I = lattice_current(gst, axis)
    return {
        "fs_energy": float(superfluid_fraction_energy(Et, E0, N, L, theta, KAPPA)),
        "fs_current": float(L * L * I / (2.0 * KAPPA * N * theta)),
        "E0": E0,
        "E_theta": Et,
        "I": I,
        "gs0": gs0,
        "gs_theta": gst,
    }


# -- single-site Gutzwiller ------------------------------------------------------------------


@dataclass
class GutzwillerResult:
    n: np.ndarray  # (2, c) <n_i>
    phi: np.ndarray  # (2, c) <b_i>
    var_n: np.ndarray  # (2, c) number variance
    mu: float  # from the on-site energy
    filling: float  # sum <n_i> / c (atoms per a^2, both layers)
    condensate_fraction: float  # sum |phi|^2 / N
    compressibility: float  # d filling / d mu (1 / (E_R a^2)), central difference
    grand_potential: float  # per site of both layers
    converged: bool
    iterations: int


def _gutzwiller_solve(
    model: LatticeModel, mu: float, n_max: int, phi0: torch.Tensor, tol: float, max_iter: int, mixing: float
):
    """Self-consistent single-site mean field at fixed mu: each site sees
    H_i = Psi_i b^dag + Psi_i^* b + (U/2) n(n - 1) + (eps_i + h_i - mu) n with
    the field Psi_i = sum_j T_ij phi_j of all its intra- and interlayer bonds
    (T = off-diagonal part of H0) and the interlayer Hartree shift
    h_i = sum_j V_ij <n_j> (if the model has V); x <- x + mixing (G(x) - x)
    for x = phi (and <n> if V). Plain damped mixing on purpose: phi = 0 is
    always a fixed point, and root finders on G(x) - x (Anderson, Newton)
    happily converge to it or to other unstable fixed points; the damped map
    only converges to stable ones. Real arithmetic when phi0 is real (T is
    real at zero twist)."""
    c2 = model.n_sites
    real = not torch.is_complex(phi0) or float(phi0.imag.abs().max()) == 0.0
    dt = torch.float64 if real else torch.complex128
    rows = torch.as_tensor(model.bonds.rows)
    cols = torch.as_tensor(model.bonds.cols)
    amp = torch.as_tensor(model.bonds.amp, dtype=torch.float64).to(dt)

    def T(x):
        return torch.zeros_like(x).index_add(0, rows, amp * x[cols])

    hartree = model.hartree_operator()
    with_V = model.V_pairs is not None
    eps = torch.as_tensor(model.onsite, dtype=torch.float64)
    d = n_max + 1
    nn = torch.arange(d, dtype=torch.float64)
    sq = torch.sqrt(nn[1:]).to(dt)
    il = torch.arange(1, d)
    D0 = (0.5 * model.U * nn * (nn - 1))[None, :]

    def site_map(phi, n_avg):
        Psi = T(phi)
        D = D0 + (eps + hartree(n_avg) - mu)[:, None] * nn[None, :]
        H = torch.diag_embed(D.to(dt))
        H[:, il, il - 1] = Psi[:, None] * sq[None, :]
        H[:, il - 1, il] = Psi.conj()[:, None] * sq[None, :]
        e, V = torch.linalg.eigh(H)
        g = V[:, :, 0]
        p = g.abs() ** 2
        b = (g[:, :-1].conj() * sq[None, :] * g[:, 1:]).sum(dim=1)
        return b, (p * nn).sum(dim=1), e[:, 0], p

    phi = phi0.real.clone().to(dt) if real else phi0.to(torch.complex128).clone()
    n_avg = torch.zeros(c2, dtype=torch.float64)
    if with_V:  # start the Hartree field from phi0's densities
        n_avg = site_map(phi, n_avg)[1]
    converged = False
    it = 0
    ratios = []
    for it in range(1, max_iter + 1):  # noqa: B007 (it is reported after the loop)
        b, n_new, e0, p = site_map(phi, n_avg)
        change = float((b - phi).abs().max())
        dn = 0.0
        if with_V:
            dn = float((n_new - n_avg).abs().max())
            change = max(change, dn)
            n_avg = n_avg + mixing * (n_new - n_avg)
        if change < tol:
            converged = True
            break
        new = phi + mixing * (b - phi)
        # Mott detection: for |phi| < 1e-5 the map is linear in phi; a decay ratio
        # that has settled below 1 means every mode decays, i.e. the phi = 0
        # (Mott / empty) state is the stable fixed point -- jump there instead of
        # creeping at rate ~1 near a lobe boundary.
        a_old, a_new = float(phi.abs().max()), float(new.abs().max())
        if 0.0 < a_new < 1e-5 and a_old > 0.0 and dn < tol:
            ratios.append(a_new / a_old)
            if len(ratios) > 20 and ratios[-1] < 1.0 and abs(ratios[-1] - ratios[-21]) < 1e-6:
                phi = torch.zeros_like(phi)
                b, n_new, e0, p = site_map(phi, n_avg)
                converged = True
                break
        else:
            ratios = []
        phi = new
    phi = b
    var = (p * nn**2).sum(dim=1) - n_new**2
    omega = float(e0.sum() - (phi.conj() * T(phi)).sum().real - 0.5 * (n_new * hartree(n_new)).sum())
    return phi.to(torch.complex128), n_new, var, omega, converged, it


def _default_mixing(model: LatticeModel) -> float:
    """Damping that keeps the staggered phase mode of the order-parameter map
    stable: its multiplier is 1 - beta (1 + r), r = |lambda_max / lambda_min|
    of T (r > 1 with J2 < 0 or Omega_R > 0), so beta = 1.6 / (1 + r) leaves a
    20 % margin; the uniform mode converges at rate 1 - beta (1 - chi)."""
    H = model.hamiltonian()
    T = (H - sp.diags(H.diagonal())).toarray()
    lam = np.linalg.eigvalsh(T)
    if lam[0] >= 0.0:
        return 0.5
    r = abs(lam[-1] / lam[0])
    return float(min(1.0, 1.6 / (1.0 + r)))


def gutzwiller(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    mu_over_U: float | None = None,
    filling: float | None = None,
    g2d: float = G2D_DEFAULT,
    delta: float = 0.0,
    n_max: int = 6,
    tol: float = 1e-10,
    max_iter: int = 5000,
    mixing: float | None = None,
    dmu_over_U: float = 0.01,
    model: LatticeModel | None = None,
    phi0=None,
    **model_kw,
) -> GutzwillerResult:
    """Single-site Gutzwiller mean field of Model C (per-site Fock space
    0..n_max, self-consistent phi_i = <b_i>; the interlayer term at mean-field
    level like the hopping: Omega_R S_ij phi_j enters site i's field). Give
    exactly one of mu_over_U (grand canonical; mu measured from the on-site
    energy) or `filling` (atoms per a^2 over both layers; mu by a safeguarded
    regula falsi).
    Compressibility = d filling / d mu by central difference +- dmu_over_U U.
    On a plain square lattice (Omega_R = 0,
    J_list = (J,)) the n = 1 lobe tip is at U / zJ = 3 + 2 sqrt 2 = 5.83,
    mu / U = sqrt 2 - 1 (foundations note S6.2)."""
    if (mu_over_U is None) == (filling is None):
        raise ValueError("give exactly one of mu_over_U and filling")
    if model is None:
        model = build_model(V0, Omega_R, m, n, delta, g2d=g2d, **model_kw)
    c, U = model.cell.c, model.U
    if mixing is None:
        mixing = _default_mixing(model)
    if phi0 is None:
        # shape of the lowest single-particle state (layer-balanced if the layers
        # are uncoupled: phi = 0 on a whole layer is a fixed point that never grows)
        v = _initial_phi(model, (0.0, 0.0), 1.0).numpy()
        phi0 = torch.as_tensor(0.7 * v / np.abs(v).max())
    else:
        phi0 = torch.as_tensor(np.asarray(phi0).reshape(-1), dtype=torch.complex128)

    def solve(mu, start):
        return _gutzwiller_solve(model, mu, n_max, start, tol, max_iter, mixing)

    def warm(sol):
        return sol[0] if float(sol[0].abs().max()) > 1e-3 else phi0

    if mu_over_U is not None:
        mu = mu_over_U * U
        sol = solve(mu, phi0)
    else:
        target = filling * c
        lo = float(np.linalg.eigvalsh(model.hamiltonian().toarray())[0]) - 1e-3  # empty below
        hi = max(U, lo + 1.0)
        sol = solve(hi, phi0)
        while float(sol[1].sum()) < target:
            lo, hi = hi, 2.0 * hi + U
            sol = solve(hi, warm(sol))
        # Safeguarded regula falsi alternating with bisection. N(mu) has kinks at
        # the Mott-lobe edges, where solves are critically slow; interpolation
        # alone creeps towards them, bisection lands inside a plateau (where
        # N = target exactly) in a few steps. lo is never solved: N(lo) = 0.
        a, fa, b, fb = lo, -target, hi, float(sol[1].sum()) - target
        sol_b = sol
        mu = hi
        for k in range(200):
            if k % 2 == 0:
                mu = b - fb * (b - a) / (fb - fa)
                mu = min(max(mu, a + 0.05 * (b - a)), b - 0.05 * (b - a))
            else:
                mu = 0.5 * (a + b)
            sol = solve(mu, warm(sol_b))
            fm = float(sol[1].sum()) - target
            if abs(fm) <= 1e-9 * target or b - a < 1e-12 * U:
                break
            if fm < 0.0:
                a, fa = mu, fm
            else:
                b, fb, sol_b = mu, fm, sol
    phi, n_avg, var, omega, conv, its = sol
    dmu = dmu_over_U * U
    Np = float(solve(mu + dmu, warm(sol))[1].sum())
    Nm = float(solve(mu - dmu, warm(sol))[1].sum())
    Ntot = float(n_avg.sum())
    return GutzwillerResult(
        n=n_avg.reshape(2, c).numpy(),
        phi=phi.reshape(2, c).numpy(),
        var_n=var.reshape(2, c).numpy(),
        mu=float(mu),
        filling=Ntot / c,
        condensate_fraction=float((phi.abs() ** 2).sum()) / max(Ntot, 1e-300),
        compressibility=(Np - Nm) / (2.0 * dmu * c),
        grand_potential=omega / (2 * c),
        converged=conv,
        iterations=its,
    )
