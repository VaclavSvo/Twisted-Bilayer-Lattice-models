"""Observables the many-body lattice methods return: `ManyBodyResult`, the momentum distribution (the
time-of-flight image), the visibilities and the phase label.

One-body density matrix. Single-site Gutzwiller: rho_ij = phi_i^* phi_j (i != j), rho_ii = <n_i>. A cluster
method adds the connected part inside its clusters, rho_ij - phi_i^* phi_j, carried as `rho_connected`
(rows, cols, values), both orders listed.

Momentum distribution of layer l (far field; the two spin states are distinguishable, so cross-layer
coherence is not imaged and each layer is computed on its own):

    n_l(k) = |w_l(k)|^2 S_l(k),
    S_l(k) = |sum_{i in l} phi_i e^{-i k.R_i}|^2 + sum_{i in l} (<n_i> - |phi_i|^2)
             + sum_{(i, j) in l, i != j} rho^conn_ij e^{i k.(R_i - R_j)},

with w_l the Fourier transform of the layer's Wannier function, separable along the layer's own axes,
w_l(k) = w1(k.e1) w1(k.e2), w1 from `_shared.engine.bands.wannier_1d`. The sums are evaluated directly at the k
points needed (the sites of the two rotated layers are not on one grid).

Visibilities (Gerbier et al., PRA 72, 053606 (2005)), per layer:
  * primary: V_p = (S(k1) - S(k1')) / (S(k1) + S(k1')), k1 = 2 pi / a along a lattice axis of the layer, k1'
    the same |k| rotated by 45 deg, each averaged over the four symmetry-related points;
  * moire: the same at the moire reciprocal vector G_1 - G_2 (|k| = 2 pi / lambda_mo, lambda_mo =
    a / (2 sin(theta / 2))) and its 45 deg partner.
  `visibility_*` use the structure factor S (Wannier envelope divided out), so a single-site Mott state has
  V = 0 exactly; `visibility_*_raw` use n(k) itself (the separable envelope is not isotropic, so the raw
  number of a Mott state is not 0); `visibility_*_caption` is the paper's Fig. 4 caption definition
  (P_max^av - P_min) / (P_max^av + P_min) on n(k), with P_max^av the mean over the four Bragg points and
  P_min the minimum of n(k) on the ring |k| = |k1| (72 angles) -- the caption does not say where P_min is
  taken; this is the choice made here.
  The Mott-state number with the perturbative short-range coherence: <b_i^dag b_j> = 2 |T_ij| n (n + 1) / U
  on the nearest-neighbour intralayer bonds of sites in an integer plateau n (Gerbier et al. 2005), added to
  rho^conn (`visibility_primary_pert`).

Phase labels (the paper's Supplement Fig. S4b made operational; thresholds are row parameters):

    SF     f_s > f_c
    SF-II  f_s <= f_c and max |phi| > phi_c       (superfluid islands without global superflow)
    I      f_s <= f_c, max |phi| <= phi_c, density contrast > c_c   (n not uniform)
    MI     f_s <= f_c, max |phi| <= phi_c, contrast <= c_c          (n uniform and integer)
    empty  no site holds an atom (max <n> < 1e-6): the vacuum, not a Mott insulator
    ?      the state (or a twisted solve) did not converge

with the contrast (n_max - n_min) / (n_max + n_min) over the sites. Optionally a set of sites is excluded
from max |phi| and the contrast (`exclude`): the reading of the paper's model in which the AA sites,
degenerate at mu / U = 1, are left out.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice._shared.engine import bands

THRESHOLDS = {"f_c": 1e-3, "phi_c": 1e-4, "c_c": 1e-2}
LABELS = ("SF", "SF-II", "I", "MI", "?", "empty")  # codes 0-5 in the maps; "empty" appended last


@dataclass
class ManyBodyResult:
    """The common result of a many-body solve on a `LatticeModel` (arrays (n_layers, c))."""

    n: np.ndarray
    var_n: np.ndarray
    phi: np.ndarray
    mu: float
    filling: float  # sum <n> / c (atoms per a^2, all layers)
    grand_potential: float  # per site (all layers)
    condensate_fraction: float  # sum |phi|^2 / N
    converged: bool
    method: str
    compressibility: float = float("nan")
    fs: float = float("nan")  # the quoted f_s (theta -> 0)
    fs_route: str = ""
    fs_theta: dict = field(default_factory=dict)  # {theta: f_s(theta)}
    visibility_primary: np.ndarray | None = None  # (n_layers,)
    visibility_moire: np.ndarray | None = None
    visibilities: dict = field(default_factory=dict)  # every visibility number, per layer
    label: str = "?"
    label_info: dict = field(default_factory=dict)
    p_nmax: float = float("nan")  # max over sites of the weight in the top Fock state
    n_max: int = -1
    starts: dict = field(default_factory=dict)
    omega_spread: float = float("nan")
    iterations: int = 0
    rho_connected: tuple | None = None  # (rows, cols, values): rho_ij - phi_i^* phi_j inside clusters
    params: dict = field(default_factory=dict)

    def scalars(self) -> dict:
        """The scalar fields (for tables and maps)."""
        d = asdict(self)
        return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}


def from_state(model: lm.LatticeModel, st, method: str = "gutzwiller", n_max: int = -1, **extra) -> ManyBodyResult:
    """A `ManyBodyResult` from a `gutzwiller.GWState` (or anything with phi, n, var_n, p_top, omega,
    mu, converged, iterations)."""
    nl, c = model.n_layers, model.cell.c
    N = float(np.sum(st.n))
    return ManyBodyResult(
        n=np.asarray(st.n).reshape(nl, c),
        var_n=np.asarray(st.var_n).reshape(nl, c),
        phi=np.asarray(st.phi).reshape(nl, c),
        mu=float(st.mu),
        filling=N / c,
        grand_potential=float(st.omega) / model.n_sites,
        condensate_fraction=float(np.sum(np.abs(st.phi) ** 2)) / max(N, 1e-300),
        converged=bool(st.converged),
        method=method,
        p_nmax=float(np.max(st.p_top)) if getattr(st, "p_top", None) is not None else float("nan"),
        n_max=int(n_max),
        iterations=int(st.iterations),
        **extra,
    )


# -- Wannier envelope -----------------------------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _wannier_ft_table(V0: float):
    x, w = bands.wannier_1d(float(V0))
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def wannier_ft_1d(V0: float, q) -> np.ndarray:
    """w1(q) = int w(x) e^{-i q x} dx of the lowest-band Wannier function (real and even: a cosine transform);
    w1(0)^2 = (int w)^2, and int |w1(q)|^2 dq / (2 pi) = 1."""
    x, w = _wannier_ft_table(V0)
    dx = float(x[1] - x[0])
    q = np.asarray(q, dtype=np.float64)
    return (np.cos(np.outer(q.ravel(), x)) @ w * dx).reshape(q.shape)


def wannier_envelope(model: lm.LatticeModel, k_points, layer: int) -> np.ndarray:
    """|w_l(k)|^2 = |w1(k.e1) w1(k.e2)|^2 along the axes of `layer`."""
    k = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    ax = model.cell.layers[layer].axes[0]  # (2, 2): ax[mu] = unit vector of axis mu
    q1, q2 = k @ ax[0], k @ ax[1]
    return (wannier_ft_1d(model.V0, q1) * wannier_ft_1d(model.V0, q2)) ** 2


# -- momentum distribution ------------------------------------------------------------------------------------------


def structure_factor(
    model: lm.LatticeModel, result: ManyBodyResult, k_points, layer: int, extra_rho=None
) -> np.ndarray:
    """S_l(k) of the module docstring at k_points (K, 2) (1/a, box frame), for the sites of `layer`."""
    k = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    c = model.cell.c
    R = model.cell.layers[layer].positions
    phi = np.asarray(result.phi).reshape(model.n_layers, c)[layer]
    n = np.asarray(result.n).reshape(model.n_layers, c)[layer]
    ph = np.exp(-1j * (k @ R.T))  # (K, c)
    S = np.abs(ph @ phi) ** 2 + float(np.sum(n - np.abs(phi) ** 2))
    for rc in (result.rho_connected, extra_rho):
        if rc is None:
            continue
        rows, cols, vals = (np.asarray(a) for a in rc)
        lo, hi = layer * c, (layer + 1) * c
        sel = (rows >= lo) & (rows < hi) & (cols >= lo) & (cols < hi) & (rows != cols)
        if sel.any():
            ri, rj = rows[sel] - lo, cols[sel] - lo
            d = R[ri] - R[rj]
            d -= model.L * np.round(d / model.L)  # minimum image: the physical separation
            S = S + np.real(np.exp(1j * (k @ d.T)) @ vals[sel])
    return S


def momentum_distribution(model: lm.LatticeModel, result: ManyBodyResult, k_points, layer: int, extra_rho=None):
    """n_l(k) = |w_l(k)|^2 S_l(k) (module docstring)."""
    return wannier_envelope(model, k_points, layer) * structure_factor(model, result, k_points, layer, extra_rho)


def _rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def primary_vectors(model: lm.LatticeModel, layer: int):
    """(the four Bragg vectors 2 pi e_mu of the layer, their four 45-deg partners)."""
    e1 = model.cell.layers[layer].axes[0][0]
    G = 2.0 * math.pi * e1
    peaks = np.array([_rot(G, q * math.pi / 2) for q in range(4)])
    partners = np.array([_rot(G, math.pi / 4 + q * math.pi / 2) for q in range(4)])
    return peaks, partners


def moire_vectors(model: lm.LatticeModel):
    """(the four moire reciprocal vectors G_1 - G_2 (and rotations by 90 deg), their 45-deg partners); |G_1 -
    G_2| = 4 pi sin(theta / 2) = 2 pi / lambda_mo. For a one-layer model (the paper's) lattice 2's axes
    are still those of the cell's second layer."""
    a1 = model.cell.layers[0].axes[0][0]
    a2 = model.cell.layers[1].axes[0][0]
    G = 2.0 * math.pi * (a1 - a2)
    peaks = np.array([_rot(G, q * math.pi / 2) for q in range(4)])
    partners = np.array([_rot(G, math.pi / 4 + q * math.pi / 2) for q in range(4)])
    return peaks, partners


def _gerbier(a, b):
    return float((a - b) / (a + b)) if a + b > 0 else 0.0


def perturbative_rho(model: lm.LatticeModel, result: ManyBodyResult, phi_c: float = THRESHOLDS["phi_c"]):
    """rho^conn_ij = 2 |T_ij| n (n + 1) / U on the strongest (nearest-neighbour) intralayer bonds between
    sites of one integer plateau n with |phi| <= phi_c (both orders): the first-order coherence of a Mott
    state (Gerbier et al. 2005; 4 J / U at n = 1). None if there is no such bond."""
    b = model.bonds
    c = model.cell.c
    n = np.asarray(result.n).reshape(-1)
    phi = np.abs(np.asarray(result.phi).reshape(-1))
    intra = (b.rows // c) == (b.cols // c)
    if not intra.any() or model.U <= 0:
        return None
    a = np.abs(b.amp)
    J1 = a[intra].max()
    sel = intra & (np.abs(a - J1) < 1e-12 * J1)
    ni, nj = n[b.rows], n[b.cols]
    plateau = (
        (np.abs(ni - np.rint(ni)) < 1e-6) & (np.abs(ni - nj) < 1e-6) & (phi[b.rows] <= phi_c) & (phi[b.cols] <= phi_c)
    )
    sel &= plateau
    if not sel.any():
        return None
    nn = np.rint(ni[sel])
    vals = 2.0 * a[sel] * nn * (nn + 1.0) / model.U
    return b.rows[sel], b.cols[sel], vals


def visibilities(model: lm.LatticeModel, result: ManyBodyResult, layers=None, n_ring: int = 72) -> dict:
    """Every visibility number per layer (module docstring): {"primary", "moire", "primary_raw", "moire_raw",
    "primary_caption", "moire_caption", "primary_pert"}, each an array (n_layers,) (NaN for a layer not
    computed). The moire numbers need a two-layer cell (the paper's model carries one; its moire vectors are
    the cell's)."""
    nl = model.n_layers
    layers = range(nl) if layers is None else layers
    keys = ("primary", "moire", "primary_raw", "moire_raw", "primary_caption", "moire_caption", "primary_pert")
    out = {k: np.full(nl, np.nan) for k in keys}
    pert = perturbative_rho(model, result)
    for l in layers:
        pk, pp = primary_vectors(model, l)
        mk, mp = moire_vectors(model)
        S = {
            "p": structure_factor(model, result, pk, l).mean(),
            "pp": structure_factor(model, result, pp, l).mean(),
            "m": structure_factor(model, result, mk, l).mean(),
            "mp": structure_factor(model, result, mp, l).mean(),
        }
        nkp = momentum_distribution(model, result, pk, l)
        nkpp = momentum_distribution(model, result, pp, l)
        nkm = momentum_distribution(model, result, mk, l)
        nkmp = momentum_distribution(model, result, mp, l)
        out["primary"][l] = _gerbier(S["p"], S["pp"])
        out["moire"][l] = _gerbier(S["m"], S["mp"])
        out["primary_raw"][l] = _gerbier(nkp.mean(), nkpp.mean())
        out["moire_raw"][l] = _gerbier(nkm.mean(), nkmp.mean())
        ang = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
        ringp = np.linalg.norm(pk[0]) * np.stack([np.cos(ang), np.sin(ang)], 1)
        ringm = np.linalg.norm(mk[0]) * np.stack([np.cos(ang), np.sin(ang)], 1)
        out["primary_caption"][l] = _gerbier(nkp.mean(), momentum_distribution(model, result, ringp, l).min())
        out["moire_caption"][l] = _gerbier(nkm.mean(), momentum_distribution(model, result, ringm, l).min())
        if pert is not None:
            a = structure_factor(model, result, pk, l, extra_rho=pert).mean()
            b = structure_factor(model, result, pp, l, extra_rho=pert).mean()
            out["primary_pert"][l] = _gerbier(a, b)
        else:
            out["primary_pert"][l] = out["primary"][l]
    return out


def attach_visibilities(model: lm.LatticeModel, result: ManyBodyResult, **kw) -> ManyBodyResult:
    v = visibilities(model, result, **kw)
    result.visibilities = {k: [float(x) for x in a] for k, a in v.items()}
    result.visibility_primary = v["primary"]
    result.visibility_moire = v["moire"]
    return result


# -- phase label ----------------------------------------------------------------------------------------------------


def density_contrast(n: np.ndarray) -> float:
    n = np.asarray(n, dtype=np.float64).ravel()
    if n.size == 0:
        return 0.0
    hi, lo = float(n.max()), float(n.min())
    return (hi - lo) / (hi + lo) if hi + lo > 0 else 0.0


def degenerate_sites(model: lm.LatticeModel, mu: float, window: float = 0.05) -> np.ndarray:
    """Boolean (n_sites,) mask of the sites whose local mu_i = mu - eps_i is within `window` U of a
    lobe-touching point k U (k >= 1 integer): at t -> 0 they are degenerate between |k> and |k + 1> and stay
    superfluid at any hopping (the mu / U = 1 degeneracy of the paper's model at its AA sites)."""
    x = (mu - np.asarray(model.onsite)) / model.U
    k = np.rint(x)
    return (k >= 1) & (np.abs(x - k) < window)


def phase_label(result: ManyBodyResult, thresholds=None, exclude=None) -> tuple:
    """(label, info) by the table of the module docstring; `exclude` = boolean (n_sites,) mask of sites left
    out of max |phi| and the contrast. info carries every number the label was decided on."""
    th = dict(THRESHOLDS)
    th.update(thresholds or {})
    phi = np.abs(np.asarray(result.phi)).ravel()
    n = np.asarray(result.n, dtype=np.float64).ravel()
    keep = np.ones(n.size, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool).ravel()
    phimax = float(phi[keep].max()) if keep.any() else 0.0
    con = density_contrast(n[keep]) if keep.any() else 0.0
    integer = bool(keep.any() and np.all(np.abs(n[keep] - np.rint(n[keep])) < 1e-6))
    info = {
        "fs": float(result.fs),
        "phi_max": phimax,
        "contrast": con,
        "integer": integer,
        "n_excluded": int((~keep).sum()),
        **{k: float(v) for k, v in th.items()},
    }
    if not result.converged or not np.isfinite(result.fs):
        return "?", info
    if result.fs > th["f_c"]:
        return "SF", info
    if phimax > th["phi_c"]:
        return "SF-II", info
    if con > th["c_c"]:
        return "I", info
    if not (keep.any() and float(n[keep].max()) >= 1e-6):
        return "empty", info
    return "MI", info


__all__ = [
    "LABELS",
    "THRESHOLDS",
    "ManyBodyResult",
    "attach_visibilities",
    "degenerate_sites",
    "density_contrast",
    "from_state",
    "moire_vectors",
    "momentum_distribution",
    "perturbative_rho",
    "phase_label",
    "primary_vectors",
    "structure_factor",
    "visibilities",
    "wannier_envelope",
    "wannier_ft_1d",
]
