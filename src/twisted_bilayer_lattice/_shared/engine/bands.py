"""Single-particle reference calculations (numpy/scipy), lattice units
a = 1, E_R = 1, hbar = 1, kinetic operator -kappa d^2/dx^2 with kappa = 1/pi^2.

These are the exact, interaction-free numbers the GPE engine is tested
against (also used by the lattice package):

  * 1D lattice V0 sin^2(pi x): Bloch bands by plane-wave diagonalisation,
    the hopping J(d), the maximally localised Wannier function of the lowest
    band, its overlap s(d) with a displaced copy and the on-site U.
    The 2D square lattice is separable, so every 2D quantity is a product
    or a sum of these.
  * The two-component moire Hamiltonian on the commensurate supercell,
    matrix-free (numpy FFT kinetic term; LOBPCG or ARPACK eigsh on a
    LinearOperator), with the potentials built by
    `engine.lattice.bilayer_potential` on an `engine.grid.Grid`, so the
    conventions (rotation sense, origin = AA, grid points) are identical to
    the engine's. Only `moire_supercell` touches torch (lazily, to build the
    potential); everything else is numpy.

Plane-wave basis for the 1D lattice: psi_q(x) = sum_l c_l e^{i(q + 2 pi l)x},
    <l|H|l>   = kappa (q + 2 pi l)^2 + V0/2,
    <l|H|l+1> = <l+1|H|l> = -V0/4,
because V0 sin^2(pi x) = V0/2 - (V0/4)(e^{2 i pi x} + e^{-2 i pi x}).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .units import KAPPA

if TYPE_CHECKING:  # annotations only; the functions import these at call time
    from .grid import Grid
    from .lattice import MoireGeometry

# -- 1D lattice -----------------------------------------------------------------


def bz_grid(n_q: int) -> np.ndarray:
    """n_q equally spaced quasi-momenta in (-pi, pi] (q = pi included,
    q = 0 included for even n_q)."""
    return -math.pi + 2.0 * math.pi * (np.arange(n_q) + 1) / n_q


def _bloch_1d(V0: float, q: np.ndarray, n_pw: int, kappa: float = KAPPA):
    """Eigen-decomposition of the plane-wave Hamiltonian at every q.
    Returns (E (n_q, n_pw), C (n_q, n_pw, n_pw), l (n_pw,)); C[iq, :, b] is
    band b at q[iq] in the basis e^{i(q + 2 pi l) x}."""
    if n_pw % 2 != 1:
        raise ValueError("n_pw must be odd (plane waves l = -n_pw//2 .. n_pw//2)")
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    l = np.arange(n_pw) - n_pw // 2
    idx = np.arange(n_pw)
    H = np.zeros((q.size, n_pw, n_pw))
    H[:, idx, idx] = kappa * (q[:, None] + 2.0 * math.pi * l[None, :]) ** 2 + 0.5 * V0
    H[:, idx[:-1], idx[1:]] = -0.25 * V0
    H[:, idx[1:], idx[:-1]] = -0.25 * V0
    E, C = np.linalg.eigh(H)
    return E, C, l


def bands_1d(
    V0: float, n_q: int = 64, n_pw: int = 31, n_bands: int = 4, kappa: float = KAPPA
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Bloch bands of V0 sin^2(pi x).
    Returns (q (n_q,) in (-pi, pi], E (n_q, n_bands)), lowest bands first."""
    q = bz_grid(n_q)
    E, _, _ = _bloch_1d(V0, q, n_pw, kappa)
    return q, E[:, :n_bands]


def hopping_1d(V0: float, d_max: int = 3, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA) -> np.ndarray:
    """J(d) = -<E_0(q) cos(q d)>_BZ for d = 1..d_max (lowest band), shape
    (d_max,). With E_0(q) = e_0 - 2 sum_d J(d) cos(q d) this is the
    tunnelling to the d-th neighbour; the sum over the n_q-point grid is
    exact up to aliasing from J(n_q - d), which is ~e^{-n_q}."""
    q, E = bands_1d(V0, n_q, n_pw, 1, kappa)
    E0 = E[:, 0]
    d = np.arange(1, d_max + 1)
    return -(E0[None, :] * np.cos(np.outer(d, q))).mean(axis=1)


def _wannier_coefficients(V0: float, n_q: int, n_pw: int, kappa: float = KAPPA):
    """Phase-fixed lowest-band Bloch coefficients: psi_q(0) = sum_l c_l(q)
    real and positive. For the even potential this is the gauge of the
    maximally localised (real, even) Wannier function (Kohn 1959).
    Returns (q, l, c (n_q, n_pw))."""
    q = bz_grid(n_q)
    _, C, l = _bloch_1d(V0, q, n_pw, kappa)
    c = C[:, :, 0].astype(np.complex128)
    s = c.sum(axis=1)
    c = c * (np.conj(s) / np.abs(s))[:, None]
    return q, l, c


def wannier_1d(
    V0: float, n_pw: int = 31, n_x: int = 64, n_sites: int | None = None, n_q: int = 64, kappa: float = KAPPA
) -> tuple[np.ndarray, np.ndarray]:
    """Maximally localised Wannier function of the lowest band, centred on
    the site at x = 0:

        w(x) = (1/n_q) sum_q psi_q(x),  psi_q phase-fixed real positive at x = 0,

    evaluated with n_x points per site on the Born-von Karman ring of n_q
    sites (where it is exactly normalised, int |w|^2 = 1) and returned on
    the window of n_sites sites around the origin (default: the whole ring).
    Returns (x, w), x = j / n_x for integer j in [-n_sites n_x / 2, n_sites n_x / 2).
    """
    q, l, c = _wannier_coefficients(V0, n_q, n_pw, kappa)
    # w(x) = sum_m a_m e^{2 pi i m x / n_q} with m = j_q + n_q l, a_m = c_l(q)/n_q
    jq = np.rint(q * n_q / (2.0 * math.pi)).astype(np.int64)
    m = (jq[:, None] + n_q * l[None, :]).ravel()
    a = (c / n_q).ravel()
    over = max(1, math.ceil((n_pw + 1) / n_x))  # no aliasing of the plane waves
    M = n_q * n_x * over
    A = np.zeros(M, dtype=np.complex128)
    np.add.at(A, np.mod(m, M), a)
    w = (np.fft.ifft(A) * M)[::over]  # x_t = t / n_x, t = 0 .. n_q n_x - 1
    if np.abs(w.imag).max() > 1e-10 * np.abs(w.real).max():
        raise RuntimeError("Wannier function is not real: phase fixing failed")
    w = np.fft.fftshift(w.real)  # x from -n_q/2 to n_q/2 - 1/n_x
    x = (np.arange(n_q * n_x) - (n_q * n_x) // 2) / n_x
    w = w / math.sqrt(np.sum(w * w) / n_x)  # = 1 already, to rounding
    if n_sites is None or n_sites >= n_q:
        return x, w
    half = (n_sites * n_x) // 2
    centre = (n_q * n_x) // 2
    sl = slice(centre - half, centre - half + n_sites * n_x)
    return x[sl], w[sl]


def overlap_1d(V0, d, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA) -> float | np.ndarray:
    """s(d) = int w(x) w(x - d) dx for any real displacement d (scalar or
    array). In the plane-wave basis

        s(d) = (1/n_q) sum_q sum_l |c_l(q)|^2 cos((q + 2 pi l) d),

    exact, and independent of the Wannier gauge; s(0) = 1 and s(integer != 0)
    = 0 (orthogonality of neighbouring Wannier functions). The 2D overlap of
    the separable lattice is S(dx, dy) = s(dx) s(dy)."""
    q, l, c = _wannier_coefficients(V0, n_q, n_pw, kappa)
    k = (q[:, None] + 2.0 * math.pi * l[None, :]).ravel()
    p = (np.abs(c) ** 2).ravel() / n_q
    d_arr = np.asarray(d, dtype=np.float64)
    out = (p[None, :] * np.cos(np.outer(d_arr.ravel(), k))).sum(axis=1)
    return out.reshape(d_arr.shape) if d_arr.ndim else float(out[0])


def _ho_sigma(V0: float, kappa: float = KAPPA) -> float:
    """Oscillator length of the site well in the harmonic approximation,
    V0 sin^2(pi x) ~ V0 pi^2 x^2: omega = 2 sqrt(V0 kappa) pi, sigma^2 = 2 kappa / omega."""
    omega = 2.0 * math.pi * math.sqrt(V0 * kappa)
    return math.sqrt(2.0 * kappa / omega)


def wannier_coefficients_band(
    V0: float, band: int = 0, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fixed Bloch coefficients of band `band` (0 = lowest) in the projection
    gauge: <g|psi_q> real and positive, with g the harmonic-oscillator state of
    order `band` of the site well, centred at x = 0 (its Fourier transform is
    (-i)^band H_band(k sigma) e^{-k^2 sigma^2 / 2}). For the even potential this is
    the gauge of the real Wannier function of parity (-1)^band centred at x = 0,
    i.e. the maximally localised one (Kohn 1959: real and symmetric fixes the gauge
    up to a sign and a lattice translation); for band 0 it coincides with
    `_wannier_coefficients` (psi_q(0) > 0). Raises if the projection weight
    |<g|psi_q>| drops below 1 % of its maximum at some q (the gauge would be
    ill-defined there). Returns (q, l, c (n_q, n_pw) complex, E (n_q,))."""
    q = bz_grid(n_q)
    E, C, l = _bloch_1d(V0, q, n_pw, kappa)
    c = C[:, :, band].astype(np.complex128)
    k = q[:, None] + 2.0 * math.pi * l[None, :]
    z = k * _ho_sigma(V0, kappa)
    herm = np.polynomial.hermite.hermval(z, [0.0] * band + [1.0])
    g_conj = (1j) ** band * herm * np.exp(-0.5 * z * z)  # conj of the trial's Fourier transform at k
    o = (c * g_conj).sum(axis=1)
    if np.abs(o).min() < 1e-2 * np.abs(o).max():
        raise RuntimeError(f"band {band}: the projection gauge is ill-defined at some q (V0 = {V0})")
    c = c * (np.conj(o) / np.abs(o))[:, None]
    return q, l, c, E[:, band]


def band_centres_1d(V0: float, n_bands: int = 4, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA) -> np.ndarray:
    """<E_b(q)>_BZ for b = 0..n_bands - 1: the energy of the Wannier function of
    band b (its on-site energy)."""
    _, E = bands_1d(V0, n_q, n_pw, n_bands, kappa)
    return E.mean(axis=0)


def overlap_bands_1d(
    V0, d, band_a: int = 0, band_b=0, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA
) -> float | np.ndarray:
    """s_ab(d) = int w_a(x) w_b(x - d) dx between the Wannier functions of bands
    a and b (`wannier_coefficients_band` gauge), for any real d:

        s_ab(d) = (1/n_q) sum_q sum_l conj(c^a_l(q)) c^b_l(q) e^{-i k_l d},  k_l = q + 2 pi l,

    exact on the Born-von Karman ring. s_00 = `overlap_1d`; s_ab(0) = delta_ab
    (orthonormality of the Wannier functions of one site); s_ab(integer n != 0) = 0;
    s_ab(-d) = (-1)^(a + b) s_ab(d) (parity). Over all n_pw bands and all sites,
    sum_b sum_n s_ab(d + n)^2 = 1 for any d (completeness). band_b may be a sequence
    of bands: then the result has a leading axis over them (one phase table for all)."""
    many = np.ndim(band_b) > 0
    bs = [int(b) for b in np.atleast_1d(band_b)]
    _, l, ca, _ = wannier_coefficients_band(V0, band_a, n_q, n_pw, kappa)
    q = bz_grid(n_q)
    k = (q[:, None] + 2.0 * math.pi * l[None, :]).ravel()
    P = np.stack([(np.conj(ca) * wannier_coefficients_band(V0, b, n_q, n_pw, kappa)[2]).ravel() / n_q for b in bs], 1)
    d_arr = np.asarray(d, dtype=np.float64)
    dflat = d_arr.ravel()
    out = np.empty((dflat.size, len(bs)), dtype=np.complex128)
    chunk = max(1, 2_000_000 // k.size)
    for i in range(0, dflat.size, chunk):
        out[i : i + chunk] = np.exp(-1j * np.outer(dflat[i : i + chunk], k)) @ P
    if np.abs(out.imag).max(initial=0.0) > 1e-10:
        raise RuntimeError("overlap_bands_1d: non-real overlap (gauge not time-reversal symmetric)")
    out = out.real.T.reshape((len(bs),) + d_arr.shape)
    if many:
        return out
    return out[0] if d_arr.ndim else float(out[0].ravel()[0])


def band_weight_1d(V0: float, d, band: int, n_q: int = 64, n_pw: int = 31, kappa: float = KAPPA) -> float | np.ndarray:
    """sum_n s_0b(d + n)^2 over all sites n of the ring: the weight of the lowest-band
    Wannier function displaced by d in band b of the undisplaced lattice. Gauge
    independent (Parseval: (1/n_q) sum_q |sum_l conj(c^0_l(q)) c^b_l(q) e^{-i k_l d}|^2,
    with the raw eigenvectors), so it is defined for every band, and
    sum_b over all n_pw bands = 1 exactly (completeness in the plane-wave space).
    1 - band_weight_1d(V0, d, 0) is the part of a displaced Wannier function that
    the lowest-band projection drops."""
    _, l, c0 = _wannier_coefficients(V0, n_q, n_pw, kappa)
    q = bz_grid(n_q)
    _, C, _ = _bloch_1d(V0, q, n_pw, kappa)
    k = q[:, None] + 2.0 * math.pi * l[None, :]
    d_arr = np.asarray(d, dtype=np.float64)
    out = np.empty(d_arr.size)
    for i, dd in enumerate(d_arr.ravel()):
        A = (np.conj(c0) * C[:, :, band] * np.exp(-1j * k * dd)).sum(axis=1)
        out[i] = float((np.abs(A) ** 2).mean())
    return out.reshape(d_arr.shape) if d_arr.ndim else float(out[0])


def I4_1d(V0: float, n_pw: int = 31, n_x: int = 64, n_q: int = 64, kappa: float = KAPPA) -> float:
    """int w^4 dx (units 1/a) of the maximally localised Wannier function,
    over the whole ring (periodic trapezoid = spectrally accurate)."""
    _, w = wannier_1d(V0, n_pw=n_pw, n_x=n_x, n_q=n_q, kappa=kappa)
    return float(np.sum(w**4) / n_x)


def onsite_U(V0: float, g2d: float, **kw) -> float:
    """Quasi-2D on-site interaction U = g2d (int w^4 dx)^2 of the separable
    square lattice; g2d in E_R a^2."""
    return g2d * I4_1d(V0, **kw) ** 2


# -- two-component moire supercell ---------------------------------------------


@dataclass
class MoireBands:
    energies: np.ndarray  # (n_bands,), ascending
    vectors: np.ndarray  # (n_bands, 2, N, N), unit 2-norm over all 2 N^2 entries
    participation: np.ndarray  # (n_bands,): (sum n)^2 / (N^2 sum n^2), n = |u_1|^2 + |u_2|^2
    weights: np.ndarray  # (n_bands, 2): sum |u_sigma|^2 per component
    N: int
    L: float
    q: tuple
    residuals: np.ndarray  # (n_bands,) ||H u - E u||


def moire_supercell(
    V0: float, m: int = 22, n: int = 1, points_per_a: float = 8.0
) -> tuple[MoireGeometry, Grid, np.ndarray, list[np.ndarray]]:
    """The supercell grid and both lattice potentials exactly as the engine
    builds them: Grid((N, N), (L, L)) with N = geometry.grid_size(points_per_a),
    V = bilayer_potential(grid, V0, geometry). Returns (geometry, grid, V
    (2, N, N) numpy float64, [kx, ky] numpy float64 FFT-ordered)."""
    import torch

    from .grid import Grid
    from .lattice import bilayer_potential, moire_geometry

    geo = moire_geometry(m, n)
    N = geo.grid_size(points_per_a)
    grid = Grid((N, N), (geo.L_box, geo.L_box), device=torch.device("cpu"), real_dtype=torch.float64)
    V = bilayer_potential(grid, V0, geometry=geo)[0].cpu().numpy().astype(np.float64)
    k = [grid.k64[ax].cpu().numpy() for ax in range(2)]
    return geo, grid, V, k


def moire_hamiltonian(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    points_per_a: float = 8.0,
    delta: float = 0.0,
    q=(0.0, 0.0),
    kappa: float = KAPPA,
) -> tuple[LinearOperator, LinearOperator, Grid]:
    """The two-component single-particle Hamiltonian on the supercell as a
    matrix-free scipy LinearOperator on vectors of length 2 N^2 (layout
    (2, N, N) raveled: the engine's (C, *N) without the batch axis),

        H = [[-kappa lap + V1 + delta/2, Omega_R], [Omega_R, -kappa lap + V2 - delta/2]],

    with -lap -> (k + q)^2 by FFT on the engine's grid (q = twist / L, as in
    BilayerGPE). Real (float64, rfft) at q = 0, complex otherwise; matmat is
    batched over the block. Also returns the kinetic preconditioner
    P = (kappa (k + q)^2 + alpha)^-1, alpha = 1 E_R, as a LinearOperator.
    Returns (H, P, grid)."""
    import scipy.fft as sfft

    _, grid, V, k = moire_supercell(V0, m, n, points_per_a)
    N = grid.N[0]
    qx, qy = float(q[0]), float(q[1])
    kin = kappa * ((k[0][:, None] + qx) ** 2 + (k[1][None, :] + qy) ** 2)
    pre = 1.0 / (kin + 1.0)
    loc = V.copy()
    loc[0] += 0.5 * delta
    loc[1] -= 0.5 * delta
    real = qx == 0.0 and qy == 0.0
    dtype = np.float64 if real else np.complex128
    dim = 2 * N * N
    kin_r = np.ascontiguousarray(kin[:, : N // 2 + 1])
    pre_r = np.ascontiguousarray(pre[:, : N // 2 + 1])
    Om = float(Omega_R)
    workers = 2

    def fourier_multiply(u, f, f_r):
        if real and np.isrealobj(u):
            return sfft.irfft2(f_r * sfft.rfft2(u, workers=workers), s=(N, N), workers=workers)
        return sfft.ifft2(f * sfft.fft2(u, workers=workers), workers=workers)

    def as_block(Y):
        Y = np.asarray(Y)
        Y2 = Y.reshape(dim, -1)
        return np.ascontiguousarray(Y2.T).reshape(Y2.shape[1], 2, N, N), Y.ndim == 1

    def from_block(u, one):
        out = u.reshape(u.shape[0], dim).T
        return out[:, 0].copy() if one else out

    def h_apply(Y):
        u, one = as_block(Y)
        out = fourier_multiply(u, kin, kin_r) + loc * u
        if Om != 0.0:
            out = out + Om * u[:, ::-1]
        return from_block(out, one)

    def p_apply(Y):
        u, one = as_block(Y)
        return from_block(fourier_multiply(u, pre, pre_r), one)

    H = LinearOperator((dim, dim), matvec=h_apply, matmat=h_apply, dtype=dtype)
    P = LinearOperator((dim, dim), matvec=p_apply, matmat=p_apply, dtype=dtype)
    return H, P, grid


def _blas_single_thread():
    """Tiny BLAS calls inside ARPACK/LOBPCG are much slower with a threaded
    OpenBLAS on small machines (thread hand-off per call); limit it to one
    thread while solving, if threadpoolctl happens to be installed."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(1, user_api="blas")
    except Exception:  # pragma: no cover - optional
        import contextlib

        return contextlib.nullcontext()


def _rayleigh_ritz(op: LinearOperator, U: np.ndarray):
    """Orthonormalise the columns of U and diagonalise H on their span.
    Returns (E ascending, U, residual norms ||H u - E u||)."""
    Q, _ = np.linalg.qr(U)
    HQ = op.matmat(Q)
    G = Q.conj().T @ HQ
    E, C = np.linalg.eigh(0.5 * (G + G.conj().T))
    U = Q @ C
    res = np.linalg.norm(HQ @ C - U * E, axis=0)
    return E, U, res


def lowest_eigenpairs(
    op: LinearOperator,
    k: int,
    method: str = "lobpcg",
    prec=None,
    tol: float | None = None,
    n_guard: int = 4,
    seed: int = 0,
    maxiter: int = 2000,
    max_rounds: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lowest k eigenpairs of a Hermitian LinearOperator, safe for
    degenerate spectra (the bilayer's C4 symmetry makes many pairs).

    method="lobpcg" (default): scipy.sparse.linalg.lobpcg with a block of
      k + n_guard vectors and the preconditioner `prec`; a block method sees
      every member of a degenerate level. tol = residual norm (default 1e-8;
      the eigenvalue error is ~ tol^2 / gap).
    method="eigsh": ARPACK Lanczos, which='SA' (tol default 1e-10, relative).
      From one start vector Lanczos sees a single direction in each exactly
      degenerate eigenspace; partners only appear through rounding, so a
      short run can return one member of a pair and skip to the next level
      (it does at m = 4, mini-zone corner). Completion: rerun on the
      Wielandt-deflated operator H + s U U^H (found levels moved up by s);
      anything it returns below the current k-th level was missed and is
      added, until a deflated run finds nothing new. Correct, but 2-4x the
      cost of one run.
    Both end with a Rayleigh-Ritz step on the vectors found.
    Returns (E (k,), U (dim, k), residual norms (k,))."""
    dim = op.shape[0]
    k = min(k, dim - 2)
    rng = np.random.default_rng(seed)
    cplx = np.dtype(op.dtype).kind == "c"

    def randn(*shape):
        x = rng.standard_normal(shape)
        return x + 1j * rng.standard_normal(shape) if cplx else x

    with _blas_single_thread():
        if method == "lobpcg":
            import warnings

            from scipy.sparse.linalg import lobpcg

            tol = 1e-8 if tol is None else tol
            nb = min(k + n_guard, dim // 4)
            X = randn(dim, nb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                E, U = lobpcg(op, X, M=prec, largest=False, tol=tol, maxiter=maxiter)
            E, U, res = _rayleigh_ritz(op, U)
        elif method == "eigsh":
            tol = 1e-10 if tol is None else tol
            kk = min(k + n_guard, dim - 2)
            nc = min(max(3 * kk + 1, 40), dim - 1)
            E, U = eigsh(op, k=kk, which="SA", tol=tol, v0=randn(dim), ncv=nc, maxiter=100000)
            E, U, _ = _rayleigh_ritz(op, U)
            atol = 1e-9 * max(1.0, float(np.abs(E).max()))
            for _ in range(max_rounds):
                n_left = min(kk, dim - 2 - U.shape[1])
                if n_left < 1:
                    break
                shift = 2.0 * float(E[-1] - E[0]) + 1.0

                def mv(v, Uf=U, shift=shift):
                    v = np.asarray(v).reshape(-1)
                    return op.matvec(v) + shift * (Uf @ (Uf.conj().T @ v))

                dop = LinearOperator((dim, dim), matvec=mv, dtype=op.dtype)
                E2, U2 = eigsh(
                    dop,
                    k=n_left,
                    which="SA",
                    tol=tol,
                    v0=randn(dim),
                    ncv=min(max(3 * n_left + 1, 40), dim - 1),
                    maxiter=100000,
                )
                new = E2 < E[min(kk, E.size) - 1] + atol
                if not new.any():
                    break
                E, U, _ = _rayleigh_ritz(op, np.concatenate([U, U2[:, new]], axis=1))
            E, U, res = _rayleigh_ritz(op, U)
        else:
            raise ValueError(f"method must be 'lobpcg' or 'eigsh', got {method!r}")
    return E[:k], U[:, :k], res[:k]


def moire_bands_q0(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    points_per_a: float = 8.0,
    n_bands: int = 8,
    delta: float = 0.0,
    q=(0.0, 0.0),
    kappa: float = KAPPA,
    method: str = "lobpcg",
    tol: float | None = None,
    n_guard: int = 4,
    seed: int = 0,
) -> MoireBands:
    """Lowest n_bands eigenpairs of the two-component single-particle
    Hamiltonian `moire_hamiltonian` on the commensurate supercell at
    quasi-momentum q (default 0, the engine's zero twist), matrix-free with
    numpy FFTs on exactly the engine's grid (`lowest_eigenpairs`: LOBPCG by
    default, method="eigsh" for ARPACK with the degeneracy completion).
    Default resolution 8 points per a: m = 22 is N = 180, ~15 s on 2 cores.

    Diagnostics: participation ratio of
    the total density (1 = spread over the whole cell, small = pocket) and
    the component weights (sum to 1)."""
    H, P, grid = moire_hamiltonian(V0, Omega_R, m, n, points_per_a, delta, q, kappa)
    N, L = grid.N[0], grid.L[0]
    E, U, res = lowest_eigenpairs(H, n_bands, method=method, prec=P, tol=tol, n_guard=n_guard, seed=seed)
    target = 1e-6
    if res.max() > target:
        import warnings

        warnings.warn(
            f"moire_bands_q0: residual {res.max():.2e} > {target:.0e} (eigenvalue error ~ residual^2 / gap)",
            stacklevel=2,
        )
    vec = U.T.reshape((n_bands, 2, N, N))
    dens = np.abs(vec) ** 2  # (n_bands, 2, N, N)
    weights = dens.sum(axis=(-2, -1))
    ntot = dens.sum(axis=1)
    pr = ntot.sum(axis=(-2, -1)) ** 2 / (N * N * (ntot**2).sum(axis=(-2, -1)))
    return MoireBands(
        energies=E,
        vectors=vec,
        participation=pr,
        weights=weights,
        N=N,
        L=L,
        q=(float(q[0]), float(q[1])),
        residuals=res,
    )


def moire_band_width(
    V0: float,
    Omega_R: float,
    m: int = 22,
    n: int = 1,
    points_per_a: float = 8.0,
    n_bands: int = 2,
    delta: float = 0.0,
    kappa: float = KAPPA,
    method: str = "lobpcg",
    tol: float | None = None,
) -> dict:
    """Lowest moire bands at the mini-zone centre Gamma = (0, 0), edge
    X = (pi/L, 0) and corner M = (pi/L, pi/L) by the twist trick k -> k + q,
    and the width of the lowest band estimated as max - min of E_0 over these
    three points (the lowest-band width versus the coupling; a convention
    with Omega = 2 Omega_R differs by that factor, this function takes the
    engine's Omega_R). At m = 22 this is three
    supercell diagonalisations (two complex), ~1 min on 2 cores with
    n_bands = 1. Returns {"Gamma", "X", "M": (n_bands,) arrays,
    "width": float, "L": box side}."""
    from .lattice import moire_geometry

    L = moire_geometry(m, n).L_box
    out = {}
    for name, q in (("Gamma", (0.0, 0.0)), ("X", (math.pi / L, 0.0)), ("M", (math.pi / L, math.pi / L))):
        out[name] = moire_bands_q0(
            V0, Omega_R, m, n, points_per_a, n_bands, delta, q=q, kappa=kappa, method=method, tol=tol
        ).energies
    e0 = np.array([out[key][0] for key in ("Gamma", "X", "M")])
    out["width"] = float(e0.max() - e0.min())
    out["L"] = L
    return out


def effective_mass_ratio(V0: float, n_pw: int = 31, kappa: float = KAPPA, h: float = 1e-3) -> float:
    """m/m* = E_0''(0) / (2 kappa) from the exact lowest band: the
    non-interacting superfluid fraction along a lattice axis (twist-energy
    definition). The nearest-neighbour value J1/kappa overestimates it because
    J2 < 0 flattens the band bottom: 0.2805 vs 0.3040 at V0 = 8 E_R."""
    q = np.array([-h, 0.0, h])
    E = _bloch_1d(V0, q, n_pw, kappa)[0][:, 0]
    curv = (E[2] - 2.0 * E[1] + E[0]) / (h * h)
    return float(curv / (2.0 * kappa))


def depth_for_equal_hopping(V0: float, period: float, n_pw: int = 31) -> float:
    """Depth V0' (in E_R of the a = 1 lattice) of a lattice of period `period`
    (in a) whose nearest-neighbour hopping equals that of the a = 1 lattice at
    depth V0 -- the 1D toy model, in which lattice 2 has its depth chosen
    for equal J. In its own recoil E_R' = E_R / period^2 the hopping is
    E_R' j(V0'/E_R'), so solve j(s') = j(V0) period^2 for s' by bisection."""
    ER2 = 1.0 / period**2
    target = hopping_1d(V0, 1, n_pw=n_pw)[0] / ER2
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if hopping_1d(mid, 1, n_pw=n_pw)[0] > target:
            lo = mid  # j(s) decreases with s
        else:
            hi = mid
    return 0.5 * (lo + hi) * ER2


def band_bottom_1d(V0: float, period: float = 1.0, n_pw: int = 31) -> float:
    """Lowest band energy at q = 0 of V0 sin^2(pi x / period), in E_R of a = 1:
    E_R' j_0(V0 / E_R') with E_R' = 1/period^2."""
    ER2 = 1.0 / period**2
    E, _, _ = _bloch_1d(V0 / ER2, np.array([0.0]), n_pw)
    return float(E[0, 0] * ER2)
