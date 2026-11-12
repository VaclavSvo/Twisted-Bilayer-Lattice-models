"""Cluster Gutzwiller (cluster.py): tilings, the reduction to single-site Gutzwiller, a cluster that
covers a whole periodic lattice against an independent exact diagonalisation, the twist, and the
square-lattice lobe tip against the single-site and QMC anchors."""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twisted_bilayer_lattice import cluster as C
from twisted_bilayer_lattice import gutzwiller as gw
from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice.paper_model import paper_model
from twisted_bilayer_lattice.references import anchor

FULL = dict(nu_z_hz=1000.0, cutoff=2.0, interlayer_interaction=True, interband=True, interband_bonds=False)


# -- tilings --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["single", "dimer", "plus", "bilayer_plus"])
def test_tilings_cover_every_site_once(family):
    """m = 22: every site in exactly one cluster (checked in make_tiling), inside + cross bonds = all bonds,
    and the cluster sizes of the family."""
    model = lm.build_model(12.0, 0.6, 22, 1, cutoff=2.0)
    t = C.make_tiling(model, family)
    size = {"single": 1, "dimer": 2, "plus": 5, "bilayer_plus": 10}[family]
    assert all(len(cl) == size for cl in t.clusters)
    assert t.n_clusters * size == model.n_sites
    flat = np.sort(np.concatenate(t.clusters))
    np.testing.assert_array_equal(flat, np.arange(model.n_sites))
    assert t.inside.sum() + (~t.inside).sum() == len(model.bonds)
    assert 0.0 <= t.lam < 1.0


def test_plus_chirality_experiment_cell():
    """Layer 1 of the (22, 1) cell tiles with {i - 2j = 0 mod 5}, layer 2 with the mirror {i + 2j = 0 mod 5}
    (opposite chiralities); each plus is a centre and its four nearest neighbours (distance 1 a)."""
    cell = lm.supercell_sites(22, 1)
    assert C.plus_chirality(cell.layers[0]) == 1
    assert C.plus_chirality(cell.layers[1]) == -1
    model = lm.build_model(12.0, 0.6, 22, 1, cutoff=2.0)
    t = C.make_tiling(model, "plus")
    c, L = model.cell.c, model.L
    for cl in t.clusters[:20] + t.clusters[-20:]:
        lay = cl[0] // c
        P = model.cell.layers[lay].positions[np.asarray(cl) - lay * c]
        d = P[1:] - P[0]
        d -= L * np.round(d / L)
        np.testing.assert_allclose(np.hypot(d[:, 0], d[:, 1]), 1.0, atol=1e-12)


def test_dimer_matching_short():
    """The interlayer dimers pair every layer-1 site with a layer-2 site within half a lattice constant
    times sqrt 2 (the largest offset of the two rotated lattices on the cell)."""
    model = lm.build_model(12.0, 0.6, 22, 1, cutoff=2.0)
    t = C.make_tiling(model, "dimer")
    assert t.info["max_distance"] <= math.sqrt(2) / 2 + 1e-9


# -- reductions -----------------------------------------------------------------------------------------------------


def test_single_site_clusters_reproduce_single_site_gutzwiller():
    """One-site clusters = single-site Gutzwiller (phi, n, Omega) to 1e-12 (both converged to tol 1e-13)."""
    model = lm.build_model(12.0, 0.3, 4, 1, **FULL)
    mu = model.U - 0.3 * model.diagnostics["F_k0_AA"]
    ref = gw.solve(model, mu, 6, tol=1e-13)
    st = C.cluster_gutzwiller(model, C.make_tiling(model, "single"), mu, 6, tol=1e-13)
    assert ref.converged and st.converged
    np.testing.assert_allclose(st.phi, ref.phi, atol=1e-12, rtol=0)
    np.testing.assert_allclose(st.n, ref.n, atol=1e-12, rtol=0)
    assert st.omega == pytest.approx(ref.omega, abs=1e-12)


def _square_model(Lx, J, U, eps=None):
    """A one-layer `LatticeModel` of an Lx x Lx periodic square lattice with nearest-neighbour -J (both
    directions listed; on Lx = 2 the +x and -x neighbours are the same site, so T_01 = -2J)."""
    c = Lx * Lx
    rows, cols, disp = [], [], []
    for x, y in itertools.product(range(Lx), repeat=2):
        i = y * Lx + x
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = ((y + dy) % Lx) * Lx + (x + dx) % Lx
            rows.append(i)
            cols.append(j)
            disp.append((dx, dy))
    bonds = lm.Bonds(
        rows=np.array(rows), cols=np.array(cols), amp=np.full(len(rows), -J), disp=np.array(disp, dtype=np.float64)
    )
    return lm.LatticeModel(
        cell=SimpleNamespace(c=c, L=float(Lx)),
        V0=0.0,
        Omega_R=0.0,
        delta=0.0,
        J=np.array([J]),
        U=U,
        S=sp.csr_matrix((c, c)),
        bonds=bonds,
        onsite=np.zeros(c) if eps is None else np.asarray(eps, dtype=np.float64),
        V_pairs=None,
        n_layers=1,
    )


def _exact_grand_canonical(T, U, eps, V, mu, n_max):
    """Independent exact diagonalisation: number-conserving basis per N (0 <= n_i <= n_max), H_N = sum T_ij
    b_i^dag b_j + (U/2) sum n (n - 1) + sum (eps_i - mu) n_i + sum_{i<j} V_ij n_i n_j, the lowest level of each
    sector (dense eigh up to 1500 states, eigsh above), the grand-canonical minimum and its rho_ij."""
    s = T.shape[0]
    best = None
    for N in range(0, s * n_max + 1):
        occ = [o for o in itertools.product(range(n_max + 1), repeat=s) if sum(o) == N]
        occ = np.array(occ, dtype=np.int64).reshape(-1, s)
        index = {tuple(o): k for k, o in enumerate(occ)}
        D = occ.shape[0]
        diag = (
            (0.5 * U * occ * (occ - 1)).sum(1)
            + ((eps - mu) * occ).sum(1)
            + np.einsum("ai,ij,aj->a", occ, np.triu(V, 1), occ)
        )
        r, c_, v = [], [], []
        for a in range(D):
            for i in range(s):
                for j in range(s):
                    if i == j or T[i, j] == 0 or occ[a, j] == 0:
                        continue
                    t = occ[a].copy()
                    amp = math.sqrt(t[j])
                    t[j] -= 1
                    amp *= math.sqrt(t[i] + 1)
                    t[i] += 1
                    b = index.get(tuple(t))
                    if b is not None:
                        r.append(b)
                        c_.append(a)
                        v.append(T[i, j] * amp)
        H = sp.csr_matrix((v, (r, c_)), shape=(D, D)) + sp.diags(diag + occ @ np.diag(T))
        if D <= 1500:
            w, X = np.linalg.eigh(H.toarray())
        else:
            w, X = spla.eigsh(H, k=1, which="SA", tol=1e-14)
        if best is None or w[0] < best[0]:
            best = (float(w[0]), X[:, 0], occ, index, N)
    E, g, occ, index, N = best
    rho = np.zeros((s, s))
    for a in range(occ.shape[0]):
        for i in range(s):
            for j in range(s):
                if occ[a, j] == 0:
                    continue
                t = occ[a].copy()
                amp = math.sqrt(t[j])
                t[j] -= 1
                amp *= math.sqrt(t[i] + 1)
                t[i] += 1
                b = index.get(tuple(t))
                if b is not None:
                    rho[i, j] += g[b] * amp * g[a]
    return E, rho, N


def _cluster_rho(model, st):
    s = model.n_sites
    rho = np.diag(st.n).astype(np.complex128) + np.outer(np.conj(st.phi), st.phi) * (1 - np.eye(s))
    if st.rho_connected is not None:
        r, c, v = st.rho_connected
        rho[r, c] += v
    return rho


def _dense_T(model):
    n = model.n_sites
    T = np.zeros((n, n))
    np.add.at(T, (model.bonds.rows, model.bonds.cols), model.bonds.amp)
    return T


def test_whole_lattice_cluster_vs_exact_diagonalisation_square_2x2():
    """A cluster covering the whole periodic 2 x 2 square lattice (no cross bonds) = the exact grand-canonical
    ground state (E to 1e-10, rho_ij to 1e-8), computed by the independent exact diagonalisation in this file."""
    model = _square_model(2, J=0.07, U=1.0, eps=[0.0, 0.03, -0.02, 0.05])
    mu, n_max = 0.43, 3
    st = C.cluster_gutzwiller(model, C.make_tiling(model, "whole"), mu, n_max, phi0=np.zeros(4), mixing=0.5)
    E, rho, N = _exact_grand_canonical(_dense_T(model), model.U, model.onsite, np.zeros((4, 4)), mu, n_max)
    assert st.converged
    assert st.omega == pytest.approx(E, abs=1e-10)
    np.testing.assert_allclose(_cluster_rho(model, st).real, rho, atol=1e-8)
    assert st.N == pytest.approx(N, abs=1e-8)


@pytest.mark.slow
def test_whole_cell_cluster_vs_exact_diagonalisation_bilayer_21():
    """The (2, 1) bilayer cell (10 sites, interlayer V on, cutoff 2: periodic-image duplicates summed in both
    codes), n_max = 2, one cluster covering it: E to 1e-10 and rho_ij to 1e-8 against the independent ED."""
    model = lm.build_model(12.0, 0.6, 2, 1, cutoff=2.0, interlayer_interaction=True, nu_z_hz=1000.0)
    c = model.cell.c
    V = np.zeros((2 * c, 2 * c))
    i, j, v = model.V_pairs
    np.add.at(V, (i, np.asarray(j) + c), v)
    V = V + V.T
    mu, n_max = 0.6 * model.U - 0.4, 2
    st = C.cluster_gutzwiller(model, C.make_tiling(model, "whole"), mu, n_max, phi0=np.zeros(2 * c), mixing=0.5)
    E, rho, N = _exact_grand_canonical(_dense_T(model), model.U, model.onsite, V, mu, n_max)
    assert st.converged
    assert st.omega == pytest.approx(E, abs=1e-10)
    np.testing.assert_allclose(_cluster_rho(model, st).real, rho, atol=1e-8)


# -- twist ----------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_cluster_twist_mott_zero():
    """Plus clusters on the paper's model (m = 22, V0 = 20, M_R = 0, mu / U = 0.5): the Mott state (each cluster
    in a number eigenstate, so phi = 0 up to the rounding of eigh) has f_s = 0 to rounding; its plus clusters
    carry the short-range coherence (rho^conn != 0) that single-site Gutzwiller lacks."""
    model = paper_model(20.0, 0.0)
    t = C.make_tiling(model, "plus")
    mu = 0.5 * model.U
    st = C.cluster_gutzwiller(model, t, mu, 2, phi0=np.zeros(model.n_sites))
    assert st.converged and np.abs(st.phi).max() < 1e-12
    r = C.cluster_fs(model, t, st, 2)
    assert abs(r["fs0"]) < 1e-10
    assert np.abs(st.rho_connected[2]).max() > 1e-3


@pytest.mark.slow
def test_cluster_twist_u0_square_pairs():
    """Twisted cluster mean field at U -> 0: a 2 x 2 periodic square lattice tiled by two horizontal pairs (each
    pair holds both x bonds of its sites, the +x and -x images, with different Peierls phases; the y bonds are
    cross bonds), U = 0.01 J, one atom per site at mu = -4J + U, n_max = 10 (a near-coherent state: P(n = 8) ~
    e^-1 / 8! = 9e-6): f_s(theta -> 0) at fixed mu equals the exact non-interacting band curvature
    (`tb_superfluid_fraction`; for a uniform condensate the GPE value at any U) to 1e-4."""
    J = 0.01
    model = _square_model(2, J=J, U=0.01 * J)
    t = C._finish(model, "pair", [[0, 1], [2, 3]])
    mu = -4.0 * J + model.U
    kw = dict(tol=1e-11, max_iter=60000, mixing=0.5)
    st = C.cluster_gutzwiller(model, t, mu, 10, phi0=np.ones(4), **kw)
    assert st.converged and st.p_top.max() < 1e-6
    # the fixed-mu twist needs J (theta / L)^2 << U n: on L = 2 that means small twists (same theta -> 0 route)
    r = C.cluster_fs(model, t, st, 10, thetas=(0.06, 0.03), **kw)
    ref = lm.tb_superfluid_fraction([J], 1e-4, 2.0)[0]
    assert r["converged"]
    assert r["fs0"] == pytest.approx(ref, abs=1e-4)


# -- the square-lattice lobe tip ------------------------------------------------------------------------------------


def test_square_clusters():
    for fam, s, lam in (("single", 1, 0.0), ("2x2", 4, 0.5), ("plus", 5, 0.4), ("3x3", 9, 2 / 3)):
        cl = C.square_cluster(fam)
        assert cl.s == s and cl.lam == pytest.approx(lam)
        # every site has four neighbours: inside (twice the internal bonds) + outside = 4 s
        assert 2 * len(cl.internal) + cl.M.sum() == 4 * s


def test_single_site_tip_is_the_anchor():
    v, tol, _ = anchor("square_lattice_UJ_c_gutzwiller")
    r = C.MottStability(C.square_cluster("single"), 1, 6).tip()
    assert r["U_over_J"] == pytest.approx(v, abs=1e-6)
    assert r["mu_over_U"] == pytest.approx(math.sqrt(2) - 1, abs=1e-5)


def test_2x2_tip_between_anchors():
    qmc, _, _ = anchor("square_lattice_UJ_c_qmc")
    ss, _, _ = anchor("square_lattice_UJ_c_gutzwiller")
    r = C.MottStability(C.square_cluster("2x2"), 1, 3).tip()
    assert qmc < r["U_over_J"] < ss
    assert r["p_nmax"] < 1e-3


@pytest.mark.slow
def test_plus_tip_between_anchors_and_decreasing_with_lambda():
    """(U/J)_c(2 x 2) and (U/J)_c(plus) lie strictly between 16.74 and 23.31 and decrease with lambda
    (plus: 0.4, 2 x 2: 0.5)."""
    qmc, _, _ = anchor("square_lattice_UJ_c_qmc")
    ss, _, _ = anchor("square_lattice_UJ_c_gutzwiller")
    r22 = C.MottStability(C.square_cluster("2x2"), 1, 3).tip()["U_over_J"]
    rp = C.MottStability(C.square_cluster("plus"), 1, 3).tip()["U_over_J"]
    assert qmc < r22 < rp < ss


@pytest.mark.slow
def test_linear_stability_matches_selfconsistent_cluster():
    """The 2 x 2 boundary from linear stability: the self-consistent uniform cluster solution has phi = 0
    slightly inside the lobe (J = 0.98 J_c) and phi > 0 slightly outside (J = 1.02 J_c), at the tip's mu."""
    ms = C.MottStability(C.square_cluster("2x2"), 1, 3)
    tip = ms.tip()
    cl = C.square_cluster("2x2")
    phi_in, _, conv_in, _ = C.square_cluster_solve(cl, 0.98 * tip["J_c"], 1.0, tip["mu_over_U"], 3, max_iter=40000)
    phi_out, _, conv_out, _ = C.square_cluster_solve(cl, 1.02 * tip["J_c"], 1.0, tip["mu_over_U"], 3, max_iter=40000)
    assert np.abs(phi_in).max() < 1e-6
    assert np.abs(phi_out).max() > 1e-2


def test_cluster_variational_omega_at_convergence():
    """The variational Omega of the converged product of cluster states equals the solver's Omega (with the
    cross-bond correction), at zero and finite twist."""
    model = lm.build_model(12.0, 0.4, 4, 1, **FULL)
    t = C.make_tiling(model, "dimer")
    mu = model.U - 0.4 * model.diagnostics["F_k0_AA"]
    for theta in ((0.0, 0.0), (0.3, 0.0)):
        st = C.cluster_gutzwiller(model, t, mu, 5, theta=theta, tol=1e-12)
        assert st.converged
        assert C.cluster_variational_omega(model, t, st, 5) == pytest.approx(st.omega, abs=1e-9)
