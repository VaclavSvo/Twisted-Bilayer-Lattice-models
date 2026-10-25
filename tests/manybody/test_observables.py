"""Common many-body observables (manybody/observables.py): the momentum distribution, the visibilities and
the phase label."""

import math

import numpy as np
import pytest

from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.manybody import observables as ob
from twisted_bilayer.manybody.paper_model import paper_model


def _square(m=22, n=1, M_R=0.0, V0=12.0):
    """The paper's one-layer model with M_R = 0: a plain square lattice of c sites (lattice 1 of the cell)."""
    return paper_model(V0, M_R, m=m, n=n)


def _result(model, phi, n):
    nl, c = model.n_layers, model.cell.c
    return ob.ManyBodyResult(
        n=np.asarray(n, dtype=np.float64).reshape(nl, c),
        var_n=np.zeros((nl, c)),
        phi=np.asarray(phi, dtype=np.complex128).reshape(nl, c),
        mu=0.0,
        filling=float(np.sum(n)) / c,
        grand_potential=0.0,
        condensate_fraction=float(np.sum(np.abs(phi) ** 2)) / max(float(np.sum(n)), 1e-300),
        converged=True,
        method="test",
    )


def test_uniform_condensate_visibility_one():
    """A uniform coherent phi (n = |phi|^2) on the plain square lattice: S(k) peaked at the reciprocal vectors,
    V_p -> 1 (1 - O(1/c) from the finite cell's off-lattice sum at the 45-deg partner)."""
    model = _square()
    c = model.cell.c
    res = _result(model, np.ones(c), np.ones(c))
    pk, pp = ob.primary_vectors(model, 0)
    S_peak = ob.structure_factor(model, res, pk, 0)
    np.testing.assert_allclose(S_peak, c * c, rtol=1e-10)  # every site in phase at a Bragg vector
    v = ob.visibilities(model, res)
    assert v["primary"][0] > 0.99
    assert v["primary_caption"][0] > 0.9


def test_mott_state_zero_visibility():
    """phi = 0 (single-site Mott state): S(k) = N at every k, so every visibility on S is exactly 0; the
    perturbative short-range coherence gives a finite V_p, positive (peak at k1, where cos(k1 . d) = 1)."""
    model = _square()
    c = model.cell.c
    res = _result(model, np.zeros(c), np.ones(c))
    v = ob.visibilities(model, res)
    assert v["primary"][0] == 0.0
    assert v["moire"][0] == 0.0
    # nearest-neighbour coherence 2 |t| n (n + 1) / U = 4 t / U at n = 1;
    # S(k) = c [1 + a sum_{d in nn} cos(k.d)]: 1 + 4a at k1 = 2 pi e1, 1 + 4a cos(sqrt2 pi) at its 45-deg partner
    t, U = model.diagnostics["t"], model.U
    a = 4.0 * t / U
    Sp, Spp = 1 + 4 * a, 1 + 4 * a * math.cos(math.sqrt(2) * math.pi)
    assert v["primary_pert"][0] == pytest.approx((Sp - Spp) / (Sp + Spp), rel=1e-10)
    assert v["primary_pert"][0] > 0


def test_structure_factor_parseval():
    """Sum of S(k) over the c allowed quasi-momenta of the layer = c N exactly (orthogonality of the plane
    waves on the periodic cell), with and without connected correlations; with the Wannier envelope summed
    over the reciprocal shifts as well, (1 / c) sum_k sum_G n(k + G) = N (orthonormal Wannier functions)."""
    model = _square(m=4, n=1, V0=12.0)
    c = model.cell.c
    rng = np.random.default_rng(3)
    phi = rng.normal(size=c) + 1j * rng.normal(size=c)
    n = np.abs(phi) ** 2 + rng.uniform(0.1, 0.5, c)
    res = _result(model, phi, n)
    # a few connected entries (Hermitian)
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 0, 3, 2])
    vals = np.array([0.1, 0.1, -0.05, -0.05])
    res.rho_connected = (rows, cols, vals)
    ly = model.cell.layers[0]
    # allowed quasi-momenta q (layer coordinates): q . t in 2 pi Z for the box translations t = columns of A
    m_, n_ = model.cell.geometry.m, model.cell.geometry.n
    A = np.array([[m_, -n_], [n_, m_]], dtype=np.float64)  # (m, n) and (-n, m) in layer-1 coordinates
    Q = []
    for z1 in range(-c, c):
        for z2 in range(-c, c):
            q = 2 * math.pi * np.linalg.solve(A.T, [z1, z2])
            Q.append(np.mod(q, 2 * math.pi))
    Q = np.unique(np.round(np.array(Q), 10), axis=0)
    assert Q.shape[0] == c
    ax = ly.axes[0]
    k_box = Q[:, :1] * ax[0][None, :] + Q[:, 1:] * ax[1][None, :]
    S = ob.structure_factor(model, res, k_box, 0)
    assert float(S.sum()) == pytest.approx(c * float(n.sum()), rel=1e-10)
    # with the envelope: sum over reciprocal shifts G = 2 pi (a, b), |a|, |b| <= 8
    tot = 0.0
    shifts = np.arange(-8, 9)
    for qa in Q:
        G = 2 * math.pi * np.array([[a, b] for a in shifts for b in shifts], dtype=np.float64)
        kk = (qa[None, :] + G) @ ax  # layer coordinates -> box frame
        tot += float(ob.momentum_distribution(model, res, kk, 0).sum())
    assert tot / c == pytest.approx(float(n.sum()), rel=1e-8)


def test_wannier_ft_norm():
    """sum_a |w1(q + 2 pi a)|^2 = 1 for every q (orthonormal Wannier functions)."""
    for V0 in (8.0, 16.0):
        for q in (0.0, 0.7, 2.0):
            s = np.sum(ob.wannier_ft_1d(V0, q + 2 * math.pi * np.arange(-12, 13)) ** 2)
            assert s == pytest.approx(1.0, abs=1e-9)


def test_moire_vector_length():
    """|G_1 - G_2| = 2 pi / lambda_mo, lambda_mo = a / (2 sin(theta / 2)), and it is a reciprocal vector of
    the box (2 pi / L times integers)."""
    model = lm.build_model(12.0, 0.3, 22, 1)
    mk, _ = ob.moire_vectors(model)
    lam = 1.0 / (2.0 * math.sin(0.5 * model.cell.geometry.theta))
    assert np.linalg.norm(mk[0]) == pytest.approx(2 * math.pi / lam, rel=1e-12)
    z = mk[0] * model.L / (2 * math.pi)
    np.testing.assert_allclose(z, np.rint(z), atol=1e-9)


def test_phase_labels():
    model = _square(m=4)
    c = model.cell.c
    base = _result(model, np.zeros(c), np.ones(c))
    base.fs = 0.0
    assert ob.phase_label(base)[0] == "MI"
    r = _result(model, np.zeros(c), np.r_[np.ones(c - 1), 0.0])
    r.fs = 0.0
    assert ob.phase_label(r)[0] == "I"
    r = _result(model, np.r_[0.01, np.zeros(c - 1)], np.ones(c))
    r.fs = 0.0
    assert ob.phase_label(r)[0] == "SF-II"
    r.fs = 0.5
    assert ob.phase_label(r)[0] == "SF"
    r.converged = False
    assert ob.phase_label(r)[0] == "?"
    # excluding the one superfluid site turns SF-II into MI
    r = _result(model, np.r_[0.01, np.zeros(c - 1)], np.ones(c))
    r.fs = 0.0
    ex = np.zeros(c, dtype=bool)
    ex[0] = True
    lab, info = ob.phase_label(r, exclude=ex)
    assert lab == "MI" and info["n_excluded"] == 1


def test_degenerate_sites_paper_model():
    """At mu / U = 1 with M_R > 0 only the sites near AA (M_i ~ 0) are within 5 % of the lobe-touching point."""
    model = paper_model(23.0, 0.15)
    deg = ob.degenerate_sites(model, model.U, window=0.05)
    assert deg[model.cell.origin_index(0)]
    assert 0 < deg.sum() < model.n_sites // 4


def test_empty_state_is_not_a_mott_insulator():
    """The vacuum (every <n_i> = 0, phi = 0, f_s = 0) gets its own label, not MI."""
    model = _square(m=4)
    c = model.cell.c
    r = _result(model, np.zeros(c), np.zeros(c))
    r.fs = 0.0
    assert ob.phase_label(r)[0] == "empty"
    r = _result(model, np.zeros(c), np.r_[1.0, np.zeros(c - 1)])  # one occupied site: contrast 1, insulator
    r.fs = 0.0
    assert ob.phase_label(r)[0] == "I"
