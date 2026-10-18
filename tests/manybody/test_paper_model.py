"""manybody.paper_model: the paper's one-layer model (Supplement Eqs. 12-13) as a one-layer LatticeModel,
and the solvers of engine.lattice_model on it.

Pinned: c = m^2 + n^2 sites and coordination 4 (exact); M_i against Eq. 12 evaluated directly at three sites
and the exact rational cos theta, sin theta; t and U equal `parameters.paper_parameters`; the atomic limit of
the Gutzwiller solve (J -> 0: n = 1 at 0 < mu/U < 1, phi = 0); the one-layer discrete GPE at M_R = 0 is the
uniform state with mu = -4t + U n (exact); the k = 0 reading of M_R equals Omega_R (S_sum_AA - S_sum_AB)/2
of `parameters.bh_parameters`.
"""

import math

import numpy as np
import pytest

from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.manybody import paper_model as pm
from twisted_bilayer.manybody import parameters as P


@pytest.fixture(scope="module")
def cell4():
    return lm.supercell_sites(4, 1)


@pytest.fixture(scope="module")
def cell22():
    return lm.supercell_sites(22, 1)


@pytest.mark.parametrize("m,n", [(4, 1), (22, 1)])
def test_sites_coordination_t_and_U(m, n, cell4, cell22):
    cell = cell4 if m == 4 else cell22
    model = pm.paper_model(8.0, 0.3, m, n, cell=cell)
    c = m * m + n * n
    assert model.n_layers == 1 and model.n_sites == c and model.variant == "paper" and not model.coupled
    assert model.mu_over_U == 1.0 and model.g2d is None and model.g2d_source == "paper"
    H = model.hamiltonian().toarray()
    assert H.shape == (c, c) and np.array_equal(H, H.T)
    off = H - np.diag(np.diag(H))
    p = P.paper_parameters(8.0)
    assert np.all((off != 0.0).sum(axis=1) == 4)
    assert np.all(np.isclose(off[off != 0.0], -p["t"], rtol=0, atol=1e-15))
    assert model.J[0] == p["t"] and model.U == p["U"] and model.diagnostics["U_over_t"] == p["U_over_t"]
    # S empty, no interlayer interaction, no Omega_R: the coupling is the site offset
    assert model.S.nnz == 0 and model.V_pairs is None and model.Omega_R == 0.0


def test_site_offsets_formula(cell22):
    """M_i = M_R [sin^2(pi (i_x cos theta + i_y sin theta)) + sin^2(pi (i_y cos theta - i_x sin theta))]
    with theta = 2 atan(n/m), at the origin (AA: 0), at three explicit sites, and against the formula
    evaluated with floating-point cos/sin at every site; the maximum sits near 2 M_R (AB)."""
    M_R = 0.37
    model = pm.paper_model(8.0, M_R, 22, 1, cell=cell22)
    theta = 2.0 * math.atan(1.0 / 22.0)
    ij = cell22.layers[0].ij
    assert model.onsite[cell22.origin_index(0)] == 0.0
    for i, j in ((1, 0), (3, 2), (-7, 4)):
        k = np.flatnonzero((ij[:, 0] == i) & (ij[:, 1] == j))
        assert k.size == 1
        want = M_R * (
            math.sin(math.pi * (i * math.cos(theta) + j * math.sin(theta))) ** 2
            + math.sin(math.pi * (j * math.cos(theta) - i * math.sin(theta))) ** 2
        )
        assert model.onsite[k[0]] == pytest.approx(want, abs=1e-12)
    assert np.allclose(model.onsite, pm.paper_site_offsets_formula(ij[:, 0], ij[:, 1], theta, M_R), atol=1e-10)
    assert 0.0 <= model.onsite.min() and model.onsite.max() == pytest.approx(2.0 * M_R, rel=0.01)
    # periodic on the supercell: the site shifted by a box translation has the same offset (exact rationals)
    assert np.allclose(pm.paper_site_offsets(cell22, M_R), model.onsite)
    assert pm.paper_site_offsets(cell22, 2.0 * M_R) == pytest.approx(2.0 * model.onsite)


def test_M_R_rules(cell22):
    b = P.bh_parameters(8.0, Omega_R=0.24, cutoff=2.0, cell=cell22)
    k0 = pm.M_R_from_Omega(8.0, 0.24, cell=cell22)
    assert k0 == pytest.approx(0.5 * 0.24 * (b["S_sum_AA"] - b["S_sum_AB"]), rel=1e-12)
    assert 0.3 < k0 / 0.24 < 0.4  # ~0.35 Omega_R at 8 E_R
    assert pm.M_R_from_Omega(8.0, 0.0, cell=cell22) == 0.0
    assert pm.M_R_from_Omega(8.0, 0.24, rule="omega") == 0.24
    assert pm.M_R_from_Omega(8.0, 0.24, rule="eq8") == 4.0
    with pytest.raises(ValueError):
        pm.M_R_from_Omega(8.0, 0.24, rule="nonsense")


def test_discrete_gpe_on_one_layer(cell4):
    """M_R = 0: the uniform state, mu = -4t + U n exactly, f_s = t/kappa (1 + O(theta^2)); M_R > 0: a
    minimum with density at AA above AB."""
    model = pm.paper_model(8.0, 0.0, 4, 1, cell=cell4)
    gs = lm.ground_state(8.0, 0.0, filling=1.0, model=model)
    assert gs.residual < 1e-10 and gs.phi.shape == (1, 17) and gs.layer_fractions == pytest.approx([1.0])
    assert gs.mu == pytest.approx(-4.0 * model.J[0] + model.U, abs=1e-12)
    assert gs.participation_ratio == pytest.approx(1.0, abs=1e-10) and gs.hessian_min > -1e-11
    r = lm.superfluid_fraction(8.0, 0.0, filling=1.0, gs0=gs)
    assert r["fs_energy"] == pytest.approx(model.J[0] / lm.KAPPA, rel=2e-3)
    assert r["fs_current"] == pytest.approx(r["fs_energy"], rel=2e-3)
    gs2 = lm.ground_state(8.0, 0.0, filling=1.0, model=pm.paper_model(8.0, 0.3, 4, 1, cell=cell4))
    assert gs2.residual < 1e-10 and gs2.hessian_min > -1e-11 and gs2.AA_AB_ratio > 1.0
    assert lm.lattice_bands(8.0, 0.0, model=model, n_bands=2).energies[0] == pytest.approx(-4.0 * model.J[0], abs=1e-12)


def test_gutzwiller_on_paper_model(cell4):
    """Deep lattice (V0 = 20: U/t = 228, M_R = 0): the n = 1 Mott state at mu/U = 0.5, and at the paper's
    mu/U = 1 (between the n = 1 and n = 2 lobes, where only J = 0 is degenerate) a superfluid with 1 < n < 2.
    Shallow (V0 = 6, U/t = 4): superfluid at mu/U = 1. The model's own mu/U is used when neither mu nor
    filling is given. With M_R > 0 the sites with the largest offset empty first."""
    deep = pm.paper_model(20.0, 0.0, 4, 1, cell=cell4)
    mi = lm.gutzwiller(20.0, 0.0, model=deep, mu_over_U=0.5)
    assert mi.converged and mi.n.shape == (1, 17) and mi.n == pytest.approx(np.ones((1, 17)), abs=1e-10)
    assert np.abs(mi.phi).max() < 1e-8 and mi.compressibility == pytest.approx(0.0, abs=1e-8)
    sf = lm.gutzwiller(20.0, 0.0, model=deep)  # mu/U = 1 from the model
    assert sf.converged and sf.mu == pytest.approx(deep.U) and 1.0 < sf.filling < 2.0 and sf.condensate_fraction > 0.1
    assert lm.gutzwiller(6.0, 0.0, model=pm.paper_model(6.0, 0.0, 4, 1, cell=cell4)).condensate_fraction > 0.5
    with pytest.raises(ValueError):
        lm.gutzwiller(20.0, 0.0, model=deep, mu_over_U=0.5, filling=1.0)
    with pytest.raises(ValueError):
        lm.gutzwiller(20.0, 0.0, model=lm.build_model(8.0, 0.0, 4, 1, cell=cell4))  # no mu/U of its own
    off = pm.paper_model(20.0, 0.3, 4, 1, cell=cell4)
    g = lm.gutzwiller(20.0, 0.0, model=off, mu_over_U=0.5)
    assert g.converged and g.n.min() < 0.5 and g.n.max() == pytest.approx(1.0, abs=1e-6)
    assert g.n[0, np.argmax(off.onsite)] < g.n[0, cell4.origin_index(0)]
