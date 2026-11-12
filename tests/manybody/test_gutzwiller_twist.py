"""The twisted, batched single-site Gutzwiller solver (gutzwiller.py) and its gates. `n_max` is
passed explicitly everywhere.

Cross-checks against the discrete lattice GPE (`model.ground_state`) run on m = 22 (slow) and on the
(4, 1) cell only after asserting that its interlayer pair list has no duplicate pairs at cutoff 2 (small cells
can carry periodic-image duplicates; (4, 1) at cutoff 2 does not)."""

import dataclasses
import math

import numpy as np
import pytest

from twisted_bilayer_lattice import gutzwiller as gw
from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice._shared.engine.units import KAPPA
from twisted_bilayer_lattice.paper_model import M_R_from_Omega, paper_model

FULL = dict(nu_z_hz=1000.0, cutoff=2.0, interlayer_interaction=True, interband=True, interband_bonds=False)


def _no_duplicate_pairs(model) -> bool:
    c = model.cell.c
    inter = (model.bonds.rows // c) != (model.bonds.cols // c)
    key = model.bonds.rows[inter] * model.n_sites + model.bonds.cols[inter]
    return np.unique(key).size == key.size


@pytest.mark.parametrize("V0, Om", [(12.0, 0.3), (16.0, 0.0), (8.0, 0.6)])
def test_theta0_reproduces_engine(V0, Om):
    """At theta = 0 the new solver is the engine's `gutzwiller` (phi, n, Omega) to 1e-12, n_max explicit."""
    model = lm.build_model(V0, Om, 4, 1, **FULL)
    ref = lm.gutzwiller(V0, Om, model=model, mu_over_U=1.0, n_max=6)
    st = gw.solve(model, 1.0 * model.U, 6)
    assert st.converged and ref.converged
    np.testing.assert_allclose(st.phi, ref.phi.ravel(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(st.n, ref.n.ravel(), atol=1e-12, rtol=0)
    assert st.omega / model.n_sites == pytest.approx(ref.grand_potential, abs=1e-12)
    assert st.iterations == ref.iterations


def test_theta0_reproduces_engine_paper_model():
    cell = lm.supercell_sites(4, 1)
    model = paper_model(20.0, M_R_from_Omega(20.0, 0.6, m=4, n=1, cell=cell), m=4, n=1, cell=cell)
    ref = lm.gutzwiller(20.0, 0.0, model=model, n_max=6)
    st = gw.solve(model, model.U, 6)
    np.testing.assert_allclose(st.phi, ref.phi.ravel(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(st.n, ref.n.ravel(), atol=1e-12, rtol=0)
    assert st.omega / model.n_sites == pytest.approx(ref.grand_potential, abs=1e-12)


def test_batch_equals_loop():
    """Three models of one structure (V0 and Omega_R > 0 differ) in one batch = three single solves, 1e-12;
    also with a twist (complex arithmetic)."""
    models = [lm.build_model(V0, Om, 4, 1, **FULL) for V0, Om in ((10.0, 0.2), (12.0, 0.6), (14.0, 1.0))]
    mus = [mdl.U for mdl in models]
    for theta in ((0.0, 0.0), (0.3, 0.0)):
        batch = gw.solve_batch(models, mus, 6, theta=theta)
        for mdl, mu, b in zip(models, mus, batch):
            one = gw.solve(mdl, mu, 6, theta=theta)
            np.testing.assert_allclose(b.phi, one.phi, atol=1e-12, rtol=0)
            np.testing.assert_allclose(b.n, one.n, atol=1e-12, rtol=0)
            assert b.omega == pytest.approx(one.omega, abs=1e-12)
            assert b.converged == one.converged and b.iterations == one.iterations


def test_mott_state_fs_exactly_zero():
    """Paper's model at V0 = 20, M_R = 0, mu / U = 0.5: the n = 1 Mott state. From phi = 0 every twisted solve
    stays at phi = 0, so Omega(theta) = Omega(0) and f_s = 0 exactly; from the default start the map stops at
    |phi| ~ 1e-11 (below tol) and f_s is zero to the rounding of Omega."""
    model = paper_model(20.0, 0.0, m=4, n=1)
    mu = 0.5 * model.U
    base = gw.solve(model, mu, 6, phi0=np.zeros(model.n_sites))
    assert np.abs(base.phi).max() == 0.0
    np.testing.assert_allclose(base.n, 1.0, atol=1e-12)
    r = gw.gutzwiller_fs0(model, mu, 6, base=base)
    assert r["fs0"] == 0.0 and r["fs"][0.3] == 0.0 and r["fs"][0.15] == 0.0
    generic = gw.solve(model, mu, 6)
    assert generic.converged and np.abs(generic.phi).max() < 1e-10
    assert (
        abs(gw.gutzwiller_fs0(model, mu, 6, base=generic)["fs0"]) < 1e-10
    )  # rounding of Omega ~ 5, amplified by L^2 / (N kappa theta^2)


def test_uncoupled_layers_identical():
    """Omega_R = 0, delta = 0, no interlayer V: the two layers are identical copies (n, phi per layer to 1e-10;
    each layer uniform). With V_ij on the layers interact through their densities and the model, factorised
    in layer 1's frame, is not exactly mirror symmetric between the layers (the site lists are each other's
    mirror images), so no exact per-site identity holds there."""
    model = lm.build_model(12.0, 0.0, 4, 1, **dict(FULL, interlayer_interaction=False))
    st = gw.solve(model, 0.9 * model.U, 6)
    c = model.cell.c
    np.testing.assert_allclose(st.n[:c], st.n[c:], atol=1e-10)
    np.testing.assert_allclose(st.phi[:c], st.phi[c:], atol=1e-10)


def test_atomic_limit():
    """J = 0 (all bond amplitudes zero): n_i = ceil(mu_i / U) for mu_i = mu - eps_i not at a lobe edge, the
    grand potential per site (U/2) n (n - 1) - mu_i n exactly, and zero compressibility inside the plateaus."""
    base = paper_model(20.0, 0.3, m=4, n=1)
    model = dataclasses.replace(base, bonds=dataclasses.replace(base.bonds, amp=np.zeros_like(base.bonds.amp)))
    mu = 1.37 * model.U
    st = gw.solve(model, mu, 6, phi0=0.5 * np.ones(model.n_sites))
    mui = mu - model.onsite
    n_exp = np.maximum(np.ceil(mui / model.U), 0.0)
    np.testing.assert_allclose(st.n, n_exp, atol=1e-12)
    om = np.sum(0.5 * model.U * n_exp * (n_exp - 1) - mui * n_exp)
    assert st.omega == pytest.approx(om, abs=1e-12)
    assert gw.compressibility(model, st, 6, dmu_over_U=0.01) == 0.0


def test_starts_agree_deep_superfluid():
    """Deep in the superfluid (full model, V0 = 8, Omega_R = 0, mu / U = 1) all four starts converge to the
    same state: Omega spread below 1e-9."""
    model = lm.build_model(8.0, 0.0, 4, 1, **FULL)
    (best, info) = gw.solve_starts([model], model.U, 8)[0]
    assert all(conv for _, conv, _ in info["starts"].values())
    assert info["omega_spread"] < 1e-9
    assert set(info["starts"]) == set(gw.START_NAMES)


def test_twist_superfluid_positive_and_smooth():
    """In the superfluid f_s(0.3) and f_s(0.15) are positive and differ at O(theta^2) only; the quoted value is
    the theta^2 extrapolation."""
    model = lm.build_model(10.0, 0.2, 4, 1, **FULL)
    r = gw.gutzwiller_fs0(model, model.U - 0.2 * model.diagnostics["F_k0_AA"], 6)
    f3, f15 = r["fs"][0.3], r["fs"][0.15]
    assert r["converged"]
    assert f3 > 0.05 and f15 > 0.05
    assert abs(f3 - f15) < 0.05 * f15
    assert r["fs0"] == pytest.approx((4 * f15 - f3) / 3, abs=1e-15)


def test_pocket_start_is_localised():
    model = lm.build_model(12.0, 0.6, 22, 1)
    p = gw.pocket_start(model)
    assert 0 < np.count_nonzero(p) < model.n_sites // 4
    AA, _ = lm.moire_points(model.cell)
    for k in range(2):
        idx = model.cell.layers[k].lookup[0, 0] + k * model.cell.c  # the origin is AA, a site of both layers
        assert p[idx] == 0.7
    assert math.isclose(np.abs(p).max(), 0.7)


def _fs_fixed_n(model, filling, n_max, theta=0.3, **kw):
    s0 = gw.solve_filling(model, filling, n_max, **kw)
    st = gw.solve_filling(model, filling, n_max, theta=(theta, 0.0), phi0=s0.phi, **kw)
    E0, Et = s0.omega + s0.mu * s0.N, st.omega + st.mu * st.N
    return (Et - E0) * model.L**2 / (s0.N * KAPPA * theta * theta), s0, st


def test_solve_with_the_interband_shift_pinned():
    """`gutzwiller.solve` on the full model with the interband shift of the sites (m = 4, V0 = 12, Omega_R = 0.6),
    mu = U - Omega_R F_k0_AA (the zero of the paper's M_i), n_max = 6: a regression pin of the shifted model
    (Omega, N, max |phi|, the sum of the site shifts) that the lowest and the pocket start reach alike. Values
    measured with this code; any change of the shift, the map or the start shows here."""
    model = lm.build_model(12.0, 0.6, 4, 1, **FULL)
    plain = lm.build_model(12.0, 0.6, 4, 1, **dict(FULL, interband=False))
    shift = np.asarray(model.onsite) - np.asarray(plain.onsite)
    assert shift.sum() == pytest.approx(-1.005460797923, rel=1e-8)
    assert shift.max() <= 1e-12  # second-order shifts only lower the sites
    mu = model.U - 0.6 * model.diagnostics["F_k0_AA"]
    for phi0 in (None, gw.pocket_start(model)):
        st = gw.solve(model, mu, 6, tol=1e-12, max_iter=20000, phi0=phi0)
        assert st.converged
        assert st.omega == pytest.approx(-0.03687087989835, rel=1e-8)
        assert st.N == pytest.approx(1.0241893568683, rel=1e-8)
        assert np.abs(st.phi).max() == pytest.approx(0.71298208264, rel=1e-7)
        assert st.p_top.max() < 1e-5


@pytest.mark.parametrize("V0, Om", [(10.0, 0.2), (12.0, 0.3)])
def test_fixed_n_fs_route_agrees_with_fixed_mu_route(V0, Om):
    """The fixed-N twist energy (`gutzwiller_fs0_fixed_n`, one atom per site on the (4, 1) cell) and the fixed-mu
    route at the filling's mu give the same f_s(theta -> 0) where both are valid: the single-twist values differ
    at O(theta^2) (here 3e-4 at theta = 0.3), the extrapolated ones to 1e-6 (5e-8 measured)."""
    model = lm.build_model(V0, Om, 4, 1, **FULL)
    r = gw.gutzwiller_fs0_fixed_n(model, 1.0, 6, max_iter=20000, tol=1e-10)
    assert r["converged"]
    assert r["N"] == pytest.approx(model.cell.c, rel=1e-8)
    r_mu = gw.gutzwiller_fs0(model, r["mu"], 6, base=r["base"], max_iter=20000, tol=1e-10)
    assert r_mu["converged"]
    assert r["fs0"] > 0.05
    assert r["fs0"] == pytest.approx(r_mu["fs0"], abs=1e-6)
    assert r["fs0"] == pytest.approx((4 * r["fs"][0.15] - r["fs"][0.3]) / 3, abs=1e-15)
    assert abs(r["fs"][0.3] - r_mu["fs"][0.3]) < 1e-3


def test_fixed_n_fs_of_a_mott_state_is_zero():
    """Deep Mott state at one atom per site (the paper's model, V0 = 20, M_R = 0, U = 100 t): phi = 0, so no twist
    energy and f_s = 0 exactly."""
    pm = paper_model(20.0, 0.0, m=4, n=1)
    model = dataclasses.replace(pm, U=100.0 * pm.diagnostics["t"])
    r = gw.gutzwiller_fs0_fixed_n(model, 1.0, 4, max_iter=20000, tol=1e-10)
    assert r["converged"]
    assert np.abs(r["base"].phi).max() < 1e-6
    assert r["fs0"] == pytest.approx(0.0, abs=1e-9)


def test_variational_omega_at_convergence():
    """The variational Omega of the converged product state equals the map's Omega (to the solver tolerance),
    at zero and finite twist: the upper bound used for unconverged twisted solves is exact at convergence."""
    model = lm.build_model(10.0, 0.2, 4, 1, **FULL)
    for theta in ((0.0, 0.0), (0.3, 0.0)):
        st = gw.solve_batch([model], [model.U], 6, theta=theta, keep_coeffs=True)[0]
        assert st.converged
        assert gw.variational_omega(model, st) == pytest.approx(st.omega, abs=1e-9)


def test_plain_square_lattice_fs_u0():
    """Plain square lattice (the paper's model with M_R = 0, U = 0.01 t, one atom per site): f_s(theta -> 0) at
    fixed mu = -4t + U equals the exact non-interacting band curvature (`tb_superfluid_fraction` at a small
    twist; uniform state, so the GPE value at any U) to 1e-4."""
    pm = paper_model(12.0, 0.0, m=4, n=1)
    t = pm.diagnostics["t"]
    model = dataclasses.replace(pm, U=0.01 * t)
    mu = -4.0 * t + model.U
    base = gw.solve(model, mu, 12, phi0=np.ones(model.n_sites), max_iter=400000, tol=1e-11)
    assert base.converged and base.p_top.max() < 1e-6
    r = gw.gutzwiller_fs0(model, mu, 12, base=base, max_iter=400000, tol=1e-11)
    ref = lm.tb_superfluid_fraction([t], 1e-4, model.L, alpha=model.cell.geometry.alpha)[0]
    assert r["fs0"] == pytest.approx(ref, abs=1e-4)


def _u0_gate(V0, Om, m, Ufac, fixed_filling, n_max=12):
    J = lm.hopping_1d(V0, 2)
    U = Ufac * J[0]
    model = lm.build_model(V0, Om, m, 1, cutoff=2.0, U=U)
    assert _no_duplicate_pairs(model)
    r3 = lm.superfluid_fraction(V0, Om, m=m, n=1, filling=2.0, theta=0.3, cutoff=2.0, U=U)
    r15 = lm.superfluid_fraction(V0, Om, m=m, n=1, filling=2.0, theta=0.15, cutoff=2.0, U=U, gs0=r3["gs0"])
    f_gpe = (4 * r15["fs_energy"] - r3["fs_energy"]) / 3
    gs = r3["gs0"]
    phi = gs.phi.reshape(-1).numpy()
    kw = dict(max_iter=400000, tol=1e-11)
    if fixed_filling:
        base = gw.solve_filling(model, 2.0, n_max, phi0=phi, **kw)
    else:
        base = gw.solve(model, gs.mu, n_max, phi0=phi, **kw)
    assert base.converged and base.p_top.max() < 1e-6
    r = gw.gutzwiller_fs0(model, base.mu, n_max, base=base, **kw)
    E_gw = (base.omega + base.mu * base.N) / base.N
    return r["fs0"], f_gpe, E_gw, gs.E_per_N, base


@pytest.mark.slow
def test_fs_u0_vs_discrete_gpe_uncoupled_layers():
    """U = 0.01 J, one atom per site, Omega_R = 0 on the (4, 1) cell: Gutzwiller f_s(theta -> 0) at the GPE's mu
    against the discrete-GPE energy route (theta -> 0 from 0.3 and 0.15 as well) to 1e-4, and E/N to 1e-4
    relative (n_max = 12)."""
    f_gw, f_gpe, E_gw, E_gpe, _ = _u0_gate(12.0, 0.0, 4, 1e-2, fixed_filling=False)
    assert f_gw == pytest.approx(f_gpe, abs=1e-4)
    assert E_gw == pytest.approx(E_gpe, rel=1e-4)


@pytest.mark.slow
def test_u0_vs_discrete_gpe_experiment_cell():
    """The same U -> 0 gate on the experiment's m = 22 cell (Omega_R = 0: two uniform layers), f_s to 1e-4 and
    E/N to 1e-4 relative."""
    f_gw, f_gpe, E_gw, E_gpe, _ = _u0_gate(12.0, 0.0, 22, 1e-2, fixed_filling=False)
    assert f_gw == pytest.approx(f_gpe, abs=1e-4)
    assert E_gw == pytest.approx(E_gpe, rel=1e-4)


@pytest.mark.slow
def test_fs_u0_vs_discrete_gpe_coupled_nmax16():
    """Omega_R = 0.05, U = 0.01 J, one atom per site at fixed filling, n_max = 16: f_s(theta -> 0) agrees with the
    discrete GPE to 5e-5 (2e-5 measured; n_max = 12 differs by 6e-4). The residual is the truncation of the
    occupation basis in the moire pockets, P(n_max = 12) = 7e-5 (not the pockets themselves)."""
    f_gw, f_gpe, _, _, _ = _u0_gate(12.0, 0.05, 4, 1e-2, fixed_filling=True, n_max=16)
    assert f_gw == pytest.approx(f_gpe, abs=5e-5)


@pytest.mark.slow
def test_fixed_filling_energy_vs_discrete_gpe_coupled():
    """Omega_R = 0.05, U = 0.01 J, one atom per site at fixed filling (solve_filling), n_max = 12: the energy per
    atom approaches the discrete GPE to 1e-4 relative. (f_s is not pinned here: at n_max = 12 it differs from
    the GPE by 6e-4, at n_max = 16 by 2e-5, so the difference is the truncation of the occupation basis in the
    moire pockets, P(n_max = 12) = 7e-5; the U -> 0 comparison of f_s is made on uniform states above.)"""
    J = lm.hopping_1d(12.0, 2)
    U = 1e-2 * J[0]
    model = lm.build_model(12.0, 0.05, 4, 1, cutoff=2.0, U=U)
    assert _no_duplicate_pairs(model)
    gs = lm.ground_state(12.0, 0.05, 4, 1, filling=2.0, cutoff=2.0, U=U)
    st = gw.solve_filling(model, 2.0, 12, phi0=gs.phi.reshape(-1).numpy(), max_iter=400000, tol=1e-11)
    assert st.converged
    assert (st.omega + st.mu * st.N) / st.N == pytest.approx(gs.E_per_N, rel=1e-4)


def test_fixed_n_route_retries_a_metastable_base_state(monkeypatch):
    """A twisted energy below the untwisted one means the untwisted state was not the minimum: the fixed-N route
    solves it again from the lowest twisted state (a quarter of the damping the second time) and quotes the twist
    energy of the lower state. Stubbed solves: the first base sits at E = 0, the twisted solves find E = -1 (a
    negative f_s), the retried base sits at E = -1, its twists at E = -1 + 0.5 theta^2."""
    from types import SimpleNamespace

    model = SimpleNamespace(cell=SimpleNamespace(c=4), L=1.0)
    calls = []

    def state(E, tag):
        return SimpleNamespace(omega=E, mu=0.0, N=4.0, converged=True, phi=tag)

    def fake(model_, filling, n_max, theta=(0.0, 0.0), phi0=None, **kw):
        calls.append((theta, phi0, kw.get("mixing")))
        if theta == (0.0, 0.0):
            return state(-1.0, "lower")  # only reached by the retry
        th = theta[0]
        return state(-1.0 + (0.5 * th * th if phi0 == "lower" else 0.0), "twisted")

    monkeypatch.setattr(gw, "solve_filling", fake)
    monkeypatch.setattr(gw, "default_mixing", lambda m: 0.8)
    r = gw.gutzwiller_fs0_fixed_n(model, 1.0, 6, base=state(0.0, "metastable"))
    assert r["retries"] == 1 and r["base"].omega == -1.0 and r["converged"]
    # f(theta) = (0.5 theta^2) L^2 / (N kappa theta^2) = 0.5 / (4 kappa): the same at both twists, so the
    # extrapolation returns it
    assert r["fs"][0.3] == pytest.approx(0.5 / (4.0 * gw.KAPPA)) and r["fs0"] == pytest.approx(r["fs"][0.3])
    retry_call = [c for c in calls if c[0] == (0.0, 0.0)]
    assert retry_call == [((0.0, 0.0), "twisted", None)]  # from a twisted state, first retry at the default damping
