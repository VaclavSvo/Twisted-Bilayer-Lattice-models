"""`solve_batch(..., eig="fast")` (the certified tridiagonal kernel) against the default dense `eigh` path on small
cells: the same fixed point, iteration counts and grand potential, with and without a twist, in the full two-layer
model and in the paper's model (including a Mott point, where the site matrices become diagonal)."""

import numpy as np
import pytest

from twisted_bilayer_lattice import fast_kernels as fk
from twisted_bilayer_lattice import gutzwiller as gw
from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice.paper_model import M_R_from_Omega, paper_model

FULL = dict(nu_z_hz=1000.0, cutoff=2.0, interlayer_interaction=True, interband=True, interband_bonds=False)


def _same(a: gw.GWState, b: gw.GWState, tol: float = 1e-9):
    assert a.converged == b.converged
    assert a.iterations == b.iterations
    np.testing.assert_allclose(a.phi, b.phi, atol=tol, rtol=0)
    np.testing.assert_allclose(a.n, b.n, atol=tol, rtol=0)
    np.testing.assert_allclose(a.p_top, b.p_top, atol=tol, rtol=0)
    assert a.omega == pytest.approx(b.omega, abs=tol * max(1.0, abs(b.omega)))


@pytest.mark.parametrize("theta", [(0.0, 0.0), (0.3, 0.0)])
def test_fast_equals_exact_full_model(theta):
    models = [lm.build_model(V0, Om, 4, 1, **FULL) for V0, Om in ((8.0, 0.6), (12.0, 0.3), (16.0, 0.2))]
    mus = [mdl.U for mdl in models]
    stats = {}
    ref = gw.solve_batch(models, mus, 6, theta=theta)
    fast = gw.solve_batch(models, mus, 6, theta=theta, eig="fast", kernel_stats=stats)
    for r, f in zip(ref, fast):
        _same(f, r)
    assert stats["sites"] > 0 and stats["rqi_certified"] > 0, stats


def test_fast_equals_exact_paper_model_mott():
    cell = lm.supercell_sites(4, 1)
    for V0, Om, mu_over_U in ((20.0, 0.6, 0.95), (8.0, 0.2, 0.95)):
        model = paper_model(V0, M_R_from_Omega(V0, Om, m=4, n=1, cell=cell), m=4, n=1, cell=cell)
        ref = gw.solve(model, mu_over_U * model.U, 6)
        fast = gw.solve(model, mu_over_U * model.U, 6, eig="fast")
        _same(fast, ref)


def test_fast_passes_through_fs_batch():
    model = lm.build_model(10.0, 0.4, 4, 1, **FULL)
    base = gw.solve(model, model.U, 6)
    ref = gw.fs_batch([model], [base], 6)[0]
    fast = gw.fs_batch([model], [base], 6, eig="fast")[0]
    assert fast["converged"] == ref["converged"] and fast["twist_converged"] == ref["twist_converged"]
    for route in ("fs", "fs_upper"):
        assert set(fast[route]) == set(ref[route])
        for th, val in ref[route].items():
            assert fast[route][th] == pytest.approx(val, abs=1e-8), (route, th)


def test_eig_default_and_unknown():
    model = lm.build_model(12.0, 0.3, 4, 1, **FULL)
    a = gw.solve(model, model.U, 6)
    b = gw.solve(model, model.U, 6, eig="exact")
    np.testing.assert_array_equal(a.phi, b.phi)
    with pytest.raises(ValueError):
        gw.solve(model, model.U, 6, eig="lanczos")


def test_kernel_stats_keys():
    stats = {}
    model = lm.build_model(12.0, 0.3, 4, 1, **FULL)
    gw.solve(model, model.U, 6, eig="fast", kernel_stats=stats)
    assert set(fk.new_stats()) <= set(stats)
    assert stats["cold"] > 0  # the first iteration is solved densely
