"""engine.lattice_model: the two-layer tight-binding (Bose-Hubbard) model and its
mean-field and Gutzwiller ground states.

Pinned numbers and their sources:
  * c = m^2 + n^2 sites per layer: index of the box superlattice (exact).
  * J1, J2 from `bands.hopping_1d` (themselves pinned to the exact
    lowest-band values in test_bands.py); coordination 4 + 4 per site (separable lattice).
  * S = 1 for the co-located AA pair (s(0) = 1); the AB value s(a/2)^2 and the
    moire depth factor 1 - s(a/2)^2 = 0.92 at V0 = 8 (exact 1D Wannier overlap).
  * U -> 0 superfluid fraction: the analytic tight-binding band at finite twist
    (`tb_superfluid_fraction`, derivation in its docstring); its q -> 0 limit
    sum_d d^2 J_d / kappa; with J3 kept it reproduces the exact band curvature
    m/m* = 0.2806 (bands.effective_mass_ratio, test_bands.py) to 1e-3.
  * Gutzwiller n = 1 lobe tip U/zJ = 3 + 2 sqrt 2 = 5.83 at mu/U = sqrt 2 - 1
    (the standard single-site mean-field result: U/zJ = 5.83, i.e. U/J = 23.3).
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer.engine import bands
from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.engine.units import KAPPA

V0 = 8.0


@pytest.fixture(scope="module")
def cell22():
    return lm.supercell_sites(22, 1)


@pytest.fixture(scope="module")
def interlayer22(cell22):
    return lm.build_interlayer(cell22.layers[0], cell22.layers[1], V0, cell22.geometry)


# -- sites ------------------------------------------------------------------------------


@pytest.mark.parametrize("m,n", [(4, 1), (22, 1), (21, 1), (5, 2)])
def test_site_counts_box_and_closure(m, n):
    cell = lm.supercell_sites(m, n)
    c = m * m + n * n
    L = math.sqrt(c)
    assert cell.c == c and cell.positions.shape == (2, c, 2)
    for layer, ly in enumerate(cell.layers):
        P = ly.positions
        assert np.all(P >= -L / 2 - 1e-12) and np.all(P < L / 2 - 1e-12 + 1e-9)
        assert len({tuple(x) for x in ly.XY}) == c
        # lattice points of the layer: position = R(-+alpha) (i, j)
        a = cell.geometry.alpha * (-1 if layer == 0 else 1)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        assert np.allclose(ly.ij @ R.T, P, atol=1e-12)
        # axes: unit, orthogonal, rotated by -+alpha
        ax = ly.axes[0]
        assert np.allclose(ax @ ax.T, np.eye(2), atol=1e-14)
        assert np.allclose(ax, R.T, atol=1e-14)
        # closure: every site shifted by +-L e_x, +-L e_y is again a site (a permutation)
        for shift in ([L, 0], [-L, 0], [0, L], [0, -L]):
            idx = ly.index(np.rint((P + np.array(shift)) * L).astype(np.int64))
            assert np.array_equal(np.sort(idx), np.arange(c))
    assert np.allclose(cell.layers[0].positions[cell.origin_index(0)], 0.0)
    assert np.allclose(cell.layers[1].positions[cell.origin_index(1)], 0.0)


# -- hopping -----------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4, 22])
def test_hopping_symmetric_and_coordination(m):
    cell = lm.supercell_sites(m, 1)
    J = bands.hopping_1d(V0, 2)
    for ly in cell.layers:
        H = lm.build_hopping(ly, J, cell.geometry).toarray()
        assert np.array_equal(H, H.T)
        assert np.all(np.diag(H) == 0.0)
        assert np.all(np.isclose(H, -J[0], rtol=0, atol=1e-15).sum(axis=1) == 4)
        assert np.all(np.isclose(H, -J[1], rtol=0, atol=1e-15).sum(axis=1) == 4)
        assert np.allclose(H.sum(axis=1), -4.0 * (J[0] + J[1]), atol=1e-15)
    # bond vectors lie along the layer's own axes (no diagonal hopping)
    b = lm._hopping_bonds(cell.layers[0], J)
    proj = b.disp @ cell.layers[0].axes[0].T
    assert np.all(np.isclose(np.abs(proj), 0.0, atol=1e-12).sum(axis=1) == 1)
    assert set(np.round(np.abs(proj).max(axis=1), 12)) == {1.0, 2.0}


def test_twisted_hopping_is_hermitian_peierls():
    cell = lm.supercell_sites(4, 1)
    J = bands.hopping_1d(V0, 2)
    H0 = lm.build_hopping(cell.layers[0], J).toarray()
    Ht = lm.build_hopping(cell.layers[0], J, theta=(0.3, -0.2)).toarray()
    assert np.allclose(Ht, Ht.conj().T, atol=1e-15)
    assert np.allclose(np.abs(Ht), np.abs(H0), atol=1e-15)


# -- interlayer overlap --------------------------------------------------------------------


def test_interlayer_AA_pair_and_AB_value(cell22, interlayer22):
    S, diag, (i, j, delta, _) = interlayer22
    o1, o2 = cell22.origin_index(0), cell22.origin_index(1)
    assert S[o1, o2] == pytest.approx(1.0, abs=1e-14)  # s(0)^2 = 1
    assert diag["S_eff_AA"] == pytest.approx(1.0, abs=1e-14)
    assert np.all(S.data <= 1.0 + 1e-14)
    # S is the product of the exact 1D overlaps in lattice 1's axes
    dp = delta @ cell22.layers[0].axes[0].T
    vals = np.asarray(S[i, j]).ravel()
    k = np.argmin(np.abs(np.abs(dp) - 0.5).max(axis=1))  # the most AB-like pair
    assert np.abs(np.abs(dp[k]) - 0.5).max() < 0.05
    assert vals[k] == pytest.approx(bands.overlap_1d(V0, dp[k, 0]) * bands.overlap_1d(V0, dp[k, 1]), abs=1e-12)
    s_half_sq = bands.overlap_1d(V0, 0.5) ** 2
    assert vals[k] == pytest.approx(s_half_sq, abs=0.02)
    # exact 1D overlap: s(a/2) = 0.286 and the depth factor 1 - s(a/2)^2 = 0.92 at V0 = 8
    assert diag["s_half_sq"] == pytest.approx(0.286**2, abs=5e-4)
    assert 1.0 - diag["s_half_sq"] == pytest.approx(0.92, abs=0.005)
    # on the discrete sites the most AB-like site has S_eff a bit above s(a/2)^2
    assert s_half_sq < diag["S_eff_AB"] < s_half_sq + 0.03
    assert 0.88 < diag["depth_one"] < 1.0 - s_half_sq
    # ... and it sits at an AB point of the moire pattern
    _, AB = lm.moire_points(cell22)
    p = cell22.layers[0].positions[diag["AB_site"]]
    dist = AB - p
    dist -= cell22.L * np.round(dist / cell22.L)
    assert np.hypot(dist[:, 0], dist[:, 1]).min() < 1.0


def test_cutoff_and_k0_coupling(cell22):
    """The coupling of a k = 0 envelope at lattice-1 site i is the row sum
    F_i = sum_j S_ij ~ prod_axes sum_n s(n + u) at local stacking shift u. The
    one-neighbour cutoff (s = 0 for |d| >= 1) keeps s(1 - u) but drops s(1 + u), an
    error linear in u (s'(1) = -0.153): F is off by ~3 % against the converged
    sum, while cutoff = 2 is within ~1e-3. At AB, F = (2 s(1/2))^2 with cutoff
    1 and (sum_n s(n + 1/2))^2 = 0.309 converged -- the band-bottom moire
    depth is 1 - F(AB) ~ 0.69, not the one-neighbour 1 - s(1/2)^2 = 0.92."""
    d = {cut: lm.build_interlayer(cell22.layers[0], cell22.layers[1], V0, cutoff=cut)[1] for cut in (1.0, 2.0, 3.0)}
    err1 = np.abs(d[1.0]["F_k0"] - d[3.0]["F_k0"]).max()
    err2 = np.abs(d[2.0]["F_k0"] - d[3.0]["F_k0"]).max()
    assert 0.01 < err1 < 0.05 and err2 < 2e-3
    half = bands.overlap_1d(V0, 0.5)
    lattice_sum = sum(bands.overlap_1d(V0, k + 0.5) for k in range(-6, 6)) ** 2
    assert d[1.0]["F_k0_AB"] == pytest.approx((2 * half) ** 2, abs=0.01)
    assert d[3.0]["F_k0_AB"] == pytest.approx(lattice_sum, abs=0.01)
    assert d[2.0]["S_dropped_max"] < 1e-3 < d[1.0]["S_dropped_max"] < 0.02


def test_overlap_frame_and_layer_symmetry():
    """The continuum model is symmetric under (y -> -y, layer 1 <-> layer 2).
    Factorising S along lattice 1's axes (frame="layer1") breaks it at the
    1e-3 level (theta = 28 deg at m = 4); along the bisector (frame="mid") it
    is exact, so the ground state is exactly layer balanced."""
    sym = lm.ground_state(V0, 1.0, 4, 1, frame="mid")
    asym = lm.ground_state(V0, 1.0, 4, 1, frame="layer1")
    assert sym.layer_fractions == pytest.approx([0.5, 0.5], abs=1e-12)
    assert 1e-5 < abs(asym.layer_fractions[0] - 0.5) < 5e-3


def test_interlayer_interaction_is_U_at_AA(cell22):
    """V_ij = g2d I2(dx') I2(dy') with I2(0) = I4, so V = U for the AA pair."""
    assert lm.density_overlap_1d(V0, 0.0) == pytest.approx(bands.I4_1d(V0), rel=1e-12)
    model = lm.build_model(V0, 0.24, 22, 1, interlayer_interaction=True, cell=cell22)
    i, j, V = model.V_pairs
    o = (i == cell22.origin_index(0)) & (j == cell22.origin_index(1))
    assert V[o][0] == pytest.approx(model.U, rel=1e-10)
    assert np.all(V > 0) and V.max() == pytest.approx(model.U, rel=1e-10)


# -- superfluid fraction, U -> 0 ------------------------------------------------------------


@pytest.mark.parametrize("d_max", [2, 3])
def test_tight_binding_fs_U0(d_max):
    m, theta = 4, 0.3
    cell = lm.supercell_sites(m, 1)
    r = lm.superfluid_fraction(V0, 0.0, m, 1, 1.0, g2d=0.0, theta=theta, d_max=d_max)
    J = bands.hopping_1d(V0, d_max)
    fE, fI = lm.tb_superfluid_fraction(J, theta, cell.L, cell.geometry.alpha)
    assert r["fs_energy"] == pytest.approx(fE, abs=1e-9)
    assert r["fs_current"] == pytest.approx(fI, abs=1e-9)
    limit = float(np.sum(np.arange(1, d_max + 1) ** 2 * J) / KAPPA)
    assert r["fs_energy"] == pytest.approx(limit, abs=2e-4)  # finite-twist correction ~1e-4
    if d_max == 2:
        assert limit == pytest.approx((J[0] + 4 * J[1]) / KAPPA, rel=1e-14)
        assert limit == pytest.approx(0.2789, abs=1e-4)
    else:
        assert r["fs_energy"] == pytest.approx(0.2806, abs=1e-3)
        assert r["fs_energy"] == pytest.approx(bands.effective_mass_ratio(V0), abs=1e-3)


# -- mean-field ground state ----------------------------------------------------------------


@pytest.mark.parametrize("m,Omega_R", [(4, 0.0), (4, 1.0), (22, 0.24)])
def test_ground_state_residual(m, Omega_R):
    gs = lm.ground_state(V0, Omega_R, m, 1, filling=1.0)
    c = m * m + 1
    assert gs.residual < 1e-8
    assert float((gs.phi.abs() ** 2).sum()) == pytest.approx(c, rel=1e-12)
    assert gs.layer_fractions.sum() == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < gs.participation_ratio <= 1.0 + 1e-12
    assert gs.hessian_min > -1e-11  # a minimum, not a saddle
    if Omega_R == 0.0:  # uniform, equally split (no interlayer interaction)
        assert gs.participation_ratio == pytest.approx(1.0, abs=1e-10)
        assert gs.layer_fractions == pytest.approx([0.5, 0.5], abs=1e-10)
        assert gs.mu == pytest.approx(-4.0 * gs.model.J.sum() + gs.model.U * 0.5, abs=1e-12)
    else:
        assert gs.AA_AB_ratio > 1.0
    # a local minimum: random normalised perturbations raise the energy
    F = lm._Functional(gs.model, (0.0, 0.0))
    phi = gs.phi.reshape(-1)
    E0 = float(F.energy(phi))
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        dphi = torch.randn(phi.shape, dtype=torch.complex128, generator=g) * 1e-3
        p = phi + dphi
        p = p * math.sqrt(gs.N) / p.norm()
        assert float(F.energy(p)) > E0


def test_current_equals_dE_dtheta_and_routes_agree():
    m, Om, theta = 4, 0.24, 0.3
    r = lm.superfluid_fraction(V0, Om, m, 1, 1.0, theta=theta)
    model = r["gs0"].model
    h = 1e-3
    Ep = lm.ground_state(V0, Om, m, 1, 1.0, theta=(theta + h, 0.0), model=model).E_per_N * r["gs0"].N
    Em = lm.ground_state(V0, Om, m, 1, 1.0, theta=(theta - h, 0.0), model=model).E_per_N * r["gs0"].N
    assert r["I"] == pytest.approx((Ep - Em) / (2 * h), rel=1e-6)
    assert r["fs_current"] == pytest.approx(r["fs_energy"], rel=2e-3)
    assert r["gs_theta"].residual < 1e-8


def test_fs_checks_and_the_retry_on_a_higher_untwisted_branch():
    """The checks of `superfluid_fraction`: on a converged m = 4 state the order holds, no retry runs and
    the error bound is at rounding level. Given an untwisted state whose energy is reported too high (the
    situation of a solve that stopped above the minimum), E(theta) comes out below E(0): the untwisted
    state is solved again from the twisted one, and f_s is the converged value again (one retry). Given an
    untwisted state reported as not converged (the twisted one converged), it is solved again in the same
    way and the converged state as low is kept (one retry, the error bound back at rounding level)."""
    import dataclasses

    m, Om, theta = 4, 0.24, 0.3
    r = lm.superfluid_fraction(V0, Om, m, 1, 1.0, theta=theta)
    assert r["fs_ordered"] and r["fs_retries"] == 0
    assert 0.0 <= r["fs_error"] < 1e-9
    gs0 = r["gs0"]
    raised = dataclasses.replace(gs0, E_per_N=gs0.E_per_N + 1e-3)
    rr = lm.superfluid_fraction(V0, Om, m, 1, 1.0, theta=theta, gs0=raised)
    assert rr["fs_retries"] == 1 and rr["fs_ordered"]
    assert rr["gs0"] is not raised and rr["gs0"].E_per_N < raised.E_per_N
    assert rr["fs_energy"] == pytest.approx(r["fs_energy"], abs=1e-9)
    loose = dataclasses.replace(gs0, residual=1.0)
    rl = lm.superfluid_fraction(V0, Om, m, 1, 1.0, theta=theta, gs0=loose)
    assert rl["fs_retries"] == 1 and rl["fs_ordered"]
    assert rl["gs0"] is not loose and rl["gs0"].residual < 1e-11
    assert 0.0 <= rl["fs_error"] < 1e-9
    assert rl["fs_energy"] == pytest.approx(r["fs_energy"], abs=1e-9)


def test_trust_region_subproblem():
    """`_trust_region_subproblem` against its definition: the Newton step when it is a minimum inside the
    region; on the boundary otherwise, with a = -c / (lam + s), s >= max(0, -lam_min), and no other point
    of the region (sampled) lower in the model; the hard case (no gradient along the negative mode) still
    reaches the boundary along that mode."""
    gen = torch.Generator().manual_seed(1)
    lam = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    c = torch.tensor([0.1, -0.2, 0.3, 0.05], dtype=torch.float64)

    def model(a):
        return float(2 * (c @ a) + (lam * a * a).sum())

    a, pred = lm._trust_region_subproblem(lam, c, 10.0)
    assert torch.allclose(a, -c / lam, rtol=1e-14) and pred == pytest.approx(-model(a), rel=1e-14)
    for lam_t in (lam, lam - 1.0):  # positive definite, step outside; indefinite
        lam = lam_t
        a, pred = lm._trust_region_subproblem(lam, c, 0.1)
        assert float(a.norm()) == pytest.approx(0.1, rel=1e-10)
        s = float((-c / a - lam).mean())
        assert torch.allclose(-c / (lam + s), a, rtol=1e-8) and s >= max(0.0, -float(lam.min())) - 1e-12
        for _ in range(200):
            b = torch.randn(4, generator=gen, dtype=torch.float64)
            b = b * 0.1 * float(torch.rand(1, generator=gen)) ** 0.25 / b.norm()
            assert model(b) >= model(a) - 1e-14
    lam = torch.tensor([-1e-3, 1.0, 2.0], dtype=torch.float64)
    c = torch.tensor([0.0, 1e-3, 0.0], dtype=torch.float64)
    a, pred = lm._trust_region_subproblem(lam, c, 0.2)
    assert float(a.norm()) == pytest.approx(0.2, rel=1e-12) and abs(float(a[0])) > 0.19 and pred > 0


def test_trust_region_polish_reaches_the_minimum():
    """From a perturbed m = 4 ground state (phases and amplitudes scrambled by 0.3) and from the saddle
    of one uncoupled layer filled alone, the trust-region polish ends on a local minimum (residual < tol,
    lowest constrained Hessian eigenvalue > 0) at the energy of `ground_state`'s solution."""
    m, Om = 4, 0.6
    gs = lm.ground_state(V0, Om, m, 1, 1.0)
    F = lm._Functional(gs.model, (0.0, 0.0))
    E_ref = gs.E_per_N * gs.N
    gen = torch.Generator().manual_seed(2)
    phi = gs.phi.reshape(-1)
    start = (
        phi
        * (1 + 0.3 * torch.randn(phi.shape, generator=gen, dtype=torch.float64))
        * torch.exp(0.3j * torch.randn(phi.shape, generator=gen, dtype=torch.float64))
    )
    p, E, res, hmin, steps = lm._trust_region_polish(F, start, gs.N, 1e-11)
    assert res < 1e-11 and hmin > 0 and 0 < steps < 400
    assert E == pytest.approx(E_ref, rel=1e-12)
    # Omega_R = 0, no interlayer interaction: one layer full and the other empty is stationary, a saddle
    gs0 = lm.ground_state(V0, 0.0, m, 1, 1.0)
    F0 = lm._Functional(gs0.model, (0.0, 0.0))
    lone = gs0.phi.clone()
    lone[1] = 0.0
    lone = lone.reshape(-1)
    assert F0.residual(lone * math.sqrt(gs0.N) / lone.norm())[0] < 1e-12
    p, E, res, hmin, _ = lm._trust_region_polish(F0, lone + 1e-6 * gs0.phi.reshape(-1), gs0.N, 1e-11)
    assert res < 1e-11 and hmin > -1e-11 and E == pytest.approx(gs0.E_per_N * gs0.N, rel=1e-12)


@pytest.mark.slow
def test_fs_m22_weak_links():
    """The bridge's (V0, Omega_R) = (16, 0.6) on the experiment's m = 22 cell, two-band model: the pockets
    are weakly linked (lowest Hessian eigenvalue ~7e-7) and LBFGS + Newton stopped at residual 5.5e-7
    with E(0) 4.1e-5 E_R above the minimum, so the twist energy came out negative (f_s = -4.2e-3). With the
    trust-region polish both states converge (residual < 1e-11, minima) and the energy and current routes
    agree (2.172e-4 and 2.168e-4, 24 Oct), with an error bound at rounding level."""
    from twisted_bilayer.manybody.bridge import BridgeSpec

    spec = BridgeSpec()
    r = lm.superfluid_fraction(
        16.0,
        0.6,
        22,
        1,
        1.0,
        g2d=lm.g2d_from_nu_z(spec.nu_z_hz),
        theta=spec.theta,
        tol=spec.tb_tol,
        cutoff=spec.cutoff,
        interlayer_interaction=True,
        frame=spec.frame,
        interband=False,
        interband_n_bands=spec.interband_n_bands,
    )
    for g in (r["gs0"], r["gs_theta"]):
        assert g.residual < 1e-11 and g.hessian_min > 0
    assert r["fs_ordered"] and r["fs_error"] < 1e-9
    assert r["fs_energy"] == pytest.approx(2.172e-4, rel=2e-3)
    assert r["fs_current"] == pytest.approx(r["fs_energy"], rel=5e-3)


# -- Gutzwiller ------------------------------------------------------------------------------


def test_gutzwiller_mott_tip_square_lattice():
    """Plain square lattice (Omega_R = 0, nearest neighbour only): at
    mu/U = sqrt 2 - 1 the order parameter is finite just below
    U/zJ = 3 + 2 sqrt 2 = 5.83 and zero just above (single-site mean-field lobe tip)."""
    tip = 3.0 + 2.0 * math.sqrt(2.0)
    assert tip == pytest.approx(5.83, abs=5e-3) and 4 * tip == pytest.approx(23.3, abs=0.05)
    muU = math.sqrt(2.0) - 1.0
    sf = lm.gutzwiller(V0, 0.0, 4, 1, mu_over_U=muU, J_list=(1.0 / (4 * 0.987 * tip),), U=1.0)
    mi = lm.gutzwiller(V0, 0.0, 4, 1, mu_over_U=muU, J_list=(1.0 / (4 * 1.013 * tip),), U=1.0)
    assert sf.converged and mi.converged
    assert np.abs(sf.phi).min() > 0.05 and sf.compressibility > 1e-3
    assert np.abs(mi.phi).max() < 1e-8
    assert mi.n == pytest.approx(np.ones_like(mi.n), abs=1e-10)
    assert mi.compressibility == pytest.approx(0.0, abs=1e-8)
    assert mi.condensate_fraction < 1e-14


# -- the interaction input: g2d, nu_z_hz, the shipped default -------------------------------


def test_g2d_from_nu_z_scaling_precedence_and_default_offset():
    """U = g2d (int w^4)^2 with g2d = g3d / (sqrt(2 pi) l_z), l_z ~ nu_z^-1/2: U ~ sqrt(nu_z) at fixed V0.
    Precedence: an explicit g2d wins, then nu_z_hz, then G2D_DEFAULT; the derived 1 kHz value is above
    the shipped default by the documented 2.4e-4 (an offset of the kind of engine.units.EXCEPTIONS)."""
    cell = lm.supercell_sites(4, 1)
    U1 = lm.build_model(V0, 0.0, 4, 1, nu_z_hz=1000.0, cell=cell).U
    U4 = lm.build_model(V0, 0.0, 4, 1, nu_z_hz=4000.0, cell=cell).U
    assert U4 / U1 == pytest.approx(2.0, rel=1e-12)
    assert lm.g2d_from_nu_z(4000.0) / lm.g2d_from_nu_z(1000.0) == pytest.approx(2.0, rel=1e-12)
    assert lm.g2d_from_nu_z(1000.0) / lm.G2D_DEFAULT - 1.0 == pytest.approx(lm.G2D_DEFAULT_OFFSET, abs=1e-5)
    assert lm.g2d_from_nu_z(1000.0) == pytest.approx(0.0157638, abs=5e-8)
    # precedence and the record on the model
    both = lm.build_model(V0, 0.0, 4, 1, g2d=0.02, nu_z_hz=1000.0, cell=cell)
    assert both.g2d == 0.02 and both.g2d_source == "explicit" and both.U == pytest.approx(bands.onsite_U(V0, 0.02))
    nu = lm.build_model(V0, 0.0, 4, 1, nu_z_hz=1000.0, cell=cell)
    assert nu.g2d_source == "nu_z" and nu.nu_z_hz == 1000.0 and nu.g2d == lm.g2d_from_nu_z(1000.0)
    default = lm.build_model(V0, 0.0, 4, 1, cell=cell)
    assert default.g2d == lm.G2D_DEFAULT and default.g2d_source == "default" and default.nu_z_hz is None
    assert default.U == pytest.approx(bands.onsite_U(V0, lm.G2D_DEFAULT), rel=1e-15)
    assert lm.resolve_g2d(0.0, 1000.0) == (0.0, "explicit")  # g2d = 0 is explicit, not "absent"
    assert lm.build_model(V0, 0.0, 4, 1, g2d=0.0, cell=cell).U == 0.0
    # the solvers pass it through
    gs = lm.ground_state(V0, 0.0, 4, 1, filling=1.0, nu_z_hz=2000.0)
    assert gs.model.g2d_source == "nu_z" and gs.model.U == pytest.approx(math.sqrt(2.0) * U1, rel=1e-12)
    gw = lm.gutzwiller(V0, 0.0, 4, 1, mu_over_U=0.5, nu_z_hz=2000.0, n_max=3)
    gw2 = lm.gutzwiller(V0, 0.0, 4, 1, mu_over_U=0.5, g2d=lm.g2d_from_nu_z(2000.0), n_max=3)
    assert gw.converged and gw.mu == gw2.mu and gw.filling == pytest.approx(gw2.filling, rel=1e-12)


# -- the interband shift ---------------------------------------------------------------------


def test_interband_shift_structure(cell22):
    """eps^(2) <= 0 everywhere, exactly 0 at AA (the Wannier function there has all its weight in the
    partner's lowest band) and largest in magnitude towards AB; proportional to Omega_R^2; the bond part
    is Hermitian, intralayer only, and small against the site part; frame="mid" is layer symmetric."""
    V0 = 12.0
    ib = lm.interband_shift(cell22, V0, 1.0, n_bands=4)
    c = cell22.c
    assert ib.onsite.shape == (2 * c,) and np.all(ib.onsite <= 1e-15)
    assert ib.onsite[cell22.origin_index(0)] == pytest.approx(0.0, abs=1e-5)  # the 5 deg rotation of the neighbours
    assert ib.onsite[c + cell22.origin_index(1)] == pytest.approx(0.0, abs=1e-5)
    d = ib.diagnostics
    assert d["lowest_band_weight_AA"] == pytest.approx(1.0, abs=1e-4)
    assert d["lowest_band_weight_AB"] == pytest.approx(0.0064, abs=5e-4)  # = lowest_band_weight_min at cutoff 1
    assert -0.12 < d["eps_min"] < -0.07 and d["eps_min"] <= d["eps_AB"] < -0.03  # ~ -Omega_R^2 (1 - w) / gap
    assert d["denominator"] == "dressed"
    bare = lm.interband_shift(cell22, V0, 1.0, n_bands=4, denominator="centre")
    assert np.all(bare.onsite <= ib.onsite + 1e-15) and bare.diagnostics["eps_min"] < d["eps_min"]
    assert d["gap_first"] == pytest.approx(bands.band_centres_1d(V0, 2)[1] - bands.band_centres_1d(V0, 2)[0])
    H = ib.bonds.matrix((2 * c, 2 * c)).toarray()
    assert np.allclose(H, H.T, atol=1e-15) and np.abs(H[:c, c:]).max() == 0.0
    assert d["bond_max"] < 0.2 * abs(d["eps_min"])
    assert np.abs(ib.onsite[:c] - ib.onsite[c:]).max() < 0.05 * abs(d["eps_min"])  # layer1 frame: small asymmetry
    cell4 = lm.supercell_sites(4, 1)
    ib4 = lm.interband_shift(cell4, V0, 1.0, n_bands=4, denominator="centre")
    ib4h = lm.interband_shift(cell4, V0, 0.5, n_bands=4, denominator="centre")
    assert np.allclose(4.0 * ib4h.onsite, ib4.onsite, atol=1e-15) and np.allclose(4.0 * ib4h.bonds.amp, ib4.bonds.amp)
    # dressed denominators: weaker than Omega_R^2 (the gap grows with the first-order dressing Omega_R S_hyb)
    dr, drh = lm.interband_shift(cell4, V0, 1.0, n_bands=4), lm.interband_shift(cell4, V0, 0.5, n_bands=4)
    assert np.all(dr.onsite >= ib4.onsite - 1e-15) and np.all(4.0 * drh.onsite <= dr.onsite + 1e-15)
    ib4 = dr
    sym = lm.interband_shift(cell4, V0, 1.0, n_bands=4, frame="mid")
    assert np.abs(sym.onsite[: cell4.c] - sym.onsite[cell4.c :]).max() < 1e-12
    assert np.abs(ib4.onsite[: cell4.c] - ib4.onsite[cell4.c :]).max() > 1e-3  # 28 deg: the layer-1 frame is one-sided
    # the model: off by default, on request the site energies and bonds are added
    off = lm.build_model(V0, 1.0, 22, 1, cutoff=2.0, cell=cell22)
    on = lm.build_model(V0, 1.0, 22, 1, cutoff=2.0, cell=cell22, interband=True, interband_n_bands=4)
    assert not off.interband and "interband" not in off.diagnostics
    assert on.interband and on.diagnostics["interband"]["bonds_included"]
    assert np.allclose(on.onsite - off.onsite, ib.onsite) and len(on.bonds) == len(off.bonds) + len(ib.bonds)
    only = lm.build_model(
        V0, 1.0, 22, 1, cutoff=2.0, cell=cell22, interband=True, interband_n_bands=4, interband_bonds=False
    )
    assert len(only.bonds) == len(off.bonds) and np.allclose(only.onsite, on.onsite)
    assert lm.build_model(V0, 0.0, 4, 1, cell=cell4, interband=True).onsite.max() == 0.0  # nothing at Omega_R = 0


@pytest.mark.parametrize("m,V0_", [(4, 12.0), (8, 8.0)])
def test_interband_bonds_layer_symmetric_across_the_boundary(m, V0_):
    """With the layer-symmetric frame the two layers are mirror images, so the second-order shift must
    give both layers the same bond count per site (as a multiset) and the same spectrum of their block.
    Both fail if the bonds of pairs whose shared partner lies across the box boundary are dropped in one
    layer (before the fix: 5712 against 22212 bond entries on m = 4, spectra 8e-3 apart)."""
    cell = lm.supercell_sites(m, 1)
    c = cell.c
    ib = lm.interband_shift(cell, V0_, 1.0, n_bands=4, frame="mid")
    count = np.bincount(ib.bonds.rows, minlength=2 * c)
    assert count[:c].sum() == count[c:].sum() > 0
    assert np.array_equal(np.sort(count[:c]), np.sort(count[c:]))
    H = ib.bonds.matrix((2 * c, 2 * c)).toarray().real + np.diag(ib.onsite)
    e1 = np.linalg.eigvalsh(H[:c, :c])
    e2 = np.linalg.eigvalsh(H[c:, c:])
    assert np.abs(e1 - e2).max() < 1e-12


def test_linearisation_matches_autograd_with_repeated_pairs():
    """On a small cell the interlayer pair list repeats (i, j) once per periodic image; the Newton
    Jacobian must accumulate the repeats like the energy does (index_add), so it equals the autograd
    Jacobian of F(phi) = H_mf phi - mu phi at fixed mu."""
    model = lm.build_model(V0, 0.5, 3, 1, cutoff=2.0, interlayer_interaction=True)
    i, j, _ = model.V_pairs
    pairs = set(zip(i.tolist(), j.tolist()))
    assert len(pairs) < len(i)  # the cell really has repeated pairs
    F = lm._Functional(model, (0.0, 0.0))
    gen = torch.Generator().manual_seed(3)
    n = model.n_sites
    phi = torch.complex(
        torch.rand(n, generator=gen, dtype=torch.float64) + 0.5, 0.3 * torch.rand(n, generator=gen, dtype=torch.float64)
    )
    Jm, _, _, _, _ = lm._linearisation(F, phi)
    _, mu = F.residual(phi)

    def f(x):
        p = torch.complex(x[:n], x[n:])
        r = F.mf_apply(p) - mu * p
        return torch.cat([r.real, r.imag])

    J_ad = torch.autograd.functional.jacobian(f, torch.cat([phi.real, phi.imag]))
    assert float((Jm - J_ad).abs().max()) < 1e-12


@pytest.mark.parametrize("Omega_R,factor", [(0.2, 10.0), (1.0, 5.0)])
def test_interband_shift_against_continuum(Omega_R, factor):
    """The key check of the interband shift: on the m = 4 cell at V0 = 12 the error of the lowest six
    k = 0 levels of the two-band model against the continuum (`compare_with_continuum`, `abs_maxdiff`
    and `spacing_maxdiff`) falls by more than `factor` when the interband shift is added. 15 Oct, with
    the dressed denominators: 0.081 -> 0.0088 at Omega_R = 1 (a factor 9; the bare band-centre
    denominators give 0.017, a factor 5), 0.0035 -> 0.0002 at 0.2. The lowest level alone (shift_tb vs
    shift_cont) is within 2e-3 of the continuum with or without it (the AA pocket, where the shift
    vanishes); the improvement is in the higher, AB-like levels."""
    V0, m = 12.0, 4
    off = lm.compare_with_continuum(V0, Omega_R, m, 1, n_bands=6, points_per_a=16.0, cutoff=2.0)
    on = lm.compare_with_continuum(V0, Omega_R, m, 1, n_bands=6, points_per_a=16.0, cutoff=2.0, interband=True)
    assert off["cont_residual"] < 1e-6 and on["cont_residual"] < 1e-6
    assert off["abs_maxdiff"] > factor * on["abs_maxdiff"]
    assert off["spacing_maxdiff"] > factor * on["spacing_maxdiff"]
    assert abs(off["shift_tb"] - off["shift_cont"]) < 2e-3 and abs(on["shift_tb"] - on["shift_cont"]) < 2e-3
    if Omega_R == 1.0:
        assert off["abs_maxdiff"] == pytest.approx(0.081, abs=0.005) and on["abs_maxdiff"] == pytest.approx(
            0.0088, abs=0.002
        )
        bare = lm.compare_with_continuum(
            V0, Omega_R, m, 1, n_bands=6, points_per_a=16.0, cutoff=2.0, interband=True, interband_denominator="centre"
        )
        assert bare["abs_maxdiff"] == pytest.approx(0.017, abs=0.003) and bare["abs_maxdiff"] > on["abs_maxdiff"]
