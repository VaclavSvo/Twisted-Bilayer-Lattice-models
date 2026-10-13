"""bilayer.lattice_model: Model C, the two-layer tight-binding model
(docs/1-theory.md §2.6, docs/2-methods.md §12.2).

Pinned numbers and their sources:
  * c = m^2 + n^2 sites per layer: index of the box superlattice (exact).
  * J1, J2 from `bands.hopping_1d` (themselves pinned to the docs/1-theory.md
    §3.2 table in test_bands.py); coordination 4 + 4 per site (separable lattice).
  * S = 1 for the co-located AA pair (s(0) = 1); the AB value s(a/2)^2 and the
    moire depth factor 1 - s(a/2)^2 = 0.92 at V0 = 8 (docs/1-theory.md §3.3).
  * U -> 0 superfluid fraction: the analytic tight-binding band at finite twist
    (`tb_superfluid_fraction`, derivation in its docstring); its q -> 0 limit
    sum_d d^2 J_d / kappa; with J3 kept it reproduces the exact band curvature
    m/m* = 0.2806 (bands.effective_mass_ratio; docs/1-theory.md §3.2) to 1e-3.
  * Gutzwiller n = 1 lobe tip U/zJ = 3 + 2 sqrt 2 = 5.83 at mu/U = sqrt 2 - 1
    (foundations note S6.2: "U/zJ = 5.83, i.e. U/J = 23.3").
"""

import math

import numpy as np
import pytest
import torch

from bilayer import bands
from bilayer import lattice_model as lm
from bilayer.units import KAPPA

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
    # 1-theory §3.3: s(a/2) = 0.286 and the depth factor 1 - s(a/2)^2 = 0.92 at V0 = 8
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


# -- Gutzwiller ------------------------------------------------------------------------------


def test_gutzwiller_mott_tip_square_lattice():
    """Plain square lattice (Omega_R = 0, nearest neighbour only): at
    mu/U = sqrt 2 - 1 the order parameter is finite just below
    U/zJ = 3 + 2 sqrt 2 = 5.83 and zero just above (foundations note S6.2)."""
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
