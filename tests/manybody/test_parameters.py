"""manybody.parameters: the Bose-Hubbard parameter audit, the transition depths and the sign audit.

Pinned numbers and their sources:
  * the paper's t and U: closed forms (Meng et al. Supplement Eq. 13; Zwerger 2003), evaluated here;
  * three rows of the audit table (V0 = 4, 10, 16; `parameters.audit_table`, computed from
    `bands` and the paper's formulas with a_s = 5.29 nm, U at 1 kHz), to 1e-3 relative: that tolerance covers
    the 2.4e-4 between the derived g2d and lattice_model.G2D_DEFAULT;
  * the four depths of the plain-square-lattice Mott transition, 11.1 / 10.0 (paper) and 17.6 / 16.1 (exact
    J1, 1 kHz U) at U/J = 23.31 / 16.74, to 0.05 E_R (same source; the U/J values are gates.references anchors);
  * the sign audit at V0 = 12, Omega_R = 0.5 on the m = 22 cell: cutoff 1 gives 3874 directed interlayer
    entries all of one sign; cutoff 2 gives 15,506 entries, 7752 of the minority sign carrying 1.27 % of the
    interlayer weight; |J2|/J1 = 7.2 % at 4 E_R, 1.2 % at 10, 0.7 % at 12 (`bands`).
"""

import math

import numpy as np
import pytest

from twisted_bilayer import config
from twisted_bilayer.engine import bands
from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.gates.references import anchor
from twisted_bilayer.manybody import parameters as P

# V0: (paper t, paper U, paper U/t, exact J1, J2/J1, U at 1 kHz, U/J1); the audit table to 4 digits, with
# the digits of the 15 Oct run where the table rounds further
AUDIT_PINS = {
    4.0: (0.11691, 0.18990, 1.624, 0.08549, -0.0719, 0.03515, 0.411),
    10.0: (0.02274, 0.37755, 16.60, 0.01918, -0.0118, 0.06531, 3.404),
    16.0: (0.00606, 0.53712, 88.69, 0.00533, -0.0028, 0.08672, 16.26),
}
DEPTH_PINS = {
    ("paper", "gutzwiller"): 11.1,
    ("paper", "qmc"): 10.0,
    ("exact", "gutzwiller"): 17.6,
    ("exact", "qmc"): 16.1,
}


def test_paper_formulas_closed_form():
    for V0 in (4.0, 12.0, 20.0):
        p = P.paper_parameters(V0)
        k_as = math.pi * config.A_S_M / config.LATTICE_CONSTANT_M
        assert p["t"] == pytest.approx(4.0 / math.sqrt(math.pi) * V0**0.75 * math.exp(-2.0 * math.sqrt(V0)), rel=1e-14)
        assert p["U"] == pytest.approx(math.sqrt(8.0 / math.pi) * k_as * V0**0.75, rel=1e-14)
        assert p["U_over_t"] == pytest.approx(p["U"] / p["t"], rel=1e-14)
    # the 3D-lattice U scales as V0^3/4 and with a_s; it is independent of the pancake
    assert P.paper_parameters(16.0)["U"] / P.paper_parameters(4.0)["U"] == pytest.approx(4.0**0.75, rel=1e-14)
    assert P.paper_parameters(4.0, a_s_m=2 * config.A_S_M)["U"] == pytest.approx(2 * P.paper_parameters(4.0)["U"])


def test_audit_table_pins():
    rows = {r["V0"]: r for r in P.audit_table()}
    assert sorted(rows) == list(range(4, 25, 2))
    for V0, pins in AUDIT_PINS.items():
        r = rows[V0]
        got = (r["t_paper"], r["U_paper"], r["U_over_t_paper"], r["J1"], r["J2_over_J1"], r["U"], r["U_over_J1"])
        for name, g, want in zip(P.AUDIT_COLUMNS[1:], got, pins):
            if name == "J2_over_J1":
                assert g == pytest.approx(want, abs=5e-4), (V0, name)
            else:
                assert g == pytest.approx(want, rel=1e-3), (V0, name)
    # the same numbers from the primitives
    r = rows[12.0]
    assert r["J1"] == pytest.approx(bands.hopping_1d(12.0, 1)[0], rel=1e-14)
    assert r["U"] == pytest.approx(bands.onsite_U(12.0, lm.g2d_from_nu_z(config.NU_Z_HZ)), rel=1e-14)
    # the shipped default moves U in the fourth digit only
    assert bands.onsite_U(12.0, lm.G2D_DEFAULT) / r["U"] - 1.0 == pytest.approx(-lm.G2D_DEFAULT_OFFSET, abs=1e-5)


def test_transition_depths_pins():
    assert P.UJ_C["gutzwiller"] == pytest.approx(4.0 * (3.0 + 2.0 * math.sqrt(2.0)), rel=1e-14)
    assert P.UJ_C["qmc"] == anchor("square_lattice_UJ_c_qmc")[0] == 16.74
    d = P.transition_depths()
    for (which, crit), want in DEPTH_PINS.items():
        assert d[which][crit] == pytest.approx(want, abs=0.05), (which, crit)
    # at the depths the ratios are the critical ones
    assert P.paper_parameters(d["paper"]["qmc"])["U_over_t"] == pytest.approx(16.74, rel=1e-8)
    V0 = d["exact"]["gutzwiller"]
    U = bands.onsite_U(V0, lm.g2d_from_nu_z(config.NU_Z_HZ))
    assert U / bands.hopping_1d(V0, 1)[0] == pytest.approx(P.UJ_C["gutzwiller"], rel=1e-8)


def test_paper_quoted_ratio_note():
    q = P.paper_quoted_ratio_check()
    assert q["quoted"] == 1.67 and q["formulas"] == pytest.approx(1.624, abs=1e-3)
    assert q["relative_gap"] == pytest.approx(0.028, abs=2e-3)
    assert q["with_a_s_100a0"] / q["formulas"] - 1.0 == pytest.approx(3.35e-4, abs=1e-5)
    assert q["Omega_R_from_quoted_ratio_2.07"] == pytest.approx(0.242, abs=1e-3)  # the 0.243 E_R reading


def test_bh_parameters_small_cell():
    b = P.bh_parameters(8.0, nu_z_hz=1000.0, Omega_R=0.5, cutoff=2.0, m=4, n=1)
    J = bands.hopping_1d(8.0, 3)
    assert np.allclose(b["J"], J) and b["J1"] == J[0] and b["J2_over_J1"] == pytest.approx(J[1] / J[0])
    assert b["U"] == pytest.approx(bands.onsite_U(8.0, lm.g2d_from_nu_z(1000.0)), rel=1e-14)
    assert b["U_over_J1"] == pytest.approx(b["U"] / J[0])
    # AA: the co-located pair (1) plus the images of the 28 deg cell; the coarse cell has no site near AB
    assert 1.0 <= b["S_sum_AA"] < 1.05 and 0.2 < b["S_sum_AB"] < 0.5
    assert b["moire_depth_k0"] == pytest.approx(0.5 * (b["S_sum_AA"] - b["S_sum_AB"]))
    assert b["interband"]["eps_AB"] < 0.0 and b["interband"]["n_bands"] == 6  # the summary of the shift (Omega_R != 0)
    # U ~ sqrt(nu_z)
    assert P.bh_parameters(8.0, nu_z_hz=4000.0, m=4)["U"] / b["U"] == pytest.approx(2.0, rel=1e-12)


def test_bh_parameters_coupling_sums_experiment_cell(cell22):
    """On the experiment's cell the AB sum with cutoff 2 is the converged lattice sum (sum_n s(n + 1/2))^2
    = 0.309 at 8 E_R (test_lattice_model.test_cutoff_and_k0_coupling), and -Omega_R (S_sum_AA - S_sum_AB)
    is the band-bottom moire depth."""
    b = P.bh_parameters(8.0, Omega_R=0.24, cutoff=2.0, cell=cell22)
    half = sum(bands.overlap_1d(8.0, k + 0.5) for k in range(-6, 6)) ** 2
    assert b["S_sum_AA"] == pytest.approx(1.0, abs=2e-3) and b["S_sum_AB"] == pytest.approx(half, abs=0.01)
    assert b["moire_depth_k0"] == pytest.approx(0.24 * (1.0 - half), abs=0.01)


@pytest.fixture(scope="module")
def cell22():
    return lm.supercell_sites(22, 1)


def test_sign_audit_pins(cell22):
    s1 = P.sign_audit(lm.build_model(12.0, 0.5, 22, 1, cutoff=1.0, cell=cell22))
    s2 = P.sign_audit(lm.build_model(12.0, 0.5, 22, 1, cutoff=2.0, cell=cell22))
    assert s1["interlayer"]["entries"] == 3874 and s1["interlayer"]["frustrated_entries"] == 0
    assert s1["interlayer"]["frustrated_weight_fraction"] == 0.0
    assert s2["interlayer"]["entries"] == 15506 and s2["interlayer"]["frustrated_entries"] == 7752
    assert s2["interlayer"]["frustrated_weight_fraction"] == pytest.approx(0.0127, abs=1e-4)
    # J2 < 0: half the intralayer entries (the d = 2 bonds) stay positive, |J2|/(J1 + |J2|) of the weight
    for s in (s1, s2):
        assert s["intralayer"]["entries"] == 2 * 8 * 485 and s["intralayer"]["frustrated_entries"] == 2 * 4 * 485
        r = abs(s["J2_over_J1"])
        assert s["intralayer"]["frustrated_weight_fraction"] == pytest.approx(r / (1.0 + r), rel=1e-12)
        assert not s["sign_free"]
    # without J2 and with the one-neighbour cutoff the model is sign-free after the gauge
    s0 = P.sign_audit(lm.build_model(12.0, 0.5, 22, 1, cutoff=1.0, d_max=1, cell=cell22))
    assert s0["sign_free"] and s0["frustrated_weight_fraction"] == 0.0
    # the gauge is needed: before it every interlayer entry is positive (Omega_R S > 0 inside |d'| < 1)
    model = lm.build_model(12.0, 0.5, 22, 1, cutoff=1.0, cell=cell22)
    inter = (model.bonds.rows // 485) != (model.bonds.cols // 485)
    assert np.all(model.bonds.amp[inter] > 0.0)


@pytest.mark.parametrize("V0,ratio", [(4.0, 0.072), (10.0, 0.012), (12.0, 0.007)])
def test_J2_over_J1_by_depth(V0, ratio):
    J = bands.hopping_1d(V0, 2)
    assert J[1] < 0.0 < J[0]
    assert abs(J[1] / J[0]) == pytest.approx(ratio, abs=5e-4)


def test_occupation_dependent_U():
    """The NPSE broadening of a site under n atoms: U_n -> U as n -> 0 (the fixed width), U_n decreases
    with n, and at the 1 kHz pancake U_2/U_1 = 0.964 at 12 E_R and 0.959 at 16 E_R (15 Oct; a 3.6-4.1 %
    effect, under the 0.95 flag of `occupation_dependent_U`), with U_1 itself 4-5 % below the fixed-width U."""
    r = P.occupation_dependent_U(12.0, n_list=(0.001, 1, 2, 3))
    assert r["n"][0]["U_n_over_U"] == pytest.approx(1.0, abs=1e-4)
    assert r["n"][1]["U_n_over_U"] > r["n"][2]["U_n_over_U"] > r["n"][3]["U_n_over_U"]
    assert r["U_fixed"] == pytest.approx(bands.onsite_U(12.0, lm.g2d_from_nu_z(config.NU_Z_HZ)), rel=1e-12)
    assert r["peak_density_1"] == pytest.approx(bands.wannier_1d(12.0)[1].max() ** 4, rel=1e-9)
    assert r["U2_over_U1"] == pytest.approx(0.964, abs=0.003) and r["n"][1]["U_n_over_U"] == pytest.approx(
        0.959, abs=0.003
    )
    r16 = P.occupation_dependent_U(16.0, n_list=(1, 2))
    assert r16["U2_over_U1"] == pytest.approx(0.959, abs=0.003)
    assert 1.0 < r16["n"][2]["x_peak"] < 1.5 and 1.2 < r16["n"][2]["s_peak"] < 1.35


def test_bh_parameters_interband_summary(cell22):
    """The interband summary of bh_parameters: eps = 0 at AA, < 0 at AB, the moire depth gets shallower
    (15 Oct, V0 = 12, Omega_R = 1: by 7 %; eps_AB = -0.061, eps_min = -0.106 E_R), and nothing at Omega_R = 0."""
    b = P.bh_parameters(12.0, Omega_R=1.0, cutoff=2.0, cell=cell22)
    ib = b["interband"]
    assert ib is not None and ib["eps_AA"] == pytest.approx(0.0, abs=1e-5) and ib["eps_AB"] < -0.03
    assert ib["eps_min"] <= ib["eps_AB"] and ib["moire_depth_change"] == pytest.approx(-0.072, abs=0.01)
    assert ib["moire_depth_with_shift"] == pytest.approx(b["moire_depth_k0"] + ib["eps_AB"] - ib["eps_AA"], rel=1e-12)
    assert ib["eps_AB"] == pytest.approx(-0.061, abs=0.005) and ib["eps_min"] == pytest.approx(-0.106, abs=0.005)
    assert P.bh_parameters(12.0, Omega_R=0.0, cell=cell22)["interband"] is None
    assert P.bh_parameters(12.0, Omega_R=1.0, cell=cell22, interband=False)["interband"] is None


def test_sign_audit_with_interband_shift(cell22):
    """The interband site energies are diagonal and change nothing in the sign audit; the interband bond
    corrections are of either sign (about 65 % of their weight positive after the gauge) and at Omega_R = 1
    their largest entry is comparable to J1, so a model with interband_bonds=True is not Monte-Carlo
    friendly (15 Oct: intralayer frustrated weight 0.70 % -> 26 % at V0 = 12, Omega_R = 1). The Monte Carlo
    design takes the site part only (interband_bonds=False)."""
    off = P.sign_audit(lm.build_model(12.0, 1.0, 22, 1, cutoff=2.0, cell=cell22))
    sites = P.sign_audit(
        lm.build_model(12.0, 1.0, 22, 1, cutoff=2.0, cell=cell22, interband=True, interband_bonds=False)
    )
    bonds = P.sign_audit(lm.build_model(12.0, 1.0, 22, 1, cutoff=2.0, cell=cell22, interband=True))
    for key in ("intralayer", "interlayer"):
        assert sites[key] == off[key]
    assert bonds["intralayer"]["entries"] > 1_000_000 and bonds["intralayer"]["frustrated_weight_fraction"] > 0.2
    assert bonds["interlayer"] == off["interlayer"]
