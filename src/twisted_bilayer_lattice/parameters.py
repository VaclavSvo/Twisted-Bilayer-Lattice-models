"""Parameters of the two-layer Bose-Hubbard Hamiltonian, and the audits around them.

Everything is in lattice units (a = 1, E_R = 1, hbar = 1) and comes from `engine.bands` (the exact lowest
band of V0 sin^2(pi x)), `engine.lattice_model` (the supercell and the interlayer overlaps) and
`engine.units` (g2d from the pancake frequency); nothing here holds a physics number of its own.

  * `bh_parameters(V0, nu_z_hz, ...)`: the Hamiltonian's J(d), U = g2d (int w^4)^2, U/J1 and the
    interlayer coupling sum_j S_ij at AA and AB (the k = 0 moire potential is -Omega_R sum_j S_ij).
  * `paper_parameters(V0)`: the formulas behind the paper's mean-field phase diagram (Meng et al., Nature
    615, 231 (2023), Supplement section IV, Eq. 13):
        t = (4/sqrt pi) E_R (V0/E_R)^{3/4} exp(-2 sqrt(V0/E_R)),
        U = sqrt(8/pi) k a_s E_R (V0/E_R)^{3/4},   k = 2 pi / lambda = pi / a,
    the deep-lattice asymptote of the hopping and the on-site interaction of a 3D cubic lattice of depth V0
    in all three directions (Zwerger, J. Opt. B 5, S9 (2003)); the second is not the 1 kHz pancake's U.
  * `audit_table`, `transition_depths`: the comparison of the two over V0, and the lattice depth of the
    plain-square-lattice n = 1 Mott transition, U/J = 23.31 (single-site mean field) or 16.74 (worm Monte
    Carlo), with either parameter set (`gates.references.ANCHORS`).
  * `sign_audit(model)`: the off-diagonal weight of H0 that a bosonic Monte Carlo cannot make negative by the
    gauge c -> -c on layer 2 (J2 < 0 bonds; interlayer pairs with S < 0).

`scripts/parameter_audit.py` prints the tables; tests/manybody/test_parameters.py pins them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from twisted_bilayer import config
from twisted_bilayer.engine import bands
from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.engine.units import lattice_units
from twisted_bilayer.gates.references import anchor

UJ_C = {
    "gutzwiller": anchor("square_lattice_UJ_c_gutzwiller")[0],  # z (3 + 2 sqrt 2) = 23.31
    "qmc": anchor("square_lattice_UJ_c_qmc")[0],  # 16.74
}
PAPER_QUOTED_U_OVER_T_4ER = 1.67  # Meng et al., Fig. 4 caption: "V = 4 E_r (U/t = 1.67, Omega_R/t = 2.07)"


def paper_parameters(V0: float, a_s_m: float = config.A_S_M, a_m: float = config.LATTICE_CONSTANT_M) -> dict:
    """The paper's t and U at depth V0 (E_R), with k a_s = pi a_s / a (a_s = config.A_S_M = 5.29 nm by
    default; the engine's 100 a0 is 3.4e-4 larger). Returns {"t", "U", "U_over_t", "k_a_s"}."""
    V0 = float(V0)
    k_a_s = math.pi * a_s_m / a_m
    t = 4.0 / math.sqrt(math.pi) * V0**0.75 * math.exp(-2.0 * math.sqrt(V0))
    U = math.sqrt(8.0 / math.pi) * k_a_s * V0**0.75
    return {"t": t, "U": U, "U_over_t": U / t, "k_a_s": k_a_s}


def bh_parameters(
    V0: float,
    nu_z_hz: float = config.NU_Z_HZ,
    Omega_R: float = 0.0,
    cutoff: float = 1.0,
    m: int = config.TWIST_M,
    n: int = config.TWIST_N,
    d_max: int = 3,
    frame: str = "layer1",
    cell: lm.Supercell | None = None,
    interband: bool = True,
    interband_n_bands: int = 6,
) -> dict:
    """The two-layer Bose-Hubbard parameters at depth V0 and pancake frequency nu_z (Hz): J(d) for
    d = 1..d_max (`bands.hopping_1d`), U = onsite_U(V0, g2d) with g2d = `lattice_model.g2d_from_nu_z`
    (U ~ sqrt(nu_z)), U/J1, and the interlayer coupling on the (m, n) supercell with the overlap S cut at
    |d'| < cutoff (`lattice_model.build_interlayer`): sum_j S_ij at AA (= 1 + the images within the
    cutoff) and at AB, so that the k = 0 moire potential is -Omega_R sum_j S_ij and its depth
    Omega_R (S_sum_AA - S_sum_AB). With interband (and Omega_R != 0) the summary of the second-order
    interband shift (`lattice_model.interband_shift`): eps^(2) at AA and AB, its extremes, per Omega_R^2,
    and the moire depth with it, Omega_R (S_sum_AA - S_sum_AB) + (eps_AB - eps_AA) (shallower: eps_AB < eps_AA = 0),
    as "moire_depth_with_shift" and the relative change."""
    V0 = float(V0)
    J = bands.hopping_1d(V0, d_max)
    g2d = lm.g2d_from_nu_z(nu_z_hz)
    U = bands.onsite_U(V0, g2d)
    cell = lm.supercell_sites(m, n) if cell is None else cell
    _, diag, _ = lm.build_interlayer(cell.layers[0], cell.layers[1], V0, cell.geometry, cutoff, frame)
    units = lattice_units(omega_z_hz=nu_z_hz)
    ib = None
    if interband and Omega_R:
        sh = lm.interband_shift(cell, V0, Omega_R, interband_n_bands, frame=frame)
        d = sh.diagnostics
        depth0 = float(Omega_R) * (diag["F_k0_AA"] - diag["F_k0_AB"])
        ib = {
            "eps_AA": d["eps_AA"],
            "eps_AB": d["eps_AB"],
            "eps_min": d["eps_min"],
            "eps_min_per_Omega2": d["eps_min"] / float(Omega_R) ** 2,
            "lowest_band_weight_AB": d["lowest_band_weight_AB"],
            "gap_first": d["gap_first"],
            "moire_depth_with_shift": depth0
            + d["eps_AB"]
            - d["eps_AA"],  # V_i = -Omega_R F_i + eps_i; eps_AB < eps_AA = 0
            "moire_depth_change": (d["eps_AB"] - d["eps_AA"]) / depth0 if depth0 else float("nan"),
            "n_bands": interband_n_bands,
            "bond_max": d["bond_max"],
        }
    return {
        "V0": V0,
        "nu_z_hz": float(nu_z_hz),
        "g2d": g2d,
        "l_z": units.l_z,
        "hbar_omega_z": units.omega_z,
        "J": J,
        "J1": float(J[0]),
        "J2_over_J1": float(J[1] / J[0]) if d_max >= 2 else None,
        "J3_over_J1": float(J[2] / J[0]) if d_max >= 3 else None,
        "U": U,
        "U_over_J1": U / float(J[0]),
        "Omega_R": float(Omega_R),
        "cutoff": float(cutoff),
        "S_sum_AA": diag["F_k0_AA"],
        "S_sum_AB": diag["F_k0_AB"],
        "S_max_AB": diag["S_eff_AB"],
        "moire_depth_k0": float(Omega_R) * (diag["F_k0_AA"] - diag["F_k0_AB"]),
        "lowest_band_weight_min": diag["lowest_band_weight_min"],
        "interband": ib,
    }


# -- the audit ------------------------------------------------------------------------------

AUDIT_V0 = tuple(range(4, 25, 2))
AUDIT_COLUMNS = ("V0", "t_paper", "U_paper", "U_over_t_paper", "J1", "J2_over_J1", "U", "U_over_J1")


def audit_row(V0: float, nu_z_hz: float = config.NU_Z_HZ) -> dict:
    """One row of the parameter audit: the paper's t, U, U/t against the exact J1, J2/J1 and the
    pancake's U(nu_z), U/J1 (no supercell needed)."""
    p = paper_parameters(V0)
    J = bands.hopping_1d(float(V0), 2)
    U = bands.onsite_U(float(V0), lm.g2d_from_nu_z(nu_z_hz))
    return {
        "V0": float(V0),
        "t_paper": p["t"],
        "U_paper": p["U"],
        "U_over_t_paper": p["U_over_t"],
        "J1": float(J[0]),
        "J2_over_J1": float(J[1] / J[0]),
        "U": U,
        "U_over_J1": U / float(J[0]),
    }


def audit_table(V0_list=AUDIT_V0, nu_z_hz: float = config.NU_Z_HZ) -> list:
    """`audit_row` over V0_list."""
    return [audit_row(V0, nu_z_hz) for V0 in V0_list]


def _ratio_paper(V0: float) -> float:
    return paper_parameters(V0)["U_over_t"]


def _ratio_exact(V0: float, nu_z_hz: float) -> float:
    return bands.onsite_U(V0, lm.g2d_from_nu_z(nu_z_hz)) / float(bands.hopping_1d(V0, 1)[0])


def transition_depth(ratio, UJ_c: float, bracket=(4.0, 40.0)) -> float:
    """The depth V0 (E_R) at which ratio(V0) = UJ_c, by brentq on log(ratio) - log(UJ_c) (U/J grows
    about exponentially with sqrt(V0))."""
    return float(brentq(lambda V0: math.log(ratio(V0)) - math.log(UJ_c), *bracket, xtol=1e-10))


def transition_depths(nu_z_hz: float = config.NU_Z_HZ) -> dict:
    """The Omega_R = 0 Mott transition of the plain square lattice at U/J = UJ_C["gutzwiller"] (23.31) and
    UJ_C["qmc"] (16.74), with the paper's formulas ("paper") and with the exact J1 and the pancake's U at
    nu_z ("exact"): {"paper": {"gutzwiller": V0, "qmc": V0}, "exact": {...}} in E_R."""
    return {
        "paper": {k: transition_depth(_ratio_paper, v) for k, v in UJ_C.items()},
        "exact": {k: transition_depth(lambda V0: _ratio_exact(V0, nu_z_hz), v) for k, v in UJ_C.items()},
    }


def paper_quoted_ratio_check() -> dict:
    """The paper quotes U/t = 1.67 at 4 E_R (Fig. 4 caption); its own formulas with a_s = 5.29 nm give
    1.62. Reports the gap and what single change would close it, as candidates only (not applied anywhere):
    the scattering length (U ~ a_s), the depth (d ln(U/t)/dV0 = 1/sqrt(V0)), or a_s = 100 a0."""
    p = paper_parameters(4.0)
    gap = PAPER_QUOTED_U_OVER_T_4ER / p["U_over_t"] - 1.0
    a_s_needed = config.A_S_M * (1.0 + gap)
    dV0_needed = math.log(1.0 + gap) * math.sqrt(4.0)
    p100 = paper_parameters(4.0, a_s_m=config.A_S_M_ENGINE)
    return {
        "quoted": PAPER_QUOTED_U_OVER_T_4ER,
        "formulas": p["U_over_t"],
        "relative_gap": gap,
        "a_s_m_needed": a_s_needed,
        "a_s_a0_needed": a_s_needed / 5.29177210903e-11,
        "dV0_needed_ER": dV0_needed,
        "with_a_s_100a0": p100["U_over_t"],
        "Omega_R_from_quoted_ratio_2.07": 2.07 * p["t"],
    }


def format_audit(rows, depths: dict | None = None, nu_z_hz: float = config.NU_Z_HZ) -> str:
    """The audit as text: the table over V0, the transition depths, and the U/t = 1.67 note."""
    u = lattice_units(omega_z_hz=nu_z_hz)
    lines = [
        f"Bose-Hubbard parameter audit (E_R units). Paper: t = (4/sqrt pi) V0^3/4 e^(-2 sqrt V0), "
        f"U = sqrt(8/pi) k a_s V0^3/4 (3D cubic lattice), a_s = {config.A_S_M * 1e9:.2f} nm.",
        f"Exact: J1, J2 of the lowest band; U = g2d (int w^4)^2 at nu_z = {nu_z_hz:g} Hz "
        f"(l_z = {u.l_z:.4f} a, hbar omega_z = {u.omega_z:.4f} E_R, g2d = {u.g2d:.7f} E_R a^2 derived, "
        f"lattice_model.G2D_DEFAULT = {lm.G2D_DEFAULT} is {lm.G2D_DEFAULT_OFFSET:.1e} below).",
        "",
        f"{'V0':>4s} {'paper t':>9s} {'paper U':>9s} {'paper U/t':>10s} {'exact J1':>9s} {'J2/J1':>8s} "
        f"{'U(nu_z)':>9s} {'U/J1':>8s}",
    ]
    for r in rows:
        lines.append(
            f"{r['V0']:4.0f} {r['t_paper']:9.5f} {r['U_paper']:9.5f} {r['U_over_t_paper']:10.3f} "
            f"{r['J1']:9.5f} {r['J2_over_J1']:+8.4f} {r['U']:9.5f} {r['U_over_J1']:8.3f}"
        )
    if depths is not None:
        lines += [
            "",
            "Omega_R = 0 Mott transition of the plain square lattice, depth in E_R:",
            f"{'':22s} {'U/J = 23.31 (Gutzwiller)':>26s} {'U/J = 16.74 (QMC)':>20s}",
            f"{'paper formulas':22s} {depths['paper']['gutzwiller']:26.2f} {depths['paper']['qmc']:20.2f}",
            f"{'exact J1, U(nu_z)':22s} {depths['exact']['gutzwiller']:26.2f} {depths['exact']['qmc']:20.2f}",
        ]
    q = paper_quoted_ratio_check()
    lines += [
        "",
        f"The paper quotes U/t = {q['quoted']} at 4 E_R; its formulas give {q['formulas']:.4f} "
        f"({100 * q['relative_gap']:.1f} % less). One change that would close it (candidates, not applied): "
        f"a_s = {q['a_s_a0_needed']:.1f} a0, or V0 = {4 + q['dV0_needed_ER']:.3f} E_R; a_s = 100 a0 gives "
        f"{q['with_a_s_100a0']:.4f}. Omega_R/t = 2.07 with the paper's t is Omega_R = "
        f"{q['Omega_R_from_quoted_ratio_2.07']:.3f} E_R.",
    ]
    return "\n".join(lines)


# -- the occupation-dependent U (report only) --------------------------------------------------


def occupation_dependent_U(V0: float, nu_z_hz: float = config.NU_Z_HZ, n_list=(1, 2, 3), n_x: int = 64) -> dict:
    """An estimate of how much the pancake broadens under n atoms on one site.
    The fixed-width reduction gives U = g2d (int w^4)^2. With the NPSE width (engine.npse: a local
    Gaussian of width s(n_2D) l_z minimising the energy per area, s^4 - x s - 1 = 0, x = g2d n_2D /
    hbar omega_z) at the local 2D density n_2D(r) = n w(x)^2 w(y)^2 of n atoms in one Wannier orbital, the
    mean-field interaction energy is E_int(n) = int e_int(n_2D(r)) d^2r (e_int includes the width's
    zero-point cost, e_int -> g2d n_2D^2 / 2 as n_2D -> 0), and U_n = 2 E_int(n) / n^2 is the U that
    reproduces it in (U/2) n^2. Returns U (fixed width), U_n and U_n / U per n, the peak density
    n w(0)^4, the NPSE parameter x at the peak and the peak width s there. A ratio U_2 / U_1 below 0.95
    means the Mott physics at double occupancy sees a weaker U than the fixed-width model (a flag for the
    Gutzwiller and Monte Carlo solvers). The sign is set by the broadening alone: no three-body or beyond-mean-field terms."""
    import torch

    from twisted_bilayer.engine.npse import NPSE2D

    units = lattice_units(omega_z_hz=nu_z_hz)
    x, w = bands.wannier_1d(V0, n_x=n_x, n_sites=8)
    dx = x[1] - x[0]
    w2 = w * w
    dens1 = w2[:, None] * w2[None, :]  # one atom: the 2D density on the (x, y) grid, int = 1
    npse = NPSE2D(units.g2d, units.omega_z)
    U_fixed = bands.onsite_U(V0, units.g2d)
    out = {
        "V0": float(V0),
        "nu_z_hz": float(nu_z_hz),
        "U_fixed": U_fixed,
        "peak_density_1": float(dens1.max()),
        "n": {},
    }
    for n in n_list:
        d = torch.as_tensor(n * dens1, dtype=torch.float64)
        E_int = float(npse.e_int(d).sum()) * dx * dx
        U_n = 2.0 * E_int / n**2
        out["n"][int(n)] = {
            "U_n": U_n,
            "U_n_over_U": U_n / U_fixed,
            "x_peak": float(units.g2d * n * dens1.max() / units.omega_z),
            "s_peak": float(npse.width(d).max()),
        }
    out["U2_over_U1"] = out["n"][2]["U_n"] / out["n"][1]["U_n"] if 1 in out["n"] and 2 in out["n"] else None
    return out


# -- the sign audit ---------------------------------------------------------------------------


def sign_audit(model: lm.LatticeModel) -> dict:
    """What a bosonic Monte Carlo (SSE, worm) would have to carry as a sign problem. The off-diagonal
    elements of H0 must be <= 0 for positive weights. Intralayer: -J1 < 0 is fine, -J2 > 0 (J2 < 0) is
    not. Interlayer: +Omega_R S_ij, with S > 0 for every pair inside |d'| < 1; the gauge c_j -> -c_j on
    layer 2 makes those negative, and leaves the pairs with S < 0 (one 1D factor with 1 < |d'| < 2, so
    only with cutoff > 1) positive. Returns, after the gauge, per class ("intralayer", "interlayer"):
    the number of directed bond entries, the number and the fraction of entries that stay positive
    ("frustrated"), and the fraction of the absolute weight sum |amp| they carry; plus J2/J1 and the
    total frustrated weight fraction. A one-layer model has no interlayer class (zeros)."""
    b = model.bonds
    c = model.cell.c
    inter = (b.rows // c) != (b.cols // c)
    amp = b.amp.astype(np.float64).copy()
    amp[inter] *= -1.0  # the gauge on layer 2
    w = np.abs(amp)
    pos = amp > 0.0

    def cls(sel):
        n = int(sel.sum())
        nf = int((sel & pos).sum())
        wt = float(w[sel].sum())
        return {
            "entries": n,
            "frustrated_entries": nf,
            "frustrated_entry_fraction": nf / n if n else 0.0,
            "frustrated_weight_fraction": float(w[sel & pos].sum()) / wt if wt > 0.0 else 0.0,
            "weight": wt,
        }

    out = {"intralayer": cls(~inter), "interlayer": cls(inter)}
    out["J2_over_J1"] = float(model.J[1] / model.J[0]) if model.J.size >= 2 else 0.0
    out["frustrated_weight_fraction"] = float(w[pos].sum() / w.sum()) if w.sum() > 0.0 else 0.0
    out["sign_free"] = not bool(pos.any())
    return out
