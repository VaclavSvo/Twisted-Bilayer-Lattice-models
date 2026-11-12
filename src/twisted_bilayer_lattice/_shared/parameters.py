"""Twisted-bilayer optical-lattice parameters (Meng et al., Nature 615, 231 (2023)), derived.

The inputs (atoms, lattice, twist, pancake, coupling, trap, imaging, numerical gates) are set in
`twisted_bilayer.config`; this module adds the constants of nature, derives the code units and couplings, and
provides the unit conversions. The engine, the time-of-flight pipeline and the gates take their numbers from here.
No dependencies beyond the standard library.

Unit systems
------------
the time-of-flight pipeline and the gates ("code units"): hbar = m(87Rb) = 1, length a = 395 nm,
    time T = m a^2 / hbar = 213.517 us, energy hbar/T  ->  E_R = pi^2/2 = 4.934802.
The engine ("lattice units"): hbar = 1, length a, energy E_R = 1, m = pi^2/2,
    time hbar/E_R = 43.268 us.
Lengths (a) and densities (atoms per a^2) are identical in both systems.
Energies, rates and couplings: code = E_R-units * E_R_CODE.  Times: code = E_R-units / E_R_CODE.
The kinetic operator is -1/2 lap (code) = -lap/pi^2 (E_R units).

The coupling J_perp is the off-diagonal matrix element of the Hamiltonian (paper's
Supplement Eq. 4, H contains Omega_R psi_2). The engine calls it Omega_R.
"""

from __future__ import annotations

import math

from . import config as _config

CONFIG = _config  # the physical inputs (twisted_bilayer.config)
_C = CONFIG

# ---------------------------------------------------------------------------
# SI constants (CODATA 2018)
# ---------------------------------------------------------------------------
H_SI = 6.62607015e-34  # J s (exact)
HBAR_SI = 1.054571817e-34  # J s (the value the pinned unit system uses)
KB_SI = 1.380649e-23  # J/K (exact)
AMU_SI = 1.66053906660e-27  # kg
M_RB87_SI = _C.MASS_U * AMU_SI  # kg
A0_SI = 5.29177210903e-11  # Bohr radius, m

# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------
A_LATTICE_SI = _C.LATTICE_CONSTANT_M  # a = lambda_L/2 (m)
A_NM = round(A_LATTICE_SI * 1e9, 9)  # 395.0
A_UM = A_NM / 1000.0  # 0.395
TWIST_M, TWIST_N = _C.TWIST_M, _C.TWIST_N  # tan(theta/2) = n/m: theta = 5.2051 deg at 22, 1
L_SUPERCELL = math.sqrt(TWIST_M**2 + TWIST_N**2)  # sqrt(485) = 22.022716 a: periodic box, 2 x 2 moire cells
L_MOIRE = L_SUPERCELL / 2.0  # 11.011358 a = 4.34949 um
C_ROT, S_ROT = TWIST_M / L_SUPERCELL, TWIST_N / L_SUPERCELL  # cos, sin of theta/2

# ---------------------------------------------------------------------------
# Interaction and pancake
# ---------------------------------------------------------------------------
A_S_SI = _C.A_S_M  # 5.29 nm (the engine uses 100 a0: see engine.units.EXCEPTIONS); SU(2): g11 = g22 = g12
NU_Z_HZ = _C.NU_Z_HZ  # vertical trap frequency (paper: ~1 kHz)

# ---------------------------------------------------------------------------
# Derived code units
# ---------------------------------------------------------------------------
T_UNIT_S = M_RB87_SI * A_LATTICE_SI**2 / HBAR_SI  # 2.135172e-4 s
T_UNIT_MS = 1e3 * T_UNIT_S  # 0.2135172 ms
HZ_PER_CODE = 1.0 / (2.0 * math.pi * T_UNIT_S)  # nu[Hz] = E[code] * 745.3965
E_R_CODE = math.pi**2 / 2.0  # 4.934802
E_R_HZ = E_R_CODE * HZ_PER_CODE  # 3678.38 Hz
OMEGA_Z = 2.0 * math.pi * NU_Z_HZ * T_UNIT_S  # 1.3415679
L_Z = 1.0 / math.sqrt(OMEGA_Z)  # 0.8633635 a = 341.03 nm
HBAR_OMEGA_Z_ER = OMEGA_Z / E_R_CODE  # 0.271858 E_R


def g2d(a_s_si: float = A_S_SI, omega_z: float = OMEGA_Z) -> float:
    """Quasi-2D contact coupling g_2D = g_3D/(sqrt(2 pi) l_z), g_3D = 4 pi a_s/a, in code
    units (energy x a^2). Divide by E_R_CODE for E_R a^2."""
    g3d = 4.0 * math.pi * a_s_si / A_LATTICE_SI
    return g3d / (math.sqrt(2.0 * math.pi) * omega_z**-0.5)


G3D = 4.0 * math.pi * A_S_SI / A_LATTICE_SI  # 0.1682939 (code, a^3)
G2D = g2d()  # 0.0777651 (code, a^2)
G2D_ER = G2D / E_R_CODE  # 0.0157585 E_R a^2


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def er_to_code(E_er: float) -> float:
    """Energies, rates and couplings (incl. g in E_R a^2) from E_R units to code units."""
    return E_er * E_R_CODE


def code_to_er(E: float) -> float:
    return E / E_R_CODE


def t_er_to_code(t_er: float) -> float:
    """Time in hbar/E_R (bilayer engine) -> code time (E*t is invariant)."""
    return t_er / E_R_CODE


def t_code_to_er(t: float) -> float:
    return t * E_R_CODE


# ---------------------------------------------------------------------------
# Experiment (Meng et al.) and model choices
# ---------------------------------------------------------------------------
# J_perp values run (code units), keyed by their E_R labels (twisted_bilayer.config section 4):
#   0.121: 0.599 = 0.121383 E_R;  0.150: 0.7404 = 0.150036 E_R;  0.243: 1.19802 = 0.242770 E_R
J_PERP_CODE = dict(_C.J_PERP_CODE)
T_TOF_MS, T_MF_STAGE_MS = _C.T_TOF_MS, _C.T_MF_STAGE_MS

# ---------------------------------------------------------------------------
# Imaging (Meng et al. Supplement I; Rb D2 probe)
# ---------------------------------------------------------------------------
LAMBDA_PROBE_UM = _C.LAMBDA_PROBE_UM
NA_IMG = _C.NA_IMG  # in-situ objective
PIXEL_INSITU_UM = _C.PIXEL_INSITU_UM  # M = 50
PIXEL_TOF_UM = _C.PIXEL_TOF_UM  # M = 11 (the enlarged inset uses M = 22: 0.73 um)
PSF_SIGMA_A = _C.PSF_SIGMA_A  # Gaussian PSF of the legacy incoherent model (0.247 um)
GAMMA_D2 = 2.0 * math.pi * _C.GAMMA_D2_HZ  # 1/s
T_PULSE_S = _C.T_PULSE_S  # imaging pulse
S_SAT = _C.S_SAT  # saturation used for C_coh (model choice)
V_RECOIL = H_SI / (M_RB87_SI * LAMBDA_PROBE_UM * 1e-6)  # 5.88454e-3 m/s

# TOF mean-field stage: kinetic phase per split step, 1/2 k2_max dt (rad); unstable near pi
MF_PHASE_TARGET, MF_PHASE_MAX, MF_DT_CAP = _C.MF_PHASE_TARGET, _C.MF_PHASE_MAX, _C.MF_DT_CAP

# ---------------------------------------------------------------------------
# Reference data (code units; provenance in comments)
# ---------------------------------------------------------------------------
F_FC_4ER = 0.8205  # <F> = |c00|^2 at 4 E_R: Franck-Condon factor of the q = 0 Bloch state
# dressed band bottoms E0(J_perp) at g = 0, V0 = 4 E_R: g = 0 ground state on the 176^2 supercell, residual 1e-11
E0_BAND_BOTTOM = {0.0: 15.24717083, 0.599: 14.72939727, 0.7404: 14.60116170, 1.19802: 14.17705565}
