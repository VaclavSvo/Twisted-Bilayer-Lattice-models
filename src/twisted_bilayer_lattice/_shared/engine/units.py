"""SI <-> lattice units.

Lattice units: a = 1 (lattice constant, lambda/2), E_R = 1 (recoil energy,
E_R = hbar^2 pi^2 / 2 m a^2), hbar = 1. Then

    m = pi^2 / 2,     kappa = hbar^2/(2m) = 1/pi^2,     t_0 = hbar/E_R,

and the kinetic operator is -kappa nabla^2 = -(1/pi^2) nabla^2. Every parameter the
engine takes (V_0, Omega_R, Delta, g, omega_z, temperature) is in these units.

The SI constants come from `twisted_bilayer.parameters` (inputs in
`twisted_bilayer.config`). Two inputs are the engine's own, set in the config
(A_S_M_ENGINE, MASS_U_ENGINE) and listed with their offset to the shared
values in `EXCEPTIONS`:

  * the s-wave length: 100 a_0 = 5.29177 nm here, parameters.A_S_SI = 5.29 nm
    (g +3.350e-4);
  * the Rb-87 mass: 86.909180 u here, 86.909180527 u in the parameters module
    (E_R, t_0 and omega_z in E_R units differ by 6.1e-9).
The two scattering lengths are a documented difference: the gate anchors use
5.29 nm, the engine's unit system 100 a_0. tests/test_units.py pins the derived
unit system and both offsets.

The code units of the time-of-flight pipeline and the gates (hbar = m = a = 1) differ from
these by E_code = E * E_R_CODE, t_code = t / E_R_CODE (E_R_CODE = pi^2/2);
convert with the re-exported helpers of the parameters module (`er_to_code`, `code_to_er`,
`t_er_to_code`, `t_code_to_er`), never with a literal.

Pure math, no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import parameters as _P

HBAR = _P.HBAR_SI  # J s
H_PLANCK = 2.0 * math.pi * HBAR  # 2 pi hbar (parameters.H_SI differs by 6e-10)
U_ATOMIC_MASS = _P.AMU_SI  # kg
BOHR_RADIUS = _P.A0_SI  # m
A_LATTICE_M = _P.A_LATTICE_SI  # default lattice constant (m)
NU_Z_HZ = _P.NU_Z_HZ  # default pancake frequency (Hz)
E_R_CODE = _P.E_R_CODE  # E_R in code units (hbar = m = a = 1): pi^2/2
er_to_code, code_to_er = _P.er_to_code, _P.code_to_er
t_er_to_code, t_code_to_er = _P.t_er_to_code, _P.t_code_to_er

# Rb-87, the species of Meng et al. (Nature 615, 231, 2023). All three
# scattering lengths of the |1,1>, |2,0> pair are ~100 a_0 (Supplement S IV),
# hence the SU(2)-symmetric g used in the paper and here.
RB87_MASS_U = _P.CONFIG.MASS_U_ENGINE  # 86.909180 (config section 1)
RB87_SCATTERING_LENGTH_M = _P.CONFIG.A_S_M_ENGINE  # 100 a0 = 5.29177 nm (config section 1)

# The engine's own inputs against the shared ones: (here, parameters, here/parameters - 1, why).
# Setting them equal in the config removes the offsets; tests/test_units.py pins them.
EXCEPTIONS = {
    "RB87_SCATTERING_LENGTH_M": (
        RB87_SCATTERING_LENGTH_M,
        _P.A_S_SI,
        RB87_SCATTERING_LENGTH_M / _P.A_S_SI - 1.0,
        "a_s = 100 a0 here vs parameters.A_S_SI = 5.29 nm (gate anchors)",
    ),
    "RB87_MASS_U": (
        RB87_MASS_U,
        _P.M_RB87_SI / _P.AMU_SI,
        RB87_MASS_U / (_P.M_RB87_SI / _P.AMU_SI) - 1.0,
        "86.909180 u here vs 86.909180527 u in parameters",
    ),
}

SPECIES = {
    "Rb87": (RB87_MASS_U, RB87_SCATTERING_LENGTH_M),
}

M_LATTICE = math.pi**2 / 2.0  # m in lattice units
KAPPA = 1.0 / math.pi**2  # hbar^2 / 2m in lattice units


@dataclass(frozen=True)
class LatticeUnits:
    """The experiment expressed in lattice units."""

    species: str
    atom_mass_kg: float
    scattering_length_m: float
    a_m: float  # lattice constant (m)
    E_R_J: float  # recoil energy (J)
    E_R_Hz: float  # E_R / h
    t0_s: float  # hbar / E_R
    a_s: float  # scattering length in units of a
    g3d: float  # 4 pi hbar^2 a_s / m, units E_R a^3
    omega_z: float  # vertical trap frequency, units E_R/hbar (0 if none)
    l_z: float  # vertical oscillator length, units of a (inf if none)
    g2d: float  # g3d / (sqrt(2 pi) l_z), units E_R a^2

    def ER_to_hz(self, E: float) -> float:
        return E * self.E_R_Hz

    def rad_s_to_ER(self, omega_rad_s: float) -> float:
        """An angular frequency omega -> hbar omega / E_R."""
        return HBAR * omega_rad_s / self.E_R_J

    def lattice_time_to_s(self, t: float) -> float:
        return t * self.t0_s


def lattice_units(species: str = "Rb87", a_m: float = A_LATTICE_M, omega_z_hz: float | None = NU_Z_HZ) -> LatticeUnits:
    """Build the unit system of the experiment: Rb-87, a = 395 nm,
    omega_z = 2 pi x 1 kHz (Meng et al.)."""
    mass_u, a_s_m = SPECIES[species]
    m_kg = mass_u * U_ATOMIC_MASS
    E_R = HBAR**2 * math.pi**2 / (2.0 * m_kg * a_m**2)
    E_R_Hz = E_R / H_PLANCK
    t0 = HBAR / E_R
    a_s = a_s_m / a_m
    g3d = 4.0 * math.pi * KAPPA * 2.0 * a_s  # 4 pi hbar^2 a_s / m, hbar^2/m = 2 kappa
    if omega_z_hz:
        omega_z = HBAR * 2.0 * math.pi * omega_z_hz / E_R
        l_z = math.sqrt(1.0 / (M_LATTICE * omega_z))
        g2d = g3d / (math.sqrt(2.0 * math.pi) * l_z)
    else:
        omega_z, l_z, g2d = 0.0, math.inf, 0.0
    return LatticeUnits(
        species=species,
        atom_mass_kg=m_kg,
        scattering_length_m=a_s_m,
        a_m=a_m,
        E_R_J=E_R,
        E_R_Hz=E_R_Hz,
        t0_s=t0,
        a_s=a_s,
        g3d=g3d,
        omega_z=omega_z,
        l_z=l_z,
        g2d=g2d,
    )


def compare_with_parameters(omega_z_hz: float = NU_Z_HZ) -> dict:
    """The engine's derived quantities next to those of the parameters module
    (relative offsets): the numbers behind EXCEPTIONS."""
    u = lattice_units(omega_z_hz=omega_z_hz)
    ref = dict(
        g2d=_P.G2D_ER,
        g3d=_P.G3D / E_R_CODE,
        E_R_Hz=_P.E_R_HZ,
        omega_z=_P.HBAR_OMEGA_Z_ER,
        l_z=_P.L_Z,
        t0_s=_P.T_UNIT_S / E_R_CODE,
    )
    return {k: dict(engine=getattr(u, k), parameters=v, rel=getattr(u, k) / v - 1.0) for k, v in ref.items()}
