"""Physical parameters of the twisted-bilayer experiment, in one place.

`twisted_bilayer.parameters` derives the code units, couplings and unit conversions from these inputs;
the engine, the time-of-flight pipeline and the physics gates all read them there. Run settings and
the named scenarios are in `twisted_bilayer.scenarios`.

As shipped, the numbers are those of the experiment of Meng et al., Nature 615, 231 (2023), and the
tests and physics gates are pinned to them: a changed value shows up as a failing anchor.

Units: SI where a name ends in _M, _HZ, _MS, _S, _UM; "E_R" = the recoil energy of one lattice,
E_R = h^2 / (8 m a^2) = h x 3678.38 Hz. The coupling J_perp is the off-diagonal matrix element of the
Hamiltonian (the paper's Supplement Eq. 4); the engine calls it Omega_R.
"""

# ── 1. Atoms ──────────────────────────────────────────────────────────────────
# Rb-87 in |F=1, m=1> and |F=2, m=0>. All three scattering lengths are ~100 a0 (Supplement S IV), so the
# interaction is taken SU(2)-symmetric: g11 = g22 = g12 = g.
MASS_U = 86.909180527                 # atomic mass (u)
A_S_M = 5.29e-9                       # s-wave scattering length (m) behind the gate anchors; exactly 100 a0
                                      # is 5.29177 nm (g differs by 3.35e-4)
# The engine's own unit system (engine/units.py) uses these two; the offsets to the values above are listed
# in engine.units.EXCEPTIONS and pinned by the tests.
A_S_M_ENGINE = 100.0 * 5.29177210903e-11   # 100 a0 = 5.29177 nm
MASS_U_ENGINE = 86.909180                  # 6e-9 in E_R: cosmetic

# ── 2. Optical lattices and the twist ─────────────────────────────────────────
# Two square lattices of the same constant, rotated by +theta/2 and -theta/2, one per spin state.
LATTICE_CONSTANT_M = 395e-9           # a = lambda_L / 2 (m)
V0_ER = 4.0                           # depth of each lattice along each axis (E_R), as in the moire images
                                      # of Fig. 4. Some scenarios (twisted_bilayer.scenarios) use 8 E_R.
# The measured angle is 5.21(5) deg. The simulations use the commensurate angle tan(theta/2) = n/m, which
# makes a periodic square box ("supercell") of side L = sqrt(m^2 + n^2) a holding 2 x 2 moire cells.
TWIST_M, TWIST_N = 22, 1              # theta = 2 atan(1/22) = 5.2051 deg, L = sqrt(485) a = 22.02 a

# ── 3. Vertical confinement (the pancake) ─────────────────────────────────────
NU_Z_HZ = 1000.0                      # vertical trap frequency (Hz; the paper: ~1 kHz). Sets g_2D through
                                      # l_z = sqrt(hbar / m omega_z) = 341 nm.

# ── 4. Microwave coupling ─────────────────────────────────────────────────────
# The paper's fitted in-lattice Rabi frequency is Omega_R = 2 pi x 893 Hz (Fig. 3c; fit form not given).
# Three readings of it, in code units (energy unit hbar^2 / (m a^2) = E_R / 4.934802), keyed by their E_R labels:
#   0.121  the Rabi fit read as sin^2(Omega t / 2): J_perp = Omega_R / 2
#   0.150  0.150 E_R (0.7404 = 0.15004 E_R): the 0.121 reading divided by the lowest-band Franck-Condon
#          factor <F> = 0.8205 is 0.1479 E_R (which gives 2 <F> J_perp / h = 893 Hz), rounded up to 0.150:
#          2 <F> J_perp / h = 906 Hz, 1.4 % above 893 Hz
#   0.243  the paper-literal Omega_R
J_PERP_CODE = {0.121: 0.599, 0.150: 0.7404, 0.243: 1.19802}

# ── 5. Radial trap, cloud and timing ──────────────────────────────────────────
NU_TRAP_HZ = 20.0                     # radial trap frequency (Hz). A model assumption: omega_r is not published
N_ATOMS = 2000.0                      # atoms in the trapped cloud (also not published)
T_TOF_MS = 18.0                       # time of flight before imaging (ms)
T_MF_STAGE_MS = 1.0                   # TOF: full mean-field stage before the per-order tail (ms)

# ── 6. Imaging (Meng et al. Supplement I; Rb D2 probe) ────────────────────────
LAMBDA_PROBE_UM = 0.780241            # probe wavelength (um)
NA_IMG = 0.69                         # numerical aperture of the in-situ objective
PIXEL_INSITU_UM = 0.32                # in-situ pixel in the atom plane (M = 50)
PIXEL_TOF_UM = 1.455                  # TOF pixel (M = 11; the enlarged inset uses M = 22: 0.73 um)
PSF_SIGMA_A = 0.625                   # Gaussian PSF of the legacy incoherent model (lattice constants; 0.247 um)
GAMMA_D2_HZ = 6.065e6                 # natural linewidth of the D2 line (Hz)
T_PULSE_S = 50e-6                     # imaging pulse (s)
S_SAT = 0.01                          # probe saturation used for the coherent-imaging contrast C_coh

# TOF mean-field stage: kinetic phase per split step k2_max dt / 2 (rad; the split step is unstable near pi):
# target of dt = 'auto', refusal threshold of an explicit dt, and the largest dt (code units)
MF_PHASE_TARGET, MF_PHASE_MAX, MF_DT_CAP = 2.0, 3.0, 0.01
