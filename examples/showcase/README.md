# Showcase: one trapped cloud, five models

`showcase.ipynb` computes the same trapped cloud of the twisted bilayer (Meng et al., Nature 615, 231 (2023);
V₀ = 4 E_R, N = 2000, ν_r = 20 Hz) with five models:

- full 3D;
- the NPSE (quasi-2D);
- fixed g (quasi-2D);
- the two-layer lattice GPE;
- single-site Gutzwiller.

It runs at three couplings, in 3 × 3 and 5 × 5 supercell boxes, and shows the images as a camera records
them, cuts, the observables and each model's deviation from full 3D. All physics comes from two packages: the
lattice models from this one (`twisted_bilayer_lattice`), the three continuum models from the Gross–Pitaevskii
package `twisted_bilayer` ([3D-GPE-Twisted-Bilayer-optical-lattices](https://github.com/VaclavSvo/3D-GPE-Twisted-Bilayer-optical-lattices)).
`showcase_helpers.py` only wires them together and plots.

## Requirements

Both packages and Jupyter in one environment: `pip install -e ".[showcase]"` from the repository root installs
this package, the Gross–Pitaevskii package and Jupyter. The notebook is stored without outputs; running it draws
everything.

## Run

1. Open `showcase.ipynb`, set `QUICK = True` and run all cells: a one-minute smoke test whose numbers are not
   physical.
2. Set `QUICK = False` and run all cells again. From scratch, on a GPU with 6 GiB of memory, this takes about
   50–80 minutes: the continuum models about 3–5 min (3 × 3) and 10–15 min (5 × 5) per coupling, the lattice
   models about 5 min per coupling on the CPU. The 5 × 5 3D solve peaks at about 2.5 GiB in complex64.
3. Results are cached in `showcase_results/` (git-ignored), so a rerun with the same parameters reloads in
   seconds, and changing one parameter (N, ν_r, a coupling, a box) computes only the new pieces.

## Parameters (first code cell)

| parameter | meaning |
| --- | --- |
| `COUPLINGS` | J⊥ in E_R. 0.121, 0.150 and 0.243 use the exact values from `scenarios.J_PERP_ER`; any other number is taken as E_R |
| `BOXES` | odd numbers of √485 a supercells |
| `MODELS` | which of the five to run |
| `N_ATOMS`, `NU_R_HZ` | the cloud (atom number and radial trap frequency) |
| `V0_ER`, `NU_Z_HZ` | the lattice depth and the pancake frequency |
| `CELL_POINTS`, `NZ`, `DTYPE_3D` | resolution and precision; the defaults are converged settings |
| `S_SAT` | imaging saturation used for C_coh |
| `SHOW_J`, `SHOW_BOX` | which cloud the image panels show |
| `FORCE` | recompute even if cached |

## Notes

- **Lattice models and the box.** The lattice models run in the 3 × 3 box and are padded with empty sites for
  5 × 5: a 5 × 5 lattice model needs more than 6 GB of RAM for its site tables, and the cloud fits in 3 × 3.
- **The lattice models' settings.** Variant C (the interlayer overlap cut at |d′| < 2, plus the interlayer
  interaction); the interband shift, site part only; U from the 1 kHz pancake; the same harmonic trap as the
  continuum, applied on the sites. Gutzwiller is solved at the lattice GPE's μ, then one Newton step brings it
  to the same N.
- **What to expect.** At 4 E_R the lattice models are the furthest from 3D: the tight-binding description is
  marginal that shallow (U/J₁ = 0.41, J₂/J₁ = −7 %). The NPSE sits within a few 10⁻³ of 3D in C_coh, and fixed
  g a few 10⁻² below it.
