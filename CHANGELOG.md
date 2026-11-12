# Changelog

User-facing changes of the package, newest first. Versions follow [semantic versioning](https://semver.org).

## 0.3.0 (November 2026)

The first version of this package on its own. The lattice models were part of `twisted-bilayer` 0.2.0
([3D-GPE-Twisted-Bilayer-optical-lattices](https://github.com/VaclavSvo/3D-GPE-Twisted-Bilayer-optical-lattices));
they moved here unchanged, so every result of 0.2.0 is reproduced.

- New import name `twisted_bilayer_lattice`, flat layout: `twisted_bilayer.engine.lattice_model` is now
  `twisted_bilayer_lattice.model`, and `twisted_bilayer.manybody.<module>` is `twisted_bilayer_lattice.<module>`
  (`gutzwiller`, `cluster`, `fast_kernels`, `observables`, `paper_model`, `parameters`, `phase_diagram`).
- The package stands alone: the modules it shares with the Gross-Pitaevskii package (inputs, derived constants,
  units, 1D bands, the moire geometry, run records) are included as copies in `twisted_bilayer_lattice/_shared/`.
- Command line `twisted-bilayer-lattice` (`list`, `run`) with the scenario kinds `tight_binding` and
  `phase_diagram`; `python -m twisted_bilayer_lattice.phase_diagram` runs a phase map on its own.
- The reference values of the square-lattice Mott transition are in `twisted_bilayer_lattice.references`.
- Not here: the comparison of the lattice mean field with the continuum Gross-Pitaevskii equation (the `bridge`
  scenario kind of 0.2.0) and `parameters.occupation_dependent_U`, which need the continuum solver.
- The five-model showcase (`examples/showcase/`) lives here; it needs the Gross-Pitaevskii package
  (`pip install -e ".[showcase]"`).

What the lattice models of 0.2.0 contain:

- A two-layer tight-binding (Bose-Hubbard) model of the twisted bilayer (`model`): discrete-GPE ground
  states with a trust-region Newton polish, the second-order interband shift (`interband=True`), superfluid fraction
  with error bounds and order checks; the paper's one-layer model (`paper_model`).
- Bose-Hubbard parameters from the continuum bands (`parameters`).
- Single-site Gutzwiller (`gutzwiller`; twisted, batched; `solve_filling`, `compressibility`, fixed-N
  superfluid fraction), cluster mean field (`cluster`), observables and phase labels
  (`observables`), and resumable phase maps over (V0, Omega_R) (`phase_diagram`, the scenario kind
  `phase_diagram`).
- `solve_batch(..., eig="fast")` with `fast_kernels`: a certified tridiagonal eigensolver for the site
  problem; the default `eig="exact"` is unchanged.
- On CUDA the batched site eigenproblem is cut into chunks (`EIGH_BATCH_MAX`), halved and retried after a cuSOLVER or
  out-of-memory failure.
