"""twisted_bilayer_lattice: lattice (Bose-Hubbard) models of a Bose-Einstein condensate in twisted-bilayer
optical lattices (the experiment of Meng et al., Nature 615, 231 (2023)).

Modules: `model` (the two-layer Bose-Hubbard model on the commensurate supercell and its mean field, the
discrete GPE), `parameters` (the Hamiltonian's parameters, the paper's formulas, the parameter and sign audits),
`paper_model` (the paper's one-layer model), `gutzwiller` (single-site Gutzwiller) with `fast_kernels`, `cluster`
(cluster Gutzwiller), `observables` (momentum distribution, visibilities, phase labels), `phase_diagram` (phase
maps over V0 and Omega_R), `references` (reference values). Command line: `twisted-bilayer-lattice` (`cli`).

`_shared` holds copies of the single-particle and bookkeeping modules of the Gross-Pitaevskii package
`twisted_bilayer` (inputs, derived constants, units, 1D bands, the moire geometry, run records), so this package
runs without it. Nothing is imported here: the device is chosen when `_shared.engine.backend` is first imported,
and the command line decides before that whether a run must stay on the CPU.
"""

__version__ = "0.3.0"
