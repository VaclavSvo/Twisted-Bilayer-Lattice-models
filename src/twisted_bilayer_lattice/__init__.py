"""Beyond mean field on the lattice: the two-layer Bose-Hubbard Hamiltonian of
`engine.lattice_model` and the methods that solve it (docs/roadmap-beyond-mean-field.md).

Rung 0 (this package so far):
  * `parameters`: the Hamiltonian's parameters from the continuum inputs (J(d), U(nu_z), the interlayer
    coupling statistics), the paper's own formulas (t, U), the parameter audit and the square-lattice
    transition depths, and the sign audit for the Monte Carlo;
  * `paper_model`: the paper's one-layer model (Supplement Eqs. 12-13) as a `LatticeModel`, variant "paper";
  * `bridge`: the lattice mean field against the continuum GPE on the same cell (scenario kind `bridge`).
"""

from . import bridge, paper_model, parameters

__all__ = ["bridge", "paper_model", "parameters"]
