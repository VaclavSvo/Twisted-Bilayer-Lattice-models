# twisted-bilayer-lattice — lattice (Bose–Hubbard) models for BECs in twisted-bilayer optical lattices

[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`twisted-bilayer-lattice` builds and solves the two-layer Bose–Hubbard model of a Bose–Einstein condensate held
in two square optical lattices twisted against each other, one lattice per hyperfine state, coupled by a
microwave field. It is written in PyTorch and runs on CPU and CUDA. The model follows the experiment of
Meng *et al.*,
[*Atomic Bose–Einstein condensate in twisted-bilayer optical lattices*, Nature **615**, 231 (2023)](https://doi.org/10.1038/s41586-023-05695-4):
⁸⁷Rb, lattice constant a = 395 nm, twist angle θ = 5.2°, a pancake-shaped vertical trap. Lattice units
throughout: a = E_R = ħ = 1.

**What it computes.** The Hamiltonian on the exactly periodic moiré supercell (tan(θ/2) = n/m), with its
parameters from the exact lowest band of each lattice (hopping, on-site interaction, interlayer overlaps, the
second-order interband shift); the paper's own one-layer model; the coherent-state mean field (a discrete
Gross–Pitaevskii equation) with the superfluid fraction from a phase twist; single-site Gutzwiller with twisted
boundaries; cluster Gutzwiller; momentum distributions, visibilities and phase labels (SF, SF-II, I, MI); and
resumable phase maps over the lattice depth V₀ and the coupling Ω_R.

The continuum (Gross–Pitaevskii) description of the same system is a separate package,
[3D-GPE-Twisted-Bilayer-optical-lattices](https://github.com/VaclavSvo/3D-GPE-Twisted-Bilayer-optical-lattices).
This package does not need it: the single-particle and bookkeeping modules both use (inputs, units, 1D bands,
the moiré geometry) are included as copies in `twisted_bilayer_lattice/_shared/`. Only the showcase example,
which compares the lattice models with the continuum, installs the other package.

## Installation

Python 3.10 or newer. Install PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/) has the
command for your CUDA version; a plain `pip install torch` on Linux pulls a CUDA build of about 6 GB), then:

```bash
git clone https://github.com/VaclavSvo/Twisted-Bilayer-Lattice-models.git
cd Twisted-Bilayer-Lattice-models
pip install -e ".[dev]"            # numpy, scipy, matplotlib, threadpoolctl + pytest, ruff
```

## Quick start

```bash
twisted-bilayer-lattice list                         # the named runs
twisted-bilayer-lattice run phase_diagram_quick      # a small phase map; files in results/phase_diagram_quick/
python check.py                                      # lint + the test suite
```

## Scenarios

`twisted-bilayer-lattice list` shows every named run; `twisted-bilayer-lattice run NAME` runs one (`--cpu` to
stay on the CPU, `--scenarios FILE` for rows of your own). A selection:

| name | what it computes | time (2 CPU cores) |
| --- | --- | --- |
| `tight_binding` | the two-layer Bose–Hubbard mean field on the experiment's supercell, three variants | 1.5 min |
| `phase_diagram_quick` | a single-site Gutzwiller phase map over V₀ and Ω_R on a small cell | minutes |
| `phase_diagram_paper_quick` | the same for the paper's one-layer model | minutes |
| `phase_diagram_cluster_quick` | the same with interlayer-dimer clusters (cluster Gutzwiller) | ~10 min |
| `phase_diagram`, `phase_diagram_paper`, `phase_diagram_cluster` | the maps on the experiment's supercell (θ = 5.2°) | hours |

The quick rows use a small tan(θ/2) = 1/4 cell; they check that everything works, not the experiment's
geometry. The full table and the options of the phase maps are in [`docs/methods.md`](docs/methods.md).

## Validation

The test suite pins the model against exact limits and independent calculations: the interband shift against
the continuum's single-particle levels, the superfluid fraction at U = 0, the single-site Gutzwiller tip of the
n = 1 Mott lobe of the square lattice at z (3 + 2√2) = 23.31, cluster Gutzwiller against exact diagonalisation
of small lattices, and the batched Gutzwiller solver against the model file's own solver. The reference values
are in
[`references.py`](src/twisted_bilayer_lattice/references.py). `python check.py --all` runs everything.

## Documentation

- [`docs/manybody.md`](docs/manybody.md): the Hamiltonian and its parameters, the paper's model, the interband
  shift, single-site and cluster Gutzwiller, the phase maps.
- [`docs/methods.md`](docs/methods.md): every scenario, the phase-map command, GPU runs, the repository layout,
  limitations.
- [`CHANGELOG.md`](CHANGELOG.md) and the [showcase example](examples/showcase/).

## How this was built

The physics, the model choices and the validation design are the author's. The implementation was
AI-assisted, written under the author's direction and reviewed by him. What the author stands behind is the
validation: the test suite and the pinned reference numbers make any unexplained change in a result fail
`python check.py`.

## Citing, licence and author

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff), and the experiment it models. MIT
licence; see [`LICENSE`](LICENSE).

Václav Svoboda, MSc Mathematical Physics, University of Vienna; Wolfgang Pauli Institute, Vienna (Mauser group).
