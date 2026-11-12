# Methods

How `twisted-bilayer-lattice` is used: the scenarios, the phase-map command, GPU runs, the repository layout.
The [README](../README.md) is the short version; [`manybody.md`](manybody.md) describes the models (the
Bose-Hubbard Hamiltonian and its parameters, Gutzwiller, the phase maps).

## Scenarios

`twisted-bilayer-lattice run <name>` runs a named row of
[`scenarios.py`](../src/twisted_bilayer_lattice/scenarios.py), or of your own file with `--scenarios FILE`.
Times are for 2 CPU cores.

| name | what it computes | time |
| --- | --- | --- |
| `tight_binding` | the two-layer Bose–Hubbard mean field on the experiment's supercell, three variants (always on the CPU) | 1.5 min |
| `phase_diagram_quick` | single-site Gutzwiller phase map (SF / SF-II / I / MI) over V₀ and Ω_R on the tan(θ/2) = 1/4 cell; for speed and plotting | minutes |
| `phase_diagram_paper_quick` | the same for the paper's one-layer model | minutes |
| `phase_diagram_cluster_quick` | the same with interlayer-dimer clusters (cluster Gutzwiller) | ~10 min |
| `phase_diagram`, `phase_diagram_noshift` | single-site Gutzwiller phase map on the experiment's supercell, 21 × 13 points, with and without the interband shift | ~2 h on one core per map |
| `phase_diagram_paper`, `phase_diagram_paper_mu095` | the paper's model on the experiment's supercell, at μ/U = 1 and 0.95 (the reading that reproduces the paper's Fig. S4c) | ~1 h |
| `phase_diagram_cluster` | interlayer-dimer cluster map on the experiment's supercell, 11 × 7 points | ~2 h |

The phase maps also run on their own, with more options (time budget, resume, U correction table):
`python -m twisted_bilayer_lattice.phase_diagram <row> --out <folder> [--device cuda] [--threads 1]
[--max-hours H] [--redo-unconverged] [--force-resume]`, where `<row>` is any entry of
`phase_diagram.ROWS`. A map is written after every V₀ column and a rerun into the same folder continues
it; a folder made with another row specification is refused.

The quick scenarios use the small tan(θ/2) = 1/4 cell (θ = 28°). Use them to check that everything works;
they do not reproduce the experiment's geometry.

## Running on a GPU

Install a CUDA build of PyTorch before the package (see Installation in the README). The phase maps run on
the CPU unless the row or the command line asks for `--device cuda`; `tight_binding` always runs on the CPU.
To keep everything on the CPU, pass `--cpu` to `twisted-bilayer-lattice run`, or set the environment variable
`TWISTED_BILAYER_FORCE_CPU=1` (PowerShell: `$env:TWISTED_BILAYER_FORCE_CPU = "1"`).

## Repository layout

```
src/twisted_bilayer_lattice/
  model.py          the two-layer Bose-Hubbard model on the commensurate supercell and its mean field
  parameters.py     Bose-Hubbard parameters, the paper's formulas, the parameter and sign audits
  paper_model.py    the paper's one-layer model
  gutzwiller.py     single-site Gutzwiller with twisted boundaries, batches and several starts
  fast_kernels.py   the certified tridiagonal site solver of `solve_batch(eig="fast")`
  cluster.py        cluster Gutzwiller
  observables.py    momentum distribution, visibilities, phase labels
  phase_diagram.py  phase maps over V0 and Omega_R
  references.py     reference values
  scenarios.py      the named runs of the command line
  runners.py        what each kind of run does
  cli.py            the `twisted-bilayer-lattice` command
  _shared/          copies of modules of the Gross-Pitaevskii package `twisted_bilayer`: config.py (the
                    experiment's physical inputs), parameters.py (derived constants and unit conversions),
                    records.py (run records), engine/ (backend.py, grid.py, units.py, lattice.py, bands.py).
                    They are kept identical to that package's files; change them there, not here
examples/        the five-model showcase (needs the Gross-Pitaevskii package)
tests/           pytest; CPU only, two threads
scripts/         bisect_eigh_batch.py and gpu_column_check.py (the batched eigensolver on a GPU)
docs/            methods.md (this page), manybody.md
check.py         lint + tests in one command
```

## Limitations

- Mean field (the discrete Gross-Pitaevskii equation), single-site and cluster Gutzwiller; finite temperature
  is not included.
- One s-wave scattering length for all three channels (SU(2)-symmetric interaction), as in the paper.
- Commensurate twist angles only (tan(θ/2) = n/m). The measured angle 5.21(5)° is represented by 1/22.
- The lowest band of each lattice only; the second-order interband shift of the site energies is optional.
