"""Named scenarios: one row per run of `twisted-bilayer-lattice run <name>`.

A row is a dict whose `kind` names the runner in `twisted_bilayer_lattice.runners`; the other keys are that
runner's parameters. Energies are in E_R and lengths in lattice constants a, unless a key says _hz.

Your own rows live in a separate Python file that defines a dict `SCENARIOS`; pass it with
`twisted-bilayer-lattice run <name> --scenarios my_rows.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib

from twisted_bilayer_lattice._shared.config import TWIST_M, TWIST_N

SCENARIOS = {
    # the two-layer tight-binding (Bose-Hubbard) mean field, variants A/B/C, on the experiment's supercell;
    # compare_m: single-particle tight binding against the continuum on the m = 4 cell (0: skip).
    # The interaction: g2d if given, else from nu_z_hz (model.g2d_from_nu_z), else the shipped
    # model.G2D_DEFAULT (both None, as here)
    "tight_binding": dict(
        kind="tight_binding",
        V0=8.0,
        m=TWIST_M,
        n=TWIST_N,
        filling=1.0,
        g2d=None,
        nu_z_hz=None,
        theta=0.3,
        Omega_R=(0.0, 0.1, 0.24, 0.5, 1.0),
        variants="ABC",
        compare_m=4,
    ),
    # phase maps over (V0, Omega_R) with single-site / cluster Gutzwiller (phase_diagram; the parameters
    # live in its ROWS, named by "base"). Resumable: a rerun continues results/<name>/maps.npz. The m = 4 rows take
    # minutes on a CPU and are for speed and plotting only; the m = 22 rows take hours (one process per map)
    "phase_diagram_quick": dict(kind="phase_diagram", base="phase_diagram_quick"),
    "phase_diagram_paper_quick": dict(kind="phase_diagram", base="phase_diagram_paper_quick"),
    "phase_diagram_cluster_quick": dict(kind="phase_diagram", base="phase_diagram_cluster_quick"),
    "phase_diagram": dict(kind="phase_diagram", base="phase_diagram"),
    "phase_diagram_noshift": dict(kind="phase_diagram", base="phase_diagram_noshift"),
    "phase_diagram_paper": dict(kind="phase_diagram", base="phase_diagram_paper"),
    "phase_diagram_paper_mu095": dict(kind="phase_diagram", base="phase_diagram_paper_mu095"),
    "phase_diagram_cluster": dict(kind="phase_diagram", base="phase_diagram_cluster"),
}


def load_scenarios(path: str | None = None) -> dict:
    """The rows above, plus the `SCENARIOS` dict of the Python file at `path` (its rows win on a name clash)."""
    rows = dict(SCENARIOS)
    if path:
        file = pathlib.Path(path).expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError(f"scenario file {file} not found")
        spec = importlib.util.spec_from_file_location(f"_scenarios_{file.stem}", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        extra = getattr(mod, "SCENARIOS", None)
        if not isinstance(extra, dict):
            raise ValueError(f"{file} must define a dict SCENARIOS")
        rows.update(extra)
    return rows
