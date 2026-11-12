"""Command line: `twisted-bilayer-lattice` (or `python -m twisted_bilayer_lattice`).

    twisted-bilayer-lattice list                             the scenarios and their kinds
    twisted-bilayer-lattice run phase_diagram_quick          run one; files in results/phase_diagram_quick/
    twisted-bilayer-lattice run tight_binding --out runs     ... in runs/tight_binding/
    twisted-bilayer-lattice run my_row --scenarios rows.py   a row of your own scenario file
    twisted-bilayer-lattice run phase_diagram_quick --cpu    stay on the CPU when a GPU is present

Exit codes of `run`: 0 done; 1 on any error; 2 a usage error (an unknown scenario).

The device is picked when the package's `_shared.engine.backend` is imported, so this module imports the runners
only after the command line has decided whether the run must stay on the CPU. The phase maps also run on their
own, with more options: `python -m twisted_bilayer_lattice.phase_diagram <row> --out <folder>`.
"""

from __future__ import annotations

import argparse
import os
import sys

FORCE_CPU = "TWISTED_BILAYER_FORCE_CPU"
CPU_ONLY_KINDS = ("tight_binding",)  # validated on the CPU only


def _cmd_list(a) -> int:
    from twisted_bilayer_lattice.scenarios import load_scenarios

    rows = load_scenarios(a.scenarios)
    width = max(len(n) for n in rows)
    for name, row in rows.items():
        print(f"{name:<{width}s}  {row['kind']}")
    return 0


def _cmd_run(a) -> int:
    from twisted_bilayer_lattice.scenarios import load_scenarios

    rows = load_scenarios(a.scenarios)
    if a.name not in rows:
        print(f"unknown scenario {a.name!r}; `twisted-bilayer-lattice list` shows the choices", file=sys.stderr)
        return 2
    row = rows[a.name]
    if a.cpu or row["kind"] in CPU_ONLY_KINDS:
        os.environ[FORCE_CPU] = "1"
    from twisted_bilayer_lattice.runners import run_scenario

    run_scenario(a.name, row, out_root=a.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(
        prog="twisted-bilayer-lattice",
        description="Lattice (Bose-Hubbard) models of a BEC in twisted-bilayer optical lattices.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list the scenarios")
    p.add_argument("--scenarios", metavar="FILE", help="a Python file with your own SCENARIOS dict")
    p.set_defaults(fn=_cmd_list)
    p = sub.add_parser("run", help="run one scenario")
    p.add_argument("name", help="scenario name (see `list`)")
    p.add_argument("--out", default="results", help="output folder; files go to OUT/NAME/ (default: results)")
    p.add_argument("--scenarios", metavar="FILE", help="a Python file with your own SCENARIOS dict")
    p.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is present")
    p.set_defaults(fn=_cmd_run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
