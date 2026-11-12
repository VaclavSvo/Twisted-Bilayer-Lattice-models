"""The scenario runners behind `twisted-bilayer-lattice run <scenario>`.

Each scenario kind is one function `run_<kind>(name, row)` that runs a row of `twisted_bilayer_lattice.scenarios`,
prints a summary, and writes its files to `<out>/<name>/`. Adding a kind: one function here, one entry in
`KINDS`, and rows in a scenarios table.

Importing this module imports `_shared.engine`, which fixes the device: `twisted_bilayer_lattice.cli` sets
TWISTED_BILAYER_FORCE_CPU before it imports this module when a run must stay on the CPU.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import torch

from twisted_bilayer_lattice._shared.engine import bands
from twisted_bilayer_lattice._shared.engine.backend import describe_device

VERBOSE = True  # progress output of the long runs
_OUT_ROOT = "results"  # set by run_scenario


def _section(title: str) -> None:
    print(f"\n-- {title} --")


def _outdir(name: str) -> str:
    d = os.path.join(_OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


# ── phase maps: single-site and cluster Gutzwiller ────────────────────────────


def run_phase_diagram(name: str, row: dict) -> None:
    """Phase map over (V0, Omega_R) with single-site or cluster Gutzwiller (phase_diagram): labels
    SF / SF-II / I / MI / empty / ?, f_s(theta -> 0), visibilities, P(n_max); maps.npz, labels.png,
    summary.json, report.txt and phase_diagram_provenance.json in results/<name>/ (resumable: a rerun continues
    the map). The row names its entry of phase_diagram.ROWS as "base"; other keys override it, except
    the run controls "device", "max_hours", "redo_unconverged" and "force_resume"."""
    from twisted_bilayer_lattice import phase_diagram as PD

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ctl = ("kind", "base", "device", "max_hours", "redo_unconverged", "force_resume")
    spec = dict(PD.ROWS[row.get("base", name)])
    spec.update({k: v for k, v in row.items() if k not in ctl})
    out = _outdir(name)
    _section("run")
    summary = PD.run_row(
        name,
        out,
        device=row.get("device", "cpu"),
        max_hours=row.get("max_hours"),
        row=spec,
        redo_unconverged=bool(row.get("redo_unconverged", False)),
        force_resume=bool(row.get("force_resume", False)),
        provenance_file="phase_diagram_provenance.json",
    )
    _section("result")
    print(PD.format_report(name, spec, PD._load_npz(os.path.join(out, "maps.npz")), summary))


# ── two-layer tight binding ───────────────────────────────────────────

MODEL_C_VARIANTS = {
    "A": ("one neighbour: S cut at |d| < 1, lattice-1 frame, no interlayer interaction", {}),
    "B": ("S cut at |d| < 2", {"cutoff": 2.0}),
    "C": ("S cut at |d| < 2 + interlayer interaction V_ij", {"cutoff": 2.0, "interlayer_interaction": True}),
    "D": (
        "C + the interband shift of the site energies (no interband bonds)",
        {"cutoff": 2.0, "interlayer_interaction": True, "interband": True, "interband_bonds": False},
    ),
}


def run_tight_binding(name: str, row: dict) -> None:
    """The two-layer tight-binding model (model) on the supercell over Omega_R: for each
    coupling the mean-field (discrete GPE) ground state, E/N and mu (relative to the on-site energy 2 eps_1D), f_s^xx by twist
    energy and by current, the PR over sites, the AA/AB density ratio, the layer-2 fraction, residuals
    and the lowest constrained Hessian eigenvalue (> 0: a minimum). Variants:
      A  the one-neighbour model (S cut at |d| < 1, lattice-1 frame, no interlayer interaction)
         -- the reference table;
      B  S cut at |d| < 2 (removes the cusp the |d| = 1 cutoff puts into the k = 0 coupling next to AA);
      C  B plus the interlayer density-density interaction V_ij (a12 = a11, as in the continuum model);
      D  C plus the second-order interband shift of the site energies (not in the shipped row).
    Also the single-particle k = 0 levels against the continuum (bands.moire_bands_q0) at compare_m."""
    from twisted_bilayer_lattice import model as lm

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    t_all = time.time()
    V0, m, n, filling, theta = row["V0"], row["m"], row["n"], row["filling"], row["theta"]
    g2d, g2d_source = lm.resolve_g2d(row.get("g2d"), row.get("nu_z_hz"))
    g2d_note = {
        "explicit": "given by the row",
        "nu_z": f"from nu_z = {row.get('nu_z_hz')} Hz (model.g2d_from_nu_z)",
        "default": "the shipped model.G2D_DEFAULT",
    }[g2d_source]
    J = bands.hopping_1d(V0, 2)
    U = bands.onsite_U(V0, g2d)
    geo = lm.supercell_sites(m, n).geometry
    head = [
        f"Two-layer tight binding, V0 = {V0} E_R, filling {filling} atoms/a^2 (both layers), "
        f"g2d = {g2d:.7f} E_R a^2 ({g2d_note})",
        f"  {geo.describe()}; {2 * (m**2 + n**2)} sites",
        f"  J1 = {J[0]:.6f}, J2 = {J[1]:.3e}, U = {U:.6f} E_R (U/J1 = {U / J[0]:.2f}); "
        f"twist theta = {theta} across the box",
        f"  E/N and mu relative to the on-site energy 2 eps_1D = {lm.onsite_energy(V0):.6f} E_R; "
        f"f_s by energy (E) and current (I); PR over the sites of both layers;",
        "  AA/AB = total density in a 1 a disc at the AA points / at the AB points; hess_min > 0: local minimum",
        "",
    ]
    cols = [
        "Omega_R",
        "Omega_R/J1",
        "E/N",
        "mu",
        "fs_xx_E",
        "fs_xx_I",
        "PR",
        "AA/AB",
        "N2/N",
        "res_0",
        "res_theta",
        "hess_min",
        "time_s",
    ]
    fmt = [
        "{:7.3f}",
        "{:10.3f}",
        "{:10.6f}",
        "{:10.6f}",
        "{:9.6f}",
        "{:9.6f}",
        "{:7.4f}",
        "{:10.4g}",
        "{:7.5f}",
        "{:8.1e}",
        "{:9.1e}",
        "{:9.2e}",
        "{:6.1f}",
    ]
    width = [7, 10, 10, 10, 9, 9, 7, 10, 7, 8, 9, 9, 6]
    lines = list(head)
    for key in row["variants"]:
        t0 = time.time()
        title, kw = MODEL_C_VARIANTS[key]
        block = [f"Model {key}: {title}", "  " + " ".join(f"{c:>{w}s}" for c, w in zip(cols, width))]
        for Om in row["Omega_R"]:
            t1 = time.time()
            r = lm.superfluid_fraction(V0, Om, m, n, filling, g2d=g2d, theta=theta, **kw)
            g = r["gs0"]
            vals = [
                Om,
                Om / J[0],
                g.E_per_N,
                g.mu,
                r["fs_energy"],
                r["fs_current"],
                g.participation_ratio,
                g.AA_AB_ratio,
                g.layer_fractions[1],
                g.residual,
                r["gs_theta"].residual,
                g.hessian_min,
                time.time() - t1,
            ]
            block.append("  " + " ".join(f.format(v) for f, v in zip(fmt, vals)))
        lines += block + [f"  ({time.time() - t0:.0f} s)", ""]
        print("\n".join(block), flush=True)
    out = _outdir(name)
    lines.append(f"total {time.time() - t_all:.0f} s on CPU ({torch.get_num_threads()} threads)")
    with open(os.path.join(out, "table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    cm = row.get("compare_m", 0)
    if cm:
        sp = [
            f"Single-particle k = 0 levels, tight binding vs continuum (moire_bands_q0, 8 pts/a), "
            f"V0 = {V0}, m = {cm}, n = {n}; energies in E_R",
            "  TB absolute = TB + on-site energy 2 eps_1D; 'shift' = E_0 - (decoupled band bottom)",
        ]
        np.set_printoptions(precision=5, suppress=True, linewidth=160)
        for cutoff in (1.0, 2.0):
            for Om in (0.24, 1.0):
                t0 = time.time()
                r = lm.compare_with_continuum(V0, Om, cm, n, n_bands=10, cutoff=cutoff)
                sp += [
                    f"  Omega_R = {Om}, S cutoff {cutoff:g} ({time.time() - t0:.0f} s)",
                    f"    spacing TB   {r['spacing_tb']}",
                    f"    spacing cont {r['spacing_cont']}",
                    f"    max |spacing difference| {r['spacing_maxdiff']:.2e}; "
                    f"shift TB {r['shift_tb']:.6f} vs cont {r['shift_cont']:.6f}; "
                    f"max |absolute difference| {r['abs_maxdiff']:.2e}",
                ]
        print("\n".join(sp))
        with open(os.path.join(out, f"single_particle_m{cm}.txt"), "w") as fh:
            fh.write("\n".join(sp) + "\n")
    print(f"wrote {out}/table.txt ({time.time() - t_all:.0f} s total)")


KINDS = {
    "tight_binding": run_tight_binding,
    "phase_diagram": run_phase_diagram,
}


def _provenance() -> dict:
    """Package version, git commit (when run from a checkout), library versions and machine."""
    from twisted_bilayer_lattice import __version__

    here = os.path.dirname(os.path.abspath(__file__))
    commit, dirty = None, None

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()

    try:  # only when the package runs from a checkout (an editable install), not from site-packages
        top = git("rev-parse", "--show-toplevel")
        if top and os.path.samefile(os.path.join(top, "src", "twisted_bilayer_lattice"), here):
            commit = git("rev-parse", "--short", "HEAD") or None
            dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(
        twisted_bilayer_lattice=__version__,
        git_commit=commit,
        git_dirty=dirty,
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        device=describe_device(),
        platform=platform.platform(),
    )


def run_scenario(name: str, row: dict, out_root: str = "results") -> None:
    """Run one scenario row; files go to <out_root>/<name>/ (provenance.json included)."""
    global _OUT_ROOT
    _OUT_ROOT = out_root
    if row["kind"] not in KINDS:
        raise SystemExit(f"scenario {name!r}: unknown kind {row['kind']!r}; known kinds: {sorted(KINDS)}")
    print(f"scenario {name} ({row['kind']}) on {describe_device()}")
    t0 = time.time()
    KINDS[row["kind"]](name, row)
    out = _outdir(name)
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(
            dict(scenario=name, row=row, seconds=round(time.time() - t0, 1), **_provenance()), f, indent=2, default=str
        )
    print(f"\ndone in {time.time() - t0:.1f} s")
