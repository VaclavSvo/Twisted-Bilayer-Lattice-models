"""Phase maps over (V0, Omega_R) with single-site Gutzwiller or cluster Gutzwiller, in the
layout of Meng et al. Fig. 5a: V0 on x, Omega_R on y, the labels SF / SF-II / I / MI / ? (`observables`).

    python -m twisted_bilayer_lattice.phase_diagram <row> --out <folder> [--device cuda] [--threads 1]
    twisted-bilayer-lattice run phase_diagram_quick            (the scenario rows of kind "phase_diagram")

Rows are the `ROWS` dict below; `scenarios.py` names some of them for the `run` command. Per grid point:

  1. the model: the full two-layer model, `build_model(V0, Omega_R, m, n, nu_z_hz, cutoff, interlayer_interaction,
     interband, interband_bonds=False)`, or the paper's one-layer model (`paper_model(V0, M_R)`, M_R from
     `M_R_from_Omega(V0, Omega_R, rule)`);
  2. mu = (mu/U) U from the row's zero `mu_ref`: "onsite" = the on-site energy (the engine's convention), "aa" =
     the AA level -Omega_R F_k0_AA of the two-layer model (the zero of the paper's M_i under rule "k0":
     M_i = Omega_R (F_AA - F_i); the interband shift vanishes at AA). For the paper's model both are M = 0;
  3. every start of `gutzwiller.START_NAMES` in one batch over the column's Omega_R points, the converged state
     of lowest Omega kept (`starts`, `omega_spread` recorded);
  4. the twists theta = 0.3 and 0.15 at the same mu, f_s(0) = (4 f(0.15) - f(0.3)) / 3 quoted. A twisted Omega
     below Omega(0) by more than the solver noise means the theta = 0 state was not the minimum: the base is
     re-solved once from the twisted state; if it stays negative the point is "?" (never hidden);
  5. the visibilities, P(n_max), the label (and for the paper's model the label with the degenerate AA sites
     excluded, `observables.degenerate_sites`).

Unconverged points are labelled "?"; tolerances are never loosened. P(n_max) > 1e-6 is flagged. Outputs in
--out: maps.npz (every field over the grid, written after each column, so an interrupted row keeps a
partial map; rerunning with the same --out resumes), labels.png, summary.json, report.txt, provenance.json.
The m = 22 rows run the coarse 11 x 7 subgrid first, then the remaining points.

Resuming: maps.npz carries a hash of the row's specification (`spec_hash`, entries that are None left out) and
`SOLVER_VERSION`; a run into a folder whose map was made with another specification or solver version is
refused (`--force-resume` overrides). A map written before these records were kept is resumed with a warning.
provenance.json keeps one entry per run in "runs".

n_max: `n_max` (and `n_max_shallow` at V0 <= `v_shallow`), or a table `n_max_table` = ((V0_upper, n_max), ...)
in ascending V0_upper, used where V0 <= V0_upper first holds (and `n_max` above the last entry). P(n_max) is
reported per point; its relative effect on N, Omega and f_s is of order 1-10 x P(n_max).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from twisted_bilayer_lattice import gutzwiller as gw
from twisted_bilayer_lattice import model as lm
from twisted_bilayer_lattice import observables as ob
from twisted_bilayer_lattice._shared import config as CFG
from twisted_bilayer_lattice._shared import records as REC
from twisted_bilayer_lattice.paper_model import M_R_from_Omega, paper_model

V0_FULL = tuple(float(v) for v in range(4, 25))  # 21 points
OM_FULL = tuple(round(0.1 * k, 10) for k in range(13))  # 0 ... 1.2, 13 points
V0_QUICK = tuple(float(v) for v in range(4, 25, 2))  # 11 points
OM_QUICK = tuple(round(0.2 * k, 10) for k in range(7))  # 7 points

COMMON = dict(
    method="gutzwiller",
    n=1,
    mu_over_U=1.0,
    nu_z_hz=CFG.NU_Z_HZ,
    cutoff=2.0,
    interlayer_interaction=True,
    interband=True,
    interband_bonds=False,
    n_max=6,
    n_max_shallow=8,  # at V0 <= v_shallow (P(n_max = 6) reaches 5e-2 at 4 E_R)
    n_max_table=None,  # optional ((V0_upper, n_max), ...), ascending; overrides n_max / n_max_shallow
    v_shallow=6.0,
    tol=1e-10,
    max_iter=5000,
    thetas=gw.THETA_FS,
    starts=gw.START_NAMES,
    thresholds=dict(ob.THRESHOLDS),
    p_nmax_flag=1e-6,
    degenerate_window=0.05,
    compressibility=False,
    coarse_first=True,
    u_factor=None,  # None or "c_v0": U multiplied by c(V0) from `u_table` (a labelled comparison row)
    u_table=None,  # {V0: c} or the path of a JSON file with it (CLI --u-table); needed with u_factor="c_v0"
)

ROWS = {
    # full two-layer model (variant C, interband shift site part on), mu from the AA level
    "phase_diagram_quick": dict(
        COMMON, variant="full", m=4, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", compressibility=True
    ),
    "phase_diagram_quick_onsite": dict(
        COMMON, variant="full", m=4, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="onsite", compressibility=True
    ),
    "phase_diagram_quick_noshift": dict(
        COMMON, variant="full", m=4, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", interband=False, compressibility=True
    ),
    "phase_diagram_quick_mu095": dict(
        COMMON, variant="full", m=4, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", mu_over_U=0.95, compressibility=True
    ),
    "phase_diagram": dict(COMMON, variant="full", m=22, V0=V0_FULL, Omega_R=OM_FULL, mu_ref="aa"),
    # the coarse 11 x 7 maps on their own (deepest columns first: a run cut short loses the all-SF shallow ones)
    "phase_diagram_coarse": dict(
        COMMON, variant="full", m=22, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", deep_first=True
    ),
    "phase_diagram_noshift_coarse": dict(
        COMMON, variant="full", m=22, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", interband=False, deep_first=True
    ),
    "phase_diagram_noshift": dict(
        COMMON, variant="full", m=22, V0=V0_FULL, Omega_R=OM_FULL, mu_ref="aa", interband=False
    ),
    # the paper's one-layer model (Supplement Eqs. 12-13)
    "phase_diagram_paper_quick": dict(
        COMMON,
        variant="paper",
        m=4,
        V0=V0_QUICK,
        Omega_R=OM_QUICK,
        mu_ref="onsite",
        M_R_rule="k0",
        compressibility=True,
    ),
    "phase_diagram_paper": dict(
        COMMON, variant="paper", m=22, V0=V0_FULL, Omega_R=OM_FULL, mu_ref="onsite", M_R_rule="k0"
    ),
    "phase_diagram_paper_mu095": dict(
        COMMON, variant="paper", m=22, V0=V0_FULL, Omega_R=OM_FULL, mu_ref="onsite", M_R_rule="k0", mu_over_U=0.95
    ),
    "phase_diagram_paper_omega": dict(
        COMMON, variant="paper", m=22, V0=V0_FULL, Omega_R=OM_FULL, mu_ref="onsite", M_R_rule="omega"
    ),
    # cluster Gutzwiller. The interlayer dimer treats the AA bond Omega_R exactly (single-site Gutzwiller
    # decouples it at mean-field level); the plus family (intralayer correlations) needs c divisible by 5, so it
    # runs on m = 22 only (the (4, 1) cell has c = 17)
    "phase_diagram_cluster_quick": dict(
        COMMON,
        method="cluster",
        family="dimer",
        variant="full",
        m=4,
        V0=V0_QUICK,
        Omega_R=OM_QUICK,
        mu_ref="aa",
        starts=("lowest", "pocket"),
    ),
    "phase_diagram_cluster": dict(
        COMMON,
        method="cluster",
        family="dimer",
        variant="full",
        m=22,
        V0=V0_QUICK,
        Omega_R=OM_QUICK,
        mu_ref="aa",
        starts=("lowest", "pocket"),
        deep_first=True,
    ),
    "phase_diagram_cluster_plus": dict(
        COMMON,
        method="cluster",
        family="plus",
        variant="full",
        m=22,
        V0=V0_QUICK,
        Omega_R=OM_QUICK,
        mu_ref="aa",
        n_max=2,
        n_max_shallow=2,
        n_total_max=None,
        starts=("lowest",),
    ),
    "phase_diagram_paper_cluster_mu095": dict(
        COMMON,
        method="cluster",
        family="plus",
        variant="paper",
        m=22,
        V0=V0_QUICK,
        Omega_R=OM_QUICK,
        mu_ref="onsite",
        M_R_rule="k0",
        mu_over_U=0.95,
        n_max=2,
        n_max_shallow=3,
        starts=("lowest",),
    ),
    # U multiplied by a correction c(V0) read from --u-table (not applied by default), one labelled map
    "phase_diagram_ucorr": dict(
        COMMON, variant="full", m=22, V0=V0_QUICK, Omega_R=OM_QUICK, mu_ref="aa", u_factor="c_v0", deep_first=True
    ),
}

LABEL_CODE = {lab: k for k, lab in enumerate(ob.LABELS)}  # SF 0, SF-II 1, I 2, MI 3, ? 4, empty 5

# A twisted Omega below Omega(0) counts as "the theta = 0 state was not the minimum" (retry, then "?") only beyond
# this level in f_s. 1e-7 (the first version) fired on rounding noise (f ~ -1e-7 is ~1e-11 in Omega) and the
# retries took half the run time; 1e-5 is still 100 x below the label threshold f_c = 1e-3.
FS_NOISE = 1e-5


def twist_verdict(fr: dict, th, f_c: float, noise: float = FS_NOISE):
    """(negative, bound) for the twisted solves `fr` (a `gutzwiller.fs_batch` / `cluster.cluster_fs` entry).

    negative: an f_s below -noise, from the converged values, or from their variational upper bounds when a
    twisted solve did not converge (an upper bound below -noise decides f_s < 0 whatever the exact value is).
    bound: when a twisted solve did not converge and none is negative, the bound on f_s(0), else None:
    f(0) <= (4/3) f_upper(theta_min) given f(theta_max) >= 0, quoted only if it is below f_c. The assumption
    f(theta_max) >= 0 is refuted when f_upper(theta_max) < -noise (f_upper >= f), which `negative` then reports and
    no bound is quoted; it is not decidable when the theta_max solve itself did not converge and f_upper(theta_max)
    is above -noise, where the bound rests on it. The stored f_upper(theta_max) is `fs_upper_03` in the maps."""
    vals = fr["fs"] if fr["twist_converged"] else fr["fs_upper"]
    negative = min(vals.values()) < -noise
    if fr["twist_converged"] or negative:
        return negative, None
    ub = (4.0 / 3.0) * max(fr["fs_upper"][min(th)], 0.0) if len(th) == 2 else fr["fs_upper"][th[0]]
    return False, (float(ub) if ub < f_c else None)


# -- the U correction (only for the labelled row) -------------------------------------------------------------------


def load_u_table(table) -> dict:
    """{V0: c} from a dict or the path of a JSON file ({"V0": c, ...}); c(V0) multiplies U in rows with
    u_factor = "c_v0" (linear in V0, held at the end values outside the table)."""
    if table is None:
        raise RuntimeError('u_factor = "c_v0" needs a table c(V0): pass --u-table <file.json> (or the row key u_table)')
    if isinstance(table, (str, os.PathLike)):
        with open(table, encoding="utf-8") as f:
            table = json.load(f)
    return {float(k): float(v) for k, v in dict(table).items()}


def u_correction(V0: float, table) -> float:
    t = load_u_table(table)
    v = np.array(sorted(t))
    c = np.array([t[k] for k in v])
    return float(np.interp(V0, v, c))


# -- models ---------------------------------------------------------------------------------------------------------


def n_max_for(row: dict, V0: float) -> int:
    table = row.get("n_max_table")
    if table:
        for v_upper, nm in table:
            if V0 <= v_upper:
                return int(nm)
        return int(row["n_max"])
    return int(row["n_max_shallow"] if V0 <= row["v_shallow"] else row["n_max"])


def build(row: dict, V0: float, Omega_R: float, cell=None) -> lm.LatticeModel:
    """The row's model at (V0, Omega_R)."""
    if row["variant"] == "paper":
        cell = lm.supercell_sites(row["m"], row["n"]) if cell is None else cell
        M_R = M_R_from_Omega(V0, Omega_R, rule=row["M_R_rule"], m=row["m"], n=row["n"], cutoff=row["cutoff"], cell=cell)
        return paper_model(V0, M_R, m=row["m"], n=row["n"], mu_over_U=row["mu_over_U"], cell=cell)
    kw = dict(
        nu_z_hz=row["nu_z_hz"],
        cutoff=row["cutoff"],
        interlayer_interaction=row["interlayer_interaction"],
        interband=row["interband"],
        interband_bonds=row["interband_bonds"],
        cell=cell,
    )
    if row.get("u_factor") == "c_v0":
        g2d = lm.g2d_from_nu_z(row["nu_z_hz"])
        U = lm.onsite_U(V0, g2d) * u_correction(V0, row.get("u_table"))
        return lm.build_model(V0, Omega_R, row["m"], row["n"], U=U, **kw)
    return lm.build_model(V0, Omega_R, row["m"], row["n"], **kw)


def mu_of(row: dict, model: lm.LatticeModel) -> float:
    mu = row["mu_over_U"] * model.U
    if row["mu_ref"] == "aa" and model.n_layers == 2 and model.Omega_R != 0.0:
        mu -= model.Omega_R * float(model.diagnostics["F_k0_AA"])
    elif row["mu_ref"] not in ("aa", "onsite"):
        raise ValueError(f"mu_ref must be 'aa' or 'onsite', got {row['mu_ref']!r}")
    return float(mu)


# -- one column of the map ------------------------------------------------------------------------------------------


def _groups(models):
    """Indices of the models grouped by identical structure (one batch each)."""
    groups = []
    for k, mdl in enumerate(models):
        for g in groups:
            if gw.same_structure(models[g[0]], mdl):
                g.append(k)
                break
        else:
            groups.append([k])
    return groups


_TILING_CACHE: dict = {}


def _tiling(model, family):
    key = (family, model.n_sites, model.cell.L, model.bonds.rows.size, model.n_layers)
    t = _TILING_CACHE.get(key)
    if t is None or not np.array_equal(t[1], model.bonds.rows) or not np.array_equal(t[2], model.bonds.cols):
        from twisted_bilayer_lattice import cluster as C

        t = (C.make_tiling(model, family), model.bonds.rows.copy(), model.bonds.cols.copy())
        _TILING_CACHE[key] = t
    return t[0]


def solve_points_cluster(row: dict, models: list, device: str = "cpu", log=print) -> list:
    """The cluster-Gutzwiller pipeline (method "cluster"): per point every start of the row (lowest
    converged Omega kept), the twists at fixed mu (theta -> 0 extrapolation), visibilities with the
    connected intra-cluster rho, the label. An unconverged twisted solve is resolved by the variational upper
    bound as in `solve_points` (`cluster.cluster_variational_omega`), or labelled "?"."""
    from twisted_bilayer_lattice import cluster as C

    out = []
    th = tuple(row["thetas"])
    for mdl in models:
        tiling = _tiling(mdl, row["family"])
        mu = mu_of(row, mdl)
        n_max = n_max_for(row, mdl.V0)
        kw = dict(n_total_max=row.get("n_total_max"), tol=row["tol"], max_iter=row["max_iter"], device=device)
        sols = []
        mix = gw.default_mixing(mdl)
        for s in row["starts"]:
            st = C.cluster_gutzwiller(mdl, tiling, mu, n_max, phi0=gw.make_start(mdl, s), mixing=mix, **kw)
            sols.append((s, st))
        conv = [x for x in sols if x[1].converged]
        best_name, best = min(conv or sols, key=lambda x: x[1].omega)
        om = [x[1].omega for x in conv]
        fsr = C.cluster_fs(mdl, tiling, best, n_max, thetas=th, mixing=mix, **kw)
        res = C.to_result(mdl, tiling, best, n_max)
        res.fs_theta = {float(t): float(v) for t, v in fsr["fs"].items()}
        res.fs = float(fsr["fs0"])
        res.fs_route = "twist energy at fixed mu, theta -> 0 from (0.3, 0.15); cluster " + row["family"]
        neg, ub = twist_verdict(fsr, th, row["thresholds"]["f_c"])
        res.converged = bool(best.converged and fsr["converged"] and not neg)
        fs_bound = False
        if best.converged and ub is not None:  # the bound of solve_points
            res.fs = ub
            res.fs_route = (
                "upper bound (twisted solve not converged): (4/3) f_upper(theta = 0.15); cluster " + row["family"]
            )
            res.converged = True
            fs_bound = True
        res.starts = {s: (st.omega, st.converged, st.iterations) for s, st in sols}
        res.omega_spread = float(max(om) - min(om)) if len(om) > 1 else 0.0
        res.params = {
            "V0": mdl.V0,
            "Omega_R": float(row_omega(mdl)),
            "mu": mu,
            "U": mdl.U,
            "best_start": best_name,
            "retried": False,
            "fs_negative": bool(neg),
            "fs_bound": fs_bound,
            "twist_converged": bool(fsr["twist_converged"]),
            "fs_upper": {float(t): float(v) for t, v in fsr["fs_upper"].items()},
            "lambda": tiling.lam,
            "p_total_cap": best.p_total_cap,
        }
        ob.attach_visibilities(mdl, res)
        res.label, res.label_info = ob.phase_label(res, row["thresholds"])
        if row["variant"] == "paper":
            ex = ob.degenerate_sites(mdl, mu, row["degenerate_window"])
            lab2, info2 = ob.phase_label(res, row["thresholds"], exclude=ex)
            res.label_info["label_excl_degenerate"] = lab2
            res.label_info["excl_info"] = info2
        out.append(res)
    return out


def solve_points(row: dict, models: list, device: str = "cpu", log=print) -> list:
    """`ManyBodyResult`s of the row's single-site Gutzwiller pipeline (module docstring) for `models`; rows with
    method "cluster" go to `solve_points_cluster`."""
    if row.get("method") == "cluster":
        return solve_points_cluster(row, models, device, log)
    out = [None] * len(models)
    th = tuple(row["thetas"])
    for g in _groups(models):
        mg = [models[k] for k in g]
        mus = [mu_of(row, mdl) for mdl in mg]
        n_max = n_max_for(row, mg[0].V0)
        kw = dict(tol=row["tol"], max_iter=row["max_iter"], device=device)
        best = gw.solve_starts(mg, mus, n_max, starts=row["starts"], **kw)
        bases = [b for b, _ in best]
        fsr = gw.fs_batch(mg, bases, n_max, thetas=th, **kw)
        # a twisted Omega below Omega(0): the base was not the minimum; re-solve it once from the twisted state
        noise = FS_NOISE
        redo = [q for q, r in enumerate(fsr) if min(r["fs"].values()) < -noise]
        retried = set()
        for q in redo:
            st_t = fsr[q]["states"][th[-1]]
            b2 = gw.solve(mg[q], mus[q], n_max, phi0=st_t.phi, **kw)
            retried.add(q)
            if b2.converged and b2.omega < bases[q].omega:
                bases[q] = b2
                best[q][1]["best"] = "retry_from_twist"
                fsr[q] = gw.fs_batch([mg[q]], [b2], n_max, thetas=th, **kw)[0]
        f_c = row["thresholds"]["f_c"]
        for q, k in enumerate(g):
            mdl, st, info, fr = mg[q], bases[q], best[q][1], fsr[q]
            res = ob.from_state(mdl, st, method="gutzwiller", n_max=n_max)
            res.fs_theta = {float(t): float(v) for t, v in fr["fs"].items()}
            res.fs = gw.extrapolate_fs0(fr["fs"], th) if len(th) == 2 else float(fr["fs"][th[0]])
            res.fs_route = "twist energy at fixed mu, theta -> 0 from (0.3, 0.15)"
            neg, ub = twist_verdict(fr, th, f_c, noise)
            res.converged = bool(st.converged and fr["converged"] and not neg)
            # twisted solves that did not converge (the soft inter-island phase mode): the variational Omega of
            # the last iterate bounds f_s from above (see `twist_verdict`). If that bound is below f_c the label is
            # decided; the bound is quoted as f_s and flagged.
            fs_bound = False
            if st.converged and ub is not None:
                res.fs = ub
                res.fs_route = "upper bound (twisted solve not converged): (4/3) f_upper(theta = 0.15)"
                res.converged = True
                fs_bound = True
            res.starts = info["starts"]
            res.omega_spread = info["omega_spread"]
            res.params = {
                "V0": mdl.V0,
                "Omega_R": float(row_omega(mdl)),
                "mu": mus[q],
                "U": mdl.U,
                "best_start": info["best"],
                "retried": q in retried,
                "fs_negative": bool(neg),
                "fs_bound": fs_bound,
                "twist_converged": bool(fr["twist_converged"]),
                "fs_upper": {float(t): float(v) for t, v in fr["fs_upper"].items()},
            }
            if row.get("compressibility"):
                res.compressibility = gw.compressibility(mdl, st, n_max, **kw)
            ob.attach_visibilities(mdl, res)
            res.label, res.label_info = ob.phase_label(res, row["thresholds"])
            if row["variant"] == "paper":
                ex = ob.degenerate_sites(mdl, mus[q], row["degenerate_window"])
                lab2, info2 = ob.phase_label(res, row["thresholds"], exclude=ex)
                res.label_info["label_excl_degenerate"] = lab2
                res.label_info["excl_info"] = info2
            out[k] = res
    return out


def row_omega(model: lm.LatticeModel) -> float:
    return float(model.diagnostics.get("Omega_R_row", model.Omega_R))


# -- the grid -------------------------------------------------------------------------------------------------------


SCALARS = (
    "filling",
    "grand_potential",
    "condensate_fraction",
    "compressibility",
    "fs",
    "p_nmax",
    "omega_spread",
    "iterations",
    "mu",
)


def _empty_maps(row: dict, nl: int, c: int) -> dict:
    nV, nO = len(row["V0"]), len(row["Omega_R"])
    f = np.full((nV, nO), np.nan)
    d = {k: f.copy() for k in SCALARS}
    d.update(
        done=np.zeros((nV, nO), dtype=bool),
        converged=np.zeros((nV, nO), dtype=bool),
        label=np.full((nV, nO), LABEL_CODE["?"], dtype=np.int64),
        label_excl=np.full((nV, nO), LABEL_CODE["?"], dtype=np.int64),
        phi_max=f.copy(),
        contrast=f.copy(),
        fs_03=f.copy(),
        fs_015=f.copy(),
        retried=np.zeros((nV, nO), dtype=bool),
        fs_negative=np.zeros((nV, nO), dtype=bool),
        fs_bound=np.zeros((nV, nO), dtype=bool),
        twist_converged=np.zeros((nV, nO), dtype=bool),
        fs_upper_03=f.copy(),
        fs_upper_015=f.copy(),
        seconds=f.copy(),
        n=np.full((nV, nO, nl, c), np.nan),
        var_n=np.full((nV, nO, nl, c), np.nan),
        phi=np.full((nV, nO, nl, c), np.nan + 0j),
    )
    for key in ("primary", "moire", "primary_raw", "moire_raw", "primary_caption", "moire_caption", "primary_pert"):
        d["vis_" + key] = np.full((nV, nO, nl), np.nan)
    for s in row["starts"]:
        d["start_omega_" + s] = f.copy()
        d["start_conv_" + s] = np.zeros((nV, nO), dtype=bool)
    d["V0"] = np.asarray(row["V0"], dtype=np.float64)
    d["Omega_R"] = np.asarray(row["Omega_R"], dtype=np.float64)
    return d


def _store(maps: dict, iv: int, io: int, res: ob.ManyBodyResult, seconds: float) -> None:
    for k in SCALARS:
        maps[k][iv, io] = float(getattr(res, k))
    maps["done"][iv, io] = True
    maps["converged"][iv, io] = res.converged
    maps["label"][iv, io] = LABEL_CODE[res.label]
    maps["label_excl"][iv, io] = LABEL_CODE[res.label_info.get("label_excl_degenerate", res.label)]
    maps["phi_max"][iv, io] = res.label_info["phi_max"]
    maps["contrast"][iv, io] = res.label_info["contrast"]
    maps["fs_03"][iv, io] = res.fs_theta.get(0.3, np.nan)
    maps["fs_015"][iv, io] = res.fs_theta.get(0.15, np.nan)
    maps["retried"][iv, io] = bool(res.params.get("retried"))
    maps["fs_negative"][iv, io] = bool(res.params.get("fs_negative"))
    maps["fs_bound"][iv, io] = bool(res.params.get("fs_bound"))
    maps["twist_converged"][iv, io] = bool(res.params.get("twist_converged"))
    fu = res.params.get("fs_upper", {})
    maps["fs_upper_03"][iv, io] = fu.get(0.3, np.nan)
    maps["fs_upper_015"][iv, io] = fu.get(0.15, np.nan)
    maps["seconds"][iv, io] = seconds
    maps["n"][iv, io] = res.n
    maps["var_n"][iv, io] = res.var_n
    maps["phi"][iv, io] = res.phi
    for key, v in res.visibilities.items():
        maps["vis_" + key][iv, io] = np.asarray(v)
    for s, (om, cv, _) in res.starts.items():
        if "start_omega_" + s in maps:
            maps["start_omega_" + s][iv, io] = om
            maps["start_conv_" + s][iv, io] = cv


def _save_npz(path: str, maps: dict) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **maps)
    os.replace(tmp, path)


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def task_order(row: dict) -> list:
    """[(V0 index, [Omega_R indices])]: the coarse subgrid (every other V0 and Omega_R point) first if the row
    asks for it and the grid is the full 21 x 13 one, then the rest, column by column."""
    nV, nO = len(row["V0"]), len(row["Omega_R"])
    if row.get("deep_first"):  # columns from the deepest lattice down (where the cluster correction matters)
        return [(iv, list(range(nO))) for iv in range(nV - 1, -1, -1)]
    if not row.get("coarse_first") or nV < 15:
        return [(iv, list(range(nO))) for iv in range(nV)]
    first = [(iv, list(range(0, nO, 2))) for iv in range(0, nV, 2)]
    rest = []
    for iv in range(nV):
        todo = list(range(nO)) if iv % 2 else list(range(1, nO, 2))
        rest.append((iv, todo))
    return first + rest


SOLVER_VERSION = "1"  # bump when a change alters the numbers a row produces (resume refuses a mismatch)


def spec_hash(row: dict) -> str:
    """Hash of the row's specification (entries that are None left out, so adding an optional key with a None
    default does not change it)."""
    spec = {k: v for k, v in sorted(row.items()) if v is not None}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def provenance() -> dict:
    from twisted_bilayer_lattice import __version__

    return {
        "package": __version__,
        "module": "twisted_bilayer_lattice.phase_diagram",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.node(),
        "threads": torch.get_num_threads(),
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def preflight(row: dict, device: str = "cpu") -> dict:
    """Time one start of one mid-grid point and extrapolate (a rough upper estimate: every point x (starts +
    twists) solves at the probe's iteration count; boundary points are slower, deep Mott / SF ones faster)."""
    V0 = row["V0"][len(row["V0"]) // 2]
    Om = row["Omega_R"][len(row["Omega_R"]) // 2]
    t0 = time.time()
    model = build(row, V0, Om)
    t_build = time.time() - t0
    t0 = time.time()
    st = gw.solve(model, mu_of(row, model), n_max_for(row, V0), tol=row["tol"], max_iter=row["max_iter"], device=device)
    t_solve = time.time() - t0
    n_pts = len(row["V0"]) * len(row["Omega_R"])
    per_point = t_build + t_solve * (len(row["starts"]) + 2.0 * len(row["thetas"]))
    return {
        "probe_point": (V0, Om),
        "build_s": t_build,
        "solve_s": t_solve,
        "iterations": st.iterations,
        "converged": st.converged,
        "per_iteration_ms": 1e3 * t_solve / max(st.iterations, 1),
        "points": n_pts,
        "estimate_h": n_pts * per_point / 3600.0,
    }


def run_row(
    name: str,
    out: str,
    device: str = "cpu",
    log=print,
    max_hours: float | None = None,
    row=None,
    redo_unconverged: bool = False,
    force_resume: bool = False,
    provenance_file: str = "provenance.json",
) -> dict:
    """Run (or resume) a row into `out`; returns the summary. A resume into a map made with another row
    specification or solver version is refused unless `force_resume` (module docstring)."""
    row = dict(ROWS[name] if row is None else row)
    if row.get("u_factor") == "c_v0":
        row["u_table"] = load_u_table(row.get("u_table"))  # the values, not a path, enter the spec
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "maps.npz")
    sid = spec_hash(row)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as f:
            old_sid = str(f["spec_hash"]) if "spec_hash" in f.files else None
            old_ver = str(f["solver_version"]) if "solver_version" in f.files else None
        if old_sid is None:
            log(f"resuming {path}: written before spec records were kept; its specification is not checked")
        elif (old_sid, old_ver) != (sid, SOLVER_VERSION) and not force_resume:
            raise RuntimeError(
                f"{path} was made with spec {old_sid} / solver {old_ver}, this run is {sid} / {SOLVER_VERSION}: "
                "use another --out, or --force-resume to mix them"
            )
    prov_path = os.path.join(out, provenance_file)
    prov = {"row": name, "spec": row, "spec_hash": sid, "solver_version": SOLVER_VERSION, "runs": []}
    if os.path.exists(prov_path):
        try:
            with open(prov_path, encoding="utf-8") as f:
                old = json.load(f)
            prov["runs"] = list(old.get("runs", [{k: v for k, v in old.items() if k not in ("row", "spec")}]))
        except (OSError, ValueError):
            pass
    prov["runs"].append(provenance())
    REC.write_json(prov_path, prov)
    pf = preflight(row, device)
    log(
        f"pre-flight {name}: probe (V0, Omega_R) = {pf['probe_point']}: build {pf['build_s']:.1f} s, one start "
        f"{pf['solve_s']:.2f} s ({pf['iterations']} iterations, {pf['per_iteration_ms']:.1f} ms each); "
        f"{pf['points']} points -> about {pf['estimate_h']:.2f} h (rough)"
    )
    cell = lm.supercell_sites(row["m"], row["n"])
    nl = 1 if row["variant"] == "paper" else 2
    fresh = _empty_maps(row, nl, cell.c)
    maps = _load_npz(path) if os.path.exists(path) else fresh
    for key, val in fresh.items():  # maps written by an earlier version of this module
        maps.setdefault(key, val)
    maps["spec_hash"] = np.array(sid)
    maps["solver_version"] = np.array(SOLVER_VERSION)
    _save_npz(path, maps)  # the spec record is on disk before the first column (and for a complete map)
    if redo_unconverged:
        bad = maps["done"] & ~maps["converged"]
        log(f"redoing {int(bad.sum())} unconverged points")
        maps["done"][bad] = False
    t_start = time.time()
    stopped = False
    for iv, ios in task_order(row):
        ios = [io for io in ios if not maps["done"][iv, io]]
        if not ios:
            continue
        if max_hours is not None and time.time() - t_start > 3600.0 * max_hours:
            stopped = True
            log(f"stopping: over the budget of {max_hours} h; the map is partial")
            break
        V0 = row["V0"][iv]
        t0 = time.time()
        models = []
        for io in ios:
            mdl = build(row, V0, row["Omega_R"][io], cell=cell)
            mdl.diagnostics["Omega_R_row"] = float(row["Omega_R"][io])
            models.append(mdl)
        res = solve_points(row, models, device=device, log=log)
        dt = (time.time() - t0) / len(ios)
        for io, r in zip(ios, res):
            _store(maps, iv, io, r, dt)
        _save_npz(path, maps)
        labs = " ".join(f"{row['Omega_R'][io]:.1f}:{r.label}" for io, r in zip(ios, res))
        log(
            f"V0 = {V0:g}: {len(ios)} points, {dt:.1f} s each | {labs} | done {int(maps['done'].sum())}/{maps['done'].size}"
        )
    summary = summarise(name, row, maps, pf, time.time() - t_start, stopped)
    REC.write_json(os.path.join(out, "summary.json"), summary)
    REC.write_text(os.path.join(out, "report.txt"), format_report(name, row, maps, summary))
    try:
        plot_labels(name, row, maps, os.path.join(out, "labels.png"))
    except Exception as exc:  # the figure must never lose the data
        log(f"figure failed: {exc}")
    return summary


# -- summary, boundary table, report, figure ------------------------------------------------------------------------


def boundary_table(row: dict, maps: dict, key: str = "label") -> list:
    """Per Omega_R: V_c = the midpoint between the last SF point and the first non-SF point along V0 (the SF
    edge), and the first V0 of each other label; None where not bracketed on the computed points."""
    V0 = np.asarray(row["V0"])
    out = []
    for io, Om in enumerate(row["Omega_R"]):
        lab = maps[key][:, io]
        done = maps["done"][:, io]
        ent = {"Omega_R": float(Om), "V_c_SF": None}
        idx = np.where(done)[0]
        seq = [(V0[i], ob.LABELS[lab[i]]) for i in idx]
        for a, b in zip(seq, seq[1:]):
            if a[1] == "SF" and b[1] not in ("SF", "?"):
                ent["V_c_SF"] = 0.5 * (a[0] + b[0])
                ent["bracket"] = (float(a[0]), float(b[0]))
                break
        for L in ("SF-II", "I", "MI"):
            first = [v for v, l in seq if l == L]
            ent["first_" + L] = float(first[0]) if first else None
        out.append(ent)
    return out


def summarise(name, row, maps, pf, seconds, stopped) -> dict:
    done = maps["done"]
    counts = {L: int(((maps["label"] == LABEL_CODE[L]) & done).sum()) for L in ob.LABELS}
    flag_p = done & (maps["p_nmax"] > row["p_nmax_flag"])
    return {
        "row": name,
        "points_done": int(done.sum()),
        "points_total": int(done.size),
        "complete": bool(done.all()),
        "stopped_over_budget": bool(stopped),
        "seconds": seconds,
        "label_counts": counts,
        "label_excl_counts": {L: int(((maps["label_excl"] == LABEL_CODE[L]) & done).sum()) for L in ob.LABELS}
        if row["variant"] == "paper"
        else None,
        "unconverged": int((done & ~maps["converged"]).sum()),
        "fs_negative": int((done & maps["fs_negative"]).sum()),
        "fs_bound": int((done & maps["fs_bound"]).sum()),
        "retried": int((done & maps["retried"]).sum()),
        "p_nmax_flagged": int(flag_p.sum()),
        "p_nmax_max": float(np.nanmax(np.where(done, maps["p_nmax"], np.nan))) if done.any() else None,
        "omega_spread_max": float(np.nanmax(np.where(done, maps["omega_spread"], np.nan))) if done.any() else None,
        "boundary": boundary_table(row, maps),
        "boundary_excl": boundary_table(row, maps, "label_excl") if row["variant"] == "paper" else None,
        "preflight": pf,
        "spec": row,
    }


def format_report(name, row, maps, s) -> str:
    L = []
    L.append(
        f"phase map {name}: {row['variant']} model, m = {row['m']}, mu/U = {row['mu_over_U']} (zero: {row['mu_ref']})"
    )
    if row["variant"] == "paper":
        L.append(f"  paper's model, M_R rule {row['M_R_rule']!r}")
    else:
        L.append(
            f"  build_model(nu_z = {row['nu_z_hz']:g} Hz, cutoff {row['cutoff']}, interlayer V {row['interlayer_interaction']}, "
            f"interband {row['interband']} (bonds {row['interband_bonds']}))"
            + (" U x c(V0)" if row.get("u_factor") else "")
        )
    if row.get("method") == "cluster":
        L.append(f"  method: cluster Gutzwiller, family {row['family']!r}, n_total_max {row.get('n_total_max')}")
    if row["m"] <= 4:
        L.append(
            "  m = 4: for speed and plotting only (small cells carry duplicate interlayer pairs; not a cross-check)"
        )
    nmax_txt = (
        f"n_max table {[tuple(x) for x in row['n_max_table']]} ({row['n_max']} above)"
        if row.get("n_max_table")
        else f"n_max {row['n_max']} ({row['n_max_shallow']} at V0 <= {row['v_shallow']:g})"
    )
    L.append(
        f"  {nmax_txt}, tol {row['tol']:g}, "
        f"max_iter {row['max_iter']}, starts {list(row['starts'])}, thresholds {row['thresholds']}"
    )
    L.append(
        f"  done {s['points_done']}/{s['points_total']}, unconverged {s['unconverged']}, f_s < 0 {s['fs_negative']}, "
        f"retried {s['retried']}, f_s from the upper bound {s['fs_bound']}, P(n_max) > {row['p_nmax_flag']:g}: {s['p_nmax_flagged']} (max {s['p_nmax_max']})"
    )
    L.append(
        f"  labels {s['label_counts']}"
        + (f"; AA-degenerate sites excluded: {s['label_excl_counts']}" if s["label_excl_counts"] else "")
    )
    L.append("")
    L.append("labels (rows Omega_R top = largest, columns V0):")
    hdr = "Om\\V0 " + " ".join(f"{v:>5g}" for v in row["V0"])
    L.append(hdr)
    for io in range(len(row["Omega_R"]) - 1, -1, -1):
        cells = []
        for iv in range(len(row["V0"])):
            cells.append(f"{ob.LABELS[maps['label'][iv, io]] if maps['done'][iv, io] else '.':>5}")
        L.append(f"{row['Omega_R'][io]:5.2f} " + " ".join(cells))
    L.append("")
    L.append("f_s(theta -> 0) (rows Omega_R, columns V0):")
    L.append(hdr)
    for io in range(len(row["Omega_R"]) - 1, -1, -1):
        cells = []
        for iv in range(len(row["V0"])):
            v = maps["fs"][iv, io]
            cells.append(f"{v:5.3f}" if maps["done"][iv, io] and np.isfinite(v) else "    .")
        L.append(f"{row['Omega_R'][io]:5.2f} " + " ".join(cells))
    L.append("")
    L.append("filling (atoms per a^2, all layers):")
    L.append(hdr)
    for io in range(len(row["Omega_R"]) - 1, -1, -1):
        cells = []
        for iv in range(len(row["V0"])):
            v = maps["filling"][iv, io]
            cells.append(f"{v:5.2f}" if maps["done"][iv, io] and np.isfinite(v) else "    .")
        L.append(f"{row['Omega_R'][io]:5.2f} " + " ".join(cells))
    L.append("")
    L.append("boundary V_c(Omega_R) (SF edge = midpoint of the bracketing V0 points; first V0 of each label):")
    for b in s["boundary"]:
        L.append(
            f"  Omega_R {b['Omega_R']:.2f}: V_c(SF) {b['V_c_SF']}  first SF-II {b['first_SF-II']}  "
            f"first I {b['first_I']}  first MI {b['first_MI']}"
        )
    return "\n".join(L) + "\n"


COLORS = {"SF": "#4C78A8", "SF-II": "#72B7B2", "I": "#F58518", "MI": "#E45756", "?": "#9D9D9D", "empty": "#E8DCC4"}
FS_CLIP = 1e-6  # f_s below this is drawn at the floor of the colour scale


def fs_for_plot(maps: dict) -> np.ndarray:
    """log10 f_s of the finished points for the colour map, shape (n_V0, n_Omega_R): clipped at `FS_CLIP`, and NaN
    (masked) at the points not run and at the points labelled "?" (undecided: an unconverged or ambiguous f_s
    drawn at the floor would look like a superfluid island or a deep insulator)."""
    show = maps["done"] & (maps["label"] != LABEL_CODE["?"])
    fs = np.where(show, maps["fs"], np.nan)
    return np.log10(np.clip(fs, FS_CLIP, None))


def plot_labels(name, row, maps, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    V0, Om = np.asarray(row["V0"]), np.asarray(row["Omega_R"])
    cmap = ListedColormap([COLORS[L] for L in ob.LABELS] + ["#FFFFFF"])
    norm = BoundaryNorm(np.arange(-0.5, len(ob.LABELS) + 1), cmap.N)
    panels = [("label", "labels")]
    if row["variant"] == "paper":
        panels.append(("label_excl", "labels, AA-degenerate sites excluded"))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5.2 * (len(panels) + 1), 4.2), constrained_layout=True)

    def edges(x):
        x = np.asarray(x, dtype=float)
        d = np.diff(x) if x.size > 1 else np.array([1.0])  # a one-point axis gets a cell of width 1
        return np.r_[x[0] - d[0] / 2, x[:-1] + d / 2, x[-1] + d[-1] / 2]

    for ax, (key, title) in zip(axes, panels):
        z = np.where(maps["done"], maps[key], len(ob.LABELS)).T
        ax.pcolormesh(edges(V0), edges(Om), z, cmap=cmap, norm=norm, shading="flat")
        ax.set_xlabel("V0 (E_R)")
        ax.set_ylabel("Omega_R (E_R)")
        ax.set_title(title, fontsize=10)
        flag = maps["done"] & (maps["p_nmax"] > row["p_nmax_flag"])
        if flag.any():
            iv, io = np.where(flag)
            ax.plot(V0[iv], Om[io], "x", color="k", ms=4)
    ax = axes[-1]
    im = ax.pcolormesh(edges(V0), edges(Om), fs_for_plot(maps).T, cmap="viridis", shading="flat")
    fig.colorbar(im, ax=ax, label=f"log10 f_s(theta -> 0), clipped at {FS_CLIP:g}")
    undecided = (maps["done"] & (maps["label"] == LABEL_CODE["?"])).T
    if undecided.any():  # grey, hatched: the f_s of these points is not a number to read
        ax.pcolor(
            edges(V0),
            edges(Om),
            np.ma.masked_where(~undecided, np.ones(undecided.shape)),
            facecolor="#CCCCCC",
            edgecolor="#777777",
            linewidth=0.0,
            hatch="////",
        )
    ax.set_xlabel("V0 (E_R)")
    ax.set_title("superfluid fraction", fontsize=10)
    from matplotlib.lines import Line2D

    handles = [Patch(color=COLORS[L], label=L) for L in ob.LABELS] + [
        Patch(facecolor="w", edgecolor="k", label="not run"),
        Line2D([], [], ls="", marker="x", color="k", label=f"P(n_max) > {row['p_nmax_flag']:g}"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7)
    sub = f"{row['variant']} model, m = {row['m']}, mu/U = {row['mu_over_U']} ({row['mu_ref']})"
    if row["variant"] == "paper":
        sub += f", M_R rule {row['M_R_rule']}"
    else:
        sub += f", interband {row['interband']}"
    if row.get("method") == "cluster":
        sub += f" | cluster {row['family']}"
    if row["m"] <= 4:
        sub += " | m = 4: speed and plotting only"
    fig.suptitle(f"{name}: {sub}", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("row", choices=sorted(ROWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None, help="stop starting new columns after this long")
    ap.add_argument("--redo-unconverged", action="store_true", help="recompute the points labelled '?' when resuming")
    ap.add_argument("--force-resume", action="store_true", help="resume a map made with another spec or solver")
    ap.add_argument("--u-table", default=None, help="JSON {V0: c} for rows with u_factor = c_v0")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    log_path = os.path.join(a.out, "log.txt")
    os.makedirs(a.out, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    row = dict(ROWS[a.row])
    if a.u_table:
        row["u_table"] = a.u_table
    s = run_row(
        a.row,
        a.out,
        a.device,
        log,
        a.max_hours,
        row=row,
        redo_unconverged=a.redo_unconverged,
        force_resume=a.force_resume,
    )
    log(json.dumps({k: s[k] for k in ("points_done", "points_total", "label_counts", "unconverged", "p_nmax_flagged")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
