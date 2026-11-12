"""Helpers of showcase.ipynb: one trapped cloud of Meng et al.'s twisted bilayer in five models, on the same box.

Continuum (twisted_bilayer.engine.trapped3d.solve_models): full 3D GPE, quasi-2D NPSE (lambda = 1), quasi-2D fixed g_2D.
Lattice (twisted_bilayer_lattice.model): the two-layer Bose-Hubbard model of the same box (variant C: S cut at
|d'| < 2, the interlayer V_ij, the interband shift's site part) with the same harmonic trap on its sites, solved
(a) as the discrete GPE (coherent-state mean field) and (b) by single-site Gutzwiller at the same atom number.

Every model ends as two column densities on the continuum's 2D grid (the lattice models through their Wannier
functions), and every observable comes from the same repository functions (trapped.cloud_numbers_from_density,
trapped.insitu_from_density), so the comparison is like for like. Results are cached per (coupling, box) in
`cache_dir`; a rerun with the same settings only reads them back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------- the repository
def find_repo(start: str | pathlib.Path = ".") -> pathlib.Path | None:
    """The `src` folder of the Gross-Pitaevskii package: `twisted_bilayer` importable already (it is installed by
    `pip install -e ".[showcase]"`), or found in this folder, a parent, or a sibling of a parent (a checkout of that
    repository kept next to this one). The lattice models come from this repository's `twisted_bilayer_lattice`."""
    try:
        import twisted_bilayer

        return pathlib.Path(twisted_bilayer.__file__).resolve().parents[1]
    except ImportError:
        pass
    here = pathlib.Path(start).resolve()
    for p in [here, *here.parents]:
        for cand in [p, *(q for q in p.iterdir() if q.is_dir())] if p.exists() else []:
            src = cand / "src"
            if (src / "twisted_bilayer" / "__init__.py").exists():
                sys.path.insert(0, str(src))
                return src
    return None


# ----------------------------------------------------------------------------- settings
@dataclasses.dataclass
class Settings:
    couplings: tuple = (0.121, 0.150, 0.243)  # keys of scenarios.J_PERP_ER (E_R)
    boxes: tuple = (3, 5)  # odd numbers of sqrt(485) a supercells
    models: tuple = ("3d", "npse", "fixed", "lattice_gpe", "gutzwiller")
    V0: float = 4.0  # E_R
    N_atoms: float = 2000.0
    nu_r_hz: float = 20.0
    nu_z_hz: float = 1000.0
    cell_points: int = 176  # points per supercell (176 = 8 per a)
    Nz: int = 16  # vertical points of the 3D grid over +-z_extent_lz l_z
    z_extent_lz: float = 5.0
    dtype_3d: str = "complex64"
    dtype_2d: str = "complex128"
    tol: float = 1e-9  # E_R, residual target of the continuum solves
    s_sat: float = 0.01  # imaging saturation of C_coh
    gutz_n_max: int = 6
    gutz_tol: float = 1e-8
    gutz_match_N: bool = True  # one Newton step in mu so that Gutzwiller holds the same N as the others
    lattice_box_max: int = 3  # the lattice models run in at most this box and are padded with empty sites beyond it
    # (a 5 x 5 lattice model needs > 6 GB of RAM for its site tables; the cloud fits the 3 x 3 box, so padding is exact
    # up to the 3 x 3 box's edge density)
    cache_dir: str = "showcase_results"

    def continuum_key(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "V0",
                "N_atoms",
                "nu_r_hz",
                "nu_z_hz",
                "cell_points",
                "Nz",
                "z_extent_lz",
                "dtype_3d",
                "dtype_2d",
                "tol",
                "s_sat",
            )
        }

    def lattice_key(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "V0",
                "N_atoms",
                "nu_r_hz",
                "nu_z_hz",
                "cell_points",
                "s_sat",
                "gutz_n_max",
                "gutz_tol",
                "gutz_match_N",
            )
        }  # not the box: see run_lattice


def _tag(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _paths(S: Settings, kind: str, J_key: float, box: int):
    key = S.continuum_key() if kind == "cont" else S.lattice_key()
    stem = f"{kind}_J{J_key:.3f}_box{box}_{_tag(key)}"
    d = pathlib.Path(S.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.json", d / f"{stem}.npz"


def device_report() -> str:
    import torch
    from twisted_bilayer.engine.backend import HAS_GPU

    if HAS_GPU:
        p = torch.cuda.get_device_properties(0)
        return f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, torch {torch.__version__}"
    return f"no GPU: CPU only (torch {torch.__version__}); use QUICK = True or expect long 3D solves"


def memory_estimate_gib(S: Settings, box: int) -> float:
    """Peak of the 3D polish, about 200 bytes per point in complex64 (400 in complex128)."""
    pts = (box * S.cell_points) ** 2 * S.Nz
    return pts * (200 if S.dtype_3d == "complex64" else 400) / 2**30


# ----------------------------------------------------------------------------- the continuum models
_SCANS: dict = {}


def _spec(S: Settings, J: float, box: int):
    from twisted_bilayer.engine import trapped as TR

    return TR.TrappedSpec(
        V0=S.V0,
        Omega_R=J,
        N_atoms=S.N_atoms,
        nu_r_hz=S.nu_r_hz,
        box_cells=box,
        cell_points=S.cell_points,
        omega_z_hz=S.nu_z_hz,
        tol=S.tol,
        dtype=S.dtype_2d,
    )


def run_continuum(S: Settings, J_key: float, box: int, force: bool = False, verbose: bool = True) -> dict:
    """3D, NPSE and fixed g for one (coupling, box); cached."""
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.scenarios import J_PERP_ER

    fj, fz = _paths(S, "cont", J_key, box)
    if fj.exists() and fz.exists() and not force:
        return load(fj, fz)
    J = J_PERP_ER.get(J_key, J_key)  # one of the three readings, or any value in E_R
    spec = _spec(S, J, box)
    vert = T3.Vertical(Nz=S.Nz, z_extent_lz=S.z_extent_lz)
    t0 = time.time()
    skey = (J_key, S.V0, S.cell_points, S.nu_z_hz)
    if skey not in _SCANS:
        _SCANS[skey] = TR.supercell_scan(spec, verbose=False)  # the same for every box at this coupling
    res = T3.solve_models(spec, vert, S.dtype_3d, scan=_SCANS[skey], verbose=False, dtype_2d=S.dtype_2d)
    obs = T3.model_observables(res, s_sat=S.s_sat)
    if verbose:
        for m in T3.MODELS:
            o = obs[m]
            mem = f", peak {o['peak_gib']:.2f} GiB" if o.get("peak_gib") is not None else ""
            print(
                f"  {NAMES[m]:>13s}: C_coh {o['C_coh']:.4f}, residual {o['residual']:.1e}, {o['steps']} steps, "
                f"{o['seconds']:.0f} s{mem}",
                flush=True,
            )
    g = res["grid2"]
    fields = {f"n_{m}": res["models"][m]["n"].astype(np.float32) for m in T3.MODELS}
    fields.update(
        x=g.x64[0].numpy(), width_3d=res["width_3d"].astype(np.float32), width_npse=res["width_npse"].astype(np.float32)
    )
    meta = dict(
        kind="continuum",
        J_key=J_key,
        J=J,
        box=box,
        settings=S.continuum_key(),
        seconds=time.time() - t0,
        E_bottom=_continuum_bottom(S.V0),
        observables=_jsonable(obs),
    )
    np.savez_compressed(fz, **fields)
    fj.write_text(json.dumps(meta, indent=1, default=float))
    return load(fj, fz)


def _continuum_bottom(V0: float) -> float:
    from twisted_bilayer.engine import bands

    return 2.0 * bands.band_bottom_1d(V0)  # 2D band bottom of one decoupled lattice (3.089723 E_R at 4 E_R)


# ----------------------------------------------------------------------------- the lattice models
def _wannier_table(V0: float):
    from twisted_bilayer.engine import bands

    x, w = bands.wannier_1d(V0, n_x=64, n_sites=8)
    return x, w * w


def deposit(cell, occupations: np.ndarray, x_grid: np.ndarray, V0: float, radius: float = 3.0) -> np.ndarray:
    """Site occupations (2, c) of the lattice model -> column densities (2, N, N) [ix, iy] on the continuum grid
    x_j = (j - N/2) dx, each site carrying its Wannier density w(x')^2 w(y')^2 along its own lattice's axes."""
    xw, w2 = _wannier_table(V0)
    N = x_grid.size
    dx = float(x_grid[1] - x_grid[0])
    K = int(math.ceil(radius / dx))
    off = np.arange(-K, K + 1)
    out = np.zeros((occupations.shape[0], N * N))
    for layer, ly in enumerate(cell.layers):
        a = ly.axes[0]  # rows: unit vectors of the two lattice axes (the same for every site)
        for s in range(0, ly.c, 512):
            p = ly.positions[s : s + 512]
            n = occupations[layer, s : s + 512]
            i0 = np.rint(p / dx).astype(np.int64) + N // 2  # nearest grid index of each site
            ix = i0[:, 0, None, None] + off[None, :, None]
            iy = i0[:, 1, None, None] + off[None, None, :]
            X = (ix - N // 2) * dx - p[:, 0, None, None]
            Y = (iy - N // 2) * dx - p[:, 1, None, None]
            u = X * a[0, 0] + Y * a[0, 1]
            v = X * a[1, 0] + Y * a[1, 1]
            val = n[:, None, None] * np.interp(u, xw, w2, 0.0, 0.0) * np.interp(v, xw, w2, 0.0, 0.0)
            flat = (np.mod(ix, N) * N + np.mod(iy, N)).ravel()
            out[layer] += np.bincount(flat, weights=val.ravel(), minlength=N * N)
    return out.reshape(-1, N, N)


def _lattice_model(S: Settings, J: float, box: int):
    from twisted_bilayer.engine import units

    from twisted_bilayer_lattice import model as lm

    cell = lm.supercell_sites(22 * box, box)  # the same box: (m, n) = box x (22, 1), the same twist
    model = lm.build_model(
        S.V0,
        J,
        22 * box,
        box,
        cell=cell,
        nu_z_hz=S.nu_z_hz,
        cutoff=2.0,
        interlayer_interaction=True,
        interband=True,
        interband_bonds=False,
    )
    w = _spec(S, J, box).omega_r()
    pos = np.concatenate([ly.positions for ly in cell.layers])
    model.onsite = model.onsite + 0.5 * units.M_LATTICE * w * w * (pos**2).sum(axis=1)  # the same trap
    return cell, model


def run_lattice(S: Settings, J_key: float, box: int, force: bool = False, verbose: bool = True) -> dict:
    """Lattice GPE and Gutzwiller for one (coupling, box); cached. Boxes above S.lattice_box_max reuse the solution of
    that box, padded with empty sites onto the larger grid."""
    lb = min(box, S.lattice_box_max)
    if box > lb:
        fj, fz = _paths(S, "latt", J_key, box)
        if fj.exists() and fz.exists() and not force:
            return load(fj, fz)
        small = run_lattice(S, J_key, lb, force, verbose)
        x = _grid_x(S, box)
        off = (box - lb) * S.cell_points // 2
        n_small = {m: small["fields"][f"n_{m}"] for m in ("lattice_gpe", "gutzwiller")}
        fields = {"x": x}
        obs = {}
        for m, n in n_small.items():
            big = np.zeros((n.shape[0], x.size, x.size), dtype=np.float32)
            big[:, off : off + n.shape[1], off : off + n.shape[2]] = n
            fields[f"n_{m}"] = big
            obs[m] = _observables(S, big.astype(np.float64), x, box)
            for k in ("mu", "residual", "seconds", "N_model", "condensate_fraction", "converged"):
                if k in small["observables"][m]:
                    obs[m][k] = small["observables"][m][k]
        meta = {k: v for k, v in small.items() if k not in ("fields", "observables")}
        meta.update(box=box, lattice_box=lb, observables=_jsonable(obs))
        np.savez_compressed(fz, **fields)
        fj.write_text(json.dumps(meta, indent=1, default=float))
        return load(fj, fz)
    return _run_lattice_box(S, J_key, box, force, verbose)


def _run_lattice_box(S: Settings, J_key: float, box: int, force: bool = False, verbose: bool = True) -> dict:
    from twisted_bilayer.engine import bands
    from twisted_bilayer.scenarios import J_PERP_ER

    from twisted_bilayer_lattice import model as lm

    fj, fz = _paths(S, "latt", J_key, box)
    if fj.exists() and fz.exists() and not force:
        return load(fj, fz)
    J = J_PERP_ER.get(J_key, J_key)  # one of the three readings, or any value in E_R
    t0 = time.time()
    cell, model = _lattice_model(S, J, box)
    if verbose:
        print(
            f"  lattice model: {2 * cell.c} sites, U = {model.U:.5f}, J1 = {model.J[0]:.5f} E_R "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )
    t1 = time.time()
    gs = lm.ground_state(S.V0, J, 22 * box, box, filling=S.N_atoms / cell.c, model=model, tol=1e-9)
    t_gpe = time.time() - t1
    if verbose:
        print(f"  lattice GPE: mu = {gs.mu:.6f} E_R, residual {gs.residual:.1e} ({t_gpe:.0f} s)", flush=True)
    t1 = time.time()
    gz = lm.gutzwiller(
        S.V0,
        J,
        22 * box,
        box,
        mu_over_U=gs.mu / model.U,
        model=model,
        phi0=gs.phi.numpy(),
        n_max=S.gutz_n_max,
        tol=S.gutz_tol,
    )
    if S.gutz_match_N and gz.compressibility > 0:
        mu1 = gz.mu + (S.N_atoms - gz.n.sum()) / (gz.compressibility * cell.c)
        gz = lm.gutzwiller(
            S.V0,
            J,
            22 * box,
            box,
            mu_over_U=mu1 / model.U,
            model=model,
            phi0=gz.phi,
            n_max=S.gutz_n_max,
            tol=S.gutz_tol,
        )
    t_gz = time.time() - t1
    if verbose:
        print(
            f"  Gutzwiller: N = {gz.n.sum():.1f}, condensate fraction {gz.condensate_fraction:.4f}, "
            f"converged {gz.converged} ({t_gz:.0f} s)",
            flush=True,
        )
    x = _grid_x(S, box)
    nG = deposit(cell, gs.density, x, S.V0)
    nZ = deposit(cell, gz.n, x, S.V0)
    Jl = bands.hopping_1d(S.V0, 2)
    Eb = float(-4.0 * Jl.sum())  # tight-binding band bottom (the lattice model's zero is the on-site energy)
    obs = {"lattice_gpe": _observables(S, nG, x, box), "gutzwiller": _observables(S, nZ, x, box)}
    obs["lattice_gpe"].update(mu=gs.mu, residual=gs.residual, seconds=t_gpe, N_model=float(gs.density.sum()))
    obs["gutzwiller"].update(
        mu=gz.mu,
        residual=float("nan"),
        seconds=t_gz,
        N_model=float(gz.n.sum()),
        condensate_fraction=gz.condensate_fraction,
        converged=bool(gz.converged),
    )
    meta = dict(
        kind="lattice",
        J_key=J_key,
        J=J,
        box=box,
        lattice_box=box,
        settings=S.lattice_key(),
        U=model.U,
        J1=float(model.J[0]),
        J2=float(model.J[1]),
        sites=2 * cell.c,
        E_bottom=Eb,
        seconds=time.time() - t0,
        observables=_jsonable(obs),
    )
    np.savez_compressed(
        fz,
        x=x,
        n_lattice_gpe=nG.astype(np.float32),
        n_gutzwiller=nZ.astype(np.float32),
        occ_gpe=gs.density,
        occ_gutz=gz.n,
        cond_gutz=np.abs(gz.phi) ** 2,
    )
    fj.write_text(json.dumps(meta, indent=1, default=float))
    return load(fj, fz)


def _grid_x(S: Settings, box: int) -> np.ndarray:
    L = box * math.sqrt(485.0)
    N = box * S.cell_points
    return (np.arange(N) - N // 2) * (L / N)


def _observables(S: Settings, n: np.ndarray, x: np.ndarray, box: int) -> dict:
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine.grid import Grid

    L = box * math.sqrt(485.0)
    grid = Grid((x.size, x.size), (L, L))
    cloud = TR.cloud_numbers_from_density(n[0], n[1], grid, math.sqrt(485.0))
    ins = TR.insitu_from_density(n[0], n[1], float(x[1] - x[0]), s_sat=S.s_sat)
    return dict(
        C_raw=ins["C_raw"],
        C_coh=ins["C_coh"],
        n_AA=ins["n_AA"],
        n_AB=ins["n_AB"],
        R_rms=cloud["R_rms"],
        N2_over_N=cloud["N2_over_N"],
        n_peak=cloud["n_peak"],
        nbar_centre_cell=cloud["nbar_centre_cell"],
        edge_density_rel=cloud["edge_density_rel"],
        N_grid=cloud["N"],
    )


# ----------------------------------------------------------------------------- results
def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.bool_):
        return bool(d)
    return d


def load(fj, fz) -> dict:
    meta = json.loads(pathlib.Path(fj).read_text())
    with np.load(fz) as z:
        meta["fields"] = {k: z[k] for k in z.files}
    return meta


def run_all(S: Settings, force: bool = False, verbose: bool = True) -> dict:
    """{(J_key, box): {"cont": ..., "latt": ...}} for every coupling and box in S."""
    out = {}
    want_c = any(m in S.models for m in ("3d", "npse", "fixed"))
    want_l = any(m in S.models for m in ("lattice_gpe", "gutzwiller"))
    for J_key in S.couplings:
        for box in S.boxes:
            out[(J_key, box)] = {}
            if want_c:
                t = time.time()
                if verbose:
                    print(
                        f"J = {J_key} E_R, {box} x {box}: continuum (3D peak ~{memory_estimate_gib(S, box):.1f} GiB)",
                        flush=True,
                    )
                out[(J_key, box)]["cont"] = run_continuum(S, J_key, box, force, verbose)
                if verbose:
                    print(f"  done in {time.time() - t:.0f} s", flush=True)
            if want_l:
                t = time.time()
                if verbose:
                    print(f"J = {J_key} E_R, {box} x {box}: lattice models", flush=True)
                out[(J_key, box)]["latt"] = run_lattice(S, J_key, box, force, verbose)
                if verbose:
                    print(f"  done in {time.time() - t:.0f} s", flush=True)
    return out


NAMES = {
    "3d": "full 3D",
    "npse": "NPSE (2D)",
    "fixed": "fixed g (2D)",
    "lattice_gpe": "lattice GPE",
    "gutzwiller": "Gutzwiller",
}
COLORS = {"3d": "#2a78d6", "npse": "#eb6834", "fixed": "#1baf7a", "lattice_gpe": "#eda100", "gutzwiller": "#e87ba4"}
A_UM = 0.395  # lattice constant in um


def observables(R: dict, J_key: float, box: int, model: str) -> dict | None:
    r = R.get((J_key, box), {})
    src = r.get("cont") if model in ("3d", "npse", "fixed") else r.get("latt")
    if src is None or model not in src["observables"]:
        return None
    o = dict(src["observables"][model])
    o["mu_rel"] = o["mu"] - src["E_bottom"]  # mu above the band bottom of one decoupled lattice
    return o


def density(R: dict, J_key: float, box: int, model: str):
    r = R.get((J_key, box), {})
    src = r.get("cont") if model in ("3d", "npse", "fixed") else r.get("latt")
    if src is None or f"n_{model}" not in src["fields"]:
        return None, None
    return src["fields"]["x"], src["fields"][f"n_{model}"]


def table(
    R: dict, S: Settings, keys=("C_coh", "C_raw", "n_AA", "n_AB", "nbar_centre_cell", "R_rms", "mu_rel", "N2_over_N")
) -> str:
    """Markdown table of every (coupling, box, model)."""
    head = {
        "C_coh": "C_coh",
        "C_raw": "C_raw",
        "n_AA": "n_AA",
        "n_AB": "n_AB",
        "nbar_centre_cell": "n̄ centre",
        "R_rms": "R_rms (µm)",
        "mu_rel": "μ − E_b (E_R)",
        "N2_over_N": "N₂/N",
    }
    lines = [
        "| J⊥ (E_R) | box | model | " + " | ".join(head[k] for k in keys) + " | time (s) |",
        "|---|---|---|" + "---|" * (len(keys) + 1),
    ]
    for J_key in S.couplings:
        for box in S.boxes:
            for m in S.models:
                o = observables(R, J_key, box, m)
                if o is None:
                    continue
                vals = []
                for k in keys:
                    v = o.get(k, float("nan"))
                    vals.append(f"{v * A_UM:.2f}" if k == "R_rms" else f"{v:.4f}")
                src = R[(J_key, box)].get("latt") if m in ("lattice_gpe", "gutzwiller") else None
                note = (
                    f" ({src['lattice_box']}×{src['lattice_box']} sites, padded)"
                    if src and src.get("lattice_box", box) != box
                    else ""
                )
                lines.append(
                    f"| {J_key:.3f} | {box}×{box} | {NAMES[m]}{note} | "
                    + " | ".join(vals)
                    + f" | {o.get('seconds', float('nan')):.0f} |"
                )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def camera(n: np.ndarray, x: np.ndarray, s_sat: float = 0.01) -> np.ndarray:
    """A density (..., N, N) through the in-situ coherent imaging transfer H(|k|) of the paper's objective (NA 0.69,
    780 nm probe, saturation s_sat; imaging.coherent_transfer): what the camera records, the atomic lattice blurred out."""
    from twisted_bilayer.engine import imaging

    N, dx = x.size, float(x[1] - x[0])
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    K = np.hypot(k[:, None], k[None, :])
    Hk = imaging.coherent_transfer(s_sat)(K)
    return np.real(np.fft.ifft2(np.fft.fft2(n, axes=(-2, -1)) * Hk, axes=(-2, -1)))


def plot_images(
    R: dict,
    S: Settings,
    J_key: float,
    box: int,
    component: int = 1,
    zoom_cells: float | None = None,
    as_imaged: bool = True,
):
    """The |2> (component=1) column density of every model through the paper's imaging (as_imaged: NA 0.69, coherent,
    s_sat), same colour scale, um axes."""
    import matplotlib.pyplot as plt

    _present()
    ms = _models(R, S, J_key, box)
    fig, axs = plt.subplots(1, len(ms), figsize=(3.2 * len(ms) + 0.6, 3.9), facecolor=BG, constrained_layout=True)
    axs = np.atleast_1d(axs)
    img = {}
    for m in ms:
        x, n = density(R, J_key, box, m)
        img[m] = camera(n[component], x, S.s_sat) if as_imaged else n[component]
    vmax = max(float(v.max()) for v in img.values())
    half = (zoom_cells * math.sqrt(485.0) / 2 * A_UM) if zoom_cells else _half_um(R, S, J_key, box, ms)
    for i, (ax, m) in enumerate(zip(axs, ms)):
        x, _ = density(R, J_key, box, m)
        xu = x * A_UM
        im = ax.imshow(
            img[m].T,
            origin="lower",
            extent=(xu[0], xu[-1], xu[0], xu[-1]),
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            interpolation="antialiased",
        )
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        if i:
            ax.set_yticks([])
        o = observables(R, J_key, box, m)
        ax.set_title(NAMES[m], color=INK, fontsize=11, pad=4)
        ax.text(0.03, 0.04, f"C_coh = {o['C_coh']:.3f}", transform=ax.transAxes, color="white", fontsize=9)
        ax.set_xlabel("x (µm)", color=MUT)
    axs[0].set_ylabel("y (µm)", color=MUT)
    what = "as the camera sees it (NA 0.69, coherent imaging)" if as_imaged else "column density"
    cb = fig.colorbar(im, ax=list(axs), shrink=0.8, pad=0.01)
    cb.set_label("atoms / site", color=MUT)
    cb.outline.set_visible(False)
    fig.suptitle(
        f"|{component + 1}⟩ {what}: J⊥ = {J_key} E_R, {box} × {box}", color=INK, fontsize=12, x=0.01, ha="left"
    )
    return fig


def plot_cuts(R: dict, S: Settings, J_key: float, box: int):
    """Total density along y = 0 (through the central AA point), every model: (a) the full density with its atomic
    lattice, (b) the same as the camera would record it (the moiré envelope)."""
    import matplotlib.pyplot as plt

    _style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for m in S.models:
        x, n = density(R, J_key, box, m)
        if n is None:
            continue
        j = n.shape[2] // 2
        tot = n.sum(axis=0)
        kw = dict(color=COLORS[m], lw=1.8 if m == "3d" else 1.1, label=NAMES[m])
        axs[0].plot(x, tot[:, j], **kw)
        axs[1].plot(x, camera(tot, x, S.s_sat)[:, j], **kw)
    for ax in axs:
        ax.set_xlabel("x (a), y = 0")
    axs[0].set_ylabel("total column density (atoms / a²)")
    axs[1].set_ylabel("as imaged (atoms / a²)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("a  cut through the cloud (atomic lattice resolved)", loc="left")
    axs[1].set_title("b  the same through the camera: the moiré envelope", loc="left")
    fig.suptitle(f"J⊥ = {J_key} E_R, {box} × {box}", fontsize=10)
    return fig


def plot_observables(R: dict, S: Settings, box: int, keys=("C_coh", "n_AA", "n_AB", "R_rms")):
    """Each observable against the coupling, one marker per model."""
    import matplotlib.pyplot as plt

    _style()
    lab = {
        "C_coh": "central-cell C_coh",
        "C_raw": "C_raw",
        "n_AA": "n_AA (atoms / a²)",
        "n_AB": "n_AB (atoms / a²)",
        "R_rms": "R_rms (µm)",
        "mu_rel": "μ − E_b (E_R)",
        "nbar_centre_cell": "central filling",
    }
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6), constrained_layout=True)
    mk = {"3d": "o", "npse": "s", "fixed": "^", "lattice_gpe": "D", "gutzwiller": "v"}
    for ax, k in zip(np.atleast_1d(axs), keys):
        for i, m in enumerate(S.models):
            xs, ys = [], []
            for J_key in S.couplings:
                o = observables(R, J_key, box, m)
                if o is not None:
                    xs.append(J_key + (i - 2) * 0.003)
                    ys.append(o[k] * (A_UM if k == "R_rms" else 1))
            ax.plot(xs, ys, mk[m], ms=8, color=COLORS[m], mec="white", mew=1, label=NAMES[m], ls="-", lw=0.8)
        ax.set_xlabel("J⊥ (E_R)")
        ax.set_title(lab.get(k, k), fontsize=9, loc="left")
        ax.set_xticks(list(S.couplings))
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Observables of the same cloud in five models ({box} × {box})", fontsize=10)
    return fig


def plot_vs_3d(R: dict, S: Settings, box: int, keys=("C_coh", "n_AA", "n_AB", "R_rms", "mu_rel")):
    """Each model's deviation from full 3D, relative (%), per coupling."""
    import matplotlib.pyplot as plt

    _style()
    ms = [m for m in S.models if m != "3d"]
    fig, axs = plt.subplots(
        1, len(S.couplings), figsize=(4.2 * len(S.couplings), 3.6), constrained_layout=True, sharey=True
    )
    w = 0.8 / max(len(ms), 1)
    for ax, J_key in zip(np.atleast_1d(axs), S.couplings):
        o3 = observables(R, J_key, box, "3d")
        if o3 is None:
            continue
        for i, m in enumerate(ms):
            o = observables(R, J_key, box, m)
            if o is None:
                continue
            d = [100 * (o[k] - o3[k]) / abs(o3[k]) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - (len(ms) - 1) / 2) * w, d, w * 0.92, color=COLORS[m], label=NAMES[m])
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(range(len(keys)), keys, rotation=30)
        ax.set_title(f"J⊥ = {J_key} E_R", fontsize=9, loc="left")
    np.atleast_1d(axs)[0].set_ylabel("(model − 3D) / |3D|  (%)")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"How far each model is from full 3D ({box} × {box})", fontsize=10)
    return fig


def plot_widths(R: dict, J_key: float, box: int):
    """The vertical width of the 3D cloud against the NPSE's, along y = 0."""
    import matplotlib.pyplot as plt

    _style()
    c = R[(J_key, box)].get("cont")
    if c is None:
        return None
    f = c["fields"]
    x = f["x"]
    n = f["n_3d"].sum(axis=0)
    j = n.shape[1] // 2
    occ = n[:, j] > 1e-3 * n.max()
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    ax.plot(x[occ], f["width_3d"][occ, j], color=COLORS["3d"], lw=1.4, label="3D ⟨z²⟩ / (l_z²/2)")
    ax.plot(x[occ], f["width_npse"][occ, j], color=COLORS["npse"], lw=1.1, label="NPSE s²")
    ax.set_xlabel("x (a), y = 0")
    ax.set_ylabel("vertical width² (1 = bare oscillator)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Vertical swelling under the atoms (J⊥ = {J_key} E_R, {box} × {box})", fontsize=10, loc="left")
    return fig


def plot_boxes(R: dict, S: Settings, keys=("C_coh", "n_AA", "R_rms")):
    """The same cloud in the boxes of S: differences from the largest box."""
    import matplotlib.pyplot as plt

    _style()
    if len(S.boxes) < 2:
        return None
    big = max(S.boxes)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 3.4), constrained_layout=True)
    for ax, k in zip(np.atleast_1d(axs), keys):
        for m in S.models:
            ys = []
            for J_key in S.couplings:
                a, b = observables(R, J_key, min(S.boxes), m), observables(R, J_key, big, m)
                ys.append(float("nan") if a is None or b is None else a[k] - b[k])
            ax.plot(S.couplings, ys, "o-", color=COLORS[m], label=NAMES[m], ms=6, lw=1)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"{k}: {min(S.boxes)}×{min(S.boxes)} − {big}×{big}", fontsize=9, loc="left")
        ax.set_xlabel("J⊥ (E_R)")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Box truncation: the same cloud in the small and the large box", fontsize=10)
    return fig


# ----------------------------------------------------------------------------- presentation figures
BG, INK, MUT = "#fcfcfb", "#1f1e1c", "#6b6a66"


def _present():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUT,
            "xtick.color": MUT,
            "ytick.color": MUT,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _models(R, S, J_key, box):
    return [m for m in S.models if density(R, J_key, box, m)[1] is not None]


def _half_um(R, S, J_key, box, ms) -> float:
    """Half-width of the view (um): the cloud out to ~2.4 R_rms, at most the box."""
    o = observables(R, J_key, box, ms[0])
    return min(box * math.sqrt(485.0) / 2, 2.4 * o["R_rms"]) * A_UM


def plot_gallery(R: dict, S: Settings, J_key: float, box: int):
    """A: the total column density of every model side by side, um axes, shared scale, C_coh in the corner."""
    import matplotlib.pyplot as plt

    _present()
    ms = _models(R, S, J_key, box)
    fig, axs = plt.subplots(1, len(ms), figsize=(3.2 * len(ms) + 0.6, 3.9), facecolor=BG, constrained_layout=True)
    axs = np.atleast_1d(axs)
    vmax = max(float(density(R, J_key, box, m)[1].sum(0).max()) for m in ms)
    half = _half_um(R, S, J_key, box, ms)
    for i, (ax, m) in enumerate(zip(axs, ms)):
        x, n = density(R, J_key, box, m)
        x = x * A_UM
        im = ax.imshow(
            n.sum(0).T,
            origin="lower",
            extent=(x[0], x[-1], x[0], x[-1]),
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            interpolation="antialiased",
        )
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        if i:
            ax.set_yticks([])
        o = observables(R, J_key, box, m)
        ax.set_title(NAMES[m], color=INK, fontsize=11, pad=4)
        ax.text(0.03, 0.04, f"C_coh = {o['C_coh']:.3f}", transform=ax.transAxes, color="white", fontsize=9)
        ax.set_xlabel("x (µm)", color=MUT)
    axs[0].set_ylabel("y (µm)", color=MUT)
    cb = fig.colorbar(im, ax=list(axs), shrink=0.8, pad=0.01)
    cb.set_label("column density (atoms / site)", color=MUT)
    cb.outline.set_visible(False)
    fig.suptitle(
        f"Column density of the trapped cloud (J⊥ = {J_key} E_R, N = {S.N_atoms:.0f}, ν_r = {S.nu_r_hz:g} Hz, "
        f"{box} × {box} box)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    return fig


def plot_differences(R: dict, S: Settings, J_key: float, box: int):
    """B: full 3D next to (model - 3D) / max n for every other model, averaged over one lattice site so the
    moire-scale error shows; each panel has its own symmetric scale (in % of the peak)."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    _present()
    ms = [m for m in _models(R, S, J_key, box) if m != "3d"]
    if density(R, J_key, box, "3d")[1] is None or not ms:
        return None
    x, n3 = density(R, J_key, box, "3d")
    n3 = n3.sum(0)
    xu = x * A_UM
    ext = (xu[0], xu[-1], xu[0], xu[-1])
    half = _half_um(R, S, J_key, box, ["3d"])
    sig = 0.5 / float(x[1] - x[0])  # half a lattice constant, in grid points
    fig = plt.figure(figsize=(3.3 * (len(ms) + 1.4), 4.5), facecolor=BG, constrained_layout=True)
    gs = fig.add_gridspec(1, len(ms) + 2, width_ratios=[1.35, 0.06] + [1] * len(ms))
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(n3.T, origin="lower", extent=ext, cmap="inferno", vmin=0, interpolation="antialiased")
    ax0.set_xlim(-half, half)
    ax0.set_ylim(-half, half)
    ax0.set_title("full 3D: column density", color=INK, fontsize=11)
    ax0.set_xlabel("x (µm)", color=MUT)
    ax0.set_ylabel("y (µm)", color=MUT)
    c0 = fig.colorbar(im0, cax=fig.add_subplot(gs[1]))
    c0.outline.set_visible(False)
    c0.set_label("atoms / site", color=MUT)
    for i, m in enumerate(ms):
        ax = fig.add_subplot(gs[2 + i])
        d = gaussian_filter((density(R, J_key, box, m)[1].sum(0) - n3) / n3.max(), sig)
        lim = float(np.abs(d).max()) or 1e-12
        ax.imshow(d.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="antialiased")
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_yticks([])
        ax.set_title(f"{NAMES[m]} − 3D\n(scale ±{100 * lim:.2g} % of the peak)", color=INK, fontsize=10)
        ax.set_xlabel("x (µm)", color=MUT)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
        cb = fig.colorbar(
            sm, ax=ax, shrink=0.7, orientation="horizontal", pad=0.02, format=lambda v, _: f"{100 * v:+.2g}%"
        )
        cb.set_ticks([-lim, 0.0, lim])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUT)
    fig.suptitle(
        "Where each model departs from full 3D (site-averaged; red: too many atoms, blue: too few; "
        f"J⊥ = {J_key} E_R, {box} × {box})",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    return fig


def _two_colour(n: np.ndarray, vmax: float) -> np.ndarray:
    r = np.clip(n[0] / vmax, 0, 1)
    c = np.clip(n[1] / vmax, 0, 1)
    rgb = np.stack([0.98 * r + 0.10 * c, 0.42 * r + 0.62 * c, 0.20 * r + 0.95 * c], -1)
    return np.clip(rgb, 0, 1).transpose(1, 0, 2)


def plot_two_state(R: dict, S: Settings, J_key: float, box: int, cells: float = 1.0):
    """D: |1> (orange) and |2> (blue) in the central `cells` x `cells` supercells; the two square lattices and the
    moire pattern where they overlap (AA, white)."""
    import matplotlib.pyplot as plt

    _present()
    ms = _models(R, S, J_key, box)
    dark = "#111110"
    fig, axs = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.9), facecolor=dark, constrained_layout=True)
    axs = np.atleast_1d(axs)
    vm = 0.8 * max(float(density(R, J_key, box, m)[1].max()) for m in ms)
    hz = cells * math.sqrt(485.0) / 2 * A_UM
    for ax, m in zip(axs, ms):
        x, n = density(R, J_key, box, m)
        xu = x * A_UM
        ax.imshow(
            _two_colour(n, vm), origin="lower", extent=(xu[0], xu[-1], xu[0], xu[-1]), interpolation="antialiased"
        )
        ax.set_xlim(-hz, hz)
        ax.set_ylim(-hz, hz)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(NAMES[m], color="#f0efec", fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
    bar = 2.0
    axs[0].plot([-hz * 0.9, -hz * 0.9 + bar], [-hz * 0.9] * 2, color="white", lw=2)
    axs[0].text(-hz * 0.9 + bar / 2, -hz * 0.82, f"{bar:g} µm", color="white", ha="center", fontsize=8)
    fig.suptitle(
        f"The two lattices in the central supercell: |1⟩ orange, |2⟩ blue, white where both are occupied (AA) "
        f"(J⊥ = {J_key} E_R)",
        color="#f0efec",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    return fig


def plot_radial(R: dict, S: Settings, J_key: float, box: int):
    """E: ring-averaged total density against the distance from the centre, site-averaged and as imaged."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    _style()
    ms = _models(R, S, J_key, box)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG, constrained_layout=True)
    half = _half_um(R, S, J_key, box, ms)
    bins = np.linspace(0, half, 90)
    rc = 0.5 * (bins[1:] + bins[:-1])
    for m in ms:
        x, n = density(R, J_key, box, m)
        X, Y = np.meshgrid(x * A_UM, x * A_UM, indexing="ij")
        idx = np.digitize(np.hypot(X, Y).ravel(), bins)
        cnt = np.maximum(np.bincount(idx, minlength=len(bins) + 1)[1 : len(bins)], 1)
        tot = n.sum(0)
        for ax, f in zip(axs, (gaussian_filter(tot, 0.5 / float(x[1] - x[0])), camera(tot, x, S.s_sat))):
            prof = np.bincount(idx, f.ravel(), len(bins) + 1)[1 : len(bins)] / cnt
            ax.plot(rc, prof, color=COLORS[m], lw=2.2 if m == "3d" else 1.4, label=NAMES[m])
    for ax, t in zip(axs, ("ring-averaged column density (site-averaged)", "ring-averaged, as the camera sees it")):
        ax.set_facecolor(BG)
        ax.set_xlabel("distance from the centre (µm)")
        ax.set_title(t, fontsize=11, loc="left")
    axs[0].set_ylabel("atoms / site")
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Radial profiles (J⊥ = {J_key} E_R, {box} × {box})", fontsize=12, x=0.01, ha="left")
    return fig
