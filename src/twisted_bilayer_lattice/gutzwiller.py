"""Single-site Gutzwiller mean field with twisted boundaries, batching and several starts.

The self-consistent damped map is the one of `engine.lattice_model._gutzwiller_solve`, rewritten here so that
it can carry a twist and a batch dimension without touching the engine: each site sees

    H_i = Psi_i b^dag + Psi_i^* b + (U/2) n (n - 1) + (eps_i + h_i - mu) n,
    Psi_i = sum_j T_ij(theta) phi_j,   h_i = sum_j V_ij <n_j>,

with T(theta) the off-diagonal part of H0 carrying the Peierls phases of the twist on every bond
(`Bonds.amplitudes`, psi = e^{i q.r} phi, q = theta / L), and x <- x + beta (G(x) - x) for x = phi (and <n>
if the model has V). The map, its Mott detection (a settled decay ratio below 1 at |phi| < 1e-5 jumps to
phi = 0), the damping beta = 1.6 / (1 + r) (`default_mixing`) and the start (`lowest_start`) are the
engine's, operation for operation, so at theta = 0 `solve` reproduces `lattice_model.gutzwiller` (pinned in
tests/manybody/test_gutzwiller_twist.py). The arithmetic is real at theta = 0 with a real start, complex
otherwise.

Only public fields of the `LatticeModel` are read (`bonds`, `onsite`, `U`, `V_pairs`, `hartree_operator()`,
`hamiltonian()`, `cell`, `n_layers`, `coupled`). `n_max` is always explicit.

Superfluid fraction by the twist at fixed mu (the energy route; to second order the same as the fixed-N
route of `lattice_model.superfluid_fraction` and the continuum):

    f_s(theta) = (Omega(theta) - Omega(0)) L^2 / (N kappa theta^2),   theta across the box along x,

and the quoted value is the theta -> 0 extrapolation f(0) = (4 f(0.15) - f(0.3)) / 3 (the error of f(theta)
is O(theta^2)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.engine.units import KAPPA

THETA_FS = (0.3, 0.15)  # the two twists of the theta -> 0 extrapolation (decision of 17 Oct)
START_NAMES = ("lowest", "pocket", "random1", "random2")

# Largest batch of small symmetric matrices handed to one CUDA `torch.linalg.eigh` call. cuSOLVER's batched
# eigensolver fails above a size limit that depends on the matrix size, not on the free memory: on a consumer GPU
# (torch 2.11.0+cu128) 9 x 9 matrices pass up to 29 230 per call (7 x 7: 30 000, 11 x 11: 28 536,
# 15 x 15: 27 192) and the next one fails, reported as "CUDA out of memory" for a ~16 GiB buffer; the 46 560
# matrices of one full-model column are above it. No NaN is involved, and the CPU has no such limit. The value
# is below the limit with a margin and was checked on a full column; `batched_eigh` halves it on a failure, so
# another card still works. Chunking splits the independent matrices of a batch.
EIGH_BATCH_MAX = 16384
_EIGH_MIN_CHUNK = 64


def _batch_too_large(exc: RuntimeError) -> bool:
    """A cuSOLVER batch-size failure: reported as a cuSOLVER error or, as on a consumer GPU, as an out-of-memory."""
    msg = str(exc).lower()
    return "cusolver" in msg or "out of memory" in msg


def batched_eigh(H: torch.Tensor, chunk: int | None = None):
    """`torch.linalg.eigh(H)` for H of shape (..., d, d), the leading dimensions flattened into one batch that is
    cut into pieces of at most `chunk` matrices (default: `EIGH_BATCH_MAX` on a CUDA tensor, no cut on the CPU;
    0 means no cut). If cuSOLVER still rejects a piece the piece size is halved and the call repeated, down to
    `_EIGH_MIN_CHUNK` (the failure is a cuSOLVER error or an out-of-memory); any other error is raised. Returns (eigenvalues, eigenvectors) shaped like `eigh`."""
    if chunk is None:
        chunk = EIGH_BATCH_MAX if H.is_cuda else 0
    lead, d = H.shape[:-2], H.shape[-1]
    flat = H.reshape(-1, d, d)
    while True:
        if chunk <= 0 or flat.shape[0] <= chunk:
            try:
                e, V = torch.linalg.eigh(flat)
                break
            except RuntimeError as exc:
                if chunk <= 0 or not _batch_too_large(exc) or chunk <= _EIGH_MIN_CHUNK:
                    raise
                chunk = max(_EIGH_MIN_CHUNK, chunk // 2)
                continue
        try:
            parts = [torch.linalg.eigh(flat[a : a + chunk]) for a in range(0, flat.shape[0], chunk)]
            e = torch.cat([p[0] for p in parts])
            V = torch.cat([p[1] for p in parts])
            break
        except RuntimeError as exc:
            if not _batch_too_large(exc) or chunk <= _EIGH_MIN_CHUNK:
                raise
            chunk = max(_EIGH_MIN_CHUNK, chunk // 2)
    return e.reshape(*lead, d), V.reshape(*lead, d, d)


@dataclass
class GWState:
    """One converged (or not) single-site Gutzwiller state of a model at fixed mu and twist."""

    phi: np.ndarray  # (n_sites,) complex128, <b_i>
    n: np.ndarray  # (n_sites,) <n_i>
    var_n: np.ndarray  # (n_sites,)
    p_top: np.ndarray  # (n_sites,) weight of the Fock state n_max (truncation monitor)
    omega: float  # grand potential of the whole cell (all layers)
    mu: float
    theta: tuple
    converged: bool
    iterations: int
    start: str = ""
    coeffs: np.ndarray | None = field(default=None, repr=False)  # (n_sites, n_max + 1) Gutzwiller amplitudes

    @property
    def N(self) -> float:
        return float(self.n.sum())


# -- the pieces the engine uses, reproduced (same calls, same order) ----------------------------------------------


def lowest_start(model: lm.LatticeModel) -> np.ndarray:
    """0.7 v / max|v| with v the lowest single-particle state at zero twist (layer-balanced sum of each
    layer's lowest state if the layers are uncoupled: phi = 0 on a whole layer never grows). The same
    calls as `lattice_model._initial_phi` and the default start of `lattice_model.gutzwiller`."""
    H = model.hamiltonian((0.0, 0.0))
    c = model.cell.c

    def lowest(A):
        if A.shape[0] <= 64:
            return np.linalg.eigh(A.toarray())[1][:, 0]
        from scipy.sparse.linalg import eigsh

        v0 = np.ones(A.shape[0], dtype=A.dtype)
        return eigsh(A, k=1, which="SA", v0=v0, tol=1e-12)[1][:, 0]

    if model.coupled:
        v = lowest(H)
    else:
        v = np.concatenate([lowest(H[s, s]) for s in (slice(k * c, (k + 1) * c) for k in range(model.n_layers))])
    v = v * 1.0 / np.linalg.norm(v)
    v = v.astype(np.complex128)
    return 0.7 * v / np.abs(v).max()


def pocket_start(model: lm.LatticeModel, radius: float = 2.0) -> np.ndarray:
    """phi = 0.7 on the sites (of every layer) within `radius` (a) of an AA point of the moire pattern,
    0 elsewhere: the start for a state of disconnected AA pockets."""
    AA, _ = lm.moire_points(model.cell)
    out = []
    L = model.cell.L
    for k in range(model.n_layers):
        pos = model.cell.layers[k].positions
        d = pos[:, None, :] - AA[None, :, :]
        d -= L * np.round(d / L)
        near = (np.hypot(d[..., 0], d[..., 1]) < radius).any(axis=1)
        out.append(np.where(near, 0.7, 0.0))
    return np.concatenate(out).astype(np.complex128)


def random_start(model: lm.LatticeModel, seed: int) -> np.ndarray:
    """phi_i = 0.7 u_i, u_i uniform in [0.1, 1) (positive, seeded): a start without the symmetry of the cell."""
    rng = np.random.default_rng(seed)
    return (0.7 * rng.uniform(0.1, 1.0, model.n_sites)).astype(np.complex128)


def make_start(model: lm.LatticeModel, name: str) -> np.ndarray:
    if name == "lowest":
        return lowest_start(model)
    if name == "pocket":
        return pocket_start(model)
    if name.startswith("random"):
        return random_start(model, int(name[len("random") :]))
    raise ValueError(f"unknown start {name!r}")


def default_mixing(model: lm.LatticeModel) -> float:
    """beta = 1.6 / (1 + r), r = |lambda_max / lambda_min| of T (0.5 if T has no negative eigenvalue), capped
    at 1: the damping of `lattice_model._default_mixing` (keeps the staggered phase mode stable)."""
    H = model.hamiltonian()
    T = (H - sp.diags(H.diagonal())).toarray()
    lam = np.linalg.eigvalsh(T)
    if lam[0] >= 0.0:
        return 0.5
    r = abs(lam[-1] / lam[0])
    return float(min(1.0, 1.6 / (1.0 + r)))


# -- the batched solver ---------------------------------------------------------------------------------------------


def same_structure(a: lm.LatticeModel, b: lm.LatticeModel) -> bool:
    """True if two models have the same sites and the same bond / interaction-pair index structure (only the
    amplitudes, U and the site energies may differ): then they can share one batch."""
    if a.n_sites != b.n_sites or a.n_layers != b.n_layers:
        return False
    if not (np.array_equal(a.bonds.rows, b.bonds.rows) and np.array_equal(a.bonds.cols, b.bonds.cols)):
        return False
    if (a.V_pairs is None) != (b.V_pairs is None):
        return False
    if a.V_pairs is not None:
        return bool(np.array_equal(a.V_pairs[0], b.V_pairs[0]) and np.array_equal(a.V_pairs[1], b.V_pairs[1]))
    return True


def solve_batch(
    models,
    mu,
    n_max: int,
    theta=(0.0, 0.0),
    phi0=None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    mixing=None,
    device: str = "cpu",
    starts=None,
    keep_coeffs: bool = False,
) -> list:
    """Solve B models of identical structure (`same_structure`) at once: phi as (B, n_sites), amplitudes as
    (B, n_bonds), one batched eigh of shape (B n_sites, n_max + 1, n_max + 1) per iteration over the members
    still running (convergence is per member; a converged member is frozen at its converged state, so the
    result of each member is the result of its own `solve`).

    mu: scalar or (B,); theta: one twist (2,) for all or (B, 2); phi0: None (each model's `lowest_start`) or a
    list / (B, n_sites) array; mixing: None (each model's `default_mixing`), a scalar or (B,).
    Returns a list of `GWState`."""
    models = list(models)
    B = len(models)
    if B == 0:
        return []
    m0 = models[0]
    for mdl in models[1:]:
        if not same_structure(m0, mdl):
            raise ValueError("models in one batch must share the bond and pair structure")
    ns = m0.n_sites
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (B,)).copy()
    th = np.asarray(theta, dtype=np.float64)
    th = np.broadcast_to(th, (B, 2)).copy() if th.ndim == 1 else th.reshape(B, 2)
    if phi0 is None:
        phi0 = [lowest_start(mdl) for mdl in models]
    phi0 = np.asarray([np.asarray(p, dtype=np.complex128).reshape(-1) for p in phi0])
    if mixing is None:
        mixing = [default_mixing(mdl) for mdl in models]
    mix = np.broadcast_to(np.asarray(mixing, dtype=np.float64), (B,)).copy()
    real = not np.any(th) and float(np.abs(phi0.imag).max()) == 0.0
    dt = torch.float64 if real else torch.complex128
    dev = torch.device(device)

    rows = torch.as_tensor(m0.bonds.rows, device=dev)
    cols = torch.as_tensor(m0.bonds.cols, device=dev)
    if real:
        amp = torch.as_tensor(np.stack([mdl.bonds.amp for mdl in models]), dtype=torch.float64, device=dev)
    else:
        amp = torch.as_tensor(
            np.stack([mdl.bonds.amplitudes(tuple(th[k]), mdl.L) for k, mdl in enumerate(models)]),
            dtype=torch.complex128,
            device=dev,
        )
    with_V = m0.V_pairs is not None
    if with_V:
        c = m0.cell.c
        vi = torch.as_tensor(m0.V_pairs[0], device=dev)
        vj = torch.as_tensor(m0.V_pairs[1], device=dev) + c
        vv = torch.as_tensor(np.stack([mdl.V_pairs[2] for mdl in models]), dtype=torch.float64, device=dev)
    eps = torch.as_tensor(np.stack([mdl.onsite for mdl in models]), dtype=torch.float64, device=dev)
    Uv = torch.as_tensor([mdl.U for mdl in models], dtype=torch.float64, device=dev)
    muv = torch.as_tensor(mu, dtype=torch.float64, device=dev)
    d = n_max + 1
    nn = torch.arange(d, dtype=torch.float64, device=dev)
    sq = torch.sqrt(nn[1:]).to(dt)
    il = torch.arange(1, d, device=dev)
    D0 = 0.5 * Uv[:, None] * nn[None, :] * (nn[None, :] - 1)  # (B, d)

    def T(x, idx):
        return torch.zeros_like(x).index_add(1, rows, amp[idx] * x[:, cols])

    def hartree(dens, idx):
        if not with_V:
            return torch.zeros_like(dens)
        v = vv[idx]
        out = torch.zeros_like(dens)
        out = out.index_add(1, vi, v * dens[:, vj])
        return out.index_add(1, vj, v * dens[:, vi])

    def site_map(phi, n_avg, idx):
        Psi = T(phi, idx)
        D = D0[idx][:, None, :] + (eps[idx] + hartree(n_avg, idx) - muv[idx][:, None])[:, :, None] * nn
        H = torch.diag_embed(D.to(dt))
        H[:, :, il, il - 1] = Psi[:, :, None] * sq
        H[:, :, il - 1, il] = Psi.conj()[:, :, None] * sq
        e, V = batched_eigh(H)
        g = V[..., 0]
        p = g.abs() ** 2
        b = (g[..., :-1].conj() * sq * g[..., 1:]).sum(dim=-1)
        return b, (p * nn).sum(dim=-1), e[..., 0], p, g

    phi = torch.as_tensor(phi0.real if real else phi0, device=dev).to(dt).clone()
    n_avg = torch.zeros(B, ns, dtype=torch.float64, device=dev)
    allidx = torch.arange(B, device=dev)
    if with_V:
        n_avg = site_map(phi, n_avg, allidx)[1]
    mixt = torch.as_tensor(mix, dtype=torch.float64, device=dev)

    # per-member outputs, filled when a member stops
    out_b = phi.clone()
    out_n = torch.zeros(B, ns, dtype=torch.float64, device=dev)
    out_e = torch.zeros(B, ns, dtype=torch.float64, device=dev)
    out_p = torch.zeros(B, ns, d, dtype=torch.float64, device=dev)
    out_g = torch.zeros(B, ns, d, dtype=dt, device=dev)
    converged = np.zeros(B, dtype=bool)
    iters = np.zeros(B, dtype=np.int64)
    ratios = [[] for _ in range(B)]
    active = list(range(B))
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 (it is reported after the loop)
        if not active:
            break
        idx = torch.as_tensor(active, device=dev)
        ph, na = phi[idx], n_avg[idx]
        b, n_new, e0, p, g = site_map(ph, na, idx)
        change = (b - ph).abs().amax(dim=1)
        dn = torch.zeros_like(change, dtype=torch.float64)
        mx = mixt[idx]
        if with_V:
            dn = (n_new - na).abs().amax(dim=1)
            change = torch.maximum(change, dn)
            n_avg[idx] = na + mx[:, None] * (n_new - na)
        new = ph + mx[:, None].to(dt) * (b - ph) if not real else ph + mx[:, None] * (b - ph)
        a_old = ph.abs().amax(dim=1)
        a_new = new.abs().amax(dim=1)
        change_l, dn_l = change.tolist(), dn.tolist()
        a_old_l, a_new_l = a_old.tolist(), a_new.tolist()
        stop, mott = [], []
        for k, mem in enumerate(active):
            if change_l[k] < tol:
                stop.append((k, mem))
                continue
            if 0.0 < a_new_l[k] < 1e-5 and a_old_l[k] > 0.0 and dn_l[k] < tol:
                r = ratios[mem]
                r.append(a_new_l[k] / a_old_l[k])
                if len(r) > 20 and r[-1] < 1.0 and abs(r[-1] - r[-21]) < 1e-6:
                    mott.append((k, mem))
            else:
                ratios[mem] = []
        for k, mem in stop:
            out_b[mem], out_n[mem], out_e[mem], out_p[mem], out_g[mem] = b[k], n_new[k], e0[k], p[k], g[k]
            converged[mem] = True
            iters[mem] = it
        if mott:
            mems = [mem for _, mem in mott]
            midx = torch.as_tensor(mems, device=dev)
            z = torch.zeros(len(mems), ns, dtype=dt, device=dev)
            bz, nz, ez, pz, gz = site_map(z, n_avg[midx], midx)
            for q, (_, mem) in enumerate(mott):
                out_b[mem], out_n[mem], out_e[mem], out_p[mem], out_g[mem] = bz[q], nz[q], ez[q], pz[q], gz[q]
                converged[mem] = True
                iters[mem] = it
        done = {mem for _, mem in stop} | {mem for _, mem in mott}
        keep = [k for k, mem in enumerate(active) if mem not in done]
        if keep:
            kk = torch.as_tensor(keep, device=dev)
            sel = idx[kk]
            phi[sel] = new[kk]
            # remember the latest map output for members that run out of iterations
            out_b[sel], out_n[sel], out_e[sel], out_p[sel], out_g[sel] = b[kk], n_new[kk], e0[kk], p[kk], g[kk]
        active = [mem for mem in active if mem not in done]
    for mem in active:
        iters[mem] = it
    results = []
    for k in range(B):
        ph = out_b[k : k + 1]
        nk = out_n[k]
        pk = out_p[k]
        var = (pk * nn**2).sum(dim=1) - nk**2
        Tphi = T(ph, torch.as_tensor([k], device=dev))[0]
        hart = hartree(nk[None, :], torch.as_tensor([k], device=dev))[0]
        omega = float(out_e[k].sum() - (ph[0].conj() * Tphi).sum().real - 0.5 * (nk * hart).sum())
        results.append(
            GWState(
                phi=ph[0].to(torch.complex128).cpu().numpy(),
                n=nk.cpu().numpy(),
                var_n=var.cpu().numpy(),
                p_top=pk[:, -1].cpu().numpy(),
                omega=omega,
                mu=float(mu[k]),
                theta=(float(th[k, 0]), float(th[k, 1])),
                converged=bool(converged[k]),
                iterations=int(iters[k]),
                start="" if starts is None else str(starts[k]),
                coeffs=out_g[k].to(torch.complex128).cpu().numpy() if keep_coeffs else None,
            )
        )
    return results


def solve(model: lm.LatticeModel, mu: float, n_max: int, theta=(0.0, 0.0), phi0=None, **kw) -> GWState:
    """One model at fixed mu and twist (a batch of one); see `solve_batch`."""
    return solve_batch([model], mu, n_max, theta, None if phi0 is None else [phi0], **kw)[0]


def solve_starts(
    models,
    mu,
    n_max: int,
    starts=START_NAMES,
    **kw,
) -> list:
    """Every model from every start in `starts` (one batch), keeping per model the converged state of lowest
    grand potential (an unconverged one only if no start converged). Returns a list of (best GWState, info),
    info = {"starts": {name: (omega, converged, iterations)}, "omega_spread": max - min omega over the
    converged starts, "best": name}."""
    models = list(models)
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(models),))
    all_models, all_mu, all_phi0, names = [], [], [], []
    for k, mdl in enumerate(models):
        for s in starts:
            all_models.append(mdl)
            all_mu.append(mu[k])
            all_phi0.append(make_start(mdl, s))
            names.append(s)
    mixing = kw.pop("mixing", None)
    if mixing is None:
        mix_m = [default_mixing(mdl) for mdl in models]
        mixing = [mix_m[k] for k in range(len(models)) for _ in starts]
    sols = solve_batch(all_models, all_mu, n_max, phi0=all_phi0, mixing=mixing, starts=names, **kw)
    out = []
    ns = len(starts)
    for k in range(len(models)):
        group = sols[k * ns : (k + 1) * ns]
        conv = [s for s in group if s.converged]
        pool = conv if conv else group
        best = min(pool, key=lambda s: s.omega)
        om = [s.omega for s in conv]
        info = {
            "starts": {s.start: (s.omega, s.converged, s.iterations) for s in group},
            "omega_spread": float(max(om) - min(om)) if len(om) > 1 else 0.0,
            "best": best.start,
        }
        out.append((best, info))
    return out


def variational_omega(model: lm.LatticeModel, state: GWState) -> float:
    """<H(theta) - mu N> of the product state built from `state.coeffs` (the Gutzwiller amplitudes of the last
    iterate): sum_i <(U/2) n (n - 1) + (eps_i - mu) n>_i + sum_bonds T_ij(theta) b_i^* b_j + sum_pairs V_ij n_i n_j
    with b_i, n_i the expectation values in those amplitudes. An upper bound on the Gutzwiller minimum at that
    twist (any product state is variational), equal to `state.omega` at a converged state. Needs a state from
    `solve_batch(..., keep_coeffs=True)`."""
    g = state.coeffs
    if g is None:
        raise ValueError("the state carries no Gutzwiller amplitudes (keep_coeffs=True)")
    d = g.shape[1]
    nn = np.arange(d, dtype=np.float64)
    p = np.abs(g) ** 2
    n = p @ nn
    b = (np.conj(g[:, :-1]) * np.sqrt(nn[1:]) * g[:, 1:]).sum(axis=1)
    loc = p @ (0.5 * model.U * nn * (nn - 1)) + (np.asarray(model.onsite) - state.mu) * n
    amp = model.bonds.amplitudes(state.theta, model.L)
    kin = np.real(np.sum(amp * np.conj(b[model.bonds.rows]) * b[model.bonds.cols]))
    inter = 0.0
    if model.V_pairs is not None:
        c = model.cell.c
        i, j, v = model.V_pairs
        inter = float(np.sum(v * n[i] * n[np.asarray(j) + c]))
    return float(loc.sum() + kin + inter)


# -- superfluid fraction by the twist -------------------------------------------------------------------------------


def fs_from_omegas(model: lm.LatticeModel, omega0: float, omega_theta: float, N: float, theta: float) -> float:
    """(Omega(theta) - Omega(0)) L^2 / (N kappa theta^2); 0 for an empty cell."""
    if N <= 0.0:
        return 0.0
    return float((omega_theta - omega0) * model.L**2 / (N * KAPPA * theta * theta))


def fs_batch(models, base_states, n_max: int, thetas=THETA_FS, axis: int = 0, **kw) -> list:
    """The twisted solves of `models` at the twists `thetas` along `axis`, warm-started from the theta = 0
    states `base_states` (same mu); returns per model {"fs": {theta: f}, "fs_upper": {theta: f}, "converged":
    bool, "states": ...}. "fs_upper" uses the variational Omega of the last iterate (`variational_omega`): an
    upper bound on f_s(theta) whether or not the twisted solve converged (given a converged base); equal to
    "fs" when it did. A Mott base state (phi = 0) gives exactly Omega(theta) = Omega(0), f_s = 0."""
    models = list(models)
    kw = dict(kw)
    kw.pop("keep_coeffs", None)
    res = [{"fs": {}, "fs_upper": {}, "converged": True, "twist_converged": True, "states": {}} for _ in models]
    for th in thetas:
        tw = [0.0, 0.0]
        tw[axis] = th
        mus = [s.mu for s in base_states]
        phi0 = [s.phi for s in base_states]
        sols = solve_batch(models, mus, n_max, theta=tuple(tw), phi0=phi0, keep_coeffs=True, **kw)
        for k, (mdl, s0, st) in enumerate(zip(models, base_states, sols)):
            res[k]["fs"][th] = fs_from_omegas(mdl, s0.omega, st.omega, s0.N, th)
            res[k]["fs_upper"][th] = fs_from_omegas(mdl, s0.omega, variational_omega(mdl, st), s0.N, th)
            res[k]["converged"] = res[k]["converged"] and st.converged and s0.converged
            res[k]["twist_converged"] = res[k]["twist_converged"] and st.converged
            st.coeffs = None
            res[k]["states"][th] = st
    return res


def extrapolate_fs0(fs: dict, thetas=THETA_FS) -> float:
    """f(0) = (4 f(theta/2) - f(theta)) / 3 for thetas = (theta, theta / 2): the O(theta^2) term removed."""
    t1, t2 = thetas
    if abs(t2 - 0.5 * t1) > 1e-12:
        raise ValueError("the extrapolation needs thetas = (theta, theta / 2)")
    return float((4.0 * fs[t2] - fs[t1]) / 3.0)


def gutzwiller_fs(model: lm.LatticeModel, mu: float, n_max: int, theta: float = 0.3, base: GWState | None = None, **kw):
    """f_s at one twist theta (along x) at fixed mu; `base` = the theta = 0 state (solved if None)."""
    base = solve(model, mu, n_max, **kw) if base is None else base
    r = fs_batch([model], [base], n_max, thetas=(theta,), **kw)[0]
    return r["fs"][theta]


def gutzwiller_fs0(model: lm.LatticeModel, mu: float, n_max: int, base: GWState | None = None, **kw) -> dict:
    """The quoted f_s: theta -> 0 from theta = 0.3 and 0.15 at fixed mu. Returns {"fs0", "fs", "converged"}."""
    base = solve(model, mu, n_max, **kw) if base is None else base
    r = fs_batch([model], [base], n_max, thetas=THETA_FS, **kw)[0]
    return {"fs0": extrapolate_fs0(r["fs"]), "fs": r["fs"], "converged": r["converged"]}


def gutzwiller_fs0_fixed_n(
    model: lm.LatticeModel,
    filling: float,
    n_max: int,
    base: GWState | None = None,
    thetas=THETA_FS,
    axis: int = 0,
    max_retry: int = 2,
    **kw,
) -> dict:
    """f_s at fixed particle number: the twist energy E(theta) - E(0), E = Omega + mu N, of the states at the same
    `filling` (`solve_filling` at each twist, warm-started from the untwisted state), theta -> 0 from `thetas`
    as in `gutzwiller_fs0`. This is the route of the continuum f_s (fixed N), and the reliable one where
    N(mu) is steep or where a fixed-mu twisted solve can fall to another branch (the fixed-mu energy of a state
    whose N drifts with the twist is not a twist energy). To second order in theta it equals the fixed-mu route;
    the two agree to O(theta^4) where both are valid.

    A twisted energy below the untwisted one (by more than 1e-5 in f_s) means the untwisted state was not the
    minimum at that filling (a metastable state of the superfluid islands: the twisted solve found the lower
    one). The untwisted state is then solved again from the lowest twisted state, the second time with a quarter of
    the damping, at most `max_retry` times; a state that is not lower and converged is not accepted.

    Returns {"fs0", "fs", "converged", "N", "mu", "base", "states", "retries"} ("converged": every solve and the
    filling of every twisted state within 1e-6 relative of the target, and no negative f_s left)."""
    base = solve_filling(model, filling, n_max, **kw) if base is None else base
    target = filling * model.cell.c

    def twist_solves(base):
        E0 = base.omega + base.mu * base.N
        fs, states = {}, {}
        ok = bool(base.converged) and abs(base.N - target) <= 1e-6 * target
        for th in thetas:
            tw = [0.0, 0.0]
            tw[axis] = th
            st = solve_filling(model, filling, n_max, theta=tuple(tw), phi0=base.phi, **kw)
            fs[th] = fs_from_omegas(model, E0, st.omega + st.mu * st.N, base.N, th)
            ok = ok and bool(st.converged) and abs(st.N - target) <= 1e-6 * target
            states[th] = st
        return E0, fs, states, ok

    E0, fs, states, ok = twist_solves(base)
    retries = 0
    while min(fs.values()) < -1e-5 and retries < max_retry:
        low = states[min(fs, key=fs.get)]
        kw2 = dict(kw)
        if retries >= 1:
            kw2["mixing"] = 0.25 * default_mixing(model)
        retries += 1
        b2 = solve_filling(model, filling, n_max, phi0=low.phi, **kw2)
        if b2.converged and b2.omega + b2.mu * b2.N < E0 - 1e-12 * max(1.0, abs(E0)):
            base = b2
            E0, fs, states, ok = twist_solves(base)
    return {
        "fs0": extrapolate_fs0(fs, thetas),
        "fs": fs,
        "converged": ok and min(fs.values()) >= -1e-5,
        "N": base.N,
        "mu": base.mu,
        "base": base,
        "states": states,
        "retries": retries,
    }


# -- fixed filling --------------------------------------------------------------------------------------------------


def solve_filling(
    model: lm.LatticeModel, filling: float, n_max: int, theta=(0.0, 0.0), phi0=None, max_bisect: int = 200, **kw
) -> GWState:
    """The state at `filling` (atoms per a^2 over all layers, sum <n> = filling c): mu by the safeguarded regula
    falsi alternating with bisection of `lattice_model.gutzwiller` (N(mu) has kinks at the lobe edges)."""
    c, U = model.cell.c, model.U
    start = lowest_start(model) if phi0 is None else np.asarray(phi0, dtype=np.complex128)

    def run(mu, p0):
        return solve(model, mu, n_max, theta, p0, **kw)

    def warm(s):
        return s.phi if float(np.abs(s.phi).max()) > 1e-3 else start

    target = filling * c
    lo = float(np.linalg.eigvalsh(model.hamiltonian().toarray())[0]) - 1e-3
    hi = max(U, lo + 1.0)
    sol = run(hi, start)
    while sol.N < target:
        lo, hi = hi, 2.0 * hi + U
        sol = run(hi, warm(sol))
    a, fa, b, fb = lo, -target, hi, sol.N - target
    sol_b = sol
    for k in range(max_bisect):
        if k % 2 == 0:
            mu = b - fb * (b - a) / (fb - fa)
            mu = min(max(mu, a + 0.05 * (b - a)), b - 0.05 * (b - a))
        else:
            mu = 0.5 * (a + b)
        sol = run(mu, warm(sol_b))
        fm = sol.N - target
        if abs(fm) <= 1e-9 * target or b - a < 1e-12 * U:
            break
        if fm < 0.0:
            a, fa = mu, fm
        else:
            b, fb, sol_b = mu, fm, sol
    return sol


def compressibility(model: lm.LatticeModel, state: GWState, n_max: int, dmu_over_U: float = 0.01, **kw) -> float:
    """d filling / d mu by central difference +- dmu_over_U U, warm-started (as `lattice_model.gutzwiller`)."""
    dmu = dmu_over_U * model.U
    p0 = state.phi if float(np.abs(state.phi).max()) > 1e-3 else lowest_start(model)
    Np = solve(model, state.mu + dmu, n_max, state.theta, p0, **kw).N
    Nm = solve(model, state.mu - dmu, n_max, state.theta, p0, **kw).N
    return (Np - Nm) / (2.0 * dmu * model.cell.c)


__all__ = [
    "GWState",
    "START_NAMES",
    "THETA_FS",
    "compressibility",
    "default_mixing",
    "extrapolate_fs0",
    "fs_batch",
    "fs_from_omegas",
    "gutzwiller_fs",
    "gutzwiller_fs0",
    "gutzwiller_fs0_fixed_n",
    "lowest_start",
    "make_start",
    "pocket_start",
    "random_start",
    "same_structure",
    "solve",
    "solve_batch",
    "solve_filling",
    "solve_starts",
]
