"""Cluster Gutzwiller (cluster mean field): clusters of sites treated exactly, coupled by a mean field.

Trial state |Psi> = prod_C |psi_C>; each cluster sees

    H_C = sum_{i,j in C} T_ij b_i^dag b_j + sum_{i in C} [(U/2) n_i (n_i - 1) + (eps_i - mu) n_i]
          + sum_{i<j in C} V_ij n_i n_j + sum_{i in C} (Psi_i b_i^dag + Psi_i^* b_i + h_i n_i),
    Psi_i = sum_{j not in C} T_ij phi_j,   h_i = sum_{j not in C} V_ij <n_j>,

T(theta) with the Peierls phases of the twist on every bond (inside and across clusters), iterated with the
damped map of single-site Gutzwiller (same mixing, same Mott detection: phi = 0 is always a fixed point and
root finders converge to it). Each cross bond's mean-field energy is in both of its clusters, so

    Omega = sum_C E_C - Re sum_{cross (i -> j)} T_ij phi_i^* phi_j - sum_{cross pairs} V_ij <n_i><n_j>

(the single-site formula of `manybody.gutzwiller`, restricted to cross bonds; one-site clusters reproduce it).

Tilings (`make_tiling`): "single"; "dimer" (a layer-1 site and a layer-2 site, by a minimum-cost perfect matching
of the periodic minimum-image distances, `scipy.optimize.linear_sum_assignment`); "plus" (a site and its four
nearest neighbours; the centres form the index-5 sublattice {i - 2j = 0 mod 5} or its mirror {i + 2j = 0 mod 5}
in the layer's own coordinates, the one containing the layer's box translations: layer 1 of the (22, 1) cell
takes the first, layer 2 the second); "bilayer_plus" (a layer-1 plus and the matched layer-2 plus).

Lowest eigenvector of each cluster: dense batched `torch.linalg.eigh` up to `dense_max` states, otherwise
`scipy.sparse.linalg.eigsh` warm-started from the previous iteration's vector.

The plain square lattice (the cluster-size scaling of the n = 1 lobe tip against 16.74) uses the
translation-invariant shortcut: with a uniform solution every cluster is equivalent, so one cluster is
solved, its outside neighbours mapped back onto its own sites by the tiling translations
(`SquareCluster`). The lobe boundary is located by the linear stability of the Mott state: the cluster
susceptibility chi_ik = <0| b_i (H_{N+1} - E_0)^-1 b_k^dag |0> + <0| b_k^dag (H_{N-1} - E_0)^-1 b_i |0> and
the outside-bond matrix M give delta phi = J chi M delta phi, so the Mott state is unstable once the largest
eigenvalue of J chi M reaches 1 (exact for the continuous transition of the cluster mean field; single site:
U / J = 4 (3 + 2 sqrt 2) = 23.31 at mu / U = sqrt 2 - 1).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.manybody import gutzwiller as gw
from twisted_bilayer.manybody import observables as ob

# -- tilings --------------------------------------------------------------------------------------------------------

PLUS_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))  # centre first, then +x, -x, +y, -y


@dataclass
class ClusterTiling:
    family: str
    clusters: list  # list of int arrays (global site indices), all of one size for the families here
    site_cluster: np.ndarray  # (n_sites,) cluster index of each site
    site_pos: np.ndarray  # (n_sites,) position of each site in its cluster
    inside: np.ndarray  # (n_bonds,) bool: both ends of the bond in one cluster
    lam: float  # fraction of the bonds (by count) inside clusters
    info: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


def _finish(model, family, clusters, info=None) -> ClusterTiling:
    ns = model.n_sites
    sc = -np.ones(ns, dtype=np.int64)
    sp_ = -np.ones(ns, dtype=np.int64)
    for k, cl in enumerate(clusters):
        if np.any(sc[cl] >= 0):
            raise RuntimeError(f"{family}: a site is in two clusters")
        sc[cl] = k
        sp_[cl] = np.arange(len(cl))
    if np.any(sc < 0):
        raise RuntimeError(f"{family}: {int((sc < 0).sum())} sites in no cluster")
    inside = sc[model.bonds.rows] == sc[model.bonds.cols]
    lam = float(inside.mean()) if inside.size else 0.0
    return ClusterTiling(family, [np.asarray(c, dtype=np.int64) for c in clusters], sc, sp_, inside, lam, info or {})


def plus_chirality(ly: lm.LayerSites) -> int:
    """+1 if the layer's box translations lie in {i - 2j = 0 mod 5}, -1 if in {i + 2j = 0 mod 5}, 0 if in neither
    (then the layer cannot be tiled by plus clusters)."""
    M = ly.M.astype(np.float64)
    c = ly.c
    tr = [np.rint(np.linalg.solve(M, v)).astype(np.int64) for v in ([c, 0], [0, c])]
    for s in (1, -1):
        if all((t[0] - s * 2 * t[1]) % 5 == 0 for t in tr):
            return s
    return 0


def _plus_layer(ly: lm.LayerSites, offset: int):
    s = plus_chirality(ly)
    if s == 0 or ly.c % 5:
        raise ValueError(
            "this layer cannot be tiled by plus clusters (c not divisible by 5 or translations off the sublattice)"
        )
    ij = ly.ij
    f = np.mod(ij[:, 0] - s * 2 * ij[:, 1], 5)
    centres = np.where(f == 0)[0]
    clusters = []
    for k in centres:
        sites = []
        for d in PLUS_OFFSETS:
            XY = ly.M @ (ij[k] + np.array(d))
            idx = int(ly.index(XY))
            if idx < 0:
                raise RuntimeError("plus arm is not a site")
            sites.append(idx + offset)
        clusters.append(sites)
    return clusters, s


def _min_image_dist(P, Q, L):
    d = P[:, None, :] - Q[None, :, :]
    d -= L * np.round(d / L)
    return np.hypot(d[..., 0], d[..., 1])


def make_tiling(model: lm.LatticeModel, family: str) -> ClusterTiling:
    """The tiling of the model's sites by one cluster family (module docstring)."""
    from scipy.optimize import linear_sum_assignment

    c, L = model.cell.c, model.cell.L
    if family == "single":
        return _finish(model, family, [[k] for k in range(model.n_sites)])
    if family == "dimer":
        if model.n_layers != 2:
            raise ValueError("the dimer family needs two layers")
        D = _min_image_dist(model.cell.layers[0].positions, model.cell.layers[1].positions, L)
        r, q = linear_sum_assignment(D)
        clusters = [[int(i), int(j) + c] for i, j in zip(r, q)]
        return _finish(
            model, family, clusters, {"max_distance": float(D[r, q].max()), "mean_distance": float(D[r, q].mean())}
        )
    if family == "plus":
        clusters, chir = [], []
        for k in range(model.n_layers):
            cl, s = _plus_layer(model.cell.layers[k], k * c)
            clusters += cl
            chir.append(s)
        return _finish(model, family, clusters, {"chirality": chir})
    if family == "bilayer_plus":
        if model.n_layers != 2:
            raise ValueError("the bilayer plus family needs two layers")
        c1, s1 = _plus_layer(model.cell.layers[0], 0)
        c2, s2 = _plus_layer(model.cell.layers[1], c)
        P1 = model.cell.layers[0].positions[[cl[0] for cl in c1]]
        P2 = model.cell.layers[1].positions[[cl[0] - c for cl in c2]]
        D = _min_image_dist(P1, P2, L)
        r, q = linear_sum_assignment(D)
        clusters = [c1[i] + c2[j] for i, j in zip(r, q)]
        return _finish(model, family, clusters, {"chirality": [s1, s2], "max_centre_distance": float(D[r, q].max())})
    if family == "whole":
        return _finish(model, family, [list(range(model.n_sites))])
    raise ValueError(f"unknown family {family!r}")


# -- cluster Fock space ---------------------------------------------------------------------------------------------


@dataclass
class FockBasis:
    s: int  # sites
    n_max: int
    n_total_max: int | None
    occ: np.ndarray  # (D, s) occupations
    B: list  # s sparse (D, D) annihilators b_i
    N: np.ndarray  # (D, s) = occ (diagonal of n_i)

    @property
    def dim(self) -> int:
        return self.occ.shape[0]


_BASIS_CACHE: dict = {}


def fock_basis(s: int, n_max: int, n_total_max: int | None = None) -> FockBasis:
    """Product basis of s sites with 0 <= n_i <= n_max and (optionally) sum n_i <= n_total_max, with the
    annihilators b_i as sparse matrices (b_i |..n_i..> = sqrt(n_i) |..n_i - 1..>; lowering stays in the basis)."""
    key = (s, n_max, n_total_max)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    d = n_max + 1
    occ = np.array(list(itertools.product(range(d), repeat=s)), dtype=np.int64).reshape(-1, s)
    if n_total_max is not None:
        occ = occ[occ.sum(axis=1) <= n_total_max]
    code = occ @ (d ** np.arange(s)[::-1])
    lookup = -np.ones(d**s, dtype=np.int64)
    lookup[code] = np.arange(occ.shape[0])
    B = []
    for i in range(s):
        src = np.where(occ[:, i] > 0)[0]
        dst = lookup[code[src] - d ** (s - 1 - i)]
        vals = np.sqrt(occ[src, i].astype(np.float64))
        B.append(sp.csr_matrix((vals, (dst, src)), shape=(occ.shape[0], occ.shape[0])))
    fb = FockBasis(s, n_max, n_total_max, occ, B, occ.astype(np.float64))
    _BASIS_CACHE[key] = fb
    return fb


# -- the cluster mean-field solver ----------------------------------------------------------------------------------


@dataclass
class ClusterState:
    phi: np.ndarray
    n: np.ndarray
    var_n: np.ndarray
    p_top: np.ndarray  # per site: probability of n_i = n_max
    omega: float
    mu: float
    theta: tuple
    converged: bool
    iterations: int
    vectors: list = field(repr=False, default_factory=list)  # lowest eigenvector of each cluster
    rho_connected: tuple | None = None
    p_total_cap: float = 0.0  # max over clusters of the weight at sum n = n_total_max (total-cutoff monitor)

    @property
    def N(self) -> float:
        return float(self.n.sum())


def _cluster_pieces(model, tiling, theta, fb, mu):
    """Per cluster: fixed Hamiltonian (sparse), and the cross-bond / cross-pair index lists."""
    rows, cols = model.bonds.rows, model.bonds.cols
    amp = model.bonds.amplitudes(theta, model.L)
    sc, spos = tiling.site_cluster, tiling.site_pos
    eps = np.asarray(model.onsite, dtype=np.float64)
    D0 = sum(0.5 * model.U * fb.N[:, k] * (fb.N[:, k] - 1) for k in range(fb.s))
    pairs = None
    if model.V_pairs is not None:
        c = model.cell.c
        pairs = (np.asarray(model.V_pairs[0]), np.asarray(model.V_pairs[1]) + c, np.asarray(model.V_pairs[2]))
    fixed = []
    cplx = np.iscomplexobj(amp)
    for k, cl in enumerate(tiling.clusters):
        diag = D0 + sum((eps[g] - mu) * fb.N[:, p] for p, g in enumerate(cl))
        H = sp.diags(diag).astype(np.complex128 if cplx else np.float64)
        sel = np.where((sc[rows] == k) & (sc[cols] == k))[0]
        for b in sel:
            i, j = spos[rows[b]], spos[cols[b]]
            if i == j:
                H = H + amp[b] * sp.diags(fb.N[:, i])
            else:
                H = H + amp[b] * (fb.B[i].T @ fb.B[j])  # b_i^dag b_j
        if pairs is not None:
            pi, pj, pv = pairs
            sel = np.where((sc[pi] == k) & (sc[pj] == k))[0]
            for q in sel:
                H = H + pv[q] * sp.diags(fb.N[:, spos[pi[q]]] * fb.N[:, spos[pj[q]]])
        fixed.append(H.tocsr())
    cross = ~tiling.inside
    return fixed, amp, cross, pairs


def cluster_gutzwiller(
    model: lm.LatticeModel,
    tiling: ClusterTiling,
    mu: float,
    n_max: int,
    n_total_max: int | None = None,
    theta=(0.0, 0.0),
    phi0=None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    mixing: float | None = None,
    dense_max: int = 2000,
    device: str = "cpu",
) -> ClusterState:
    """Self-consistent cluster mean field at fixed mu and twist (module docstring). All clusters must have the
    same number of sites. phi0: start (default `gutzwiller.lowest_start`); mixing: default
    `gutzwiller.default_mixing` (the single-site damping, which keeps the staggered mode stable)."""
    sizes = {len(cl) for cl in tiling.clusters}
    if len(sizes) != 1:
        raise ValueError("clusters of one tiling must have one size")
    s = sizes.pop()
    fb = fock_basis(s, n_max, n_total_max)
    fixed, amp, cross, pairs = _cluster_pieces(model, tiling, theta, fb, mu)
    rows, cols = model.bonds.rows, model.bonds.cols
    ns = model.n_sites
    nc = len(tiling.clusters)
    cl_idx = np.stack(tiling.clusters)  # (nc, s)
    cr_rows, cr_cols, cr_amp = rows[cross], cols[cross], amp[cross]
    if pairs is not None:
        pi, pj, pv = pairs
        xpair = tiling.site_cluster[pi] != tiling.site_cluster[pj]
        xi, xj, xv = pi[xpair], pj[xpair], pv[xpair]
    if mixing is None:
        mixing = gw.default_mixing(model)
    phi = (gw.lowest_start(model) if phi0 is None else np.asarray(phi0, dtype=np.complex128)).copy()
    complex_ = np.iscomplexobj(amp) or np.abs(phi.imag).max() > 0
    if not complex_:
        phi = phi.real.copy()
    n_avg = np.zeros(ns)
    dense = fb.dim <= dense_max
    Bd = [b.toarray() for b in fb.B] if dense else None
    Fd = np.stack([f.toarray() for f in fixed]) if dense else None  # (nc, D, D); dense_max keeps it small
    vecs = [None] * nc

    def fields(phi_, n_):
        Psi = np.zeros(ns, dtype=np.complex128 if complex_ else np.float64)
        np.add.at(Psi, cr_rows, cr_amp * phi_[cr_cols])
        h = np.zeros(ns)
        if pairs is not None and xi.size:
            np.add.at(h, xi, xv * n_[xj])
            np.add.at(h, xj, xv * n_[xi])
        return Psi, h

    def cluster_map(phi_, n_):
        Psi, h = fields(phi_, n_)
        P = Psi[cl_idx]  # (nc, s)
        Hh = h[cl_idx]
        out_b = np.zeros((nc, s), dtype=np.complex128)
        e0 = np.zeros(nc)
        if dense:
            dt = np.complex128 if complex_ else np.float64
            chunk = max(1, int(2e8 // (16 * fb.dim * fb.dim)))  # at most ~200 MB of matrices at once
            for a in range(0, nc, chunk):
                sl = slice(a, min(nc, a + chunk))
                H = Fd[sl].astype(dt)
                for p in range(s):
                    Bp = Bd[p]
                    H += P[sl, p, None, None] * Bp.T[None] + np.conj(P[sl, p])[:, None, None] * Bp[None]
                    H += Hh[sl, p, None, None] * np.diag(fb.N[:, p])[None]
                e, V = torch.linalg.eigh(torch.as_tensor(H, device=device))
                e, V = e[:, 0].cpu().numpy(), V[:, :, 0].cpu().numpy()
                for q, k in enumerate(range(sl.start, sl.stop)):
                    vecs[k] = V[q]
                e0[sl] = e
        else:
            for k in range(nc):
                H = fixed[k]
                for p in range(s):
                    H = H + P[k, p] * fb.B[p].T + np.conj(P[k, p]) * fb.B[p] + Hh[k, p] * sp.diags(fb.N[:, p])
                v0 = vecs[k] if vecs[k] is not None else np.ones(fb.dim)
                w, v = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-13)
                vecs[k] = v[:, 0]
                e0[k] = w[0]
        Vs = np.stack(vecs)  # (nc, D)
        pr = np.abs(Vs) ** 2
        out_n = pr @ fb.N
        out_n2 = pr @ fb.N**2
        out_top = pr @ (fb.occ == n_max).astype(np.float64)
        for p in range(s):
            out_b[:, p] = np.einsum("kd,kd->k", np.conj(Vs), (fb.B[p] @ Vs.T).T)
        b = np.zeros(ns, dtype=np.complex128)
        nn_ = np.zeros(ns)
        n2 = np.zeros(ns)
        top = np.zeros(ns)
        b[cl_idx] = out_b
        nn_[cl_idx] = out_n
        n2[cl_idx] = out_n2
        top[cl_idx] = out_top
        if not complex_:
            b = b.real
        return b, nn_, n2, top, e0

    with_V = pairs is not None
    if with_V:
        n_avg = cluster_map(phi, n_avg)[1]
    converged, ratios, it = False, [], 0
    for it in range(1, max_iter + 1):  # noqa: B007
        b, n_new, n2, top, e0 = cluster_map(phi, n_avg)
        change = float(np.abs(b - phi).max())
        dn = 0.0
        if with_V:
            dn = float(np.abs(n_new - n_avg).max())
            change = max(change, dn)
            n_avg = n_avg + mixing * (n_new - n_avg)
        if change < tol:
            converged = True
            break
        new = phi + mixing * (b - phi)
        a_old, a_new = float(np.abs(phi).max()), float(np.abs(new).max())
        if 0.0 < a_new < 1e-5 and a_old > 0.0 and dn < tol:
            ratios.append(a_new / a_old)
            if len(ratios) > 20 and ratios[-1] < 1.0 and abs(ratios[-1] - ratios[-21]) < 1e-6:
                phi = np.zeros_like(phi)
                b, n_new, n2, top, e0 = cluster_map(phi, n_avg)
                converged = True
                break
        else:
            ratios = []
        phi = new
    phi = b
    # Omega: cluster energies minus the doubly counted cross-bond and cross-pair mean-field energies
    omega = float(e0.sum() - np.real(np.sum(cr_amp * np.conj(phi[cr_rows]) * phi[cr_cols])))
    if with_V and xi.size:
        omega -= float(np.sum(xv * n_new[xi] * n_new[xj]))
    # connected one-body density matrix inside each cluster
    rr, cc, vv = [], [], []
    if s > 1:
        for k, cl in enumerate(tiling.clusters):
            v = vecs[k]
            Bv = [fb.B[p] @ v for p in range(s)]
            for p in range(s):
                for q in range(s):
                    if p == q:
                        continue
                    rho = np.vdot(Bv[p], Bv[q])  # <b_p^dag b_q>
                    rr.append(cl[p])
                    cc.append(cl[q])
                    vv.append(rho - np.conj(phi[cl[p]]) * phi[cl[q]])
    rho_c = (np.array(rr, dtype=np.int64), np.array(cc, dtype=np.int64), np.array(vv)) if rr else None
    p_cap = 0.0
    if n_total_max is not None:
        capm = fb.occ.sum(axis=1) == n_total_max
        p_cap = max(float((np.abs(v) ** 2)[capm].sum()) for v in vecs)
    return ClusterState(
        phi=np.asarray(phi, dtype=np.complex128),
        n=n_new,
        var_n=n2 - n_new**2,
        p_top=top,
        omega=omega,
        mu=float(mu),
        theta=tuple(float(t) for t in theta),
        converged=converged,
        iterations=it,
        vectors=list(vecs),
        rho_connected=rho_c,
        p_total_cap=p_cap,
    )


def cluster_variational_omega(model, tiling, st: ClusterState, n_max: int, n_total_max: int | None = None) -> float:
    """<H(theta) - mu N> of the product of the cluster states `st.vectors` (the last iterate): the fixed cluster
    Hamiltonians in those states, plus the cross bonds and cross pairs with the states' own <b_i>, <n_i>.
    An upper bound on the cluster-Gutzwiller minimum at that twist; equal to `st.omega` at convergence."""
    s = len(tiling.clusters[0])
    fb = fock_basis(s, n_max, n_total_max)
    fixed, amp, cross, pairs = _cluster_pieces(model, tiling, st.theta, fb, st.mu)
    ns = model.n_sites
    b = np.zeros(ns, dtype=np.complex128)
    n = np.zeros(ns)
    E = 0.0
    for k, cl in enumerate(tiling.clusters):
        v = st.vectors[k]
        E += float(np.real(np.vdot(v, fixed[k] @ v)))
        pr = np.abs(v) ** 2
        for p in range(s):
            b[cl[p]] = np.vdot(v, fb.B[p] @ v)
            n[cl[p]] = pr @ fb.N[:, p]
    rows, cols = model.bonds.rows[cross], model.bonds.cols[cross]
    E += float(np.real(np.sum(amp[cross] * np.conj(b[rows]) * b[cols])))
    if pairs is not None:
        pi, pj, pv = pairs
        x = tiling.site_cluster[pi] != tiling.site_cluster[pj]
        E += float(np.sum(pv[x] * n[pi[x]] * n[pj[x]]))
    return E


def to_result(model, tiling, st: ClusterState, n_max: int) -> ob.ManyBodyResult:
    res = ob.from_state(model, st, method=f"cluster:{tiling.family}", n_max=n_max)
    res.rho_connected = st.rho_connected
    res.params["lambda"] = tiling.lam
    res.params["p_total_cap"] = st.p_total_cap
    return res


def cluster_fs(model, tiling, base: ClusterState, n_max: int, thetas=gw.THETA_FS, **kw) -> dict:
    """Twist f_s at fixed mu (the same definition as `gutzwiller.fs_from_omegas`), warm-started from base.phi;
    returns {"fs": {theta: f}, "fs0", "converged"}."""
    out, upper, conv, tconv = {}, {}, base.converged, True
    for th in thetas:
        st = cluster_gutzwiller(model, tiling, base.mu, n_max, theta=(th, 0.0), phi0=base.phi, **kw)
        out[th] = gw.fs_from_omegas(model, base.omega, st.omega, base.N, th)
        ov = cluster_variational_omega(model, tiling, st, n_max, kw.get("n_total_max"))
        upper[th] = gw.fs_from_omegas(model, base.omega, ov, base.N, th)
        conv = conv and st.converged
        tconv = tconv and st.converged
    fs0 = gw.extrapolate_fs0(out, thetas) if len(thetas) == 2 else out[thetas[0]]
    return {"fs": out, "fs_upper": upper, "fs0": fs0, "converged": conv, "twist_converged": tconv}


# -- the plain square lattice: translation-invariant clusters and the lobe tip --------------------------------------

SQUARE_FAMILIES = {
    "single": (((0, 0),), ((1, 0), (0, 1))),
    "2x2": (((0, 0), (1, 0), (0, 1), (1, 1)), ((2, 0), (0, 2))),
    "plus": (PLUS_OFFSETS, ((2, 1), (1, -2))),
    "3x3": (tuple((x, y) for y in range(3) for x in range(3)), ((3, 0), (0, 3))),
}


@dataclass
class SquareCluster:
    """One cluster of the infinite square lattice with nearest-neighbour hopping J, tiled by the translations
    `trans`. `internal` lists the bonds inside (i < j once); `M[k, j]` counts the outside neighbours of site k
    that are copies of cluster site j (the mean field Psi_k = -J sum_j M_kj phi_j of a uniform solution)."""

    family: str
    sites: tuple
    trans: tuple
    internal: list
    M: np.ndarray

    @property
    def s(self) -> int:
        return len(self.sites)

    @property
    def lam(self) -> float:
        """Fraction of the lattice bonds inside clusters: internal bonds / (2 s)."""
        return len(self.internal) / (2.0 * self.s)


def square_cluster(family: str) -> SquareCluster:
    sites, trans = SQUARE_FAMILIES[family]
    sites = tuple(tuple(p) for p in sites)
    pos = {p: k for k, p in enumerate(sites)}
    A = np.array(trans, dtype=np.float64).T  # columns: translation vectors
    if abs(abs(np.linalg.det(A)) - len(sites)) > 1e-9:
        raise RuntimeError(f"{family}: translations do not have index {len(sites)}")

    def home(p):
        for q, k in pos.items():
            z = np.linalg.solve(A, np.subtract(p, q))
            if np.all(np.abs(z - np.rint(z)) < 1e-9):
                return k
        raise RuntimeError(f"{family}: {p} is not covered")

    for p in itertools.product(range(-4, 5), repeat=2):  # tiling covers every site exactly once
        n_hits = sum(
            1 for q in sites if np.all(np.abs((z := np.linalg.solve(A, np.subtract(p, q))) - np.rint(z)) < 1e-9)
        )
        if n_hits != 1:
            raise RuntimeError(f"{family}: site {p} covered {n_hits} times")
    internal, M = [], np.zeros((len(sites), len(sites)))
    for k, p in enumerate(sites):
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + d[0], p[1] + d[1])
            if q in pos:
                if pos[q] > k:
                    internal.append((k, pos[q]))
            else:
                M[k, home(q)] += 1.0
    return SquareCluster(family, sites, tuple(trans), internal, M)


def _sector_basis(s: int, N: int, n_max: int):
    occ = [c for c in itertools.product(range(n_max + 1), repeat=s) if sum(c) == N] if N >= 0 else []
    occ = np.array(occ, dtype=np.int64).reshape(-1, s)
    index = {tuple(r): k for k, r in enumerate(occ)}
    return occ, index


def _sector_hamiltonian(cl: SquareCluster, occ, index, J: float, U: float):
    """H_N = -J sum_internal (b_i^dag b_j + h.c.) + (U/2) sum n (n - 1) (mu enters as -mu N, a constant)."""
    D = occ.shape[0]
    diag = 0.5 * U * (occ * (occ - 1)).sum(axis=1).astype(np.float64)
    r, c, v = [], [], []
    for a in range(D):
        for i, j in cl.internal:
            for x, y in ((i, j), (j, i)):  # b_x^dag b_y
                if occ[a, y] == 0:
                    continue
                t = occ[a].copy()
                amp = math.sqrt(t[y])
                t[y] -= 1
                amp *= math.sqrt(t[x] + 1)
                t[x] += 1
                bidx = index.get(tuple(t))
                if bidx is not None:
                    r.append(bidx)
                    c.append(a)
                    v.append(-J * amp)
    H = sp.csr_matrix((v, (r, c)), shape=(D, D)) + sp.diags(diag)
    return H.tocsr()


def _raising(occ_from, occ_to, index_to, site: int, n_max: int):
    """b_site^dag from sector N to N + 1 (sparse)."""
    r, c, v = [], [], []
    for a, row in enumerate(occ_from):
        if row[site] >= n_max:
            continue
        t = row.copy()
        t[site] += 1
        b = index_to.get(tuple(t))
        if b is not None:
            r.append(b)
            c.append(a)
            v.append(math.sqrt(t[site]))
    return sp.csr_matrix((v, (r, c)), shape=(occ_to.shape[0], occ_from.shape[0]))


class MottStability:
    """Linear stability of the n-filled Mott state of a square cluster (module docstring), cached per J."""

    def __init__(self, cl: SquareCluster, n_fill: int = 1, n_max: int = 3, U: float = 1.0):
        self.cl, self.n, self.n_max, self.U = cl, n_fill, n_max, U
        s = cl.s
        U = float(U)
        N0 = n_fill * s
        self.occ0, self.idx0 = _sector_basis(s, N0, n_max)
        self.occp, self.idxp = _sector_basis(s, N0 + 1, n_max)
        self.occm, self.idxm = _sector_basis(s, N0 - 1, n_max)
        self.Rp = [_raising(self.occ0, self.occp, self.idxp, i, n_max) for i in range(s)]  # N0 -> N0 + 1
        self.Rm = [_raising(self.occm, self.occ0, self.idx0, i, n_max) for i in range(s)]  # N0 - 1 -> N0
        # H_N(J) = J K_N + D_N: the hopping and interaction parts are built once
        self._KD = {}
        for key, (occ, idx) in {
            "0": (self.occ0, self.idx0),
            "p": (self.occp, self.idxp),
            "m": (self.occm, self.idxm),
        }.items():
            D = _sector_hamiltonian(cl, occ, idx, 0.0, U)
            K = _sector_hamiltonian(cl, occ, idx, 1.0, U) - D
            self._KD[key] = (K.tocsr(), D.tocsr())
        self._cache = {}
        self._cg_prev = {}

    def _H(self, key, J):
        K, D = self._KD[key]
        return (J * K + D).tocsr()

    def _parts(self, J):
        if J in self._cache:
            return self._cache[J]
        H0, Hp, Hm = self._H("0", J), self._H("p", J), self._H("m", J)
        if H0.shape[0] <= 400:
            w, V = np.linalg.eigh(H0.toarray())
            E0, g = float(w[0]), V[:, 0]
        else:
            w, V = spla.eigsh(H0, k=1, which="SA", tol=1e-14, v0=np.ones(H0.shape[0]))
            E0, g = float(w[0]), V[:, 0]
        if len(self._cache) > 64:
            self._cache.clear()
        up = [R @ g for R in self.Rp]  # b_k^dag |0>, in N0 + 1
        dn = [R.T @ g for R in self.Rm]  # b_i |0>, in N0 - 1
        top = float(np.sum(np.abs(g) ** 2 * (self.occ0 == self.n_max).any(axis=1)))
        parts = (Hp, Hm, E0, up, dn, top)
        self._cache[J] = parts
        return parts

    def chi(self, J: float, mu: float) -> np.ndarray:
        """chi_ik at hopping J and chemical potential mu (U as given). Raises if the Mott state is not the
        ground state in the N0 +- 1 sectors (a denominator is not positive)."""
        Hp, Hm, E0, up, dn, _ = self._parts(J)
        Ap = (Hp - sp.identity(Hp.shape[0]) * (E0 + mu)).tocsr()
        Am = (Hm - sp.identity(Hm.shape[0]) * (E0 - mu)).tocsr()

        def solve_all(A, rhs, key):
            # dense below 2000 states; above, conjugate gradients (A is positive definite inside the lobe; a
            # sparse LU of the 3 x 3 sectors fills in badly), warm-started from the previous solutions
            if A.shape[0] <= 2000:
                return np.linalg.solve(A.toarray(), np.stack(rhs, 1))
            prev = self._cg_prev.get(key)
            cols = []
            for q, r in enumerate(rhs):
                x0 = None if prev is None else prev[:, q]
                x, info = spla.cg(A, r, x0=x0, rtol=1e-13, atol=0.0, maxiter=20000)
                if info != 0:
                    raise ValueError("conjugate gradients did not converge (Mott state not stable?)")
                cols.append(x)
            X = np.stack(cols, 1)
            self._cg_prev[key] = X
            return X

        Xp = solve_all(Ap, up, "p")
        Xm = solve_all(Am, dn, "m")
        U_ = np.stack(up, 1)
        Dm = np.stack(dn, 1)
        chi = U_.T @ Xp + (Dm.T @ Xm).T  # chi_ik = <0|b_i Gp b_k^dag|0> + <0|b_k^dag Gm b_i|0>
        chi = 0.5 * (chi + chi.T)
        if np.any(np.linalg.eigvalsh(chi) <= 0):
            raise ValueError("Mott state not stable against particle / hole addition at this (J, mu)")
        return chi

    def lam_max(self, J: float, mu: float) -> float:
        return float(np.max(np.real(np.linalg.eigvals(J * self.chi(J, mu) @ self.cl.M))))

    def J_c(self, mu: float, J_hi: float = 0.08) -> float:
        """The J at which the largest eigenvalue of J chi M reaches 1 at this mu (brentq on log J)."""
        from scipy.optimize import brentq

        def f(lj):
            # beyond the cluster's own lobe edge the N0 state is not its ground state and chi diverged on the
            # way there, so lam_max crossed 1 before: count the invalid region as unstable
            try:
                return self.lam_max(math.exp(lj), mu) - 1.0
            except ValueError:
                return 1.0

        lo, hi = math.log(1e-4), math.log(J_hi)
        while f(hi) < 0:
            hi += 0.5
        return math.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12))

    def tip(self, mu_bracket=(0.15, 0.7)) -> dict:
        """The lobe tip: the maximum of J_c(mu) over mu / U (golden-section search). Returns {"mu", "J_c",
        "U_over_J", "p_nmax"}."""
        from scipy.optimize import minimize_scalar

        r = minimize_scalar(
            lambda m: -self.J_c(m * self.U), bounds=mu_bracket, method="bounded", options={"xatol": 1e-7}
        )
        Jc = -float(r.fun)
        return {"mu_over_U": float(r.x), "J_c": Jc, "U_over_J": self.U / Jc, "p_nmax": self._parts(Jc)[5]}


def square_cluster_solve(
    cl: SquareCluster, J: float, U: float, mu: float, n_max: int, tol=1e-12, max_iter=20000, phi0=0.5
):
    """The self-consistent uniform cluster mean field on the infinite square lattice (one cluster, its outside
    neighbours mapped onto its own sites by the tiling): returns (phi per cluster site, n per site,
    converged, iterations). Used to cross-check the linear-stability boundary."""
    fb = fock_basis(cl.s, n_max)
    Bd = [b.toarray() for b in fb.B]
    H0 = np.diag(sum(0.5 * U * fb.N[:, k] * (fb.N[:, k] - 1) - mu * fb.N[:, k] for k in range(cl.s)))
    for i, j in cl.internal:
        H0 += -J * (Bd[i].T @ Bd[j] + Bd[j].T @ Bd[i])
    phi = np.full(cl.s, phi0)
    beta = 0.5
    conv, it = False, 0
    for it in range(1, max_iter + 1):  # noqa: B007
        Psi = -J * cl.M @ phi
        H = H0 + sum(Psi[k] * (Bd[k].T + Bd[k]) for k in range(cl.s))
        w, V = np.linalg.eigh(H)
        v = V[:, 0]
        b = np.array([v @ Bd[k] @ v for k in range(cl.s)])
        if np.abs(b - phi).max() < tol:
            conv = True
            phi = b
            break
        phi = phi + beta * (b - phi)
    n = np.array([(v**2) @ fb.N[:, k] for k in range(cl.s)])
    return phi, n, conv, it


__all__ = [
    "ClusterState",
    "ClusterTiling",
    "FockBasis",
    "MottStability",
    "PLUS_OFFSETS",
    "SQUARE_FAMILIES",
    "SquareCluster",
    "cluster_fs",
    "cluster_gutzwiller",
    "cluster_variational_omega",
    "fock_basis",
    "make_tiling",
    "plus_chirality",
    "square_cluster",
    "square_cluster_solve",
    "to_result",
]
