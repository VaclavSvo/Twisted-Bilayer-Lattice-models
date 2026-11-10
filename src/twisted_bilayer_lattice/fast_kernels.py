"""A certified, warm-started kernel for the single-site Gutzwiller site problem (torch, float64).

`gutzwiller.solve_batch(..., eig="fast")` uses it in place of the batched dense `eigh`. The site matrix of
`solve_batch` is H_kk = D_k (= U k(k-1)/2 + w k), H_{k,k-1} = sqrt(k) Psi, H_{k-1,k} = sqrt(k) Psi^*,
k = 0..d-1: every term moves one atom or none, so H is tridiagonal. The gauge change Psi = |Psi| e^{i alpha},
g_k = e^{i k alpha} f_k turns it into the real symmetric tridiagonal T with diagonal a = D and off-diagonal
t_k = sqrt(k) |Psi| >= 0; then e0 = lambda_0(T), p_k = f_k^2, b = e^{i alpha} sum_k sqrt(k) f_{k-1} f_k.
Only the lowest eigenpair is needed: at fixed fields the grand potential restricted to one site is the Rayleigh
quotient of H, minimised by its ground vector.

`ground_state(D, Psi, warm=None, warm_valid=None)`, D (S, d) real, Psi (S,) real or complex, returns a dict with
e0 (S,), p (S, d), b (S,) (dtype of Psi), f (S, d) real (the gauge-reduced vector, the next warm start), g (S, d)
(original gauge, dtype of Psi) and `stats` (per-reason counters). Per site:

  1. Psi == 0 (diagonal): the ground state is the unit vector at argmin a_k, exactly; a tie (second-smallest a
     within eta = 1e-12 scale) falls back to eigh ('tie').
  2. no warm vector: eigh ('cold').
  3. otherwise at most MAX_RQI = 3 Rayleigh-quotient steps, sigma = f^T T f, (T - sigma) y = f by the LDL^T
     recurrence with the pivot guard |q| < eps_p = 1e-14 scale -> +-eps_p; after every step the certificate:
       r = |T f - sigma f|;  Sturm count at sigma - r - eta == 0 and at sigma + r + eta >= 1 ('lowest');
       Sturm count at sigma + gamma == 1, gamma = 1e-3 scale ('gap');  Davis-Kahan r / (gamma - r) <= 1e-13 ('dk');
     scale = max|a| + 2 max t (Gershgorin). Certified sites stop; the others take another step.
  4. sites still uncertified after MAX_RQI steps (reason = the first failing test of the last step: 'nonfinite',
     'lowest', 'gap', 'dk'), and every 'cold' / 'tie' site, go to ONE gathered batched eigh of their dense T.

The certificate is strict: a site it does not accept is solved by the same dense `eigh` as the default path, so
the output is the lowest eigenpair either way; near-degenerate sites (gap below gamma) always fall back. The
global phase of g differs from that of `eigh` (b, p, e0 and the grand potential do not depend on it).

Layout inside the kernel is (d, S), so each recurrence level is a contiguous vector over the sites. No compiled
code and no new dependencies; the same code runs on CPU and CUDA (data-dependent gathers sync on CUDA once per
call; not yet measured on a GPU).

References: B. N. Parlett, The Symmetric Eigenvalue Problem (SIAM, 1998) (Rayleigh-quotient iteration, Sturm
sequences); W. Barth, R. S. Martin, J. H. Wilkinson, Numer. Math. 9, 386 (1967) (Sturm-sequence bisection);
C. Davis, W. M. Kahan, SIAM J. Numer. Anal. 7, 1 (1970) (the sin-theta bound).
"""

from __future__ import annotations

import torch

GAMMA_REL = 1e-3
GAP_PROBES = (GAMMA_REL,)  # one probe at 1e-3 scale; a ladder such as (1e-3, 1e-2, 1e-1) proves a larger gap
ETA_REL = 1e-12
PIV_REL = 1e-14
DK_TOL = 1e-13
MAX_RQI = 3
REASONS = ("cold", "tie", "lowest", "gap", "dk", "nonfinite")
_TINY = 2.2250738585072014e-308  # smallest normal float64 (LAPACK safmin), for the Sturm-count pivmin


def new_stats() -> dict:
    s = {r: 0 for r in REASONS}
    s.update(sites=0, diag=0, rqi_certified=0, rqi_steps=[0] * MAX_RQI, calls=0)
    return s


def merge_stats(acc: dict, s: dict) -> dict:
    for k, v in s.items():
        if k == "rqi_steps":
            acc[k] = [x + y for x, y in zip(acc[k], v)]
        else:
            acc[k] += v
    return acc


# -- tridiagonal primitives, layout (d, S) ---------------------------------------------------------------------------


def sturm_count(a: torch.Tensor, t2: torch.Tensor, sigma: torch.Tensor, pivmin: torch.Tensor) -> torch.Tensor:
    """Number of eigenvalues < sigma for each column of the tridiagonal (a, t), t2 = t^2 (Sylvester inertia of
    T - sigma by the LDL^T pivots; a tiny pivot is replaced by -pivmin, as LAPACK dstebz)."""
    q = a[0] - sigma
    q = torch.where(q.abs() < pivmin, -pivmin, q)
    cnt = (q < 0).to(torch.int32)
    for k in range(1, a.shape[0]):
        q = a[k] - sigma - t2[k - 1] / q
        q = torch.where(q.abs() < pivmin, -pivmin, q)
        cnt = cnt + (q < 0).to(torch.int32)
    return cnt


def sturm_counts(a, t2, sigmas, pivmin):
    """sturm_count for P shifts at once: sigmas (P, S) -> counts (P, S); one pass over the d levels."""
    q = a[0][None, :] - sigmas
    q = torch.where(q.abs() < pivmin, -pivmin, q)
    cnt = (q < 0).to(torch.int32)
    for k in range(1, a.shape[0]):
        q = a[k][None, :] - sigmas - t2[k - 1][None, :] / q
        q = torch.where(q.abs() < pivmin, -pivmin, q)
        cnt = cnt + (q < 0).to(torch.int32)
    return cnt


def ldl_solve(a, t, sigma, v, piv):
    """y = (T - sigma)^{-1} v by LDL^T without pivoting; |pivot| < piv is replaced by +-piv (sign kept, 0 -> +piv)."""
    d = a.shape[0]
    D = torch.empty_like(a)
    L = torch.empty_like(t)

    def guard(q):
        return torch.where(q.abs() < piv, torch.where(q < 0, -piv, piv), q)

    D[0] = guard(a[0] - sigma)
    for k in range(d - 1):
        L[k] = t[k] / D[k]
        D[k + 1] = guard(a[k + 1] - sigma - L[k] * t[k])
    y = torch.empty_like(v)
    y[0] = v[0]
    for k in range(1, d):
        y[k] = v[k] - L[k - 1] * y[k - 1]
    w = torch.empty_like(v)
    w[d - 1] = y[d - 1] / D[d - 1]
    for k in range(d - 2, -1, -1):
        w[k] = y[k] / D[k] - L[k] * w[k + 1]
    return w


def matvec(a, t, v):
    w = a * v
    w[:-1] += t * v[1:]
    w[1:] += t * v[:-1]
    return w


def _normalise(v, alt):
    """Unit columns with a fixed sign: sum_k (-1)^k v_k > 0 (the ground vector of T with t >= 0 alternates)."""
    v = v / torch.sqrt((v * v).sum(dim=0))
    s = torch.sign((alt * v).sum(dim=0))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return v * s


def lowest_tridiagonal(a, t, warm=None, warm_valid=None, stats=None, gap_probes=None):
    """Certified lowest eigenpair of the real symmetric tridiagonals with diagonal a (S, d) and off-diagonal
    t (S, d-1) >= 0. Returns (e0 (S,), f (S, d), reason (list of fallback reasons per index, for tests)) and
    updates `stats`. warm: (S, d) start vectors or None; warm_valid: (S,) bool. gap_probes: relative gap probes
    gamma_j / scale (default GAP_PROBES = (1e-3,)); the Davis-Kahan bound uses the largest gamma_j whose
    Sturm count is exactly 1 (each extra probe is one more O(d) count; the 1e-13 acceptance is unchanged)."""
    probes = GAP_PROBES if gap_probes is None else tuple(gap_probes)
    stats = new_stats() if stats is None else stats
    S, d = a.shape
    dev, dt = a.device, a.dtype
    stats["calls"] += 1
    stats["sites"] += S
    aT = a.T.contiguous()  # (d, S)
    tT = t.T.contiguous()  # (d-1, S)
    t2 = tT * tT
    tmax = tT.abs().amax(dim=0) if d > 1 else torch.zeros(S, dtype=dt, device=dev)
    scale = aT.abs().amax(dim=0) + 2.0 * tmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    eta, piv = ETA_REL * scale, PIV_REL * scale
    pivmin = _TINY * torch.clamp(tmax * tmax, min=1.0)
    alt = ((-1.0) ** torch.arange(d, device=dev, dtype=dt))[:, None]

    e0 = torch.empty(S, dtype=dt, device=dev)
    F = torch.zeros(d, S, dtype=dt, device=dev)
    reason = torch.full((S,), -1, dtype=torch.int64, device=dev)  # -1: done by the fast path
    code = {r: i for i, r in enumerate(REASONS)}

    # 1. diagonal sites
    diag = tmax == 0
    if bool(diag.any()):
        idx = torch.nonzero(diag).squeeze(1)
        ad = aT[:, idx]
        srt, order = torch.sort(ad, dim=0)
        kmin = order[0]
        tie = (srt[1] - srt[0] <= eta[idx]) if d > 1 else torch.zeros_like(kmin, dtype=torch.bool)
        ok = idx[~tie]
        F[:, ok] = 0.0
        F[kmin[~tie], ok] = 1.0
        F[:, ok] = F[:, ok] * torch.sign(alt[:, 0][kmin[~tie]])  # sign convention of _normalise
        e0[ok] = srt[0][~tie]
        reason[idx[tie]] = code["tie"]
        stats["diag"] += int((~tie).sum())
        stats["tie"] += int(tie.sum())

    # 2. cold sites
    has_warm = torch.zeros(S, dtype=torch.bool, device=dev)
    if warm is not None:
        has_warm = torch.ones(S, dtype=torch.bool, device=dev) if warm_valid is None else warm_valid.to(dev)
        has_warm = has_warm & torch.isfinite(warm).all(dim=1) & (warm.abs().sum(dim=1) > 0)
    cold = (~diag) & (~has_warm)
    reason[cold] = code["cold"]
    stats["cold"] += int(cold.sum())

    # 3. warm RQI with the certificate after every step
    act = torch.nonzero((~diag) & has_warm).squeeze(1)
    if act.numel():
        v = _normalise(warm[act].T.to(dt), alt)
        last_reason = torch.zeros(act.numel(), dtype=torch.int64, device=dev)
        full = act.numel() == S
        for step in range(MAX_RQI):
            if full:
                a_, t_, t2_, sc_, e_, pm, pv = aT, tT, t2, scale, eta, pivmin, piv
            else:
                a_, t_, t2_ = aT[:, act], tT[:, act], t2[:, act]
                sc_, e_, pm, pv = scale[act], eta[act], pivmin[act], piv[act]
            sig = (v * matvec(a_, t_, v)).sum(dim=0)
            v = _normalise(ldl_solve(a_, t_, sig, v, pv), alt)
            Tv = matvec(a_, t_, v)
            sig = (v * Tv).sum(dim=0)
            res = Tv - sig * v
            r = torch.sqrt((res * res).sum(dim=0))
            fin = torch.isfinite(v).all(dim=0) & torch.isfinite(sig) & torch.isfinite(r)
            shifts = torch.stack([sig - r - e_, sig + r + e_] + [sig + pr * sc_ for pr in probes])
            cnt = sturm_counts(a_, t2_, shifts, pm[None, :])
            lo_ok = (cnt[0] == 0) & (cnt[1] >= 1)
            g_eff = torch.zeros_like(sig)
            for j, pr in enumerate(probes):
                gp = pr * sc_
                g_eff = torch.where((cnt[2 + j] == 1) & (gp > g_eff), gp, g_eff)
            gap_ok = g_eff > 0
            dk_ok = (g_eff > r) & (r / torch.clamp(g_eff - r, min=1e-300) <= DK_TOL)
            good = fin & lo_ok & gap_ok & dk_ok
            last_reason = torch.where(
                ~fin,
                code["nonfinite"],
                torch.where(~lo_ok, code["lowest"], torch.where(~gap_ok, code["gap"], code["dk"])),
            )
            if bool(good.any()):
                gi = act[good]
                F[:, gi] = v[:, good]
                e0[gi] = sig[good]
                stats["rqi_steps"][step] += int(good.sum())
                stats["rqi_certified"] += int(good.sum())
            keep = ~good
            act, v, last_reason = act[keep], v[:, keep], last_reason[keep]
            full = False
            if act.numel() == 0:
                break
        if act.numel():
            reason[act] = last_reason
            for nm in ("nonfinite", "lowest", "gap", "dk"):
                stats[nm] += int((last_reason == code[nm]).sum())

    # 4. one gathered eigh for every remaining site
    fb = torch.nonzero(reason >= 0).squeeze(1)
    if fb.numel():
        m = fb.numel()
        H = torch.zeros(m, d, d, dtype=dt, device=dev)
        ii = torch.arange(d, device=dev)
        H[:, ii, ii] = a[fb]
        if d > 1:
            H[:, ii[1:], ii[:-1]] = t[fb]
            H[:, ii[:-1], ii[1:]] = t[fb]
        w, V = torch.linalg.eigh(H)
        e0[fb] = w[:, 0]
        F[:, fb] = _normalise(V[:, :, 0].T, alt)
    return e0, F.T.contiguous(), reason


def ground_state(D, Psi, warm=None, warm_valid=None, stats=None, gap_probes=None) -> dict:
    """The site ground states of `solve_batch` (see the module docstring). D (S, d) real; Psi (S,) real/complex."""
    stats = new_stats() if stats is None else stats
    S, d = D.shape
    dev = D.device
    k = torch.arange(d, device=dev, dtype=D.dtype)
    sq = torch.sqrt(k[1:])
    absPsi = Psi.abs().to(D.dtype)
    t = absPsi[:, None] * sq[None, :]
    e0, f, _ = lowest_tridiagonal(D, t, warm, warm_valid, stats, gap_probes)
    nz = absPsi > 0
    if torch.is_complex(Psi):
        ph = torch.where(nz, Psi / torch.where(nz, absPsi, torch.ones_like(absPsi)).to(Psi.dtype), torch.ones_like(Psi))
        g = f.to(Psi.dtype) * ph[:, None] ** k.to(Psi.dtype)[None, :]
    else:
        ph = torch.where(Psi < 0, -torch.ones_like(Psi), torch.ones_like(Psi))
        g = f * ph[:, None] ** k[None, :]
    b = ph * (sq[None, :] * f[:, :-1] * f[:, 1:]).sum(dim=1).to(ph.dtype)
    return dict(e0=e0, p=f * f, b=b, f=f, g=g, stats=stats)


__all__ = [
    "ground_state",
    "lowest_tridiagonal",
    "sturm_count",
    "sturm_counts",
    "ldl_solve",
    "new_stats",
    "merge_stats",
    "REASONS",
]
