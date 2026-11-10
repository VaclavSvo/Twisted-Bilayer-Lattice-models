"""The certified tridiagonal kernel of `manybody/fast_kernels.py` against dense `eigh`: random unreduced
tridiagonals from warm, cold and random starts; the complex gauge identity with the site matrix of `solve_batch`;
excited warm starts; near-degeneracy; diagonal sites and ties; the pivot guard."""

import math

import pytest
import torch

from twisted_bilayer.manybody import fast_kernels as fk

DT = torch.float64
LADDER = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def dense_T(a, t):
    S, d = a.shape
    H = torch.zeros(S, d, d, dtype=a.dtype)
    i = torch.arange(d)
    H[:, i, i] = a
    if d > 1:
        H[:, i[1:], i[:-1]] = t
        H[:, i[:-1], i[1:]] = t
    return H


def ref_lowest(a, t):
    w, V = torch.linalg.eigh(dense_T(a, t))
    v = V[:, :, 0]
    alt = (-1.0) ** torch.arange(a.shape[1], dtype=DT)
    v = v * torch.sign((v * alt).sum(dim=1, keepdim=True))
    return w, V, v


def vec_err(f, v):
    """distance between unit vectors up to sign"""
    s = torch.sign((f * v).sum(dim=1, keepdim=True))
    return (f - s * v).abs().amax(dim=1)


def gw_site(S, d, rng, U=1.0, psi_lo=1e-6, psi_hi=0.3):
    k = torch.arange(d, dtype=DT)
    w = torch.empty(S, dtype=DT).uniform_(-2.5, 0.5, generator=rng)
    D = 0.5 * U * k * (k - 1) + w[:, None] * U * k
    lp = torch.empty(S, dtype=DT).uniform_(math.log10(psi_lo), math.log10(psi_hi), generator=rng)
    return D, 10.0**lp


# 1. random unreduced tridiagonals, d = 2..17, scales 1e-3..1e3: warm (near-exact), cold and random starts
@pytest.mark.parametrize("d", list(range(2, 18)))
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_random_unreduced(d, scale):
    rng = torch.Generator().manual_seed(1000 * d + int(math.log10(scale) + 3))
    S = 300
    a = scale * torch.empty(S, d, dtype=DT).uniform_(-1, 1, generator=rng)
    t = scale * torch.empty(S, d - 1, dtype=DT).uniform_(0.05, 1.0, generator=rng)
    w, V, v = ref_lowest(a, t)
    sc = a.abs().amax(dim=1) + 2 * t.abs().amax(dim=1)
    # warm start from the ground vector of a slightly perturbed matrix (one SCF step earlier)
    ap = a * (1 + 1e-6 * torch.empty_like(a).uniform_(-1, 1, generator=rng))
    _, _, vp = ref_lowest(ap, t)
    for warm in (vp, None, torch.empty(S, d, dtype=DT).uniform_(-1, 1, generator=rng)):
        st = fk.new_stats()
        e0, f, reason = fk.lowest_tridiagonal(a, t, warm, None, st)
        assert torch.isfinite(e0).all() and torch.isfinite(f).all()
        assert ((e0 - w[:, 0]).abs() <= 1e-12 * sc).all()
        gap = w[:, 1] - w[:, 0] if d > 1 else torch.full((S,), float("inf"), dtype=DT)
        # vector accuracy: eigh itself is good to ~eps |T| / gap; the certified ones to 1e-13 + that
        tol = 1e-12 + 64 * 2.2e-16 * sc / gap
        assert (vec_err(f, v) <= tol).all()
        assert st["sites"] == S
        if warm is None:
            assert st["cold"] == S
    # the near-exact warm start is certified (the fast path is exercised). With the single gap probe
    # gamma = 1e-3 scale, the Davis-Kahan test needs r <= 1e-16 scale, at the rounding floor of the residual, so a
    # fraction of converged random sites falls back; the probe ladder certifies almost all (same 1e-13 bound).
    st = fk.new_stats()
    fk.lowest_tridiagonal(a, t, vp, None, st)
    assert st["rqi_certified"] >= 0.3 * S, st
    st = fk.new_stats()
    e0, f, _ = fk.lowest_tridiagonal(a, t, vp, None, st, LADDER)
    assert st["rqi_certified"] >= 0.9 * S, st
    assert ((e0 - w[:, 0]).abs() <= 1e-12 * sc).all()
    assert (vec_err(f, v) <= 1e-12 + 64 * 2.2e-16 * sc / (w[:, 1] - w[:, 0])).all()


# 2. the complex gauge identity: the fast path equals eigh of the complex Hermitian site matrix of solve_batch
def test_complex_gauge_identity():
    rng = torch.Generator().manual_seed(7)
    S, d = 2000, 9
    D, apsi = gw_site(S, d, rng)
    alpha = torch.empty(S, dtype=DT).uniform_(-math.pi, math.pi, generator=rng)
    Psi = apsi * torch.exp(1j * alpha)
    k = torch.arange(d, dtype=DT)
    sq = torch.sqrt(k[1:]).to(torch.complex128)
    H = torch.diag_embed(D.to(torch.complex128))
    il = torch.arange(1, d)
    H[:, il, il - 1] = Psi[:, None] * sq
    H[:, il - 1, il] = Psi.conj()[:, None] * sq
    w, V = torch.linalg.eigh(H)
    g0 = V[:, :, 0]
    b_ref = (g0[:, :-1].conj() * sq * g0[:, 1:]).sum(dim=1)
    p_ref = g0.abs() ** 2
    cold = fk.ground_state(D, Psi)  # all eigh (cold)
    st = fk.new_stats()
    out = fk.ground_state(D, Psi, warm=cold["f"], stats=st)  # warm: the certified path
    assert st["rqi_certified"] > 0.9 * S, st
    for o in (cold, out):
        assert (o["e0"] - w[:, 0]).abs().max() <= 1e-12 * D.abs().amax()
        assert (o["b"] - b_ref).abs().max() <= 1e-11
        assert (o["p"] - p_ref).abs().max() <= 1e-11
        Hg = (H @ o["g"][:, :, None])[:, :, 0]
        assert (Hg - o["e0"][:, None] * o["g"]).abs().max() <= 1e-11 * D.abs().amax()
    # real negative / positive Psi (theta = 0, alpha = 0 or pi)
    Psi_r = apsi * torch.where(torch.rand(S, generator=rng, dtype=DT) < 0.5, -1.0, 1.0)
    Hr = dense_T(D, (Psi_r[:, None] * torch.sqrt(k[1:])))
    wr, Vr = torch.linalg.eigh(Hr)
    gr = Vr[:, :, 0]
    br = (gr[:, :-1] * torch.sqrt(k[1:]) * gr[:, 1:]).sum(dim=1)
    o = fk.ground_state(D, Psi_r, warm=fk.ground_state(D, Psi_r)["f"])
    assert o["b"].dtype == DT and (o["b"] - br).abs().max() <= 1e-11


# 3. a warm start from the first excited vector must never return it
def test_excited_warm_start_rejected():
    rng = torch.Generator().manual_seed(3)
    for d in (3, 7, 9, 13, 21):
        S = 500
        D, apsi = gw_site(S, d, rng, psi_lo=1e-2)
        t = apsi[:, None] * torch.sqrt(torch.arange(1, d, dtype=DT))
        w, V, v = ref_lowest(D, t)
        exc = V[:, :, 1]
        st = fk.new_stats()
        e0, f, reason = fk.lowest_tridiagonal(D, t, exc, None, st)
        gap = w[:, 1] - w[:, 0]
        assert ((e0 - w[:, 0]).abs() < 0.5 * gap).all()  # never the excited eigenvalue
        assert (vec_err(f, v) <= 1e-12 + 64 * 2.2e-16 * D.abs().amax(dim=1) / gap).all()
        assert st["lowest"] > 0.9 * S, st  # caught by the 'lowest' Sturm test, then eigh


# 4. near-degeneracy (gap below gamma) must fall back
def test_near_degeneracy_falls_back():
    d, S = 7, 200
    k = torch.arange(d, dtype=DT)
    w_int = -1.0 + torch.linspace(-1e-10, 1e-10, S, dtype=DT)  # n = 1 and n = 2 degenerate at w = -1
    D = 0.5 * k * (k - 1) + w_int[:, None] * k
    Psi = torch.full((S,), 1e-9, dtype=DT)
    t = Psi[:, None] * torch.sqrt(k[1:])
    w, V, v = ref_lowest(D, t)
    assert ((w[:, 1] - w[:, 0]) < 1e-3 * D.abs().amax(dim=1)).all()
    st = fk.new_stats()
    e0, f, reason = fk.lowest_tridiagonal(D, t, v.clone(), None, st)  # even the exact vector is not accepted
    assert st["gap"] == S and st["rqi_certified"] == 0, st
    assert torch.equal(e0, w[:, 0])  # the gathered eigh gives the reference result
    assert (vec_err(f, v) <= 4e-16).all()  # the same eigh, renormalised (rounding only)


# 5. diagonal sites: unique minimum (analytic), tie (fallback to eigh)
def test_diagonal_sites():
    d = 7
    k = torch.arange(d, dtype=DT)
    w_vals = torch.tensor([-0.3, -1.4, -2.2, 0.2], dtype=DT)
    D = 0.5 * k * (k - 1) + w_vals[:, None] * k
    Psi = torch.zeros(4, dtype=DT)
    st = fk.new_stats()
    out = fk.ground_state(D, Psi, stats=st)
    assert st["diag"] == 4 and st["tie"] == 0 and st["cold"] == 0
    kmin = D.argmin(dim=1)
    assert torch.equal(out["e0"], D.min(dim=1).values)
    assert torch.equal(out["p"], torch.nn.functional.one_hot(kmin, d).to(DT))
    assert (out["b"] == 0).all()
    # ties: w = -1 makes n = 1 and n = 2 degenerate; w = 0 makes n = 0 and 1 degenerate
    Dt = 0.5 * k * (k - 1) + torch.tensor([-1.0, 0.0], dtype=DT)[:, None] * k
    st = fk.new_stats()
    out = fk.ground_state(Dt, torch.zeros(2, dtype=DT), stats=st)
    assert st["tie"] == 2, st
    w, V = torch.linalg.eigh(torch.diag_embed(Dt))
    assert torch.equal(out["e0"], w[:, 0])


# 6. the pivot guard: an exactly singular LDL^T (sigma equal to a diagonal entry at the first pivot)
def test_pivot_guard():
    d, S = 7, 50
    rng = torch.Generator().manual_seed(11)
    D, apsi = gw_site(S, d, rng, psi_lo=1e-3)
    t = apsi[:, None] * torch.sqrt(torch.arange(1, d, dtype=DT))
    # direct: sigma = a_0 makes the first pivot exactly 0
    v = torch.ones(d, S, dtype=DT)
    piv = 1e-14 * (D.abs().amax(dim=1) + 2 * t.amax(dim=1))
    y = fk.ldl_solve(D.T.contiguous(), t.T.contiguous(), D[:, 0].clone(), v, piv)
    assert torch.isfinite(y).all()
    # through the kernel: warm = e_0 gives sigma = a_0 exactly in the first RQI step
    warm = torch.zeros(S, d, dtype=DT)
    warm[:, 0] = 1.0
    st = fk.new_stats()
    e0, f, reason = fk.lowest_tridiagonal(D, t, warm, None, st)
    w, V, vr = ref_lowest(D, t)
    assert torch.isfinite(e0).all() and torch.isfinite(f).all() and st["nonfinite"] == 0
    gap = w[:, 1] - w[:, 0]
    assert ((e0 - w[:, 0]).abs() <= 1e-12 * D.abs().amax(dim=1)).all()
    assert (vec_err(f, vr) <= 1e-12 + 64 * 2.2e-16 * D.abs().amax(dim=1) / gap).all()
