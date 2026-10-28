"""The chunked batched eigh used by the Gutzwiller site map: same result as one call, and the cuSOLVER fallback."""

import numpy as np
import pytest
import torch

from twisted_bilayer.manybody import gutzwiller as gw


def _batch(shape, d, dtype=torch.float64, seed=3):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*shape, d, d, dtype=dtype, generator=g)
    return 0.5 * (A + A.transpose(-1, -2).conj())


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("chunk", [1, 7, 40, 10_000])
def test_chunked_reproduces_the_eigensystem(dtype, chunk):
    """A batch that fits in one piece is the same call, bitwise. A split batch is a different LAPACK call per
    piece, and the last bits of a result can depend on where each matrix sits in the buffer (float64 on a Windows
    CPU differs at chunk 1 and 7), so there it is the eigensystem that must agree, to rounding."""
    H = _batch((5, 8), 9, dtype)  # (members, sites, d, d) as in the site map
    e0, V0 = torch.linalg.eigh(H)
    e1, V1 = gw.batched_eigh(H, chunk=chunk)
    assert e1.shape == e0.shape and V1.shape == V0.shape
    if chunk >= 40:
        assert torch.equal(e1, e0) and torch.equal(V1, V0)
    scale = e0.abs().max().item()
    eye = torch.eye(9, dtype=dtype)
    assert (e1 - e0).abs().max().item() <= 1e-12 * scale
    assert (H @ V1 - V1 * e1[..., None, :]).abs().max().item() <= 1e-12 * scale  # eigen-equation
    assert (V1.conj().transpose(-1, -2) @ V1 - eye).abs().max().item() <= 1e-12  # orthonormal columns
    overlap = (V0.conj() * V1).sum(-2).abs()  # same eigenvector up to a phase (random spectra: non-degenerate)
    assert (overlap - 1).abs().max().item() <= 1e-9


def test_default_is_no_cut_on_the_cpu():
    H = _batch((3, 5), 4)
    e0, V0 = torch.linalg.eigh(H)
    e1, V1 = gw.batched_eigh(H)
    assert torch.equal(e1, e0) and torch.equal(V1, V0)


@pytest.mark.parametrize(
    "message",
    [
        "cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling cusolverDnXsyevBatched",
        "CUDA out of memory. Tried to allocate 15.83 GiB.",  # how a consumer GPU reports a too large batch
    ],
)
def test_halves_the_chunk_when_cusolver_rejects_a_batch(monkeypatch, message):
    """A rejected batch above 100 matrices (either message above) is retried in smaller pieces; the answer is
    unchanged."""
    real = torch.linalg.eigh
    sizes = []

    def fake(x):
        sizes.append(x.shape[0])
        if x.shape[0] > 100:
            raise RuntimeError(message)
        return real(x)

    H = _batch((4, 100), 9)  # 400 matrices
    monkeypatch.setattr(torch.linalg, "eigh", fake)
    e1, V1 = gw.batched_eigh(H, chunk=1024)
    monkeypatch.undo()
    e0, V0 = torch.linalg.eigh(H)
    assert torch.equal(e1, e0) and torch.equal(V1, V0)
    assert sizes[0] == 400 and sizes[-1] <= 100  # the first try was the whole batch, the last pieces fit


def test_other_errors_are_not_swallowed(monkeypatch):
    def fake(x):
        raise RuntimeError("the solver did not converge")

    monkeypatch.setattr(torch.linalg, "eigh", fake)
    with pytest.raises(RuntimeError, match="did not converge"):
        gw.batched_eigh(_batch((2, 3), 4), chunk=8)


def test_batch_error_below_the_minimum_chunk_is_raised(monkeypatch):
    def fake(x):
        raise RuntimeError("cusolver says no")

    monkeypatch.setattr(torch.linalg, "eigh", fake)
    with pytest.raises(RuntimeError, match="cusolver"):
        gw.batched_eigh(_batch((3, 50), 4), chunk=256)


def test_solve_batch_with_forced_chunking_is_bitwise(monkeypatch):
    """The whole batched solver with a tiny chunk gives the same state as the default (the constant is only a
    size limit for CUDA)."""
    from twisted_bilayer.engine import lattice_model as lm

    model = lm.build_model(10.0, 0.3, 4, 1, nu_z_hz=1000.0, cutoff=2.0, interband=True, interband_bonds=False)
    ref = gw.solve_batch([model], [0.9 * model.U], 4, tol=1e-9)[0]
    real = gw.batched_eigh
    monkeypatch.setattr(gw, "batched_eigh", lambda H, chunk=None: real(H, chunk=5))
    got = gw.solve_batch([model], [0.9 * model.U], 4, tol=1e-9)[0]
    assert np.array_equal(got.phi, ref.phi) and got.omega == ref.omega and got.iterations == ref.iterations
