"""Bisect the largest batch of small symmetric matrices that torch.linalg.eigh accepts on the CUDA device.

cuSOLVER's batched eigensolver rejected 46 560 matrices of 9 x 9 (a consumer GPU, torch 2.11.0+cu128), which is what
`gutzwiller.EIGH_BATCH_MAX` guards against. This finds the limit per matrix size, so the constant can be
set below it. Standalone: needs only torch.

    python scripts/bisect_eigh_batch.py
    python scripts/bisect_eigh_batch.py --sizes 7 9 11 15 --upper 200000

Writes bisect_eigh_batch.json into the working directory and prints a summary. Takes about a minute.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def try_batch(n_batch, dim, dtype=torch.float64):
    """Return (ok, message). ok means eigh ran and the eigen-decomposition has a small residual."""
    try:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(12345)
        a = torch.randn(n_batch, dim, dim, dtype=dtype, device="cuda", generator=gen)
        a = 0.5 * (a + a.transpose(-1, -2))
        w, v = torch.linalg.eigh(a)
        torch.cuda.synchronize()
        # residual on a spread of matrices (first, middle, last) - the whole batch would cost memory
        idx = torch.tensor(sorted({0, n_batch // 2, n_batch - 1}), device="cuda")
        asel, wsel, vsel = a[idx], w[idx], v[idx]
        res = (asel @ vsel - vsel * wsel.unsqueeze(-2)).abs().max().item()
        if not torch.isfinite(w).all().item():
            return False, "non-finite eigenvalues"
        if res > 1e-9:
            return False, f"residual {res:.2e}"
        return True, f"ok (residual {res:.1e})"
    except Exception as exc:  # cuSOLVER errors surface as RuntimeError; keep the message
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def bisect(dim, upper, lower=1):
    """Largest passing batch in [lower, upper]; assumes lower passes. Also tests upper first."""
    ok_hi, msg_hi = try_batch(upper, dim)
    if ok_hi:
        return upper, upper, f"upper bound {upper} passes - no limit found below it ({msg_hi})"
    lo, hi = lower, upper  # lo passes, hi fails
    ok_lo, msg_lo = try_batch(lo, dim)
    if not ok_lo:
        return None, hi, f"even batch {lo} fails: {msg_lo}"
    last_fail = msg_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, msg = try_batch(mid, dim)
        if ok:
            lo = mid
        else:
            hi = mid
            last_fail = msg
    return lo, hi, last_fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[7, 9, 11, 15], help="matrix dimensions (n_max + 1)")
    p.add_argument("--upper", type=int, default=120000, help="largest batch to try")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: run this on a CUDA machine")
    info = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print(info)
    # Warm-up so the first call's library load is not counted as a failure.
    try_batch(8, 9)

    results = {}
    for dim in args.sizes:
        t0 = time.time()
        largest_ok, first_fail, msg = bisect(dim, args.upper)
        results[dim] = {"largest_passing": largest_ok, "smallest_failing": first_fail, "failure": msg}
        print(
            f"dim {dim:3d}: largest passing batch {largest_ok}, smallest failing {first_fail} "
            f"[{time.time() - t0:.1f} s]\n          {msg}"
        )

    # Check the failure is a clean threshold, not an erratic one: retest a few values below it.
    for dim, r in results.items():
        if r["largest_passing"]:
            lp = r["largest_passing"]
            r["recheck_below"] = {str(b): try_batch(b, dim)[0] for b in (lp // 2, int(lp * 0.9), lp)}
    out = Path("bisect_eigh_batch.json")
    out.write_text(json.dumps({"info": info, "results": results}, indent=2))
    print(f"\nwritten {out}")
    print(
        "Suggestion: chunk = largest passing batch of the LARGEST dimension you use, rounded down to a "
        "power of two with a margin of at least 2x (e.g. 16384 if the limit is 46559)."
    )


if __name__ == "__main__":
    main()
