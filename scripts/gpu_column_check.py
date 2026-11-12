"""Run one column (one V0, all Omega_R) of a Gutzwiller phase map on a device and compare it with a stored map.

    python scripts/gpu_column_check.py --v0 5 --reference <folder or maps.npz> [--row phase_diagram] [--device cuda]

The column of the full two-layer model at V0 = 5 puts 13 Omega_R x 4 starts x 970 sites = 50 440 matrices of
9 x 9 through one batched eigh per iteration, the size cuSOLVER used to reject (`gutzwiller.batched_eigh`). The
script runs the column, prints the labels and the largest differences to the stored column (same row
specification), and writes `column_check.json` into --out. Differences between a GPU and a CPU run are rounding
(1e-12 relative in the grand potential is typical); a label that differs is reported, not hidden.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from twisted_bilayer_lattice import gutzwiller as gw
from twisted_bilayer_lattice import phase_diagram as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v0", type=float, default=5.0)
    ap.add_argument("--reference", required=True, help="a maps.npz or the folder holding it")
    ap.add_argument("--row", default="phase_diagram")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="column_check")
    a = ap.parse_args()

    ref_path = a.reference if a.reference.endswith(".npz") else os.path.join(a.reference, "maps.npz")
    ref = pd._load_npz(ref_path)
    iv = int(np.argmin(np.abs(ref["V0"] - a.v0)))
    if abs(ref["V0"][iv] - a.v0) > 1e-9:
        raise SystemExit(f"V0 = {a.v0} is not in the reference map {ref['V0']}")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    info = {
        "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "EIGH_BATCH_MAX": gw.EIGH_BATCH_MAX,
    }
    print(info, flush=True)
    row = dict(pd.ROWS[a.row], V0=(a.v0,))
    t0 = time.time()
    s = pd.run_row("column_check", a.out, a.device, print, None, row=row)
    seconds = time.time() - t0
    new = pd._load_npz(os.path.join(a.out, "maps.npz"))
    lab_new, lab_ref = new["label"][0], ref["label"][iv]
    same = lab_new == lab_ref
    out = {
        "info": info,
        "seconds": seconds,
        "points_done": s["points_done"],
        "labels_new": [int(x) for x in lab_new],
        "labels_reference": [int(x) for x in lab_ref],
        "labels_equal": bool(same.all()),
        "differing_Omega_R": [float(x) for x in new["Omega_R"][~same]],
    }
    for key in ("grand_potential", "fs", "filling", "p_nmax"):
        d = np.abs(new[key][0] - ref[key][iv])
        out[f"max_abs_diff_{key}"] = float(np.nanmax(d))
        out[f"max_rel_diff_{key}"] = float(np.nanmax(d / np.maximum(np.abs(ref[key][iv]), 1e-300)))
    with open(os.path.join(a.out, "column_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if out["labels_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
