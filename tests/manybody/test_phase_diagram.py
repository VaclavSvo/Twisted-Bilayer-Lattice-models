"""The phase-map driver (manybody/phase_diagram.py) and the S4c gate (the four panels of Meng et al.'s Fig. S4c)."""

import json
import os

import numpy as np
import pytest

from twisted_bilayer.engine import lattice_model as lm
from twisted_bilayer.manybody import observables as ob
from twisted_bilayer.manybody import phase_diagram as pd


def _tiny_row():
    return dict(pd.ROWS["phase_diagram_paper_quick"], V0=(12.0, 20.0), Omega_R=(0.0, 0.6), compressibility=False)


def test_driver_writes_and_resumes(tmp_path):
    row = _tiny_row()
    out = str(tmp_path / "row")
    s = pd.run_row("tiny", out, row=row, log=lambda *_: None)
    for f in ("maps.npz", "summary.json", "report.txt", "provenance.json", "labels.png"):
        assert os.path.exists(os.path.join(out, f))
    assert s["complete"] and s["points_done"] == 4
    m = pd._load_npz(os.path.join(out, "maps.npz"))
    assert m["n"].shape == (2, 2, 1, 17) and m["phi"].dtype == np.complex128
    assert set(np.unique(m["label"])) <= set(range(len(ob.LABELS)))
    # resuming a complete row computes nothing new and keeps the numbers
    before = m["fs"].copy()
    s2 = pd.run_row("tiny", out, row=row, log=lambda *_: None)
    assert s2["points_done"] == 4
    np.testing.assert_array_equal(pd._load_npz(os.path.join(out, "maps.npz"))["fs"], before)
    json.loads(open(os.path.join(out, "summary.json")).read())


def test_task_order_coarse_first():
    """The full 21 x 13 grid starts with its 11 x 7 even subgrid, then visits every other point once."""
    row = pd.ROWS["phase_diagram"]
    order = pd.task_order(row)
    first = order[:11]
    assert all(iv % 2 == 0 and ios == list(range(0, 13, 2)) for iv, ios in first)
    seen = np.zeros((21, 13), dtype=int)
    for iv, ios in order:
        seen[iv, ios] += 1
    assert np.all(seen == 1)


def test_mu_reference():
    """mu_ref "aa": mu = (mu/U) U - Omega_R F_k0_AA (the zero of the paper's M_i under rule k0); "onsite": (mu/U) U."""
    model = lm.build_model(12.0, 0.6, 4, 1, cutoff=2.0)
    row = dict(pd.ROWS["phase_diagram_quick"])
    assert pd.mu_of(row, model) == pytest.approx(model.U - 0.6 * model.diagnostics["F_k0_AA"], rel=1e-14)
    row["mu_ref"] = "onsite"
    assert pd.mu_of(row, model) == pytest.approx(model.U, rel=1e-14)


def test_boundary_table():
    row = dict(_tiny_row(), V0=(4.0, 6.0, 8.0), Omega_R=(0.0,))
    maps = {"done": np.ones((3, 1), bool), "label": np.array([[0], [0], [3]])}
    b = pd.boundary_table(row, maps)[0]
    assert b["V_c_SF"] == 7.0 and b["first_MI"] == 8.0 and b["first_SF-II"] is None


S4C_POINTS = ((10.0, 0.6), (15.0, 0.6), (23.0, 0.3), (23.0, 1.1))
# The S4c panels. The paper's Fig. S4a puts MI at large V and small Omega_R and I at large V and large Omega_R, so
# (23, 0.3) is MI and (23, 1.1) is I.
S4C_EXPECTED = ("SF", "SF-II", "MI", "I")


def _s4c_labels(mu_over_U, rule="k0", excluded=False):
    row = dict(pd.ROWS["phase_diagram_paper"], mu_over_U=mu_over_U, M_R_rule=rule)
    cell = lm.supercell_sites(22, 1)
    labels = []
    for V0, Om in S4C_POINTS:
        model = pd.build(row, V0, Om, cell=cell)
        model.diagnostics["Omega_R_row"] = Om
        res = pd.solve_points(row, [model])[0]
        labels.append(res.label_info["label_excl_degenerate"] if excluded else res.label)
    return tuple(labels)


@pytest.mark.slow
def test_s4c_paper_model_mu095():
    """The paper's model (rule k0) at mu / U = 0.95: the four S4c panels come out SF, SF-II, MI, I. This is the
    reading that reproduces S4c (m = 22, single-site Gutzwiller, n_max 6)."""
    assert _s4c_labels(0.95) == S4C_EXPECTED


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "mu / U = 1 is the lobe-touching point: the AA sites (M_i ~ 0) stay degenerate between |1> and |2> "
        "(n = 1.5, |phi| = 0.71) and polarise their neighbours (|phi| = 0.025-0.06 > phi_c = 1e-4), so (23, 0.3) "
        "and (23, 1.1) come out SF-II instead of MI and I (labels SF, SF-II, SF-II, SF-II on m = 22)"
    ),
)
def test_s4c_paper_model_mu1():
    assert _s4c_labels(1.0) == S4C_EXPECTED


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "excluding the AA-degenerate sites (|mu_i/U - k| < 0.05) does not remove the induced coherence of their "
        "neighbours: max |phi| over the remaining sites is 0.025 at (23, 0.3) and 0.057 at (23, 1.1), above "
        "phi_c, so the labels stay SF-II"
    ),
)
def test_s4c_paper_model_mu1_aa_excluded():
    assert _s4c_labels(1.0, excluded=True) == S4C_EXPECTED


def test_resume_refuses_another_spec(tmp_path):
    """A map made with one row spec is not resumed with another (unless forced); a map written before the spec
    records were kept is resumed; provenance.json keeps one entry per run."""
    row = dict(_tiny_row(), V0=(20.0,), Omega_R=(0.6,))
    out = str(tmp_path / "row")
    pd.run_row("tiny", out, row=row, log=lambda *_: None)
    other = dict(row, tol=1e-9)
    with pytest.raises(RuntimeError, match="spec"):
        pd.run_row("tiny", out, row=other, log=lambda *_: None)
    pd.run_row("tiny", out, row=other, log=lambda *_: None, force_resume=True)
    prov = json.loads(open(os.path.join(out, "provenance.json")).read())
    assert len(prov["runs"]) == 2 and prov["spec_hash"] == pd.spec_hash(other)  # the refused run left no entry
    # a legacy map (no spec_hash): resumed with a message
    m = pd._load_npz(os.path.join(out, "maps.npz"))
    m.pop("spec_hash")
    m.pop("solver_version")
    pd._save_npz(os.path.join(out, "maps.npz"), m)
    msgs = []
    pd.run_row("tiny", out, row=row, log=msgs.append)
    assert any("not checked" in x for x in msgs)
    assert str(pd._load_npz(os.path.join(out, "maps.npz"))["spec_hash"]) == pd.spec_hash(row)


def test_spec_hash_ignores_none_entries():
    row = dict(pd.ROWS["phase_diagram_quick"])
    assert pd.spec_hash(row) == pd.spec_hash(dict(row, some_new_option=None))
    assert pd.spec_hash(row) != pd.spec_hash(dict(row, n_max=7))


def test_n_max_table():
    row = dict(pd.ROWS["phase_diagram"], n_max_table=((6.0, 14), (10.0, 12), (14.0, 8)))
    assert [pd.n_max_for(row, v) for v in (4.0, 6.0, 8.0, 10.0, 12.0, 20.0)] == [14, 14, 12, 12, 8, 6]
    assert pd.n_max_for(pd.ROWS["phase_diagram"], 4.0) == 8 and pd.n_max_for(pd.ROWS["phase_diagram"], 12.0) == 6


def test_u_correction_table(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"4": 0.99, "20": 0.97}))
    assert pd.u_correction(12.0, str(path)) == pytest.approx(0.98)
    assert pd.u_correction(24.0, {4.0: 0.99, 20.0: 0.97}) == pytest.approx(0.97)
    with pytest.raises(RuntimeError, match="u-table"):
        pd.u_correction(12.0, None)


def test_runner_kind_keeps_both_provenance_files(tmp_path):
    """`twisted-bilayer run` with a phase-map row: the runner's provenance.json does not overwrite the map's own
    run history (phase_diagram_provenance.json)."""
    from twisted_bilayer import runners

    verbose, out_root = runners.VERBOSE, runners._OUT_ROOT
    runners.VERBOSE = False
    row = dict(
        kind="phase_diagram", base="phase_diagram_paper_quick", V0=(20.0,), Omega_R=(0.6,), compressibility=False
    )
    try:
        runners.run_scenario("pdq", row, out_root=str(tmp_path))
    finally:
        runners.VERBOSE, runners._OUT_ROOT = verbose, out_root
    out = tmp_path / "pdq"
    for f in ("maps.npz", "summary.json", "provenance.json", "phase_diagram_provenance.json", "labels.png"):
        assert (out / f).exists(), f
    assert json.loads((out / "phase_diagram_provenance.json").read_text())["runs"]
