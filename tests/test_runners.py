"""runners: every scenario row has a runner, and the tight-binding variants are what the shipped row names."""

from twisted_bilayer_lattice import runners
from twisted_bilayer_lattice.scenarios import SCENARIOS


def test_every_scenario_kind_has_a_runner():
    missing = {name: row.get("kind") for name, row in SCENARIOS.items() if row.get("kind") not in runners.KINDS}
    assert not missing, missing
    kinds = {row.get("kind") for row in SCENARIOS.values()}
    assert {"phase_diagram", "tight_binding"} <= kinds  # the check sees them


def test_tight_binding_variants_and_the_shipped_row():
    """Variant D is C plus the interband shift of the site energies only (same bonds, same interlayer pairs);
    the shipped row keeps the variants A, B and C."""
    import numpy as np

    from twisted_bilayer_lattice import model as lm

    assert SCENARIOS["tight_binding"]["variants"] == "ABC"
    assert set(runners.MODEL_C_VARIANTS) == set("ABCD")
    c = lm.build_model(12.0, 0.5, 4, 1, **runners.MODEL_C_VARIANTS["C"][1])
    d = lm.build_model(12.0, 0.5, 4, 1, **runners.MODEL_C_VARIANTS["D"][1])
    np.testing.assert_array_equal(c.bonds.rows, d.bonds.rows)
    np.testing.assert_array_equal(c.bonds.amp, d.bonds.amp)
    np.testing.assert_array_equal(c.V_pairs[2], d.V_pairs[2])
    shift = np.asarray(d.onsite) - np.asarray(c.onsite)
    assert np.abs(shift).max() > 1e-4 and shift.max() <= 1e-12  # second-order shifts lower the sites
