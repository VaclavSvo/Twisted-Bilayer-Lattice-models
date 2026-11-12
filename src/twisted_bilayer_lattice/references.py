"""Reference values of the lattice models, dimensionless. Every number here is either exact (a closed form
computed on the spot) or a quoted value; its source string says which.
"""

from __future__ import annotations

import math

# name: (value, tolerance, source)
ANCHORS = {
    # The lattice model (dimensionless): the n = 1 superfluid-Mott transition of the plain square
    # lattice (z = 4) in U/J. Single-site mean field: the lobe tip U/zJ = 3 + 2 sqrt 2 (Fisher et al.,
    # Phys. Rev. B 40, 546 (1989); Sheshadri et al., Europhys. Lett. 22, 257 (1993)), exact to rounding.
    # Worm quantum Monte Carlo: (J/U)_c = 0.05974(3), i.e. (U/J)_c = 16.739(8), Capogrosso-Sansone, Soyler,
    # Prokof'ev and Svistunov, Phys. Rev. A 77, 015602 (2008), arXiv:0710.2703 (the 2D paper; Phys. Rev. B
    # 75, 134302 (2007) by the same group is the 3D cubic lattice, (U/J)_c = 29.34(2)).
    "square_lattice_UJ_c_gutzwiller": (4.0 * (3.0 + 2.0 * math.sqrt(2.0)), 1e-12, "z (3 + 2 sqrt 2) = 23.3137"),
    "square_lattice_UJ_c_qmc": (16.74, 1e-3, "Capogrosso-Sansone et al. 2008, worm QMC: 1/0.05974(3)"),
}


def anchor(name: str) -> tuple[float, float, str]:
    v, tol, src = ANCHORS[name]
    return v, tol, src
