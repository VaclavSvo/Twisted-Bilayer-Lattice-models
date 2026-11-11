# The lattice model beyond mean field

The plan is `roadmap-beyond-mean-field.md`; this page documents what the `manybody` package does so far: the
Hamiltonian and its parameters, the bridge to the continuum, single-site and cluster Gutzwiller and the phase
maps. Lattice units: a = 1, E_R = 1, ħ = 1.

## The Hamiltonian and its inputs

`engine.lattice_model.build_model` assembles the two-layer Bose–Hubbard model on the commensurate supercell
(its docstring lists what the model keeps and drops):

    H = − Σ_σ Σ_{i≠i'} J(|R_i − R_i'|) b_i† b_i' + Ω_R Σ_ij S(R_i − R'_j)(b_i† c_j + h.c.)
        + (U/2) Σ n_i(n_i − 1) + Σ_ij V_ij n_i⁽¹⁾ n_j⁽²⁾ + Σ_i (ε_i − μ) n_i.

Every input comes from `config.py`: V₀ gives J(d) (the exact lowest band, `bands.hopping_1d`) and the
overlaps S (`bands.overlap_1d`); the pancake frequency ν_z gives g₂D = g₃D/(√(2π) l_z) through
`engine.units.lattice_units` and U = g₂D (∫w⁴)², so U ∝ √ν_z. The interaction input of every lattice
function (`build_model`, `ground_state`, `superfluid_fraction`, `gutzwiller`, the `tight_binding` row) is
resolved by `lattice_model.resolve_g2d`: an explicit `g2d` wins; otherwise `nu_z_hz` gives
`g2d_from_nu_z(nu_z_hz)`; otherwise the shipped `G2D_DEFAULT = 0.01576`. The derived 1 kHz value,
0.0157638, is 2.4 × 10⁻⁴ above the shipped default (the engine's a_s = 100 a₀ against the rounded constant;
an offset of the kind listed in `engine.units.EXCEPTIONS`). The default is unchanged on purpose: moving it
would move every lattice-model number, and that is a decision for the author. The model records which
value it used (`g2d`, `g2d_source`, `nu_z_hz`).

## The parameter audit (`manybody.parameters`, `scripts/parameter_audit.py`)

The paper's mean-field phase diagram (Meng et al., Nature 615, 231 (2023), Supplement Sec. IV, Eq. 13) uses

    t = (4/√π) E_R (V₀/E_R)^{3/4} e^{−2√(V₀/E_R)},   U = √(8/π) k a_s E_R (V₀/E_R)^{3/4},   k = π/a,

the deep-lattice asymptote of the hopping and the on-site interaction of a 3D cubic lattice of depth V₀ in
all three directions (Zwerger, J. Opt. B 5, S9 (2003)). `paper_parameters(V0)` evaluates them with
a_s = `config.A_S_M` (5.29 nm); `audit_table` puts them next to the exact J₁, J₂/J₁ and the pancake's
U(ν_z), U/J₁, and `transition_depths` solves U/J = (U/J)_c of the plain square lattice for V₀ with either
set, at the single-site mean-field lobe tip U/zJ = 3 + 2√2 (U/J = 23.31) and the worm-Monte-Carlo value
(U/J)_c = 1/0.05974(3) = 16.74 (Capogrosso-Sansone, Söyler, Prokof'ev and Svistunov, Phys. Rev. A 77,
015602 (2008)); both are anchors in `gates.references`. The tests pin three rows and the four depths.

The paper quotes U/t = 1.67 at 4 E_R (Fig. 4 caption) where its formulas give 1.62; the audit reports the
gap and the single changes that would close it (a_s = 102.8 a₀, or V₀ = 4.055 E_R) without applying any.
Ω_R/t = 2.07 in the same caption with the paper's t is Ω_R = 0.242 E_R, the "paper-literal" reading of
`config.J_PERP_CODE`.

## The paper's own model (`manybody.paper_model`, variant "paper")

Supplement Eqs. 12–13: one layer, nearest-neighbour t, the paper's U, μ/U = 1, and the other lattice as a
site offset

    M_i = M_R [sin²(π(i_x cos θ + i_y sin θ)) + sin²(π(i_y cos θ − i_x sin θ))],

built on the c = m² + n² sites of lattice 1 of the supercell with the exact rational cos θ = (m² − n²)/c,
sin θ = 2mn/c of the commensurate twist (M = 0 at AA, 2 M_R at AB). `paper_model(V0, M_R, ...)` returns a
one-layer `LatticeModel`, and every solver of `lattice_model` takes it (`ground_state`,
`superfluid_fraction`, `lattice_bands`, `gutzwiller`, which uses the model's μ/U when given neither μ/U
nor a filling). The paper does not say how M_R depends on Ω_R. M_R is an explicit input, and
`M_R_from_Omega(V0, Omega_R, rule)` offers three documented readings, all assumptions:

* `"k0"` (default): the two-layer model's own k = 0 coupling. The band-bottom moiré potential at a
  lattice-1 site is −Ω_R Σ_j S_ij (`build_interlayer`, F_k0); matching M_R f_i at AA and AB gives
  M_R = Ω_R (S_sum_AA − S_sum_AB)/2, about 0.35 Ω_R at 8 E_R. It vanishes at Ω_R = 0 and is what the
  two-layer Hamiltonian contains at first order.
* `"omega"`: M_R = Ω_R.
* `"eq8"`: M_R = V₀/2, the linear term of the strong-coupling effective potential of the Supplement's
  Eq. 8 at a lattice-1 site (V_eff = V₂/2 − V₂²/(8Ω_R) + const; equivalently Eq. 9). Independent of Ω_R,
  and the dropped quadratic term is the Ω_R ≫ V₀ expansion, outside the experiment's range.

## The sign audit (`manybody.parameters.sign_audit`)

For a bosonic Monte Carlo the off-diagonal elements of H₀ must be ≤ 0. Intralayer, −J₁ < 0 is fine and
−J₂ > 0 (J₂ < 0) is not. Interlayer, +Ω_R S_ij is positive for every pair inside |d'| < 1; the gauge
c_j → −c_j on layer 2 makes those negative and leaves the pairs with S < 0 (one 1D factor with
1 < |d'| < 2, so only with cutoff > 1) positive. `sign_audit(model)` applies the gauge and reports, per
class, the directed entries that stay positive and the fraction of the weight Σ|amp| they carry. The
numbers at V₀ = 12, Ω_R = 0.5 are pinned in `tests/manybody/test_parameters.py`.

## The interband shift (`engine.lattice_model.interband_shift`, `build_model(interband=True)`)

Ω_R couples the lowest-band Wannier function of one layer to the higher bands β of the other, which the
two-band model drops. Second order in that coupling (Schrieffer–Wolff) gives, for lattice-1 sites i, i',

    H⁽²⁾_{ii'} = −Ω_R² Σ_{β≠0} Σ_j ⟨w₀ᵢ|w_βⱼ⟩⟨w_βⱼ|w₀ᵢ'⟩ / D_β,

a site energy (i = i') and an intralayer bond correction (i ≠ i'), and the same for lattice 2 through
lattice 1's higher bands. With separable Wannier functions in the frame axes (the approximation of
`build_interlayer`), ⟨w₀ᵢ|w_βⱼ⟩ = s_{0β_x}(Δx') s_{0β_y}(Δy'), where `bands.overlap_bands_1d` gives
s_{0β}(d) = ∫ w₀(x) w_β(x − d) dx from the higher-band Wannier functions in the projection gauge
(`bands.wannier_coefficients_band`: ⟨g_β|ψ_q⟩ > 0 with g_β the oscillator state of order β of the site
well; for the even potential this is the real Wannier function of parity (−1)^β, the maximally localised
one, and for β = 0 it coincides with the existing gauge). The checks: s₀₀ = `overlap_1d`, s_ab(0) = δ_ab,
s_ab(n) = 0, s_ab(−d) = (−1)^{a+b} s_ab(d), and the gauge-free band weights `bands.band_weight_1d` sum to
exactly 1 over all plane-wave bands, with 6 bands per axis holding > 99.9 % of a half-site-displaced
function. The energy denominator uses the 1D band centres, E_β − E₀ = (e_{β_x} − e₀) + (e_{β_y} − e₀),
and by default (`denominator="dressed"`) adds the site's first-order dressing Ω_R S_hyb,i (1 at AA): the
two-band model already lowers the state by that much, so the intermediate states are that much further
away (the Brillouin–Wigner choice for the states being corrected). ε⁽²⁾ = 0 at AA, where the whole Wannier
function lies in the partner's lowest band, and is most negative where it does not, so the moiré
potential gets shallower and reshaped.

The bond corrections are grouped by the intermediate site of the other layer, by its wrapped index in both
layers: two sites share an intermediate orbital when their pairs reach the same partner site, possibly
through different periodic images, and the bond then joins one site to the image of the other that shares
it (the displacement δ_a − δ_b carries the Peierls phase). With the layer-symmetric frame (`frame="mid"`)
both layers have the same bond count per site and the same spectrum of their H⁽²⁾ block
(`test_interband_bonds_layer_symmetric_across_the_boundary`).

Validation (`tests/test_lattice_model.py::test_interband_shift_against_continuum`): the lowest six k = 0 levels of the
tight-binding model against the continuum (`compare_with_continuum`, m = 4, V₀ = 12, 16 points per a, S cut
at |d'| < 2, 6 bands). The maximum level error falls from 0.081 to 0.0083 E_R at Ω_R = 1 (with the bare
band-centre denominators 0.016), from 0.031 to 0.0021 at Ω_R = 0.6, and from 0.0035 to 0.0001 at 0.2; at
V₀ = 8 and 16 (Ω_R = 1) from 0.083 to 0.015 and from 0.073 to 0.0055. The lowest level alone (the AA pocket, where the shift vanishes) is within
4 × 10⁻³ either way. On the experiment's cell (m = 22, V₀ = 12, six lowest levels, all AA-pocket states) the
error falls from 0.0074 to 0.0014 at Ω_R = 1 and from 0.0032 to 0.0004 at 0.6 (the site part alone:
0.0018 and 0.0006).
The residual is the second-order truncation and the separable-frame approximation. Resolution (16, 24, 32
points per a), the number of bands (4, 6, 8) and the overlap cutoff (2, 3, 4) change the result in the
fourth digit. The bond corrections are not small: on m = 4 they move the k = 0 levels by 3.5 × 10⁻⁴
(Ω_R = 0.2), 3.2 × 10⁻³ (0.6) and 8.6 × 10⁻³ E_R (1.0) at V₀ = 12, and by 2.2 × 10⁻² at V₀ = 8,
Ω_R = 1 (the site part alone leaves 0.0116 at V₀ = 12, Ω_R = 1, against 0.0083 with the bonds). They are
included with the site energies by default (`interband_bonds=True`); the shipped phase-map rows use the
site part only (`interband_bonds=False`). On the experiment's cell that model's mean field misses the continuum
by up to 0.016 E_R in μ − E_bottom and 0.022 in f_s at V₀ = 8 (Ω_R = 0.6–1), against ≤ 0.005 and ≤ 0.002 with the
bonds; at V₀ = 12 and 16 it is within 0.01 E_R and 0.002.

`interband` stays **off by default** in `build_model`; `bh_parameters(V0, Omega_R=...)` reports the
shift's summary (ε at AA and AB, its minimum, the moiré depth with it) and the bridge runs both. Whether
to make it the default is the author's decision (it passed its validation).

## The bridge (`manybody.bridge`, scenarios `bridge_quick` and `bridge`)

The same mean field on both sides: the discrete GPE of the two-layer model (variant C: S cut at |d'| < 2,
the interlayer V_ij, J₁ + J₂, with and without the interband shift) against the continuum GPE
(`engine.sweep.coupling_scan` with `vertical="fixed"` at the same g₂D(ν_z); NPSE would compare different
models) on the same cell, filling and twist, for V₀ ∈ {8, 12, 16} and Ω_R ∈ {0, 0.24, 0.6, 1.0}. Compared:
μ − E_bottom and E/N − E_bottom from each model's own decoupled band bottom (−4 Σ J_d on the lattice,
2 E_1D(q = 0) in the continuum; the lattice measures energies from the on-site energy 2 ε_1D), f_s by the
twist energy, the AA/AB density ratio with the same 1 a discs at the moiré points, N₂/N, and the
participation ratio (over sites against over grid points, not the same object). Output: `bridge.txt`
(the table), `bridge.json`, `bridge.png` (the differences against Ω_R per V₀). One gate: at V₀ = 16,
Ω_R = 0 the two μ − E_bottom agree to 3 % (measured: 1.45 % on m = 4, 1.46 % on m = 22); if not the run stops,
since that would be a units or reference-energy mismatch. Always complex128 (energy differences of 10⁻³ E_R).

## The occupation-dependent U (`manybody.parameters.occupation_dependent_U`, report only)

With n atoms in one Wannier orbital the pancake broadens locally. The NPSE width s(n₂D) (`engine.npse`,
s⁴ − x s − 1 = 0, x = g₂D n₂D / ħω_z) at the local density n w(x)² w(y)² gives the mean-field interaction
energy E_int(n) = ∫ e_int d²r and U_n = 2 E_int/n² (U_n → U as n → 0). At 1 kHz the peak density of one
atom is 9.5 (12 E_R) to 11.2 (16 E_R) atoms per a², x = 0.55–0.65 for one atom and 1.1–1.3 for two, so
the site's width grows by 13–15 % (one atom) and 24–28 % (two); U₁ is 4–5 % below the fixed-width U and
U₂/U₁ = 0.964 (12 E_R), 0.959 (16 E_R). This is below a 5 % flag for the ratio
U₂/U₁, but U₂ is 8–9 % below the fixed-width U: a systematic of the same size as the ν_z question.

## Single-site Gutzwiller (`manybody.gutzwiller`)

A variational ground-state (T = 0) method: Ω = ⟨Ψ|H − μN|Ψ⟩ is minimised over product states
|Ψ⟩ = ⊗ᵢ Σₙ cᵢₙ |n⟩ᵢ, n = 0 … n_max. At a stationary point every site's amplitudes are the ground state of

    Hᵢ = (U/2) n(n − 1) + (εᵢ + hᵢ − μ) n + Ψᵢ b† + Ψᵢ* b,   Ψᵢ = Σⱼ Tᵢⱼ(θ) φⱼ,   hᵢ = Σⱼ Vᵢⱼ ⟨nⱼ⟩,

an (n_max + 1)-dimensional tridiagonal matrix that depends on the neighbours' φⱼ = ⟨bⱼ⟩ and ⟨nⱼ⟩. The conditions
are solved by the damped map φ ← φ + β (G(φ) − φ) of `lattice_model._gutzwiller_solve` (β = 1.6/(1 + r), the
Mott jump to φ = 0 once the decay ratio settles below 1 at |φ| < 10⁻⁵), operation for operation, so at θ = 0 the
solver reproduces `lattice_model.gutzwiller` bit for bit. The twist enters as Peierls phases on every bond
(complex arithmetic only at θ ≠ 0); a batch runs models of one bond structure together with per-member
convergence; four starts are solved and the lowest converged Ω is kept.

* **f_s** by the twist energy at fixed μ, f_s(θ) = (Ω(θ) − Ω(0)) L²/(N κ θ²), quoted as the θ → 0 extrapolation
  (4 f(0.15) − f(0.3))/3. Where a twisted solve does not converge (the soft relative phase of nearly separate
  islands), ⟨H(θ) − μN⟩ of the last iterate's product state bounds Ω(θ) from above, so f(0) ≤ (4/3) f_upper(0.15)
  if f(0.3) ≥ 0 (not guaranteed with couplings of both signs); a bound below f_c decides the label (`fs_bound`).
* **Truncation.** The relative effect of the occupation cutoff on N, Ω and f_s is about 1–10 × P(n_max), the
  weight of the top Fock state (measured on the experiment's cell, n_max 3 … 14). P(n_max) ≤ 10⁻⁶ needs n_max ≈ 6 at
  V₀ ≥ 20, 8 at 12–18, 10–12 at 7–10 and more than 14 at V₀ ≤ 6 when μ/U = 1 from the AA level (about 7 atoms per
  site at V₀ = 4). `phase_diagram`'s `n_max_table` sets n_max per V₀ range.
* **Several minima.** Where the occupied pockets are weakly coupled (SF-II), the product-state energy has several
  local minima that differ in the relative phases (signs of φ) of the islands, a few 10⁻⁵ per site apart in Ω;
  they share the label. Root-finding accelerators (Anderson mixing, Newton–Krylov) applied to the map converge to
  saddle points of Ω there, which satisfy the stopping test; the damped map does not.

## Cluster Gutzwiller (`manybody.cluster`)

The product runs over clusters C of sites treated exactly (every bond inside C, with its twist phase), coupled to
the rest by the mean fields Ψᵢ and hᵢ of the cross bonds; Ω = Σ_C E_C − Re Σ_cross Tᵢⱼ φᵢ* φⱼ −
Σ_cross Vᵢⱼ ⟨nᵢ⟩⟨nⱼ⟩. One-site clusters reproduce single-site Gutzwiller; a cluster covering a whole small periodic
lattice reproduces exact diagonalisation (tests). Tilings: single, interlayer dimer (minimum-cost matching), plus
(a site and its four neighbours, one chirality per layer), bilayer plus. Fock dimensions: dimer 49 (n_max 6), plus
243 (n_max 2), bilayer plus 59 049. The n = 1 lobe tip of the plain square lattice comes from the linear stability
of the Mott state with the exact cluster susceptibility: 23.31 (single site), 21.76 (plus), 21.19 (2 × 2), 20.30
(3 × 3), against 16.74 from quantum Monte Carlo.

## Phase maps (`manybody.phase_diagram`, scenario kind `phase_diagram`)

Per grid point (V₀, Ω_R): the model (the full two-layer model of `build_model`, or the paper's one-layer model
with M_R from `M_R_from_Omega`), μ = (μ/U) U from the row's zero (`mu_ref`: "onsite", or "aa" = the AA level
−Ω_R F_k0_AA of the two-layer model, the zero of the paper's M_i under rule k0), the starts, the twists, the
visibilities and the label:

| label | condition |
| --- | --- |
| SF | f_s > f_c |
| SF-II | f_s ≤ f_c and max ∣φ∣ > φ_c (phase-coherent islands without global superflow) |
| I | f_s ≤ f_c, max ∣φ∣ ≤ φ_c, density contrast > c_c |
| MI | f_s ≤ f_c, max ∣φ∣ ≤ φ_c, uniform density |
| empty | no site occupied |
| ? | the state or a twisted solve did not converge, or a twisted Ω fell below Ω(0) beyond the noise |

f_c = 10⁻³, φ_c = 10⁻⁴, c_c = 10⁻². At μ/U = 1 the paper's model sits where the n = 1 and n = 2 lobes touch at its
AA sites, and mean field gives neither MI nor I there; μ/U = 0.95 with M_R rule k0 reproduces the four panels of the
paper's Fig. S4c (SF, SF-II, MI, I). A map is resumable; maps.npz records the row's specification hash and the
solver version, and a resume with another specification is refused.

