"""
Spectrum physics module for the GCR/SPE Geant4 pipeline.

Implements:
  - The force-field-modulated Local Interstellar Spectrum (LIS) for GCR
    protons, using a real fitted parametrization (rigidity-space, 3
    smoothly-joined power laws) from an AMS-02 + Voyager 1 combined fit.
  - The SAME functional shape, rescaled by mass/charge and relative
    abundance, as a documented approximation for heavier GCR species.
  - The Xapsos et al. (2000) Weibull-form SPE proton spectrum, with a
    "severity" parameter (0-1) interpolating "average" -> "worst-case".
  - Conversion from rigidity-space flux to kinetic-energy-space flux
    (with the required Jacobian), and from there to (energy,
    relative_intensity) tables Geant4's G4GeneralParticleSource can
    consume via /gps/hist/point.

GCR flux physics is naturally a function of KINETIC ENERGY PER NUCLEON,
not total kinetic energy. Total energy = A * (energy per nucleon), so a
single shared TOTAL-energy grid across species badly truncates heavy ions
-- for iron (A=56), a 10 MeV-10 GeV TOTAL-energy window only covers
~0.18-179 MeV/nucleon, nowhere near where iron's spectrum actually peaks.
Each species gets its OWN total-kinetic-energy grid, derived from a shared
per-nucleon-energy range via gcr_species_energy_table() -- this is the
function to use for GCR, not a single shared grid.

=======================================================================
KNOWN APPROXIMATIONS -- state these on the poster, do not let a judge
find them first.
=======================================================================

1. SPECTRAL SHAPE FOR HEAVY SPECIES. species_lis_flux_vs_rigidity()
   rescales the PROTON LIS by a relative abundance. Real helium and iron
   LIS have genuinely different spectral SHAPES in rigidity, not merely
   different amplitudes. This is adequate for a first-order dose study
   but it is NOT adequate for cross-validation against OLTARIS, whose
   Badhwar-O'Neill model carries proper per-species spectra.

   -> For OLTARIS validation, compare the PROTON component only. OLTARIS
      reports differential fluence per isotope, so protons can be
      extracted cleanly, and the proton LIS here is a real fit. Say
      explicitly in the write-up that He and Fe were not cross-validated.

2. SPECIES COVERAGE. Only Z = 1, 2, 26 are modelled by default. The
   intermediate charges (C, N, O, Ne, Mg, Si) carry real dose and are
   absent. See GCR_ABUNDANCE_MODE below -- the choice of how to handle
   that gap is explicit rather than buried.

3. MODULATION. The force-field approximation with a single potential is a
   simplification of real heliospheric transport. When cross-validating,
   the potential MUST match whatever OLTARIS is configured with --
   different modulation means a different source, which invalidates the
   comparison no matter how carefully everything else is matched.
"""

import numpy as np

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------
PROTON_MASS_MEV = 938.272
U_MEV = 931.494  # atomic mass unit; the correct per-nucleon rest mass for
                 # bound nucleons in a nucleus. Using the free-proton mass
                 # for heavy ions is a ~0.7% systematic in rigidity.

# =====================================================================
# GCR COMPOSITION
# =====================================================================
# Two defensible choices, made explicit because they give materially
# different dose-equivalent answers and the difference is invisible if
# the abundances are just numbers in a dict.
#
#   "realistic"  Each species carries its own measured abundance.
#                Z = 3-25 is simply absent, so dose equivalent is
#                UNDERESTIMATED. Honest and simple; state the gap.
#
#   "fe_proxy"   Iron absorbs the entire Z > 2 abundance budget, standing
#                in for all heavy ions. Because iron has the highest
#                quality factor of the group, this OVERESTIMATES dose
#                equivalent. Conservative, but it must be declared --
#                a 1% iron abundance is roughly 10-20x the real value and
#                a judge who knows GCR composition will notice.
#
# VERIFY these fractions against a primary reference (e.g. a Badhwar-
# O'Neill tabulation or an AMS-02 composition paper) before publishing
# numbers. They are approximate values for ~1 GeV/n at solar minimum.
GCR_ABUNDANCE_MODE = "realistic"

_ABUNDANCE_REALISTIC = {
    "proton": 0.870,
    "helium": 0.120,
    "carbon": 0.0035,
    "oxygen": 0.0035,
    "iron":   0.0007,
}

_ABUNDANCE_FE_PROXY = {
    "proton": 0.87,
    "helium": 0.12,
    "iron":   0.01,   # stands in for ALL Z > 2
}

_SPECIES_ZA = {
    "proton": dict(Z=1,  A=1),
    "helium": dict(Z=2,  A=4),
    "carbon": dict(Z=6,  A=12),
    "oxygen": dict(Z=8,  A=16),
    "iron":   dict(Z=26, A=56),
}


def _build_species():
    ab = (_ABUNDANCE_REALISTIC if GCR_ABUNDANCE_MODE == "realistic"
          else _ABUNDANCE_FE_PROXY)
    return {name: dict(_SPECIES_ZA[name], gcr_abundance=frac)
            for name, frac in ab.items()}


SPECIES = _build_species()

# Geant4 particle names. Anything heavier than helium is launched through
# GPS's generic ion interface, which also needs /gps/ion Z A.
G4_PARTICLE_NAME = {
    "proton": "proton",
    "helium": "alpha",
    "carbon": "ion",
    "oxygen": "ion",
    "iron":   "ion",
}


def g4_source_lines(species):
    """GPS commands selecting this species. Use instead of hardcoding."""
    name = G4_PARTICLE_NAME[species]
    if name != "ion":
        return [f"/gps/particle {name}"]
    Z, A = SPECIES[species]["Z"], SPECIES[species]["A"]
    # /gps/particle ion must precede /gps/ion, and /gps/ion takes
    # Z A Q E where Q is charge state and E excitation energy.
    return ["/gps/particle ion", f"/gps/ion {Z} {A} {Z} 0"]


# ---------------------------------------------------------------------
# 1. PROTON LOCAL INTERSTELLAR SPECTRUM -- real fitted parameters
# ---------------------------------------------------------------------
_PROTON_LIS_PARAMS = dict(
    N=5658.0,
    gamma0=1.669,
    breaks=[
        dict(R=0.572, delta=-4.117, s=1.78),
        dict(R=6.2,   delta=-0.423, s=3.89),
        dict(R=540.0, delta=-0.26,  s=1.53),
    ],
)


def proton_lis_flux_vs_rigidity(R_GV):
    R_GV = np.asarray(R_GV, dtype=float)
    R_safe = np.clip(R_GV, 1e-6, None)
    p = _PROTON_LIS_PARAMS
    flux = p["N"] * (R_safe / 1.0) ** p["gamma0"]
    for b in p["breaks"]:
        term = (1.0 + (R_safe / b["R"]) ** b["s"]) / (1.0 + b["R"] ** (-b["s"]))
        flux = flux * term ** (b["delta"] / b["s"])
    return flux


def species_lis_flux_vs_rigidity(species, R_GV):
    """
    APPROXIMATION (see module docstring item 1): the proton LIS shape
    rescaled by relative abundance. Real heavy-species LIS differ in shape,
    not just amplitude. Acceptable for a first-order dose study; not
    acceptable as the basis of an OLTARIS cross-validation for Z > 1.
    """
    base = proton_lis_flux_vs_rigidity(R_GV)
    ratio = SPECIES[species]["gcr_abundance"] / SPECIES["proton"]["gcr_abundance"]
    return base * ratio


def nucleon_rest_mass_mev(A):
    """
    Per-nucleon rest mass. A free proton is 938.272 MeV; a nucleon bound in
    a nucleus averages the atomic mass unit, 931.494 MeV. Using the proton
    mass for iron is a ~0.7% systematic in rigidity, small but free to fix.
    """
    return PROTON_MASS_MEV if A == 1 else U_MEV


def rigidity_from_kinetic_energy_per_nucleon(T_pn_MeV, A, Z):
    T_pn_MeV = np.asarray(T_pn_MeV, dtype=float)
    m = nucleon_rest_mass_mev(A)
    pc_per_nucleon_MeV = np.sqrt(np.clip(T_pn_MeV, 0, None)
                                 * (T_pn_MeV + 2.0 * m))
    return (A / Z) * pc_per_nucleon_MeV / 1000.0


def gcr_flux_vs_energy_per_nucleon(species, T_pn_grid_MeV,
                                   modulation_potential_MV=400.0):
    """
    Force-field-modulated GCR differential flux dJ/dT_pn (relative units)
    at 1 AU, as a function of kinetic energy PER NUCLEON.

    modulation_potential_MV must match the value used in any code being
    compared against (see module docstring item 3). 400 MV is a solar-
    minimum-ish default, not a universal constant.
    """
    Z = SPECIES[species]["Z"]
    A = SPECIES[species]["A"]
    m = nucleon_rest_mass_mev(A)

    T = np.clip(np.asarray(T_pn_grid_MeV, dtype=float), 1e-3, None)

    # Force-field energy loss per nucleon entering the heliosphere.
    Phi = modulation_potential_MV * (Z / A)
    T_shifted = T + Phi

    R_shifted_GV = rigidity_from_kinetic_energy_per_nucleon(T_shifted, A, Z)
    J_LIS_at_shifted = species_lis_flux_vs_rigidity(species, R_shifted_GV)

    modulation_factor = (
        (T * (T + 2.0 * m)) / (T_shifted * (T_shifted + 2.0 * m))
    )
    dJdR = J_LIS_at_shifted * modulation_factor

    pc_shifted_MeV = np.clip(np.sqrt(T_shifted * (T_shifted + 2.0 * m)),
                             1e-6, None)
    dR_dT = (A / Z) * (T_shifted + m) / (pc_shifted_MeV * 1000.0)

    return np.clip(dJdR * dR_dT, 0.0, None)


# ---------------------------------------------------------------------
# GCR energy grids -- per-species, derived from a shared per-nucleon range
# ---------------------------------------------------------------------
def per_nucleon_energy_grid_gcr(n_points=60):
    """
    Kinetic energy PER NUCLEON grid (MeV/n), shared across GCR species.
    Each species' own TOTAL-kinetic-energy grid (what GPS needs) is derived
    from this via T_total = A * T_pn in gcr_species_energy_table().

    NOTE FOR THE WRITE-UP: the cap is 10 GeV PER NUCLEON, so iron reaches
    560 GeV total. Earlier framing of this project described a "10 GeV cap
    to avoid multi-scale gradient pathologies" -- that was in total energy
    and no longer describes what is sampled. Per-nucleon is the standard
    framing in space radiation and the physics justification (the MIP
    transition, the E^-2.7 falloff) is naturally expressed that way, but
    the two framings must not both appear in the materials.
    """
    return np.logspace(np.log10(10.0), np.log10(1.0e4), n_points)


def gcr_species_energy_table(species, n_points=60,
                             modulation_potential_MV=400.0):
    """
    Returns (T_total_grid_MeV, flux) for one GCR species, on THIS species'
    own total-kinetic-energy grid (A * the shared per-nucleon range), so
    its full spectral shape including the peak is represented.
    """
    A = SPECIES[species]["A"]
    T_pn_grid = per_nucleon_energy_grid_gcr(n_points=n_points)
    flux = gcr_flux_vs_energy_per_nucleon(
        species, T_pn_grid, modulation_potential_MV=modulation_potential_MV)
    return A * T_pn_grid, flux


def gcr_species_event_weights(n_points=60, modulation_potential_MV=400.0):
    """
    Relative number of EVENTS to simulate per species.

    GPS cannot sample across sources by intensity in the way this pipeline
    needs (/gps/source/multiplevertex fires one primary from EVERY source
    each event rather than sampling one), so species are run as separate
    sub-jobs and the event counts must themselves carry the abundance
    weighting. Returns a dict summing to 1.0.

    Weighting by integrated flux rather than by raw abundance matters
    because the modulated spectra have different shapes over the sampled
    range, so equal abundance does not mean equal sampled flux.
    """
    integ = {}
    for s in SPECIES:
        E, F = gcr_species_energy_table(
            s, n_points=n_points,
            modulation_potential_MV=modulation_potential_MV)
        integ[s] = float(np.trapezoid(F, E)) if hasattr(np, "trapezoid") \
            else float(np.trapz(F, E))
    tot = sum(integ.values())
    return {s: v / tot for s, v in integ.items()}


# ---------------------------------------------------------------------
# 2. SPE PROTON SPECTRUM -- Xapsos et al. (2000) Weibull form
# ---------------------------------------------------------------------
# Integral fluence  N(>E) = A * exp(-kappa * E^alpha), differentiated below
# to give dN/dE. CITE the source for these parameter values in the write-up.
_SPE_PARAMS_AVERAGE = dict(A=1.0, kappa=0.15, alpha=0.4)
_SPE_PARAMS_WORST_CASE = dict(A=50.0, kappa=0.05, alpha=0.35)


def spe_flux_vs_kinetic_energy(T_MeV_grid, severity=0.5):
    severity = float(np.clip(severity, 0.0, 1.0))
    A = _SPE_PARAMS_AVERAGE["A"] + severity * (_SPE_PARAMS_WORST_CASE["A"] - _SPE_PARAMS_AVERAGE["A"])
    kappa = _SPE_PARAMS_AVERAGE["kappa"] + severity * (_SPE_PARAMS_WORST_CASE["kappa"] - _SPE_PARAMS_AVERAGE["kappa"])
    alpha = _SPE_PARAMS_AVERAGE["alpha"] + severity * (_SPE_PARAMS_WORST_CASE["alpha"] - _SPE_PARAMS_AVERAGE["alpha"])

    T_safe = np.clip(np.asarray(T_MeV_grid, dtype=float), 1e-3, None)
    flux = A * kappa * alpha * T_safe ** (alpha - 1.0) * np.exp(-kappa * T_safe ** alpha)
    return np.clip(flux, 0.0, None)


def energy_grid_for_environment(env_type, n_points=60):
    """SPE only (protons, so per-nucleon == total kinetic energy). For GCR,
    use gcr_species_energy_table(species)."""
    if env_type == "SPE":
        return np.logspace(np.log10(1.0), np.log10(500.0), n_points)
    raise ValueError(
        f"energy_grid_for_environment no longer supports '{env_type}' -- "
        f"use gcr_species_energy_table(species) for GCR.")


# ---------------------------------------------------------------------
# 3. GPS-READY TABLES
# ---------------------------------------------------------------------
def gps_histogram_commands(energy_grid_MeV, flux_values, interpolation="Lin"):
    """
    Macro lines implementing this spectrum as a GPS arbitrary point
    histogram for the CURRENTLY ACTIVE GPS source (call /gps/source/set
    <idx> first if using multiple sources).

    Order and content confirmed against a real Geant4 run: an earlier
    version omitted /gps/hist/inter and G4SPSEneDistribution::
    GenArbPointEnergies aborted with "IntType unknown type" because no
    interpolation scheme had been set. The required sequence is
    ene/type Arb -> ene/diffspec -> hist/type arb -> hist/point (xN) ->
    hist/inter <Lin|Log|Spline|Exp>. "Lin" is appropriate because the flux
    table is already densely sampled here, so GPS is not being asked to
    infer shape between sparse points.
    """
    flux_values = np.clip(np.asarray(flux_values, dtype=float), 0.0, None)
    if flux_values.sum() <= 0:
        raise ValueError("Flux histogram is entirely zero -- check energy range/parameters.")

    lines = ["/gps/ene/type Arb", "/gps/ene/diffspec 1", "/gps/hist/type arb"]
    for E, F in zip(energy_grid_MeV, flux_values):
        lines.append(f"/gps/hist/point {E:.6f} {F:.6e}")
    lines.append(f"/gps/hist/inter {interpolation}")
    return lines


if __name__ == "__main__":
    print(f"GCR_ABUNDANCE_MODE = {GCR_ABUNDANCE_MODE}\n")
    print(f"{'species':<10}{'Z':>4}{'A':>4}{'abundance':>12}"
          f"{'E_total range (MeV)':>26}{'event weight':>14}")
    w = gcr_species_event_weights()
    for s, d in SPECIES.items():
        E, _ = gcr_species_energy_table(s)
        print(f"{s:<10}{d['Z']:>4}{d['A']:>4}{d['gcr_abundance']:>12.4f}"
              f"{E[0]:>13.1f} - {E[-1]:>9.0f}{w[s]:>14.4f}")
    print("\nEvent weights sum:", round(sum(w.values()), 6))