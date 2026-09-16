"""
Spectrum physics module for the GCR/SPE Geant4 pipeline.

Implements:
  - The force-field-modulated Local Interstellar Spectrum (LIS) for GCR
    protons, using a real fitted parametrization (rigidity-space, 3
    smoothly-joined power laws) from an AMS-02 + Voyager 1 combined fit.
  - The SAME functional shape, rescaled by mass/charge and relative
    abundance, as a documented approximation for GCR helium and iron.
    NOTE: this exact technique (reusing the proton LIS shape) is used in
    the literature itself for helium. Extending it to iron here is a
    flagged simplification, not an independently-fitted result -- if you
    want higher fidelity later, sourcing a real iron LIS fit is the
    highest-value next step.
  - The Xapsos et al. (2000) Weibull-form SPE proton spectrum, with a
    single "severity" parameter (0-1) interpolating between an
    "average" and "worst-case" event.
  - Conversion from rigidity-space flux to kinetic-energy-space flux
    (with the required Jacobian), and from there to (energy,
    relative_intensity) tables that Geant4's G4GeneralParticleSource can
    consume directly via /gps/hist/point.

Everything here works in KINETIC ENERGY PER NUCLEON (MeV/n) internally
for the GCR treatment, matching the standard form of the force-field
equation (Gleeson & Axford 1968), then converts to TOTAL kinetic energy
(what Geant4 actually needs) only at the point of building the GPS
table. This keeps the physics species-agnostic in the middle of the
calculation and species-specific only at the boundaries (rigidity<->
energy conversion depends on each species' own A/Z).
"""

import numpy as np

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------
PROTON_MASS_MEV = 938.272   # standard per-nucleon reference mass
U_MEV = 931.494             # atomic mass unit, MeV/c^2 (nuclear rest mass
                             # approximated as A * U_MEV -- ignores the
                             # ~1% binding-energy correction, acceptable
                             # at the energies used here)

SPECIES = {
    # name: Z, A, relative GCR abundance BY PARTICLE COUNT (matches the
    # standard ~87% H / ~12% He / ~1% HZE composition figures)
    "proton": dict(Z=1, A=1, gcr_abundance=0.87),
    "helium": dict(Z=2, A=4, gcr_abundance=0.12),
    "iron":   dict(Z=26, A=56, gcr_abundance=0.01),
}

# Geant4 particle naming (GPS convention)
G4_PARTICLE_NAME = {
    "proton": "proton",
    "helium": "alpha",   # He-4 nucleus
    "iron": "ion",        # requires /gps/ion 26 56 as well
}


# ---------------------------------------------------------------------
# 1. PROTON LOCAL INTERSTELLAR SPECTRUM -- real fitted parameters
#
# Rigidity-space, 3 smoothly-joined power laws (AMS-02 + Voyager 1
# combined fit, force-field-approximation framework). R in GV,
# dJ/dR in m^-2 sr^-1 s^-1 GV^-1. Only the SHAPE matters for our use
# (Geant4's GPS normalizes any histogram it's given internally), so the
# absolute units don't need to carry through exactly -- what matters is
# getting the relative weighting across energy right.
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
    """dJ_LIS/dR at rigidity R (GV) for protons. Vectorized."""
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
    LIS flux (dJ/dR) for a given species, using the proton-fitted SHAPE
    rescaled by that species' relative GCR abundance (see module
    docstring for the caveat on helium vs iron).
    """
    base = proton_lis_flux_vs_rigidity(R_GV)
    abundance_ratio = SPECIES[species]["gcr_abundance"] / SPECIES["proton"]["gcr_abundance"]
    return base * abundance_ratio


def rigidity_from_kinetic_energy_per_nucleon(T_pn_MeV, A, Z):
    """
    Rigidity R (GV) for a nucleus (A, Z) at kinetic energy per nucleon
    T_pn (MeV/n). pc_per_nucleon = sqrt(T_pn * (T_pn + 2*m_p)); the
    A/Z factor is the standard mass-to-charge dependence of rigidity.
    """
    T_pn_MeV = np.asarray(T_pn_MeV, dtype=float)
    pc_per_nucleon_MeV = np.sqrt(np.clip(T_pn_MeV, 0, None) * (T_pn_MeV + 2.0 * PROTON_MASS_MEV))
    return (A / Z) * pc_per_nucleon_MeV / 1000.0


def gcr_flux_vs_energy_per_nucleon(species, T_pn_grid_MeV, modulation_potential_MV=400.0):
    """
    Force-field-modulated GCR differential flux dJ/dT_pn (relative
    units -- shape only) at 1 AU. Gleeson & Axford (1968):

        J_TOA(T) = J_LIS(T + Phi) * [T(T+2m_p)] / [(T+Phi)(T+Phi+2m_p)]

    with T = kinetic energy PER NUCLEON (MeV/n), Phi = phi_MV * Z/A
    (MeV/n, the per-nucleon energy loss from solar modulation), and m_p
    the proton rest mass used as the standard per-nucleon reference mass.
    Default modulation potential (400 MV) = solar minimum, matching the
    project's own stated choice ("conservative worst case").
    """
    Z = SPECIES[species]["Z"]
    A = SPECIES[species]["A"]
    T = np.asarray(T_pn_grid_MeV, dtype=float)
    T = np.clip(T, 1e-3, None)

    Phi = modulation_potential_MV * (Z / A)
    T_shifted = T + Phi

    R_shifted_GV = rigidity_from_kinetic_energy_per_nucleon(T_shifted, A, Z)
    J_LIS_at_shifted = species_lis_flux_vs_rigidity(species, R_shifted_GV)

    modulation_factor = (
        (T * (T + 2.0 * PROTON_MASS_MEV))
        / (T_shifted * (T_shifted + 2.0 * PROTON_MASS_MEV))
    )
    dJdR = J_LIS_at_shifted * modulation_factor

    # Jacobian dR/dT_shifted, needed to convert dJ/dR into dJ/dT:
    pc_shifted_MeV = np.sqrt(T_shifted * (T_shifted + 2.0 * PROTON_MASS_MEV))
    pc_shifted_MeV = np.clip(pc_shifted_MeV, 1e-6, None)
    dR_dT = (A / Z) * (T_shifted + PROTON_MASS_MEV) / (pc_shifted_MeV * 1000.0)

    dJdT = dJdR * dR_dT
    return np.clip(dJdT, 0.0, None)


# ---------------------------------------------------------------------
# 2. SPE PROTON SPECTRUM -- Xapsos et al. (2000) Weibull form
#
#   J(E) = A * kappa * alpha * E^(alpha-1) * exp(-kappa * E^alpha)
#
# Parameter sets below span a representative "average" -> "worst-case"
# range, consistent with the Weibull functional form used by ESP/
# SPENVIS for real historical SPE fits. These specific numbers are
# illustrative of that range, not a reproduction of one named event's
# published fit -- if you want to match a specific historical event
# (e.g. Oct 1989), the exact published A/kappa/alpha for that event
# should replace these.
# ---------------------------------------------------------------------
_SPE_PARAMS_AVERAGE = dict(A=1.0, kappa=0.15, alpha=0.4)
_SPE_PARAMS_WORST_CASE = dict(A=50.0, kappa=0.05, alpha=0.35)


def spe_flux_vs_kinetic_energy(T_MeV_grid, severity=0.5):
    """
    SPE differential proton flux (relative units) at kinetic energy
    T_MeV_grid. severity in [0, 1]: 0.0 = "average" event, 1.0 =
    "worst-case" event, linear interpolation of Weibull parameters
    in between.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    A = _SPE_PARAMS_AVERAGE["A"] + severity * (_SPE_PARAMS_WORST_CASE["A"] - _SPE_PARAMS_AVERAGE["A"])
    kappa = _SPE_PARAMS_AVERAGE["kappa"] + severity * (_SPE_PARAMS_WORST_CASE["kappa"] - _SPE_PARAMS_AVERAGE["kappa"])
    alpha = _SPE_PARAMS_AVERAGE["alpha"] + severity * (_SPE_PARAMS_WORST_CASE["alpha"] - _SPE_PARAMS_AVERAGE["alpha"])

    T = np.asarray(T_MeV_grid, dtype=float)
    T_safe = np.clip(T, 1e-3, None)
    flux = A * kappa * alpha * T_safe ** (alpha - 1.0) * np.exp(-kappa * T_safe ** alpha)
    return np.clip(flux, 0.0, None)


# ---------------------------------------------------------------------
# 3. GPS-READY TABLES
# ---------------------------------------------------------------------
def energy_grid_for_environment(env_type, n_points=60):
    """Default TOTAL kinetic energy grids (MeV) per environment."""
    if env_type == "GCR":
        return np.logspace(np.log10(10.0), np.log10(1.0e4), n_points)  # 10 MeV - 10 GeV
    elif env_type == "SPE":
        return np.logspace(np.log10(1.0), np.log10(500.0), n_points)   # 1 - 500 MeV
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def gps_histogram_commands(energy_grid_MeV, flux_values):
    """
    Macro command lines implementing this spectrum as a GPS arbitrary
    point histogram for the CURRENTLY ACTIVE GPS source (call
    /gps/source/set <idx> before these if using multiple sources).
    """
    flux_values = np.clip(np.asarray(flux_values, dtype=float), 0.0, None)
    if flux_values.sum() <= 0:
        raise ValueError("Flux histogram is entirely zero -- check energy range/parameters.")

    lines = ["/gps/ene/type Arb", "/gps/hist/type arb"]
    for E, F in zip(energy_grid_MeV, flux_values):
        lines.append(f"/gps/hist/point {E:.6f} {F:.6e}")
    lines.append("/gps/ene/diffspec 1")  # histogram is a differential spectrum
    return lines