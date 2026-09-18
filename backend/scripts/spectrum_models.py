"""
Spectrum physics module for the GCR/SPE Geant4 pipeline.

Implements:
  - The force-field-modulated Local Interstellar Spectrum (LIS) for GCR
    protons, using a real fitted parametrization (rigidity-space, 3
    smoothly-joined power laws) from an AMS-02 + Voyager 1 combined fit.
  - The SAME functional shape, rescaled by mass/charge and relative
    abundance, as a documented approximation for GCR helium and iron.
  - The Xapsos et al. (2000) Weibull-form SPE proton spectrum, with a
    "severity" parameter (0-1) interpolating "average" -> "worst-case".
  - Conversion from rigidity-space flux to kinetic-energy-space flux
    (with the required Jacobian), and from there to (energy,
    relative_intensity) tables Geant4's G4GeneralParticleSource can
    consume via /gps/hist/point.

IMPORTANT (fixed in this version): GCR flux physics is naturally a
function of KINETIC ENERGY PER NUCLEON, not total kinetic energy. Total
energy = A * (energy per nucleon), so a single shared TOTAL-energy grid
across species badly truncates heavy ions -- for iron (A=56), a
10 MeV-10 GeV TOTAL-energy window only covers ~0.18-179 MeV/nucleon,
nowhere near where iron's spectrum actually peaks (a few hundred MeV/n
to multi-GeV/n, same per-nucleon region as protons and helium). Each
species now gets its OWN total-kinetic-energy grid, derived from a
shared per-nucleon-energy range via gcr_species_energy_table() below --
this is the function to use for GCR now, not a single shared grid.
"""

import numpy as np

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------
PROTON_MASS_MEV = 938.272
U_MEV = 931.494

SPECIES = {
    "proton": dict(Z=1, A=1, gcr_abundance=0.87),
    "helium": dict(Z=2, A=4, gcr_abundance=0.12),
    "iron":   dict(Z=26, A=56, gcr_abundance=0.01),
}

G4_PARTICLE_NAME = {
    "proton": "proton",
    "helium": "alpha",
    "iron": "ion",
}


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
    base = proton_lis_flux_vs_rigidity(R_GV)
    abundance_ratio = SPECIES[species]["gcr_abundance"] / SPECIES["proton"]["gcr_abundance"]
    return base * abundance_ratio


def rigidity_from_kinetic_energy_per_nucleon(T_pn_MeV, A, Z):
    T_pn_MeV = np.asarray(T_pn_MeV, dtype=float)
    pc_per_nucleon_MeV = np.sqrt(np.clip(T_pn_MeV, 0, None) * (T_pn_MeV + 2.0 * PROTON_MASS_MEV))
    return (A / Z) * pc_per_nucleon_MeV / 1000.0


def gcr_flux_vs_energy_per_nucleon(species, T_pn_grid_MeV, modulation_potential_MV=400.0):
    """
    Force-field-modulated GCR differential flux dJ/dT_pn (relative
    units) at 1 AU, as a function of kinetic energy PER NUCLEON.
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

    pc_shifted_MeV = np.sqrt(T_shifted * (T_shifted + 2.0 * PROTON_MASS_MEV))
    pc_shifted_MeV = np.clip(pc_shifted_MeV, 1e-6, None)
    dR_dT = (A / Z) * (T_shifted + PROTON_MASS_MEV) / (pc_shifted_MeV * 1000.0)

    dJdT = dJdR * dR_dT
    return np.clip(dJdT, 0.0, None)


# ---------------------------------------------------------------------
# GCR energy grids -- per-species, derived from a shared per-nucleon range
# ---------------------------------------------------------------------
def per_nucleon_energy_grid_gcr(n_points=60):
    """
    Kinetic energy PER NUCLEON grid (MeV/n), shared across GCR species.
    Each species' own TOTAL-kinetic-energy grid (what Geant4/GPS needs)
    is derived from this via T_total = A * T_pn in
    gcr_species_energy_table() below -- NOT by sharing one total-energy
    grid across species (see module docstring for why that was wrong).
    """
    return np.logspace(np.log10(10.0), np.log10(1.0e4), n_points)  # 10 MeV/n - 10 GeV/n


def gcr_species_energy_table(species, n_points=60, modulation_potential_MV=400.0):
    """
    Returns (T_total_grid_MeV, flux) for one GCR species. T_total_grid is
    THIS species' own total-kinetic-energy grid (A * the shared
    per-nucleon range), so its full characteristic spectral shape --
    including its peak -- is actually represented, not truncated.
    """
    A = SPECIES[species]["A"]
    T_pn_grid = per_nucleon_energy_grid_gcr(n_points=n_points)
    flux = gcr_flux_vs_energy_per_nucleon(species, T_pn_grid, modulation_potential_MV=modulation_potential_MV)
    T_total_grid = A * T_pn_grid
    return T_total_grid, flux


# ---------------------------------------------------------------------
# 2. SPE PROTON SPECTRUM -- Xapsos et al. (2000) Weibull form
# ---------------------------------------------------------------------
_SPE_PARAMS_AVERAGE = dict(A=1.0, kappa=0.15, alpha=0.4)
_SPE_PARAMS_WORST_CASE = dict(A=50.0, kappa=0.05, alpha=0.35)


def spe_flux_vs_kinetic_energy(T_MeV_grid, severity=0.5):
    severity = float(np.clip(severity, 0.0, 1.0))
    A = _SPE_PARAMS_AVERAGE["A"] + severity * (_SPE_PARAMS_WORST_CASE["A"] - _SPE_PARAMS_AVERAGE["A"])
    kappa = _SPE_PARAMS_AVERAGE["kappa"] + severity * (_SPE_PARAMS_WORST_CASE["kappa"] - _SPE_PARAMS_AVERAGE["kappa"])
    alpha = _SPE_PARAMS_AVERAGE["alpha"] + severity * (_SPE_PARAMS_WORST_CASE["alpha"] - _SPE_PARAMS_AVERAGE["alpha"])

    T = np.asarray(T_MeV_grid, dtype=float)
    T_safe = np.clip(T, 1e-3, None)
    flux = A * kappa * alpha * T_safe ** (alpha - 1.0) * np.exp(-kappa * T_safe ** alpha)
    return np.clip(flux, 0.0, None)


def energy_grid_for_environment(env_type, n_points=60):
    """SPE only (protons, so per-nucleon == total kinetic energy). For
    GCR, use gcr_species_energy_table(species) instead -- see above."""
    if env_type == "SPE":
        return np.logspace(np.log10(1.0), np.log10(500.0), n_points)
    else:
        raise ValueError(
            f"energy_grid_for_environment no longer supports '{env_type}' -- "
            f"use gcr_species_energy_table(species) for GCR."
        )


# ---------------------------------------------------------------------
# 3. GPS-READY TABLES
# ---------------------------------------------------------------------
def gps_histogram_commands(energy_grid_MeV, flux_values):
    flux_values = np.clip(np.asarray(flux_values, dtype=float), 0.0, None)
    if flux_values.sum() <= 0:
        raise ValueError("Flux histogram is entirely zero -- check energy range/parameters.")

    lines = ["/gps/ene/type Arb", "/gps/hist/type arb"]
    for E, F in zip(energy_grid_MeV, flux_values):
        lines.append(f"/gps/hist/point {E:.6f} {F:.6e}")
    lines.append("/gps/ene/diffspec 1")
    return lines