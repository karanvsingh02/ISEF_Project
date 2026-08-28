import math
from typing import Dict

# Standard Z and A values for HDPE and LHS-1 Lunar Regolith constituents
ELEMENTS = {
    "H":  {"Z": 1,  "A": 1.008},
    "C":  {"Z": 6,  "A": 12.011},
    "O":  {"Z": 8,  "A": 15.999},
    "Mg": {"Z": 12, "A": 24.305},
    "Al": {"Z": 13, "A": 26.982},
    "Si": {"Z": 14, "A": 28.085},
    "Ca": {"Z": 20, "A": 40.078},
    "Fe": {"Z": 26, "A": 55.845}
}

# Local Cache for NIST XCOM Mass Attenuation Coefficients (mu/rho in cm^2/g)
# Cached at 662 keV (Cs-137 gamma marker, common for physical testing)
NIST_XCOM_CACHE_662KEV = {
    "H":  0.1537,
    "C":  0.0772,
    "O":  0.0776,
    "Si": 0.0780,
    "Al": 0.0746,
    "Fe": 0.0732
}

def calculate_zeff(mass_fractions: Dict[str, float]) -> float:
    """
    Calculates the Effective Atomic Number (Z_eff) of a composite material
    using the Mayneord power-law equation for gamma/photon interactions.
    """
    # Step 1: Calculate the total electron denominator sum(w_i * Z_i / A_i)
    denominator = 0.0
    for element, w_i in mass_fractions.items():
        if element in ELEMENTS and w_i > 0:
            Z = ELEMENTS[element]["Z"]
            A = ELEMENTS[element]["A"]
            denominator += w_i * (Z / A)
            
    if denominator == 0:
        return 0.0

    # Step 2: Calculate fractional electron content (f_i) and sum for Z_eff
    z_eff_power_sum = 0.0
    for element, w_i in mass_fractions.items():
        if element in ELEMENTS and w_i > 0:
            Z = ELEMENTS[element]["Z"]
            A = ELEMENTS[element]["A"]
            
            f_i = (w_i * (Z / A)) / denominator
            z_eff_power_sum += f_i * math.pow(Z, 2.94)
            
    # Step 3: Take the 2.94 root
    z_eff = math.pow(z_eff_power_sum, 1 / 2.94)
    return round(z_eff, 4)

def calculate_mixture_attenuation(mass_fractions: Dict[str, float]) -> float:
    """
    Calculates the bulk mass attenuation coefficient (mu/rho) for the mixture
    at 662 keV using the standard rule of mixtures.
    """
    mu_rho_total = 0.0
    for element, w_i in mass_fractions.items():
        if element in NIST_XCOM_CACHE_662KEV:
            mu_rho_total += w_i * NIST_XCOM_CACHE_662KEV[element]
    return round(mu_rho_total, 4)