import math
from typing import Dict, Any

def calculate_bicerano_properties(w_H: float, w_C: float, w_O: float = 0.0) -> Dict[str, float]:
    """
    Estimates polymer density (rho g/cm^3) and effective parameters
    using empirical group-contribution approximations based on Bicerano relations.
    """
    # Empirical Bicerano correlation for aliphatic/aromatic hydrocarbon polymers:
    # High hydrogen mass fraction correlates directly with lower density and higher stopping power.
    estimated_density = 0.85 + (0.55 * w_H) - (0.10 * w_O)
    
    # Restrict density to realistic polymer bounds (0.85 to 1.65 g/cm^3)
    density_g_cm3 = max(0.85, min(1.65, estimated_density))
    
    # Mean Excitation Energy I (eV) using Bragg additivity rule approximation
    # I_H ~ 19.2 eV, I_C ~ 78.0 eV, I_O ~ 95.0 eV
    mean_excitation_energy_ev = (w_H * 19.2) + (w_C * 78.0) + (w_O * 95.0)

    return {
        "density_g_cm3": round(density_g_cm3, 4),
        "mean_excitation_energy_ev": round(mean_excitation_energy_ev, 2),
        "hydrogen_mass_fraction": round(w_H, 4)
    }