import math
import json
import os
from typing import Dict, Any

# Add this import at the top
from app.core.rdkit_service import calculate_true_3d_density 

# 1. Load the ICRU-37 database ONCE when the module is imported
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ICRU_PATH = os.path.join(DATA_DIR, "icru37_data.json")

try:
    with open(ICRU_PATH, "r") as f:
        ICRU_DB = json.load(f)
except FileNotFoundError:
    print(f"Warning: Could not find {ICRU_PATH}. Did you run build_icru_database.py?")
    ICRU_DB = {}

def calculate_bicerano_properties(
    mass_fractions: Dict[str, float], 
    is_polymer: bool = True,
    use_3d_fidelity: bool = False, # NEW FLAG
    smiles_string: str = None      # NEW PARAMETER
) -> Dict[str, float]:
    """
    Calculates bulk density and mean excitation energy.
    Can switch between a fast linear heuristic (for GAN loops) 
    and exact 3D Van der Waals volume (for Geant4 validation).
    """
    
    # 1. Density Estimation 
    density_g_cm3 = 0.0
    if is_polymer:
        if use_3d_fidelity and smiles_string:
            # Tier 2: The Validator (High-Fidelity)
            try:
                density_g_cm3 = calculate_true_3d_density(smiles_string)
                if density_g_cm3 == 0.0: # Fallback if 3D embedding fails
                     print("Falling back to linear heuristic.")
                     w_H = mass_fractions.get("H", 0.0)
                     w_O = mass_fractions.get("O", 0.0)
                     estimated_density = 0.85 + (0.55 * w_H) - (0.10 * w_O)
                     density_g_cm3 = max(0.85, min(1.65, estimated_density))
            except Exception as e:
                print(f"3D Density calculation failed: {e}. Falling back to heuristic.")
                w_H = mass_fractions.get("H", 0.0)
                w_O = mass_fractions.get("O", 0.0)
                estimated_density = 0.85 + (0.55 * w_H) - (0.10 * w_O)
                density_g_cm3 = max(0.85, min(1.65, estimated_density))

        else:
        # Dynamic solid mixture rule using standard constituent densities
            w_h = mass_fractions.get("H", 0.0)
            w_c = mass_fractions.get("C", 0.0)
            w_hdpe = w_h + w_c
            w_regolith = 1.0 - w_hdpe if w_hdpe < 1.0 else 0.0
            
            rho_hdpe = 0.95
            rho_regolith = 2.75
            
            if w_regolith > 0:
                density_g_cm3 = 1.0 / ((w_hdpe / rho_hdpe) + (w_regolith / rho_regolith))
            else:
                density_g_cm3 = 0.95
    else:
        # Default Lunar Regolith (LHS-1) bulk density
        density_g_cm3 = 1.30 

    # 2. Bragg Additivity Rule for Mean Excitation Energy
    numerator = 0.0
    denominator = 0.0

    for symbol, w_i in mass_fractions.items():
        if w_i <= 0:
            continue
            
        if symbol not in ICRU_DB:
            print(f"Warning: {symbol} not in ICRU DB. Skipping in I-value calculation.")
            continue
            
        element_data = ICRU_DB[symbol]
        A_i = element_data["A"]
        Z_i = element_data["Z"]
        I_i = element_data["I_eV"]
        
        weighting_factor = w_i * (Z_i / A_i)
        numerator += weighting_factor * math.log(I_i)
        denominator += weighting_factor

    mean_excitation_energy_ev = math.exp(numerator / denominator) if denominator > 0 else 0.0

    return {
        "density_g_cm3": round(density_g_cm3, 4),
        "mean_excitation_energy_ev": round(mean_excitation_energy_ev, 2),
        "hydrogen_mass_fraction": round(mass_fractions.get("H", 0.0), 4)
    }