import math
import json
import os
from typing import Dict

# ---------------------------------------------------------
# Load Dynamic XCOM Database from JSON
# ---------------------------------------------------------
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xcom_data.json")

def load_xcom_database() -> dict:
    """Loads the local master NIST XCOM JSON database."""
    if os.path.exists(DATA_FILE_PATH):
        with open(DATA_FILE_PATH, "r") as f:
            return json.load(f)["elements"]
    return {}

# Load database into memory once when the backend boots
XCOM_DB = load_xcom_database()

# ---------------------------------------------------------
# Log-Log Physics Interpolation
# ---------------------------------------------------------
def log_interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Performs log-log linear interpolation for radiation cross-sections."""
    if x <= x1: return y1
    if x >= x2: return y2
    
    log_x, log_x1, log_x2 = math.log(x), math.log(x1), math.log(x2)
    log_y1, log_y2 = math.log(y1), math.log(y2)
    
    slope = (log_y2 - log_y1) / (log_x2 - log_x1)
    return math.exp(log_y1 + slope * (log_x - log_x1))

def get_element_mu_rho(element: str, target_energy_mev: float) -> float:
    """Retrieves and interpolates mass attenuation coefficient dynamically from JSON."""
    if element not in XCOM_DB:
        return 0.075 # Safe fallback if an exotic element is missing
        
    grid = XCOM_DB[element]["grid"]
    
    # Boundary checks
    if target_energy_mev <= grid[0][0]: return grid[0][1]
    if target_energy_mev >= grid[-1][0]: return grid[-1][1]
        
    # Find the surrounding energy bins and interpolate
    for i in range(len(grid) - 1):
        e1, mu1 = grid[i]
        e2, mu2 = grid[i+1]
        if e1 <= target_energy_mev <= e2:
            return log_interpolate(target_energy_mev, e1, mu1, e2, mu2)
            
    return grid[-1][1]

# ---------------------------------------------------------
# Dynamic Z_eff and Mixture Calculators
# ---------------------------------------------------------
def calculate_zeff(mass_fractions: Dict[str, float]) -> float:
    """Calculates Effective Atomic Number (Z_eff) dynamically for any element set."""
    denominator = sum(
        w_i * (XCOM_DB[el]["Z"] / XCOM_DB[el]["A"]) 
        for el, w_i in mass_fractions.items() if el in XCOM_DB
    )
    if denominator == 0:
        return 0.0

    z_eff_power_sum = sum(
        ((w_i * (XCOM_DB[el]["Z"] / XCOM_DB[el]["A"])) / denominator) * math.pow(XCOM_DB[el]["Z"], 2.94)
        for el, w_i in mass_fractions.items() if el in XCOM_DB
    )
    return round(math.pow(z_eff_power_sum, 1 / 2.94), 4)

def calculate_mixture_attenuation_at_energy(mass_fractions: Dict[str, float], target_energy_mev: float) -> float:
    """Calculates bulk mass attenuation coefficient for any material blend at any energy."""
    mu_rho_total = 0.0
    for element, w_i in mass_fractions.items():
        if w_i > 0 and element in XCOM_DB:
            mu_rho_total += w_i * get_element_mu_rho(element, target_energy_mev)
    return round(mu_rho_total, 4)