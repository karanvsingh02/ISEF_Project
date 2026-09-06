import torch
import math
import json
import os
from typing import Optional

# Load Sternheimer DB
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STERNHEIMER_PATH = os.path.join(DATA_DIR, "sternheimer_data.json")

try:
    with open(STERNHEIMER_PATH, "r") as f:
        STERNHEIMER_DB = json.load(f)
except FileNotFoundError:
    print(f"Warning: Could not find {STERNHEIMER_PATH}.")
    STERNHEIMER_DB = {}

# Riemann zeta values for the exact Bloch (1933) correction series[cite: 3].
_ZETA3 = 1.2020569032
_ZETA5 = 1.0369277551
_ZETA7 = 1.0083492774


def bethe_bloch_stopping_power(
    kinetic_energy_MeV: torch.Tensor,
    density_g_cm3: float,
    z_over_a: float,
    mean_excitation_eV: float,
    projectile_charge: float = 1.0,
    projectile_mass_MeV: float = 938.27208816,
    target_Z: float = 6.0,   # Baseline default to Carbon (Polyethylene approximation)
    target_A: float = 12.011,
    material_name: str = "Polyethylene",
    effective_charge_model: str = "zbl",   # "barkas" or "zbl"
    barkas_L1_override: Optional[torch.Tensor] = None,
    low_energy_k: Optional[float] = None,  # Hook for SRIM-extracted CAB proportionality constant
) -> torch.Tensor:
    """
    Calculates total linear stopping power (-dE/dx) in MeV/cm for protons
    and heavy ions, achieving full relativistic correctness while seamlessly 
    blending low-energy velocity-proportional models.

    Incorporates:
    - Ziegler-Biersack-Littmark (1985) or Barkas effective charge fits[cite: 3].
    - General Sternheimer-Peierls (1971) density effect (insulator & conductor branches)[cite: 3].
    - FULL Barkas-Berger-Seltzer shell-correction polynomial[cite: 3].
    - Exact Bloch (1933) correction as a 3-term zeta-function series[cite: 3].
    - Dual Method: Smooth Sigmoid transition into the Lindhard-Scharff (1961) 
      velocity-proportional low-energy regime.
    - ZBL Universal Nuclear Stopping[cite: 3].
    """
    # Physical constants (NIST / PDG)
    m_e = 0.51099895          # Electron rest energy, MeV
    K = 0.307075              # MeV cm^2 / mol
    alpha = 1.0 / 137.035999  # Fine structure constant

    # 1. Enforce numerical stability floor (Allowed to go extremely low for LSS)
    T = torch.clamp(kinetic_energy_MeV, min=1e-4)

    # 2. Relativistic kinematics using total projectile mass
    gamma = 1.0 + (T / projectile_mass_MeV)
    beta2 = 1.0 - (1.0 / (gamma ** 2))
    beta2 = torch.clamp(beta2, min=1e-12)
    beta = torch.sqrt(beta2)

    # 3. Effective charge
    if effective_charge_model == "zbl":
        v_over_v0 = beta / alpha
        y_r = v_over_v0 / (projectile_charge ** (2.0 / 3.0))
        q_fraction = 1.0 - torch.exp(
            -0.803 * y_r ** 0.3 - 1.3167 * y_r ** 0.6 - 0.38157 * y_r - 0.008983 * y_r ** 2
        )
        z_eff = projectile_charge * q_fraction
    else:
        z_eff = projectile_charge * (1.0 - torch.exp(-125.0 * beta * (projectile_charge ** (-2.0 / 3.0))))

    # 4. Maximum kinetic energy transferable to an electron (T_max)
    mass_ratio = m_e / projectile_mass_MeV
    T_max = (2.0 * m_e * beta2 * (gamma ** 2)) / (
        1.0 + (2.0 * gamma * mass_ratio) + (mass_ratio ** 2)
    )

    # 5. Sternheimer Density Effect (Sternheimer-Peierls 1971 general formulation)
    X = torch.log10(beta * gamma)
    delta = torch.zeros_like(X)
    
    if material_name in STERNHEIMER_DB:
        params = STERNHEIMER_DB[material_name]
        C_stern = params["C"]
        X0 = params["X0"]
        X1 = params["X1"]
        a = params["a"]
        m = params["m"]
        is_conductor = params.get("conductor", False)
        delta0 = params.get("delta0", 0.0)

        delta = torch.where(X >= X1, 2.0 * math.log(10.0) * X + C_stern, delta)
        transition_term = 2.0 * math.log(10.0) * X + C_stern + a * ((X1 - X) ** m)
        delta = torch.where((X >= X0) & (X < X1), transition_term, delta)

        if is_conductor:
            conductor_term = delta0 * (10.0 ** (2.0 * (X - X0)))
            delta = torch.where(X < X0, conductor_term, delta)
    else:
        # Fallback to general Peierls asymptotic formula (Tensor-Compatible for PINN)
        plasma_eV = 28.816 * torch.sqrt(density_g_cm3 * z_over_a)
        C_stern = -2.0 * torch.log(mean_excitation_eV / plasma_eV) - 1.0
        X_a = -C_stern / (2.0 * math.log(10.0)) # math.log is fine here because 10.0 is a static float
        delta = torch.where(X > X_a, 2.0 * math.log(10.0) * X + C_stern, torch.zeros_like(X))

    # 6. Shell Correction (C/Z)
    eta = beta * gamma
    eta_clamped = torch.clamp(eta, min=0.13)   # this is the formula's real validity floor[cite: 3]
    I_eV = mean_excitation_eV
    bracket_I2 = (0.422377 / eta_clamped**2 + 0.0304043 / eta_clamped**4 - 0.00038106 / eta_clamped**6)
    bracket_I3 = (3.858019 / eta_clamped**2 - 0.1667989 / eta_clamped**4 + 0.00157955 / eta_clamped**6)
    C_over_Z = bracket_I2 * 1e-6 * (I_eV ** 2) + bracket_I3 * 1e-9 * (I_eV ** 3)

    # 7. Bloch and Barkas Corrections
    y = z_eff * alpha / beta
    y_clamped = torch.clamp(y, max=0.5) 
    L_bloch = -_ZETA3 * (y_clamped ** 2) + _ZETA5 * (y_clamped ** 4) - _ZETA7 * (y_clamped ** 6)

    if barkas_L1_override is not None:
        L_barkas = barkas_L1_override
    else:
        L_barkas = (0.00129 * z_eff) / torch.sqrt(beta)

    # 8. Core Electronic Stopping Number
    I_MeV = mean_excitation_eV * 1e-6
    log_arg = (2.0 * m_e * beta2 * (gamma ** 2) * T_max) / (I_MeV ** 2)
    stopping_number = (0.5 * torch.log(log_arg)) - beta2 - (0.5 * delta) - C_over_Z + L_barkas + L_bloch
    stopping_number = torch.clamp(stopping_number, min=1e-12)

    # --- HIGH-ENERGY BETHE-BLOCH REGIME ---
    S_bethe = (K * (z_eff ** 2) * z_over_a / beta2) * stopping_number

    # --- LOW-ENERGY LINDHARD-SCHARFF (LSS) REGIME ---
    # Velocity proportional regime: S_e = k * sqrt(T)
    if low_energy_k is not None:
        S_lss = low_energy_k * torch.sqrt(T)
    else:
        # Exact analytical LSS cross-section proportional to velocity (beta)
        xi_e = projectile_charge ** (1.0 / 6.0)
        Z_lss = (projectile_charge**(2.0/3.0) + target_Z**(2.0/3.0))**(1.5)
        v_over_v0 = beta / alpha
        
        # 8 * pi * e^2 * a_0 * N_A reduces flawlessly to exactly (2 * K / alpha^2)
        lss_constant = (2.0 * K) / (alpha ** 2) 
        S_lss = (lss_constant * xi_e * projectile_charge * target_Z / Z_lss) * v_over_v0 * (1.0 / target_A)

    # --- THE DIFFERENTIABLE PIECEWISE BLEND ---
    # Transition from Lindhard-Scharff to Bethe-Bloch at the proton Bragg Peak (~0.1 MeV/amu)
    # A steep Sigmoid ensures LSS cannot unphysically overshoot into the >1 MeV regime.
    T_per_amu = T / (projectile_mass_MeV / 931.5) 
    
    # Center at 0.1 MeV/amu, steepened slope (30.0) to strictly bound the low-energy regime
    W = torch.sigmoid(30.0 * (T_per_amu - 0.1))
    
    electronic_mass_stopping = (1.0 - W) * S_lss + W * S_bethe
    

    # 9. ZBL Universal Nuclear Stopping Power
    T_keV = T * 1000.0
    epsilon = (32.53 * target_A * T_keV) / (
        (projectile_mass_MeV/931.5 + target_A) * projectile_charge * target_Z *
        math.sqrt((projectile_charge ** (2.0/3.0)) + (target_Z ** (2.0/3.0)))
    )

    S_n_reduced = (0.5 * torch.log(1.0 + 1.1383 * epsilon)) / (
        epsilon + 0.01321 * (epsilon ** 0.21226) + 0.19593 * torch.sqrt(epsilon)
    )

    nuclear_conversion_factor = (8.462 * projectile_charge * target_Z * (projectile_mass_MeV/931.5)) / (
        (projectile_mass_MeV/931.5 + target_A) * math.sqrt((projectile_charge ** (2.0/3.0)) + (target_Z ** (2.0/3.0)))
    )
    nuclear_mass_stopping = (S_n_reduced * nuclear_conversion_factor) / 1000.0

    # 10. Total Linear Stopping Power (MeV / cm)
    total_mass_stopping = electronic_mass_stopping + nuclear_mass_stopping
    linear_stopping_power = total_mass_stopping * density_g_cm3

    return linear_stopping_power


def bohr_energy_loss_straggling(
    kinetic_energy_MeV: torch.Tensor,
    density_g_cm3: float,
    z_over_a: float,
    path_length_cm: float,
    projectile_charge: float = 1.0,
    projectile_mass_MeV: float = 938.27208816,
) -> torch.Tensor:
    """
    Bohr (1915) energy-loss straggling: the variance Omega^2 (MeV^2) of the
    stopping fluctuation distribution over `path_length_cm`[cite: 3].
    """
    m_e = 0.51099895
    K = 0.307075

    T = torch.clamp(kinetic_energy_MeV, min=1e-3)
    gamma = 1.0 + (T / projectile_mass_MeV)
    beta2 = torch.clamp(1.0 - (1.0 / (gamma ** 2)), min=1e-12)

    omega2_per_gcm2 = K * (projectile_charge ** 2) * z_over_a * m_e * (1.0 - beta2 / 2.0) / (1.0 - beta2)
    path_length_g_cm2 = path_length_cm * density_g_cm3

    return omega2_per_gcm2 * path_length_g_cm2