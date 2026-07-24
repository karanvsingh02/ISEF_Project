import torch

def bethe_bloch_stopping_power(
    kinetic_energy_MeV: torch.Tensor,
    density_g_cm3: float = 1.0,
    z_over_a: float = 0.55,  # High hydrogen fraction (e.g., Polyethylene/Water)
    mean_excitation_eV: float = 57.4
) -> torch.Tensor:
    """
    Calculates mass stopping power -dE/dx (MeV cm^2/g) for protons using 
    the relativistic Bethe-Bloch formulation.
    """
    m_e = 0.511   # Electron rest mass in MeV
    m_p = 938.27  # Proton rest mass in MeV
    
    # Relativistic factor calculations
    gamma = 1.0 + (kinetic_energy_MeV / m_p)
    beta = torch.sqrt(1.0 - (1.0 / (gamma ** 2)))
    beta = torch.clamp(beta, min=1e-6)  # Guard against division by zero
    
    I_MeV = mean_excitation_eV * 1e-6
    
    # Maximum kinematic energy transfer in a single collision
    T_max = (2.0 * m_e * (beta ** 2) * (gamma ** 2)) / (
        1.0 + (2.0 * gamma * (m_e / m_p)) + ((m_e / m_p) ** 2)
    )
    
    K = 0.307075  # Constant in MeV cm^2 / mol
    
    stopping_power = (K * (z_over_a) / (beta ** 2)) * (
        0.5 * torch.log((2.0 * m_e * (beta ** 2) * (gamma ** 2) * T_max) / (I_MeV ** 2)) - (beta ** 2)
    )
    
    return stopping_power * density_g_cm3