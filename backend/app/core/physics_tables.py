import torch
import math
import json
import os
from typing import Dict

# 1. Load the Elemental Database
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NIST_PATH = os.path.join(DATA_DIR, "nist_elemental_pstar.json")

try:
    with open(NIST_PATH, "r") as f:
        NIST_DB = json.load(f)
        for element in NIST_DB:
            NIST_DB[element]["energy_MeV"] = torch.tensor(NIST_DB[element]["energy_MeV"], dtype=torch.float32)
            NIST_DB[element]["stopping_power"] = torch.tensor(NIST_DB[element]["stopping_power"], dtype=torch.float32)
except FileNotFoundError:
    print(f"Warning: Could not find {NIST_PATH}.")
    NIST_DB = {}

# 2. Elemental Physical Properties (Z, A, I-value, and now each element's own
# natural bulk density -- the density is needed for the new density-effect
# "phase correction" below). Source: ICRU-37 I-values as tabulated by NIST,
# physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html.
#
# NOTE on Carbon: NIST's Table 1 lists I(C, graphite) = 78.0 eV. This file
# keeps the original 81.0 eV (the ICRU "amorphous carbon" value) for
# backward compatibility -- swap it to 78.0 eV if your compounds are
# graphite-based rather than amorphous/polymeric carbon.
ELEMENTAL_PROPS = {
    "H":  {"Z": 1.0,  "A": 1.008,  "I_eV": 19.2,  "density_g_cm3": 8.375e-05},
    "C":  {"Z": 6.0,  "A": 12.011, "I_eV": 81.0,  "density_g_cm3": 1.700},
    "N":  {"Z": 7.0,  "A": 14.007, "I_eV": 82.0,  "density_g_cm3": 1.165e-03},
    "O":  {"Z": 8.0,  "A": 15.999, "I_eV": 95.0,  "density_g_cm3": 1.332e-03},
    "Na": {"Z": 11.0, "A": 22.990, "I_eV": 149.0, "density_g_cm3": 0.971},
    "Mg": {"Z": 12.0, "A": 24.305, "I_eV": 156.0, "density_g_cm3": 1.740},
    "Al": {"Z": 13.0, "A": 26.982, "I_eV": 166.0, "density_g_cm3": 2.699},
    "Si": {"Z": 14.0, "A": 28.085, "I_eV": 173.0, "density_g_cm3": 2.330},
    "P":  {"Z": 15.0, "A": 30.974, "I_eV": 173.0, "density_g_cm3": 2.200},
    "S":  {"Z": 16.0, "A": 32.06,  "I_eV": 180.0, "density_g_cm3": 2.000},
    "Cl": {"Z": 17.0, "A": 35.45,  "I_eV": 174.0, "density_g_cm3": 2.995e-03},
    "K":  {"Z": 19.0, "A": 39.098, "I_eV": 190.0, "density_g_cm3": 0.862},
    "Ca": {"Z": 20.0, "A": 40.078, "I_eV": 191.0, "density_g_cm3": 1.550},
    "Fe": {"Z": 26.0, "A": 55.845, "I_eV": 286.0, "density_g_cm3": 7.874},
}


def differentiable_1d_interp(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """Differentiable 1D linear interpolation in PyTorch."""
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]

    weight = (x - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def _peierls_density_effect(
    X: torch.Tensor, density_g_cm3, z_over_a, mean_excitation_eV
) -> torch.Tensor:
    """
    General (Sternheimer-Peierls 1971) asymptotic density-effect delta(X),
    X = log10(beta*gamma). Used here purely as a RELATIVE correction (compound
    vs Bragg-weighted elemental average) -- see the "phase/state correction"
    block in table_based_stopping_power(). The asymptotic form (rather than
    the full 3-region piecewise fit) is adequate for a difference term, since
    near-threshold behavior mostly cancels between the compound and the
    elemental estimate anyway.
    """
    density_t = torch.as_tensor(density_g_cm3, dtype=X.dtype, device=X.device)
    za_t = torch.as_tensor(z_over_a, dtype=X.dtype, device=X.device)
    mean_i_t = torch.as_tensor(mean_excitation_eV, dtype=X.dtype, device=X.device)

    valid = (density_t > 0) & (za_t > 0)
    safe_density = torch.clamp(density_t, min=1e-8)
    safe_za = torch.clamp(za_t, min=1e-8)
    
    plasma_eV = 28.816 * torch.sqrt(safe_density * safe_za)
    C_stern = -2.0 * torch.log(mean_i_t / plasma_eV) - 1.0
    X_a = -C_stern / (2.0 * math.log(10.0))
    res = torch.where(X > X_a, 2.0 * math.log(10.0) * X + C_stern, torch.zeros_like(X))
    
    if isinstance(valid, torch.Tensor) and valid.numel() > 1:
        res = torch.where(valid, res, torch.zeros_like(X))
    elif isinstance(valid, torch.Tensor) and not valid.item():
        res = torch.zeros_like(X)
    return res


def table_based_stopping_power(
    kinetic_energy_MeV: torch.Tensor,
    mass_fractions: Dict[str, torch.Tensor],
    density_g_cm3: torch.Tensor,
    mean_excitation_eV: torch.Tensor = None,
    projectile_mass_MeV: float = 938.27208816,
    apply_density_effect_correction: bool = True,
) -> torch.Tensor:
    """
    Calculates linear stopping power using NIST PSTAR tables and Bragg's Rule,
    with a relativistic hybrid correction for molecular binding energies, PLUS
    (new) a compound density-effect "phase/state" correction.

    Why the new correction matters: Bragg's rule sums NIST elemental stopping
    powers, and each of those tables already has that ELEMENT's own density
    effect baked in (e.g. gaseous H2/O2, solid graphite). That is only correct
    if the compound has the same bulk dielectric response as the weighted mix
    of pure elements -- which is false whenever the compound's actual
    density/phase differs from its constituents' natural states (e.g. a solid
    polymer built conceptually from H2/O2 gas + graphite). This correction
    replaces the implicit Bragg-weighted elemental density effect with the
    density effect computed for the compound as a whole, using its own real
    density and mean excitation energy.

    NOT implemented here (see companion notes for bethe_bloch_stopping_power.py
    for the full rationale): Ashley-Ritchie-Brandt Barkas-correction tables,
    Lindhard-Sorensen relativistic Bloch, Lindhard-Scharff low-energy regime,
    Brandt-Kitagawa effective charge, Vavilov/Landau straggling.
    """
    total_mass_stopping_power = torch.zeros_like(kinetic_energy_MeV, dtype=torch.float32)
    z_over_a_eff = torch.zeros_like(kinetic_energy_MeV, dtype=torch.float32)
    log_I_bragg_num = torch.zeros_like(kinetic_energy_MeV, dtype=torch.float32)

    # Apply Bragg's Additivity Rule: Sum(w_i * S_i)
    for element, fraction in mass_fractions.items():
        if element in NIST_DB:
            xp = NIST_DB[element]["energy_MeV"]
            yp = NIST_DB[element]["stopping_power"]

            element_stopping = differentiable_1d_interp(kinetic_energy_MeV, xp, yp)
            total_mass_stopping_power = total_mass_stopping_power + (fraction * element_stopping)

            if element in ELEMENTAL_PROPS and mean_excitation_eV is not None:
                Z_i = ELEMENTAL_PROPS[element]["Z"]
                A_i = ELEMENTAL_PROPS[element]["A"]
                I_i = ELEMENTAL_PROPS[element]["I_eV"]

                term = fraction * (Z_i / A_i)
                z_over_a_eff = z_over_a_eff + term
                log_I_bragg_num = log_I_bragg_num + term * math.log(I_i)

    # Apply Hybrid Molecular Binding Correction (if I_comp is provided)
    if mean_excitation_eV is not None:
        K = 0.307075
        
        # Avoid division by zero on z_over_a_eff
        z_safe_eff = torch.clamp(z_over_a_eff, min=1e-6)
        ln_I_bragg = log_I_bragg_num / z_safe_eff
        ln_I_comp = torch.log(torch.clamp(mean_excitation_eV, min=1.0))

        T = torch.clamp(kinetic_energy_MeV, min=0.1)
        gamma = 1.0 + (T / projectile_mass_MeV)
        beta2 = 1.0 - (1.0 / (gamma ** 2))
        beta2 = torch.clamp(beta2, min=1e-8)

        delta_S_mass = (K * z_over_a_eff / beta2) * (ln_I_bragg - ln_I_comp)
        dampener = 1.0 - torch.exp(-kinetic_energy_MeV / 2.0)
        total_mass_stopping_power = total_mass_stopping_power + (delta_S_mass * dampener)

        if apply_density_effect_correction:
            beta_gamma = torch.sqrt(torch.clamp(beta2, min=1e-12)) * gamma
            X = torch.log10(beta_gamma)

            delta_compound = _peierls_density_effect(X, density_g_cm3, z_over_a_eff, mean_excitation_eV)

            delta_bragg_weighted = torch.zeros_like(kinetic_energy_MeV)
            for element, fraction in mass_fractions.items():
                if element in ELEMENTAL_PROPS:
                    props = ELEMENTAL_PROPS[element]
                    z_over_a_i = props["Z"] / props["A"]
                    delta_i = _peierls_density_effect(
                        X, props["density_g_cm3"], z_over_a_i, props["I_eV"]
                    )
                    delta_bragg_weighted = delta_bragg_weighted + fraction * (z_over_a_i / z_safe_eff) * delta_i

            delta_state_correction = -(K * z_over_a_eff / beta2) * 0.5 * (delta_compound - delta_bragg_weighted)
            total_mass_stopping_power = total_mass_stopping_power + (delta_state_correction * dampener)

    linear_stopping_power = total_mass_stopping_power * density_g_cm3
    return linear_stopping_power