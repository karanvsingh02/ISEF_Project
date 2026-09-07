import math
import json
import os
from typing import Dict

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)
ICRU_PATH = os.path.join(DATA_DIR, "icru37_data.json")

try:
    with open(ICRU_PATH, "r") as f:
        ICRU_DB = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find ICRU-37 database at {ICRU_PATH}."
    )


def calculate_bragg_properties(
    mass_fractions: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute the Bragg-additivity mean excitation energy (I-value)
    from elemental mass fractions using ICRU-37 reference data.

    Density is intentionally decoupled and must be calculated by
    the physical mixture model.
    """

    if not mass_fractions:
        raise ValueError("mass_fractions cannot be empty.")

    # Validate elemental mass fractions.
    total_mass_fraction = sum(mass_fractions.values())

    assert all(
        math.isfinite(w) and w >= 0.0
        for w in mass_fractions.values()
    ), "Mass fractions must be finite and non-negative."

    assert math.isclose(
        total_mass_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-10
    ), (
        f"Mass fractions must sum to 1.0, "
        f"got {total_mass_fraction}"
    )

    numerator = 0.0
    denominator = 0.0

    for symbol, w_i in mass_fractions.items():

        if w_i <= 0.0:
            continue

        if symbol not in ICRU_DB:
            raise KeyError(
                f"Element '{symbol}' is missing from ICRU database."
            )

        element_data = ICRU_DB[symbol]

        A_i = element_data["A"]
        Z_i = element_data["Z"]
        I_i = element_data["I_eV"]

        assert A_i > 0.0, f"Invalid atomic mass for {symbol}."
        assert Z_i > 0.0, f"Invalid atomic number for {symbol}."
        assert I_i > 0.0, f"Invalid mean excitation energy for {symbol}."

        weighting_factor = w_i * (Z_i / A_i)

        numerator += weighting_factor * math.log(I_i)
        denominator += weighting_factor

    assert denominator > 0.0, (
        "Bragg additivity denominator is zero or negative."
    )

    mean_excitation_energy_ev = math.exp(
        numerator / denominator
    )

    assert (
        math.isfinite(mean_excitation_energy_ev)
        and mean_excitation_energy_ev > 0.0
    ), "Invalid calculated mean excitation energy."

    return {
        "mean_excitation_energy_ev": mean_excitation_energy_ev,
        "hydrogen_mass_fraction": mass_fractions.get("H", 0.0),
    }