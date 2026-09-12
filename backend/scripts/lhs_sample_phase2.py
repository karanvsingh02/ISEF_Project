import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.core.bicerano import calculate_bragg_properties


# ============================================================
# Tier 3 voxel-grid configuration
# ============================================================

TOPOLOGY_TIER = 3
TOPOLOGY_NAME = "tier3_voxel_grid"

VOXEL_NX = 64
VOXEL_NY = 64
VOXEL_NZ = 64

# Dedicated geometry/topology seed base.
# Geant4 must use this seed ONLY for geometry construction,
# never for event/physics randomness.
VOXEL_SEED_BASE = 20260907


def generate_lhs_design(n_samples: int = 1500):
    print(f"Generating {n_samples} Phase-2 (3D) Latin Hypercube design samples...")

    # ========================================================
    # LHS generation (Phase 2: 3D Design Space)
    # ========================================================

    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=n_samples)

    # Design variables:
    #   1. Shield thickness: 1.0–15.0 cm
    #   2. Regolith mass fraction: 0.0–0.70
    #   3. Incident Energy: 10.0–1000.0 MeV (GCR/SPE spectrum bounds)
    # Sample log10(Energy) from log10(10 MeV) = 1.0 to log10(10000 MeV) = 4.0
    l_bounds = [1.0, 0.0, 1.0]
    u_bounds = [15.0, 0.70, 4.0]

    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

    # ========================================================
    # LHS-1 XRF oxide composition
    # Normalized excluding reported LOI
    # Reference: Frontiers 2023 / Exolith Lab baseline
    # ========================================================

    raw_oxides = {
        "SiO2": 49.12,
        "TiO2": 0.63,
        "Al2O3": 26.29,
        "FeO": 3.20,
        "MnO": 0.06,
        "MgO": 2.86,
        "CaO": 13.52,
        "Na2O": 2.55,
        "K2O": 0.34,
        "P2O5": 0.17,
    }

    oxide_total = sum(raw_oxides.values())

    lhs1_oxides = {
        k: v / oxide_total
        for k, v in raw_oxides.items()
    }

    # ========================================================
    # Atomic masses (g/mol)
    # ========================================================

    am = {
        "O": 15.999,
        "Si": 28.085,
        "Ti": 47.867,
        "Al": 26.982,
        "Fe": 55.845,
        "Mn": 54.938,
        "Mg": 24.305,
        "Ca": 40.078,
        "Na": 22.990,
        "K": 39.098,
        "P": 30.974,
    }

    # ========================================================
    # Oxide molar masses
    # ========================================================

    ox_mass = {
        "SiO2": am["Si"] + 2 * am["O"],
        "TiO2": am["Ti"] + 2 * am["O"],
        "Al2O3": 2 * am["Al"] + 3 * am["O"],
        "FeO": am["Fe"] + am["O"],
        "MnO": am["Mn"] + am["O"],
        "MgO": am["Mg"] + am["O"],
        "CaO": am["Ca"] + am["O"],
        "Na2O": 2 * am["Na"] + am["O"],
        "K2O": 2 * am["K"] + am["O"],
        "P2O5": 2 * am["P"] + 5 * am["O"],
    }

    # ========================================================
    # Explicit stoichiometric oxygen counts
    # ========================================================

    oxygen_counts = {
        "SiO2": 2,
        "TiO2": 2,
        "Al2O3": 3,
        "FeO": 1,
        "MnO": 1,
        "MgO": 1,
        "CaO": 1,
        "Na2O": 1,
        "K2O": 1,
        "P2O5": 5,
    }

    # ========================================================
    # Elemental composition of normalized LHS-1
    # ========================================================

    lhs1_elements = [
        "Si",
        "Ti",
        "Al",
        "Fe",
        "Mn",
        "Mg",
        "Ca",
        "Na",
        "K",
        "P",
        "O",
    ]

    lhs1_elem = {
        el: 0.0
        for el in lhs1_elements
    }

    mapping = [
        ("SiO2", "Si", 1, am["Si"]),
        ("TiO2", "Ti", 1, am["Ti"]),
        ("Al2O3", "Al", 2, am["Al"]),
        ("FeO", "Fe", 1, am["Fe"]),
        ("MnO", "Mn", 1, am["Mn"]),
        ("MgO", "Mg", 1, am["Mg"]),
        ("CaO", "Ca", 1, am["Ca"]),
        ("Na2O", "Na", 2, am["Na"]),
        ("K2O", "K", 2, am["K"]),
        ("P2O5", "P", 2, am["P"]),
    ]

    for ox, el, count, mass in mapping:
        lhs1_elem[el] += (
            lhs1_oxides[ox]
            * (count * mass / ox_mass[ox])
        )

    for ox, wt_frac in lhs1_oxides.items():
        lhs1_elem["O"] += (
            wt_frac
            * (
                oxygen_counts[ox]
                * am["O"]
                / ox_mass[ox]
            )
        )

    lhs1_sum = sum(lhs1_elem.values())

    assert abs(lhs1_sum - 1.0) < 1e-10, (
        f"LHS-1 elemental fractions sum to {lhs1_sum}"
    )

    # ========================================================
    # Physical constants
    # ========================================================

    rho_hdpe = 0.95       # g/cm^3
    rho_regolith = 2.75   # g/cm^3

    # Exact HDPE stoichiometry: (C2H4)n
    mw_hdpe = (
        2 * 12.011
        + 4 * 1.008
    )

    # ========================================================
    # Dataset construction
    # ========================================================

    data = []

    for i, row in enumerate(scaled_samples):

        thickness_cm = float(row[0])
        w_regolith = float(row[1])
        log_energy = float(row[2])
        energy_mev = float(10.0 ** log_energy)  # Yields values spanning 10 MeV to 10,000 MeV  # <--- PHASE 2: Extracted Energy
        w_hdpe = 1.0 - w_regolith

        # ----------------------------------------------------
        # Mixture density
        # ----------------------------------------------------

        density_g_cm3 = 1.0 / (
            (w_hdpe / rho_hdpe)
            + (w_regolith / rho_regolith)
        )

        areal_density_g_cm2 = (
            density_g_cm3 * thickness_cm
        )

        # ----------------------------------------------------
        # Convert regolith MASS fraction -> VOLUME fraction
        # ----------------------------------------------------

        regolith_volume_fraction = (
            w_regolith / rho_regolith
        ) / (
            (w_regolith / rho_regolith)
            + (w_hdpe / rho_hdpe)
        )

        hdpe_volume_fraction = (
            1.0 - regolith_volume_fraction
        )

        # ----------------------------------------------------
        # Dedicated topology seed.
        # ----------------------------------------------------

        voxel_seed = (
            VOXEL_SEED_BASE + i
        )

        # ----------------------------------------------------
        # HDPE elemental mass fractions
        # ----------------------------------------------------

        h_hdpe = (
            w_hdpe
            * (4 * 1.008 / mw_hdpe)
        )

        c_hdpe = (
            w_hdpe
            * (2 * 12.011 / mw_hdpe)
        )

        # ----------------------------------------------------
        # Composite elemental mass fractions
        # ----------------------------------------------------

        mass_fractions = {
            "H": h_hdpe,
            "C": c_hdpe,
            "Si": w_regolith * lhs1_elem["Si"],
            "Ti": w_regolith * lhs1_elem["Ti"],
            "Al": w_regolith * lhs1_elem["Al"],
            "Fe": w_regolith * lhs1_elem["Fe"],
            "Mn": w_regolith * lhs1_elem["Mn"],
            "Mg": w_regolith * lhs1_elem["Mg"],
            "Ca": w_regolith * lhs1_elem["Ca"],
            "Na": w_regolith * lhs1_elem["Na"],
            "K": w_regolith * lhs1_elem["K"],
            "P": w_regolith * lhs1_elem["P"],
            "O": w_regolith * lhs1_elem["O"],
        }

        row_sum = sum(
            mass_fractions.values()
        )

        assert abs(row_sum - 1.0) < 1e-10, (
            f"Row {i} mass fractions sum to {row_sum}"
        )

        # ----------------------------------------------------
        # Stage-1 analytical approximation
        # ----------------------------------------------------

        props = calculate_bragg_properties(
            mass_fractions
        )

        # ----------------------------------------------------
        # Store complete design-point record
        # ----------------------------------------------------

        data.append({

            # -----------------------------------------------
            # Identity
            # -----------------------------------------------

            "run_id": i,

            # -----------------------------------------------
            # Tier-3 topology metadata
            # -----------------------------------------------

            "topology_tier": TOPOLOGY_TIER,
            "topology_name": TOPOLOGY_NAME,
            "topology_idx": TOPOLOGY_TIER,
            "voxel_seed": voxel_seed,
            "voxel_nx": VOXEL_NX,
            "voxel_ny": VOXEL_NY,
            "voxel_nz": VOXEL_NZ,
            "regolith_volume_fraction_target": regolith_volume_fraction,
            "hdpe_volume_fraction_target": hdpe_volume_fraction,

            # -----------------------------------------------
            # Primary design variables (Now 3D!)
            # -----------------------------------------------

            "thickness_cm": thickness_cm,
            "w_regolith": w_regolith,
            "w_hdpe": w_hdpe,
            "incident_energy_mev": energy_mev,  # <--- PHASE 2: Energy Stored

            # -----------------------------------------------
            # Derived physical quantities
            # -----------------------------------------------

            "density_g_cm3": density_g_cm3,
            "areal_density_g_cm2": areal_density_g_cm2,

            # -----------------------------------------------
            # Elemental mass fractions
            # -----------------------------------------------

            "H_mass_fraction": mass_fractions["H"],
            "C_mass_fraction": mass_fractions["C"],
            "O_mass_fraction": mass_fractions["O"],
            "Si_mass_fraction": mass_fractions["Si"],
            "Al_mass_fraction": mass_fractions["Al"],
            "Ca_mass_fraction": mass_fractions["Ca"],
            "Fe_mass_fraction": mass_fractions["Fe"],
            "Mg_mass_fraction": mass_fractions["Mg"],
            "Na_mass_fraction": mass_fractions["Na"],
            "K_mass_fraction": mass_fractions["K"],
            "Mn_mass_fraction": mass_fractions["Mn"],
            "P_mass_fraction": mass_fractions["P"],
            "Ti_mass_fraction": mass_fractions["Ti"],

            # -----------------------------------------------
            # Stage-1 analytical physics
            # -----------------------------------------------

            "mean_excitation_ev": props["mean_excitation_energy_ev"],
        })

    df = pd.DataFrame(data)

    # ========================================================
    # Dataset-level validation
    # ========================================================

    assert len(df) == n_samples
    assert df["run_id"].is_unique
    assert df["run_id"].min() == 0
    assert df["run_id"].max() == n_samples - 1

    # --------------------------------------------------------
    # Topology validation
    # --------------------------------------------------------
    assert df["topology_tier"].eq(TOPOLOGY_TIER).all()
    assert df["topology_idx"].eq(TOPOLOGY_TIER).all()
    assert df["voxel_seed"].is_unique

    # --------------------------------------------------------
    # Design-variable bounds (Added Energy Validation)
    # --------------------------------------------------------
    assert df["thickness_cm"].between(1.0, 15.0).all()
    assert df["w_regolith"].between(0.0, 0.70).all()
    assert df["w_hdpe"].between(0.30, 1.0).all()
    assert df["incident_energy_mev"].between(10.0, 10000.0).all()

    # --------------------------------------------------------
    # Explicit LHS stratum validation (Now 3D!)
    # --------------------------------------------------------
    u_thickness = (scaled_samples[:, 0] - 1.0) / (15.0 - 1.0)
    u_regolith = (scaled_samples[:, 1] / 0.70)
    u_energy = (scaled_samples[:, 2] - 1.0) / (4.0 - 1.0)

    thickness_strata = np.floor(u_thickness * n_samples).astype(int)
    regolith_strata = np.floor(u_regolith * n_samples).astype(int)
    energy_strata = np.floor(u_energy * n_samples).astype(int)

    assert np.array_equal(np.sort(thickness_strata), np.arange(n_samples)), "Thickness strata failed!"
    assert np.array_equal(np.sort(regolith_strata), np.arange(n_samples)), "Regolith strata failed!"
    assert np.array_equal(np.sort(energy_strata), np.arange(n_samples)), "Energy strata failed!"

    # ========================================================
    # Summary
    # ========================================================

    print("Phase 2 Dataset Summary Statistics:")
    print(
        df[
            [
                "thickness_cm",
                "w_regolith",
                "incident_energy_mev",
                "density_g_cm3",
                "areal_density_g_cm2",
                "regolith_volume_fraction_target",
            ]
        ].describe()
    )

    # ========================================================
    # Save (New Filename for Phase 2)
    # ========================================================

    output_path = os.path.join(
        os.path.dirname(__file__),
        "lhs_design_space_phase2.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        float_format="%.17g"
    )

    print(f"✅ Phase-2 (3D) LHS dataset saved safely to {output_path}")


if __name__ == "__main__":
    generate_lhs_design(1500)