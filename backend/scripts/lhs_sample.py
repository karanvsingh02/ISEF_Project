import os
import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from app.core.bicerano import calculate_bragg_properties

def generate_lhs_design(n_samples: int = 1500):
    print(f"Generating {n_samples} Latin Hypercube design samples...")
    
    sampler = qmc.LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=n_samples)
    
    # Design variables:
    #   Shield thickness: 1.0–15.0 cm
    #   Regolith mass fraction: 0.0–0.70
    l_bounds = [1.0, 0.0]
    u_bounds = [15.0, 0.70]
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
    
    # LHS-1 XRF oxide composition, normalized excluding reported LOI
    # Reference: Frontiers 2023 / Exolith Lab baseline characterization table
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
        "P2O5": 0.17
    }
    oxide_total = sum(raw_oxides.values())
    lhs1_oxides = {k: v / oxide_total for k, v in raw_oxides.items()}
    
    # Atomic masses (g/mol) based on standard isotopic reference weights
    am = {
        "O": 15.999, "Si": 28.085, "Ti": 47.867, "Al": 26.982,
        "Fe": 55.845, "Mn": 54.938, "Mg": 24.305, "Ca": 40.078,
        "Na": 22.990, "K": 39.098, "P": 30.974
    }
    
    # Oxide molar masses
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
        "P2O5": 2 * am["P"] + 5 * am["O"]
    }
    
    # Explicit stoichiometric oxygen counts per oxide molecule
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
    
    # Pre-calculate elemental fractions per unit mass of normalized LHS-1
    lhs1_elem = {el: 0.0 for el in ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P", "O"]}
    
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
        ("P2O5", "P", 2, am["P"])
    ]
    
    for ox, el, count, mass in mapping:
        lhs1_elem[el] += lhs1_oxides[ox] * (count * mass / ox_mass[ox])
        
    for ox, wt_frac in lhs1_oxides.items():
        lhs1_elem["O"] += wt_frac * (oxygen_counts[ox] * am["O"] / ox_mass[ox])

    # Strict construction-time assertion for baseline elemental sum
    lhs1_sum = sum(lhs1_elem.values())
    assert abs(lhs1_sum - 1.0) < 1e-10, f"LHS-1 elemental fractions sum to {lhs1_sum}"

    print("\nNormalized LHS-1 elemental composition baseline:")
    for el, frac in lhs1_elem.items():
        print(f"  {el:>2}: {frac:.8f}")
    print(f"  Total: {lhs1_sum:.8f}\n")

    data = []
    
    for i, row in enumerate(scaled_samples):
        thickness_cm = row[0]
        w_regolith = row[1]
        w_hdpe = 1.0 - w_regolith
        
        # Zero-porosity constituent mixture model (HDPE: 0.95 g/cm³, LHS-1 grain density: 2.75 g/cm³)
        rho_hdpe = 0.95
        rho_regolith = 2.75
        density_g_cm3 = 1.0 / ((w_hdpe / rho_hdpe) + (w_regolith / rho_regolith))
        areal_density_g_cm2 = density_g_cm3 * thickness_cm
        
        # HDPE exact molar breakdown ((C2H4)n)
        mw_hdpe = 2 * 12.011 + 4 * 1.008
        h_hdpe = w_hdpe * (4 * 1.008 / mw_hdpe)
        c_hdpe = w_hdpe * (2 * 12.011 / mw_hdpe)
        
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
            "O": w_regolith * lhs1_elem["O"]
        }
        
        row_sum = sum(mass_fractions.values())
        assert abs(row_sum - 1.0) < 1e-10, f"Row {i} mass fractions sum to {row_sum}"
        
        # Updated clean function call
        props = calculate_bragg_properties(mass_fractions)
        
        data.append({
            "run_id": i,
            "topology_idx": 0,
            "thickness_cm": thickness_cm,
            "w_regolith": w_regolith,
            "w_hdpe": w_hdpe,
            "density_g_cm3": density_g_cm3,
            "areal_density_g_cm2": areal_density_g_cm2,
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
            "mean_excitation_ev": props["mean_excitation_energy_ev"]
        })
        
    df = pd.DataFrame(data)
    
    # Dataset-level audits and boundary assertions
    assert len(df) == n_samples
    assert df["run_id"].is_unique
    assert df["run_id"].min() == 0
    assert df["run_id"].max() == n_samples - 1
    
    element_cols = [c for c in df.columns if "mass_fraction" in c]
    element_sum = df[element_cols].sum(axis=1)
    assert (abs(element_sum - 1.0) < 1e-10).all(), "Dataset elemental mass fractions do not sum to 1.0!"
    
    assert df["thickness_cm"].between(1.0, 15.0).all()
    assert df["w_regolith"].between(0.0, 0.70).all()
    assert (df["density_g_cm3"] > 0).all()
    assert (df["areal_density_g_cm2"] > 0).all()
    assert np.isfinite(df["mean_excitation_ev"]).all()
    
    # Explicit Stratum-Level LHS Validation
    u_thickness = (scaled_samples[:, 0] - 1.0) / (15.0 - 1.0)
    u_regolith = scaled_samples[:, 1] / 0.70
    
    thickness_strata = np.floor(u_thickness * n_samples).astype(int)
    regolith_strata = np.floor(u_regolith * n_samples).astype(int)

    assert np.array_equal(np.sort(thickness_strata), np.arange(n_samples)), "Thickness LHS strata validation failed!"
    assert np.array_equal(np.sort(regolith_strata), np.arange(n_samples)), "Regolith LHS strata validation failed!"
    
    # Print summary statistics for physical sanity checking
    print("Dataset Summary Statistics:")
    print(df[[
        "thickness_cm",
        "w_regolith",
        "density_g_cm3",
        "areal_density_g_cm2",
        "mean_excitation_ev"
    ]].describe())
    print(f"\nMean excitation energy range: {df['mean_excitation_ev'].min():.3f} to {df['mean_excitation_ev'].max():.3f} eV\n")
    
    output_path = os.path.join(os.path.dirname(__file__), "lhs_design_space.csv")
    df.to_csv(output_path, index=False, float_format="%.10g")
    print(f"✅ Fully validated LHS design-space dataset saved to {output_path}")

if __name__ == "__main__":
    generate_lhs_design(1500)