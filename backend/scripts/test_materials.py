import os
import sys
import math

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rdkit_service import parse_smiles_composition, KNOWN_POLYMERS
from app.core.bicerano import calculate_bicerano_properties
from app.core.xcom_service import (
    calculate_zeff,
    calculate_mixture_attenuation_at_energy,
    get_element_mu_rho
)
from app.core.materials_project_client import fetch_reference_density

def test_materials_api_load():
    
    print("\n--- Testing Materials Project API Key Load ---")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("MATERIALS_PROJECT_API_KEY")

        if api_key and api_key != "your_actual_api_key_here":
            print("PASS: API Key successfully found in environment.")
            return True
        else:
            print("FAIL: API Key missing or left as default template. Check your .env file.")
            return False
    except ImportError:
        print("FAIL: python-dotenv is not installed. Run: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_hdpe_literature_baseline():
    print("\n--- Testing Cheminformatics & Bicerano (HDPE) ---")

    # Use the long-chain alkane approximation for HDPE
    hdpe_smiles = KNOWN_POLYMERS.get("HDPE", "C" * 100)

    try:
        # 1. Parse SMILES
        comp = parse_smiles_composition(hdpe_smiles)

        # 2. Calculate Properties using the Bragg additivity rule
        # 2. Calculate Properties using the Bragg additivity rule
        mass_fractions = {
            "H": comp["hydrogen_fraction_wH"],
            "C": comp["carbon_fraction_wC"],
            "O": comp["oxygen_fraction_wO"]
        }
        
        props = calculate_bicerano_properties(mass_fractions, is_polymer=True)

        # 3. Official Literature Baselines for HDPE
        LIT_WH = 0.1437
        LIT_DENSITY = 0.95  # g/cm^3
        LIT_I_EV = 57.4     # eV (ICRU-37 standard)

        # 4. Calculate Residuals
        wH_error = abs(props["hydrogen_mass_fraction"] - LIT_WH)
        density_error = abs(props["density_g_cm3"] - LIT_DENSITY)
        I_error = abs(props["mean_excitation_energy_ev"] - LIT_I_EV)

        wH_pct = (wH_error / LIT_WH) * 100
        density_pct = (density_error / LIT_DENSITY) * 100
        I_pct = (I_error / LIT_I_EV) * 100

        print(f"Hydrogen Fraction: {props['hydrogen_mass_fraction']:.4f} (Lit: {LIT_WH}) | Error: {wH_pct:.2f}%")
        print(f"Density (g/cm3):   {props['density_g_cm3']:.4f} (Lit: {LIT_DENSITY}) | Error: {density_pct:.2f}%")
        print(f"Mean Excitation:   {props['mean_excitation_energy_ev']:.2f} eV (Lit: {LIT_I_EV} eV) | Error: {I_pct:.2f}%")
        # NOTE: residual I-value error here comes from the Bragg additivity rule
        # not fully capturing molecular (chemical-bond) electron binding effects
        # vs. the free-atom values it's built from (see ICRU Report 37). This is
        # NOT the Sternheimer density-effect correction, which is an unrelated,
        # projectile-energy-dependent term in the stopping-power formula.

        checks = [
            ("Hydrogen fraction", wH_pct, 5.0),
            ("Density", density_pct, 5.0),
            ("Mean excitation energy", I_pct, 5.0),
        ]
        all_passed = True
        for name, pct, tol in checks:
            if pct >= tol:
                print(f"FAIL: {name} error ({pct:.2f}%) exceeds the {tol}% tolerance threshold!")
                all_passed = False

        if all_passed:
            print("PASS: All HDPE literature benchmarks within tolerance.")
        return all_passed

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_dynamic_xcom_engine():
    print("\n--- Testing Dynamic NIST XCOM & Z_eff Engine ---")
    all_passed = True

    try:
        # 1. The Lead (Pb) Sanity Check
        pb_mu = get_element_mu_rho("Pb", 1.0)
        print(f"Pb Attenuation at 1.0 MeV: {pb_mu:.5f} cm^2/g (Expected: ~0.07102)")
        if abs(pb_mu - 0.07102) >= 0.001:
            print("FAIL: Lead attenuation check failed!")
            all_passed = False

        # 2. Test HDPE Z_eff
        hdpe_mass_fractions = {"H": 0.1437, "C": 0.8563}
        z_eff = calculate_zeff(hdpe_mass_fractions)
        print(f"Calculated HDPE Z_eff:     {z_eff} (Expected: ~5.444)")
        # Verified: Zeff = (sum f_i * Z_i^m)^(1/m), m=2.94, electron-fraction
        # weighted (f_H=0.25, f_C=0.75) -> 5.4439 for HDPE. Confirmed
        # independently by direct computation, not just trusting the output.
        if abs(z_eff - 5.444) >= 0.01:
            print("FAIL: Z_eff calculation failed!")
            all_passed = False

        # 3. Test Dynamic Mixture Attenuation (HDPE at 662 keV / 0.662 MeV)
        hdpe_mu = calculate_mixture_attenuation_at_energy(hdpe_mass_fractions, 0.662)
        print(f"HDPE Attenuation at 662 keV: {hdpe_mu:.4f} cm^2/g")
        if not (hdpe_mu > 0):
            print("FAIL: Mixture attenuation returned 0!")
            all_passed = False

        if all_passed:
            print("PASS: Dynamic XCOM engine checks passed.")
        return all_passed

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_actinide_k_edge_patch():
    print("\n--- Testing Actinide K-Edge Patch ---")
    try:
        # Uranium K-edge is at 115.606 keV (0.115606 MeV)
        # The patch inserts the post-edge point at +0.1 eV (0.1156061 MeV)

        e_below = 0.115606
        e_above = 0.1156061

        mu_below = get_element_mu_rho("U", e_below)
        mu_above = get_element_mu_rho("U", e_above)

        jump_ratio = mu_above / mu_below

        print(f"Uranium mu/rho just BELOW edge ({e_below*1000:.3f} keV): {mu_below:.4f} cm^2/g")
        print(f"Uranium mu/rho just ABOVE edge ({e_above*1000:.4f} keV): {mu_above:.4f} cm^2/g")
        print(f"Observed discontinuity jump: {jump_ratio:.3f}x (expected ~3.95x, see build_database_final.py)")

        # Tightened from ">2.0" -- the expected total-mu jump for U is ~3.95x
        # (photoelectric-only jump ratio 4.74x, diluted by the non-jumping
        # coherent/incoherent/pair terms). A loose ">2.0" bound would pass
        # even if the patch regressed to roughly half its correct magnitude.
        if not (3.5 < jump_ratio < 4.4):
            print(f"FAIL: K-edge jump ratio {jump_ratio:.3f}x outside expected 3.5-4.4x band!")
            return False

        print("PASS: Actinide K-edge patch produces the expected discontinuity.")
        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False
def test_3d_density_integration():
    print("\n--- Testing 3D RDKit Density Integration ---")
    try:
        # Use a short polyethylene chain (Decane) for fast, reliable 3D embedding
        test_smiles = "CCCCCCCCCC" 
        comp = parse_smiles_composition(test_smiles)
        
        mass_fractions = {
            "H": comp["hydrogen_fraction_wH"],
            "C": comp["carbon_fraction_wC"],
            "O": comp["oxygen_fraction_wO"]
        }

        # 1. Tier 1: The Fast Heuristic (GAN Mode)
        fast_props = calculate_bicerano_properties(
            mass_fractions, 
            is_polymer=True, 
            use_3d_fidelity=False
        )
        
        # 2. Tier 2: The Exact 3D Engine (Geant4 Validation Mode)
        exact_props = calculate_bicerano_properties(
            mass_fractions, 
            is_polymer=True, 
            use_3d_fidelity=True, 
            smiles_string=test_smiles
        )

        fast_density = fast_props["density_g_cm3"]
        exact_density = exact_props["density_g_cm3"]

        print(f"Fast Heuristic Density:  {fast_density:.4f} g/cm^3")
        print(f"Exact 3D Render Density: {exact_density:.4f} g/cm^3")

        # Validation Check 1: Ensure the 3D density is physically realistic for a hydrocarbon (0.6 - 1.1 g/cm3)
        if not (0.6 < exact_density < 1.1):
            print(f"FAIL: 3D density {exact_density} is outside realistic physical bounds!")
            return False

        # Validation Check 2: Ensure the flag actually triggered the heavy calculation
        # (The numbers should be slightly different)
        if fast_density == exact_density:
            print("FAIL: The 3D flag did not change the density calculation.")
            return False

        print("PASS: 3D density calculator successfully integrated and produced physically valid bounds.")
        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False
def test_materials_project_api():
    print("\n--- Testing Materials Project API Client ---")
    try:
        # mp-149 is the Materials Project ID for elemental Silicon (Si)
        # We expect a density around 2.33 g/cm^3
        test_id = "mp-149"
        print(f"Fetching density for {test_id} (Silicon) over the network...")
        
        density = fetch_reference_density(test_id)
        
        if density is None:
            print("FAIL: API returned None. Check your internet connection, API key, and MPRester status.")
            return False
            
        print(f"Success! Retrieved Density: {density} g/cm^3")
        
        # Validation Check: Silicon's density should be approximately 2.33 g/cm^3
        if not (2.2 < density < 2.4):
            print(f"FAIL: Density {density} is outside expected bounds for Silicon!")
            return False
            
        print("PASS: Materials Project API integration is working flawlessly.")
        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False

    
if __name__ == "__main__":
    print("==================================================")
    print("AEGIS PHASE 1 CAPSTONE VALIDATION")
    print("==================================================")

    results = {
        "Materials API load": test_materials_api_load(),
        "HDPE literature baseline": test_hdpe_literature_baseline(),
        "Dynamic XCOM engine": test_dynamic_xcom_engine(),
        "Actinide K-edge patch": test_actinide_k_edge_patch(),
        "3D Density Integration": test_3d_density_integration(),
        "Materials Project API": test_materials_project_api(),
    }

    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        sys.exit(1)