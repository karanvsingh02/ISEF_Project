import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rdkit_service import parse_smiles_composition, KNOWN_POLYMERS
from app.core.bicerano import calculate_bicerano_properties

def test_material_engine():
    print("🧪 Running Task 2.4 Cheminformatics & Bicerano Verification...\n")
    
    for name, smiles in KNOWN_POLYMERS.items():
        try:
            comp = parse_smiles_composition(smiles)
            props = calculate_bicerano_properties(
                comp["hydrogen_fraction_wH"], 
                comp["carbon_fraction_wC"],
                comp["oxygen_fraction_wO"]
            )
            print(f"🔹 Polymer: {name}")
            print(f"   SMILES: {smiles[:30]}...")
            print(f"   Hydrogen Mass Fraction (w_H): {props['hydrogen_mass_fraction'] * 100:.2f}%")
            print(f"   Estimated Density (rho):       {props['density_g_cm3']} g/cm³")
            print(f"   Mean Excitation Energy (I):   {props['mean_excitation_energy_ev']} eV\n")
        except Exception as e:
            print(f"❌ Error processing {name}: {e}\n")

if __name__ == "__main__":
    test_material_engine()