from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict

# Dictionary of standard polymers for quick lookup
KNOWN_POLYMERS = {
    "HDPE": "C"*100,  # Polyethylene repeat unit representation
    "Polypropylene": "C(C)C"*50,
    "Polyimide": "O=C1NC(=O)c2ccc(Oc3ccc4C(=O)NC(=O)c4c3)cc21",
    "PKEK": "O=C1C=CC(=O)C=C1",  # Polyetherketoneketone baseline
}

def parse_smiles_composition(smiles_str: str) -> Dict[str, float]:
    """
    Parses a SMILES string using RDKit and calculates molecular weight
    and elemental mass fractions (Hydrogen, Carbon, Oxygen, Nitrogen, etc.).
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string provided: {smiles_str}")

    # Add explicit hydrogens to molecular structure
    mol_with_h = Chem.AddHs(mol)
    total_mw = Descriptors.MolWt(mol_with_h)

    element_masses = {}
    for atom in mol_with_h.GetAtoms():
        symbol = atom.GetSymbol()
        atomic_mass = atom.GetMass()
        element_masses[symbol] = element_masses.get(symbol, 0.0) + atomic_mass

    # Calculate mass fractions
    mass_fractions = {element: mass / total_mw for element, mass in element_masses.items()}
    
    return {
        "molecular_weight": total_mw,
        "hydrogen_fraction_wH": mass_fractions.get("H", 0.0),
        "carbon_fraction_wC": mass_fractions.get("C", 0.0),
        "oxygen_fraction_wO": mass_fractions.get("O", 0.0),
        "mass_fractions": mass_fractions
    }