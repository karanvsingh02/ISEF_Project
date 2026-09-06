from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict
from rdkit.Chem import AllChem


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

def calculate_true_3d_density(smiles: str) -> float:
    """
    Calculates the exact theoretical density of a polymer candidate by rendering 
    it in 3D space, calculating its Van der Waals volume, and applying a 
    macroscopic packing coefficient.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
        
    mol = Chem.AddHs(mol)
    
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status != 0:
        print(f"Warning: 3D Embedding failed for {smiles}. Falling back to standard polymer heuristic.")
        return 1.0000  # Safe generic polymer density to prevent divide-by-zero crashes
        
    AllChem.MMFFOptimizeMolecule(mol)
    
    volume_A3 = AllChem.ComputeMolVolume(mol)
    molar_mass = Descriptors.MolWt(mol)
    
    # Intrinsic Van der Waals density (100% packing)
    intrinsic_density = (molar_mass / volume_A3) * 1.660539
    
    # Macroscopic Packing Coefficient (Kitaigorodskii rule for polymers ~ 0.65)
    # This accounts for the free volume (empty space) between molecular chains
    PACKING_COEFFICIENT = 0.68
    macroscopic_density = intrinsic_density * PACKING_COEFFICIENT
    
    return round(macroscopic_density, 4)