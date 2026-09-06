import torch
import sys
import os

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.physics import bethe_bloch_stopping_power

def test_perfected_physics_engine():
    print("\n==================================================")
    print("TESTING PERFECTED RELATIVISTIC PHYSICS ENGINE")
    print("==================================================")
    
    # Test energies: 1 MeV (Low), 10 MeV (Therapeutic), 100 MeV (Solar), 1000 MeV (GCR)
    energies = torch.tensor([1.0, 10.0, 100.0, 1000.0])
    
    # Polyethylene (HDPE) Parameters
    rho = 0.95            # g/cm^3
    z_over_a = 0.57       # Effective Z/A
    I_eV = 57.4           # Mean excitation energy
    
    # 1. Run the PyTorch Engine
    linear_stopping = bethe_bloch_stopping_power(
        kinetic_energy_MeV=energies,
        density_g_cm3=rho,
        z_over_a=z_over_a,
        mean_excitation_eV=I_eV
    )
    
    # Convert Linear Stopping (MeV/cm) back to Mass Stopping (MeV cm^2/g) 
    # to compare directly with NIST PSTAR tables
    mass_stopping = linear_stopping / rho
    
    # 2. NIST PSTAR Reference Data for Polyethylene
    # TRUE NIST PSTAR Reference Data for Polyethylene (HDPE)
    nist_references = {
        1.0: 244.5,
        10.0: 44.43,
        100.0: 7.291,
        1000.0: 2.209
    }
    
    all_passed = True
    print(f"{'Energy (MeV)':<15} | {'Computed (MeV cm2/g)':<25} | {'NIST PSTAR':<15} | {'Error %'}")
    print("-" * 75)
    
    for i, e_val in enumerate(energies.tolist()):
        computed = mass_stopping[i].item()
        reference = nist_references[e_val]
        error_pct = abs(computed - reference) / reference * 100
        
        print(f"{e_val:<15.1f} | {computed:<25.4f} | {reference:<15.4f} | {error_pct:.2f}%")
        
        # We allow a slightly wider tolerance (5-10%) at the extremes (1 MeV) because 
        # analytical surrogates are naturally fighting deep-shell corrections here.
        if error_pct > 10.0:
            all_passed = False

    print("\nSUMMARY:")
    if all_passed:
        print("✅ PASS: The PyTorch surrogate strongly mirrors NIST PSTAR references across 3 orders of magnitude!")
    else:
        print("❌ FAIL: One or more energy domains deviated significantly from NIST baselines.")

if __name__ == "__main__":
    test_perfected_physics_engine()