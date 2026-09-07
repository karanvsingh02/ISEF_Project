import pandas as pd
import subprocess
import os

def run_geant4_batch(csv_path="lhs_design_space.csv", geant4_exe="./ShieldSim"):
    df = pd.read_csv(csv_path)
    
    # Ensure output directory exists
    os.makedirs("pinn_training_data", exist_ok=True)
    
    for index, row in df.iterrows():
        run_id = int(row["run_id"])
        mac_filename = f"run_{run_id}.mac"
        
        # Write the Geant4 macro for this specific LHS sample
        with open(mac_filename, "w") as f:
            f.write("/control/verbose 0\n")
            f.write("/run/verbose 0\n")
            
            # Pass Geometry & Density
            f.write(f"/shield/setThickness {row['thickness_cm']} cm\n")
            f.write(f"/shield/setDensity {row['density_g_cm3']} g/cm3\n")
            
            # Pass Elemental Mass Fractions
            f.write(f"/shield/fracH {row['H_mass_fraction']}\n")
            f.write(f"/shield/fracC {row['C_mass_fraction']}\n")
            f.write(f"/shield/fracO {row['O_mass_fraction']}\n")
            f.write(f"/shield/fracSi {row['Si_mass_fraction']}\n")
            f.write(f"/shield/fracAl {row['Al_mass_fraction']}\n")
            f.write(f"/shield/fracCa {row['Ca_mass_fraction']}\n")
            f.write(f"/shield/fracFe {row['Fe_mass_fraction']}\n")
            f.write(f"/shield/fracMg {row['Mg_mass_fraction']}\n")
            f.write(f"/shield/fracNa {row['Na_mass_fraction']}\n")
            f.write(f"/shield/fracK {row['K_mass_fraction']}\n")
            f.write(f"/shield/fracMn {row['Mn_mass_fraction']}\n")
            f.write(f"/shield/fracP {row['P_mass_fraction']}\n")
            f.write(f"/shield/fracTi {row['Ti_mass_fraction']}\n")
            
            # Update geometry and run 10,000 primary particles
            f.write("/run/initialize\n")
            f.write(f"/analysis/setFileName pinn_training_data/output_run_{run_id}.csv\n")
            f.write("/run/beamOn 10000\n")
            
        print(f"🚀 Executing Geant4 for Run ID {run_id} (Thickness: {row['thickness_cm']} cm)...")
        subprocess.run([geant4_exe, mac_filename], stdout=subprocess.DEVNULL)
        
        # Clean up macro to save space
        os.remove(mac_filename)

if __name__ == "__main__":
    run_geant4_batch()