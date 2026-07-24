import os
import sys
import csv
import random

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join("app", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "oltaris_tracks.csv")

def generate_oltaris_dataset(target_records: int = 5000):
    """
    Generates/structures the baseline dataset schema for OLTARIS space particle tracks.
    Includes Hydrogen Mass Fraction, Thickness, Incident Energy, and Transmitted Dose.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"🛰️ Initializing OLTARIS Space Radiation Batch Data Generation ({target_records} tracks)...")

    headers = ["thickness_cm", "energy_mev", "w_H", "density_g_cm3", "transmitted_dose_mgy"]
    
    records = []
    for _ in range(target_records):
        thickness = round(random.uniform(0.5, 20.0), 2)
        energy = round(random.uniform(10.0, 500.0), 2)
        w_H = round(random.uniform(0.02, 0.15), 4)
        density = round(0.85 + (0.55 * w_H), 4)
        
        # Approximate particle transport dose curve based on CSDA attenuation physics
        # Transmitted dose decreases exponentially with thickness and density
        attenuation_factor = max(0.0, 1.0 - (thickness / (0.08 * energy)))
        base_dose = energy * 0.15 * attenuation_factor
        transmitted_dose = round(max(0.0, base_dose + random.gauss(0, 0.2)), 4)
        
        records.append([thickness, energy, w_H, density, transmitted_dose])

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)

    print(f"✅ Successfully wrote {target_records} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_oltaris_dataset()