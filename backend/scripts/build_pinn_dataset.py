import os
import numpy as np
import pandas as pd

def build_macroscopic_dataset(
    lhs_file="lhs_design_space_phase2.csv",
    output_file="pinn_training_tensor_phase2.csv",
):
    print("🚀 Building macroscopic PINN dataset with 9-Column Hadronic Physics...")

    lhs_df = pd.read_csv(lhs_file)

    # UPDATED: Now mapped to your exact 9-column Geant4 output
    g4_columns = [
        "Incident_Energy_MeV",
        "Absorbed_Dose_Gy",
        "Dose_Equivalent_Sv",
        "Secondary_Neutrons",
        "Secondary_Gammas",
        "Transmitted_Primary_Count",
        "Secondary_Protons",
        "Secondary_Charged_Pions",
        "Secondary_Light_Ions"
    ]

    rows = []
    skipped = []

    for run_id in range(len(lhs_df)):
        # --- SPLIT FOLDER ROUTING ---
        # --- SPLIT FOLDER ROUTING ---
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder_name = "Part1" if run_id < 750 else "Part2"
        g4_file = os.path.join(base_dir, folder_name, f"output_run_{run_id}.csv")
        if not os.path.exists(g4_file):
            skipped.append(run_id)
            continue

        g4_df = pd.read_csv(g4_file, names=g4_columns)
        n = len(g4_df)

        dose = g4_df["Dose_Equivalent_Sv"].to_numpy()
        neutrons = g4_df["Secondary_Neutrons"].to_numpy()
        gammas = g4_df["Secondary_Gammas"].to_numpy()
        transmitted = g4_df["Transmitted_Primary_Count"].to_numpy()
        
        # New Spallation Fragments
        protons = g4_df["Secondary_Protons"].to_numpy()
        pions = g4_df["Secondary_Charged_Pions"].to_numpy()
        light_ions = g4_df["Secondary_Light_Ions"].to_numpy()

        # --- 1. Mean + standard error (for uncertainty-weighted loss) ---
        mean_dose = dose.mean()
        sem_dose = dose.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        mean_neutrons = neutrons.mean()
        sem_neutrons = neutrons.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        mean_gammas = gammas.mean()
        mean_protons = protons.mean()
        mean_pions = pions.mean()
        mean_light_ions = light_ions.mean()
        transmitted_ratio = transmitted.sum() / n

        # --- 2. Hurdle decomposition for the spike-dominated neutron channel ---
        interaction_mask = neutrons > 0
        p_interaction = interaction_mask.mean()
        mean_neutrons_given_event = (
            neutrons[interaction_mask].mean() if interaction_mask.any() else 0.0
        )

        # --- 3. Log-space diagnostics for the heavy-tailed dose channel ---
        positive_dose = dose[dose > 0]
        if len(positive_dose) > 0:
            log_mean_dose = np.log(positive_dose).mean()
            log_std_dose = np.log(positive_dose).std(ddof=1)
        else:
            log_mean_dose, log_std_dose = np.nan, np.nan

        # --- 4. Convergence diagnostics ---
        rel_sem_dose = sem_dose / mean_dose if mean_dose > 0 else np.nan
        rel_sem_neutrons = sem_neutrons / mean_neutrons if mean_neutrons > 0 else np.nan

        rows.append({
            "run_id": run_id,
            "thickness_cm": lhs_df.loc[run_id, "thickness_cm"],
            "w_regolith": lhs_df.loc[run_id, "w_regolith"],
            "incident_energy_mev": lhs_df.loc[run_id, "incident_energy_mev"],

            # Primary training targets
            "target_dose_sv_per_particle": mean_dose,
            "target_secondary_neutrons": mean_neutrons,
            "target_secondary_gammas": mean_gammas,
            "transmission_probability": transmitted_ratio,
            
            # New Spallation Targets
            "target_secondary_protons": mean_protons,
            "target_secondary_pions": mean_pions,
            "target_secondary_light_ions": mean_light_ions,

            # Uncertainty -> use as inverse-variance weights in the loss
            "sem_dose": sem_dose,
            "sem_neutrons": sem_neutrons,
            "rel_sem_dose": rel_sem_dose,
            "rel_sem_neutrons": rel_sem_neutrons,

            # Hurdle-model decomposition of the neutron channel
            "p_nuclear_interaction": p_interaction,
            "mean_neutrons_given_interaction": mean_neutrons_given_event,

            # Optional secondary targets / normalization diagnostics
            "log_mean_dose": log_mean_dose,
            "log_std_dose": log_std_dose,

            "n_events": n,
        })

    if skipped:
        print(f"⚠️ Warning: {len(skipped)} runs missing (e.g. {skipped[:5]}...)")

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_file, index=False)

    print(f"✅ Compiled {len(final_df)} macroscopic data points -> {output_file}")
    
    shaky = final_df[final_df["rel_sem_neutrons"] > 0.10]
    if len(shaky):
        print(
            f"⚠️ {len(shaky)} runs have >10% relative SEM on neutron yield — "
            f"ensure your training loss down-weights these via sem_neutrons before you trust them."
        )

    return final_df

if __name__ == "__main__":
    build_macroscopic_dataset()