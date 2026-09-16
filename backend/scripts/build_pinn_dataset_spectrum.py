import os
import numpy as np
import pandas as pd


def build_spectrum_dataset(
    lhs_file="lhs_design_space_spectrum.csv",
    g4_output_dir="pinn_training_data_spectrum",
    output_file="pinn_training_tensor_spectrum.csv",
):
    """
    Same aggregation logic as build_macroscopic_dataset.py (mean + SEM
    per run, hurdle decomposition for neutrons, log-space dose
    diagnostics), applied to the GCR/SPE spectrum runs. The key
    difference: there is no incident_energy_mev to join per row --
    energy was sampled per-event from a spectrum inside Geant4, so the
    design-space columns joined back from the LHS file are
    'environment' and 'spe_severity' instead.
    """
    print("🚀 Building GCR/SPE spectrum PINN dataset...")

    lhs_df = pd.read_csv(lhs_file)

    g4_columns = [
        "Incident_Energy_MeV",
        "Absorbed_Dose_Gy",
        "Dose_Equivalent_Sv",
        "Secondary_Neutrons",
        "Secondary_Gammas",
        "Transmitted_Primary_Count",
    ]

    rows = []
    skipped = []

    for run_id in range(len(lhs_df)):
        g4_file = os.path.join(g4_output_dir, f"output_run_{run_id}.csv")
        if not os.path.exists(g4_file):
            skipped.append(run_id)
            continue

        g4_df = pd.read_csv(g4_file, names=g4_columns)
        n = len(g4_df)

        dose = g4_df["Dose_Equivalent_Sv"].to_numpy()
        neutrons = g4_df["Secondary_Neutrons"].to_numpy()
        gammas = g4_df["Secondary_Gammas"].to_numpy()
        transmitted = g4_df["Transmitted_Primary_Count"].to_numpy()
        primary_energy = g4_df["Incident_Energy_MeV"].to_numpy()

        mean_dose = dose.mean()
        sem_dose = dose.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        mean_neutrons = neutrons.mean()
        sem_neutrons = neutrons.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

        mean_gammas = gammas.mean()
        transmitted_ratio = transmitted.sum() / n

        interaction_mask = neutrons > 0
        p_interaction = interaction_mask.mean()
        mean_neutrons_given_event = (
            neutrons[interaction_mask].mean() if interaction_mask.any() else 0.0
        )

        positive_dose = dose[dose > 0]
        if len(positive_dose) > 0:
            log_mean_dose = np.log(positive_dose).mean()
            log_std_dose = np.log(positive_dose).std(ddof=1)
        else:
            log_mean_dose, log_std_dose = np.nan, np.nan

        rel_sem_dose = sem_dose / mean_dose if mean_dose > 0 else np.nan
        rel_sem_neutrons = sem_neutrons / mean_neutrons if mean_neutrons > 0 else np.nan

        rows.append({
            "run_id": run_id,
            "thickness_cm": lhs_df.loc[run_id, "thickness_cm"],
            "w_regolith": lhs_df.loc[run_id, "w_regolith"],

            # Design-space columns that REPLACE incident_energy_mev for
            # the spectrum dataset -- there is no single incident energy
            # per run anymore.
            "environment": lhs_df.loc[run_id, "environment"],
            "spe_severity": lhs_df.loc[run_id, "spe_severity"],

            # Diagnostic only: the mean/spread of the ACTUAL per-event
            # sampled energies for this run (useful as a sanity check
            # that the GPS spectrum sampling is behaving as expected --
            # not a model input).
            "mean_sampled_primary_energy_mev": primary_energy.mean(),
            "std_sampled_primary_energy_mev": primary_energy.std(ddof=1) if n > 1 else 0.0,

            "target_dose_sv_per_particle": mean_dose,
            "target_secondary_neutrons": mean_neutrons,
            "target_secondary_gammas": mean_gammas,
            "transmission_probability": transmitted_ratio,

            "sem_dose": sem_dose,
            "sem_neutrons": sem_neutrons,
            "rel_sem_dose": rel_sem_dose,
            "rel_sem_neutrons": rel_sem_neutrons,

            "p_nuclear_interaction": p_interaction,
            "mean_neutrons_given_interaction": mean_neutrons_given_event,

            "log_mean_dose": log_mean_dose,
            "log_std_dose": log_std_dose,

            "n_events": n,
        })

    if skipped:
        print(f"⚠️ Warning: {len(skipped)} runs missing (e.g. {skipped[:5]}...)")

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_file, index=False)

    print(f"✅ Compiled {len(final_df)} spectrum data points -> {output_file}")

    shaky = final_df[final_df["rel_sem_neutrons"] > 0.10]
    if len(shaky):
        print(f"⚠️ {len(shaky)} runs have >10% relative SEM on neutron yield.")

    # Sanity-check print: confirm the sampled energy distribution actually
    # looks like a spectrum (wide spread), not something that accidentally
    # collapsed back to a near-constant value per run.
    print("\nSampled primary energy sanity check (should show real spread, "
          "not near-zero std, especially for GCR rows):")
    print(final_df.groupby("environment")["std_sampled_primary_energy_mev"].describe())

    return final_df


if __name__ == "__main__":
    build_spectrum_dataset()