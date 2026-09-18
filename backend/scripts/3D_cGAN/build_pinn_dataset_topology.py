import os
import numpy as np
import pandas as pd

from topology_io import read_topology_file

GRID_SIZE = 64  # must match DetectorConstruction.cc / lhs_sample_topology.py


def build_topology_dataset(
    lhs_file="lhs_design_space_topology.csv",
    g4_output_dir="pinn_training_data_topology",
    topology_file_dir="topology_files",
    output_file="pinn_training_tensor_topology.csv",
):
    """
    Same aggregation logic as build_macroscopic_dataset.py / build_pinn_
    dataset_spectrum.py (mean + SEM per run, hurdle decomposition for
    neutrons, log-space dose diagnostics), applied to the topology runs.
    Design-space columns joined per row: topology_family, family_param,
    and the ACHIEVED regolith volume fraction (recomputed directly from
    the saved topology file, not just the LHS target -- see
    run_pinn_batch_topology.py's note on target vs. achieved fraction).
    """
    print("🚀 Building topology PINN dataset...")

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
        topo_file = os.path.join(topology_file_dir, f"topology_{run_id}.txt")
        if not os.path.exists(g4_file) or not os.path.exists(topo_file):
            skipped.append(run_id)
            continue

        g4_df = pd.read_csv(g4_file, names=g4_columns)
        n = len(g4_df)

        dose = g4_df["Dose_Equivalent_Sv"].to_numpy()
        neutrons = g4_df["Secondary_Neutrons"].to_numpy()
        gammas = g4_df["Secondary_Gammas"].to_numpy()
        transmitted = g4_df["Transmitted_Primary_Count"].to_numpy()

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

        topo_vol = read_topology_file(topo_file, GRID_SIZE, GRID_SIZE, GRID_SIZE)
        achieved_fraction = float(topo_vol.mean())

        rows.append({
            "run_id": run_id,
            "thickness_cm": lhs_df.loc[run_id, "thickness_cm"],
            "w_regolith_target": lhs_df.loc[run_id, "w_regolith"],
            "w_regolith_achieved": achieved_fraction,

            "topology_family": lhs_df.loc[run_id, "topology_family"],
            "family_param": lhs_df.loc[run_id, "family_param"],

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

    print(f"✅ Compiled {len(final_df)} topology data points -> {output_file}")

    if len(final_df) > 0:
        drift = (final_df["w_regolith_achieved"] - final_df["w_regolith_target"]).abs()
        print(f"\nTarget vs. achieved volume fraction drift: mean={drift.mean():.3f}, max={drift.max():.3f}")
        print("(Expected to be near-zero for random/clustered/core_shell; the 'layered'")
        print(" family can show larger drift at low n_layers due to whole-layer quantization --")
        print(" not a bug, just that family's inherent resolution limit.)")
        print(final_df.groupby("topology_family")[["w_regolith_target", "w_regolith_achieved"]].apply(
            lambda g: (g["w_regolith_achieved"] - g["w_regolith_target"]).abs().mean()
        ))

    return final_df


if __name__ == "__main__":
    build_topology_dataset()