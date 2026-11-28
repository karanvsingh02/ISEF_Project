import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spectrum_models import SPECIES, gcr_species_event_weights  # noqa: E402

# ============================================================
# Aggregate the per-event Geant4 output into one row per design point.
#
# ---------------------------------------------------------------------
# CHANGED: weighted aggregation for GCR
# ---------------------------------------------------------------------
# run_pinn_batch_spectrum.py no longer proportions events to abundance
# and pools them. It allocates events more evenly (importance sampling)
# and keeps one file per species, because proportional allocation gave
# iron ~7 events out of 10,000 at realistic abundances -- and iron
# contributes disproportionately to dose equivalent through its quality
# factor, so the most important term was the worst measured.
#
# The flux weights are therefore reapplied HERE:
#
#     mean  = sum_s  w_s * mean_s
#     sem   = sqrt( sum_s  w_s^2 * sem_s^2 )
#
# which is the standard weighted combination of independent estimates.
# Per-species contributions are also recorded, which answers a question
# worth putting on the poster directly: what fraction of dose equivalent
# comes from iron, against its fraction of the flux?
#
# ---------------------------------------------------------------------
# CHANGED: topology descriptors
# ---------------------------------------------------------------------
# areal_density_cv and z_centroid are joined in from
# topology_descriptors.csv. These are the two descriptors that survived
# validation against 88 Geant4 runs (r^2 = 0.97 against transmission and
# 0.98 against neutron yield, both robust to dropping the high-leverage
# channeled family). interface_density and anisotropy are carried through
# for completeness but should NOT be used as surrogate inputs -- see the
# comments in topology_generators.descriptors().
# ============================================================

G4_COLUMNS = [
    "Incident_Energy_MeV",
    "Absorbed_Dose_Gy",
    "Dose_Equivalent_Sv",
    "Secondary_Neutrons",
    "Secondary_Gammas",
    "Transmitted_Primary_Count",
    "Secondary_Protons",
    "Secondary_Charged_Pions",
    "Secondary_Light_Ions",
]

# Descriptor columns joined from topology_descriptors.csv. The first two
# are the validated surrogate inputs; the rest are diagnostics.
TOPOLOGY_COLUMNS = [
    "areal_density_cv", "z_centroid", "areal_density_p10",
    "topology_family", "topology_param_a", "topology_param_b",
    "interface_density", "z_slope",
]

# A run whose relative SEM exceeds this on dose is too noisy to train on
# cleanly. The previous version only checked neutrons, so a run at 6.5%
# relative error on DOSE -- the surrogate's primary target -- passed
# silently.
REL_SEM_DOSE_WARN = 0.05
REL_SEM_NEUTRON_WARN = 0.10


def _summarise(df):
    """Per-event statistics for one species file (or one SPE file)."""
    n = len(df)
    dose = df["Dose_Equivalent_Sv"].to_numpy()
    absorbed = df["Absorbed_Dose_Gy"].to_numpy()
    neutrons = df["Secondary_Neutrons"].to_numpy()

    def sem(a):
        return a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

    interaction = neutrons > 0
    positive = dose[dose > 0]

    return dict(
        n=n,
        dose=dose.mean(), dose_sem=sem(dose),
        absorbed=absorbed.mean(), absorbed_sem=sem(absorbed),
        neutrons=neutrons.mean(), neutrons_sem=sem(neutrons),
        gammas=df["Secondary_Gammas"].to_numpy().mean(),
        protons=df["Secondary_Protons"].to_numpy().mean(),
        pions=df["Secondary_Charged_Pions"].to_numpy().mean(),
        light_ions=df["Secondary_Light_Ions"].to_numpy().mean(),
        transmitted=df["Transmitted_Primary_Count"].to_numpy().mean(),
        primary_energy=df["Incident_Energy_MeV"].to_numpy().mean(),
        primary_energy_std=(df["Incident_Energy_MeV"].to_numpy().std(ddof=1)
                            if n > 1 else 0.0),
        p_interaction=interaction.mean(),
        neutrons_given_interaction=(neutrons[interaction].mean()
                                    if interaction.any() else 0.0),
        log_mean_dose=np.log(positive).mean() if len(positive) else np.nan,
        log_std_dose=(np.log(positive).std(ddof=1)
                      if len(positive) > 1 else np.nan),
    )


def _combine(parts, weights):
    """
    Weighted combination of independent per-species estimates.

    Means combine linearly. Standard errors combine in quadrature with
    the weights SQUARED, which is what makes the importance-sampling
    scheme pay off: a rare species gets a small w_s, so even a relatively
    noisy per-species estimate contributes little to the total error.
    """
    out = {}
    scalar_means = ["dose", "absorbed", "neutrons", "gammas", "protons",
                    "pions", "light_ions", "transmitted", "primary_energy",
                    "p_interaction", "neutrons_given_interaction"]
    for k in scalar_means:
        out[k] = sum(weights[s] * p[k] for s, p in parts.items())

    for k in ["dose_sem", "absorbed_sem", "neutrons_sem"]:
        out[k] = float(np.sqrt(sum((weights[s] ** 2) * (p[k] ** 2)
                                   for s, p in parts.items())))

    # Energy spread across the whole population: within-species variance
    # plus between-species variance of the means. Ignoring the second term
    # would badly understate the spread, since species means differ by
    # more than an order of magnitude (iron's grid is A = 56 times
    # protons').
    mu = out["primary_energy"]
    within = sum(weights[s] * p["primary_energy_std"] ** 2
                 for s, p in parts.items())
    between = sum(weights[s] * (p["primary_energy"] - mu) ** 2
                  for s, p in parts.items())
    out["primary_energy_std"] = float(np.sqrt(within + between))

    for k in ["log_mean_dose", "log_std_dose"]:
        vals = [(weights[s], p[k]) for s, p in parts.items()
                if not np.isnan(p[k])]
        wsum = sum(w for w, _ in vals)
        out[k] = sum(w * v for w, v in vals) / wsum if wsum > 0 else np.nan

    out["n"] = sum(p["n"] for p in parts.values())
    return out


def build_spectrum_dataset(
    lhs_file="lhs_design_space_spectrum.csv",
    topology_file="topology_descriptors.csv",
    g4_output_dir="pinn_training_data_spectrum",
    output_file="pinn_training_tensor_spectrum.csv",
):
    print("🚀 Building GCR/SPE spectrum PINN dataset...")

    lhs_df = pd.read_csv(lhs_file).set_index("run_id", drop=False)

    topo_df = None
    if os.path.exists(topology_file):
        topo_df = pd.read_csv(topology_file).set_index("run_id", drop=False)
        print(f"   joined topology descriptors from {topology_file}")
    else:
        print(f"⚠️  {topology_file} not found -- descriptor columns will be "
              f"absent. Run make_topology_maps.py.")

    weights_all = gcr_species_event_weights()

    rows, skipped, partial = [], [], []

    for run_id in lhs_df["run_id"].astype(int):
        lhs_row = lhs_df.loc[run_id]
        env = lhs_row["environment"]

        if env == "GCR":
            pattern = os.path.join(
                g4_output_dir, f"output_run_{run_id}_species_*.csv")
            files = sorted(glob.glob(pattern))
            if not files:
                skipped.append(run_id)
                continue

            parts = {}
            for f in files:
                species = os.path.basename(f).rsplit("_species_", 1)[1][:-4]
                if species not in SPECIES:
                    continue
                parts[species] = _summarise(
                    pd.read_csv(f, names=G4_COLUMNS))

            if not parts:
                skipped.append(run_id)
                continue
            if len(parts) < len(SPECIES):
                partial.append((run_id, sorted(set(SPECIES) - set(parts))))

            # Renormalise over the species actually present, so a missing
            # sub-job biases the result as little as possible rather than
            # silently dropping that share of the flux.
            w = {s: weights_all[s] for s in parts}
            tot = sum(w.values())
            w = {s: v / tot for s, v in w.items()}

            agg = _combine(parts, w)
            species_share = {
                f"dose_frac_{s}": w[s] * p["dose"] / agg["dose"]
                if agg["dose"] > 0 else np.nan
                for s, p in parts.items()
            }
            flux_share = {f"flux_frac_{s}": w[s] for s in parts}

        else:
            f = os.path.join(g4_output_dir, f"output_run_{run_id}.csv")
            if not os.path.exists(f):
                skipped.append(run_id)
                continue
            agg = _summarise(pd.read_csv(f, names=G4_COLUMNS))
            species_share, flux_share = {}, {}

        rel_sem_dose = (agg["dose_sem"] / agg["dose"]
                        if agg["dose"] > 0 else np.nan)
        rel_sem_neut = (agg["neutrons_sem"] / agg["neutrons"]
                        if agg["neutrons"] > 0 else np.nan)

        rec = {
            "run_id": run_id,
            "thickness_cm": lhs_row["thickness_cm"],
            "w_regolith": lhs_row["w_regolith"],
            "regolith_volume_fraction_target":
                lhs_row["regolith_volume_fraction_target"],
            "environment": env,
            "spe_severity": lhs_row["spe_severity"],

            "mean_sampled_primary_energy_mev": agg["primary_energy"],
            "std_sampled_primary_energy_mev": agg["primary_energy_std"],

            "target_dose_sv_per_particle": agg["dose"],
            "target_absorbed_dose_gy_per_particle": agg["absorbed"],
            "target_secondary_neutrons": agg["neutrons"],
            "target_secondary_gammas": agg["gammas"],
            "target_secondary_protons": agg["protons"],
            "target_secondary_pions": agg["pions"],
            "target_secondary_light_ions": agg["light_ions"],
            "transmission_probability": agg["transmitted"],

            "sem_dose": agg["dose_sem"],
            "sem_neutrons": agg["neutrons_sem"],
            "rel_sem_dose": rel_sem_dose,
            "rel_sem_neutrons": rel_sem_neut,

            "p_nuclear_interaction": agg["p_interaction"],
            "mean_neutrons_given_interaction": agg["neutrons_given_interaction"],
            "log_mean_dose": agg["log_mean_dose"],
            "log_std_dose": agg["log_std_dose"],

            "n_events": agg["n"],
        }
        rec.update(species_share)
        rec.update(flux_share)

        if topo_df is not None and run_id in topo_df.index:
            for c in TOPOLOGY_COLUMNS:
                if c in topo_df.columns:
                    rec[c] = topo_df.loc[run_id, c]

        rows.append(rec)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_file, index=False)

    print(f"✅ Compiled {len(final_df)} spectrum data points -> {output_file}")

    if skipped:
        print(f"⚠️  {len(skipped)} runs missing entirely "
              f"(e.g. {skipped[:5]}...)")
    if partial:
        print(f"⚠️  {len(partial)} GCR runs missing some species sub-jobs; "
              f"weights were renormalised over what was present. "
              f"First few: {partial[:3]}")

    noisy_d = final_df[final_df["rel_sem_dose"] > REL_SEM_DOSE_WARN]
    noisy_n = final_df[final_df["rel_sem_neutrons"] > REL_SEM_NEUTRON_WARN]
    print(f"\nStatistical quality")
    print(f"  rel SEM on dose     > {REL_SEM_DOSE_WARN:.0%}: "
          f"{len(noisy_d)} runs  (median {final_df['rel_sem_dose'].median():.4f})")
    print(f"  rel SEM on neutrons > {REL_SEM_NEUTRON_WARN:.0%}: "
          f"{len(noisy_n)} runs  (median {final_df['rel_sem_neutrons'].median():.4f})")
    if len(noisy_d):
        print("  Dose is the surrogate's primary target, so these cap how "
              "well any\n  model can fit. Raise BEAM_ON_EVENTS if the count "
              "is large.")

    dose_frac_cols = [c for c in final_df.columns if c.startswith("dose_frac_")]
    if dose_frac_cols:
        print("\nDose-equivalent contribution vs flux share (GCR rows)")
        gcr = final_df[final_df.environment == "GCR"]
        for c in sorted(dose_frac_cols):
            s = c.replace("dose_frac_", "")
            fc = f"flux_frac_{s}"
            if fc in gcr.columns:
                print(f"  {s:<10} dose {gcr[c].mean():>7.3%}   "
                      f"flux {gcr[fc].mean():>8.4%}   "
                      f"enhancement {gcr[c].mean()/max(gcr[fc].mean(),1e-12):>6.1f}x")
        print("  Enhancement >> 1 means that species punches far above its "
              "abundance,\n  which is exactly why it needed more events than "
              "its flux share.")

    print("\nSampled primary energy by environment:")
    print(final_df.groupby("environment")["std_sampled_primary_energy_mev"]
                  .describe())

    return final_df


if __name__ == "__main__":
    build_spectrum_dataset()