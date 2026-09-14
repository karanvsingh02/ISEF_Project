"""
Stage 2 of the PINN dataset pipeline.
Applies Gaussian Process (GP) empirical-Bayes shrinkage to the Geant4 data.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import concurrent.futures
from functools import partial

def gp_shrink(df, target_col, sem_col, n_restarts=5):
    # ---------------------------------------------------------
    # 1. LOG-ENERGY TRANSFORMATION
    # Ensures the GP kernel evaluates distances evenly across 
    # all orders of magnitude (10 MeV to 10 GeV).
    # ---------------------------------------------------------
    thickness = df["thickness_cm"].to_numpy(dtype=float)
    w_regolith = df["w_regolith"].to_numpy(dtype=float)
    log_energy = np.log10(df["incident_energy_mev"].to_numpy(dtype=float))
    
    X = np.column_stack([thickness, w_regolith, log_energy])
    y = df[target_col].to_numpy(dtype=float)
    mc_sem = df[sem_col].to_numpy(dtype=float)

    # Normalize inputs
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # Build and fit the GP
    kernel = C(1.0, (1e-2, 1e3)) * RBF(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e3))

    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts)
    gp.fit(X_norm, y)
    gp_mean, gp_std = gp.predict(X_norm, return_std=True)

    # Calculate variances
    mc_var = mc_sem ** 2
    gp_var = gp_std ** 2

    # ---------------------------------------------------------
    # SCALE THE SAFETY EPSILON TO THIS COLUMN'S OWN MAGNITUDE
    #
    # A fixed epsilon (the previous code used 1e-30) silently dominates
    # the precision-combination formula below whenever the column's
    # natural variance is smaller than that epsilon -- which is exactly
    # what happens for absorbed dose (values ~1e-14, so variances
    # ~1e-32) and for any exact-zero-interaction row (mc_var == 0 for a
    # column like neutron yield). When that happens, EVERY affected
    # row's combined variance collapses to ~epsilon regardless of its
    # real precision, so the "shrunk_sem" column stops being a genuine
    # per-row estimate and becomes an almost-constant floor instead
    # (this showed up as target_dose_..._shrunk_sem sitting around
    # 8e-16-1.4e-15 for nearly every row, and target_secondary_
    # neutrons_shrunk_sem being exactly sqrt(1e-30) = 1e-15 for every
    # zero-interaction row).
    #
    # Scaling epsilon to a tiny fraction of this column's own typical
    # variance keeps it acting purely as a division-by-zero guard
    # (needed for exact-zero mc_var rows) without ever swamping a real,
    # non-degenerate signal.
    # ---------------------------------------------------------
    positive_var = np.concatenate([mc_var[mc_var > 0], gp_var[gp_var > 0]])
    eps = np.median(positive_var) * 1e-6 if positive_var.size > 0 else 1e-30
    eps = max(eps, 1e-300)  # guard against a literal zero (e.g. an all-zero column)

    # Precision-weighted combination
    w_mc = gp_var / (mc_var + gp_var + eps)
    w_gp = 1.0 - w_mc

    shrunk = w_mc * y + w_gp * gp_mean
    shrunk_var = 1.0 / (1.0 / (mc_var + eps) + 1.0 / (gp_var + eps))

    # ---------------------------------------------------------
    # 2. NON-NEGATIVITY CONSTRAINT
    # Prevents unconstrained GP from outputting infinitesimal 
    # negative targets (e.g. -1e-18 dose) on zero-yield runs.
    # ---------------------------------------------------------
    shrunk = np.maximum(shrunk, 0.0)

    return shrunk, np.sqrt(shrunk_var), gp_mean, gp_std

def _process_single_target(df_copy, target_col, sem_col):
    """Wrapper function to execute GP shrink on a single core."""
    shrunk, shrunk_sem, gp_mean, gp_sem = gp_shrink(df_copy, target_col, sem_col)
    
    # Calculate improvement metric directly in the thread
    improvement = 1 - (shrunk_sem / df_copy[sem_col].replace(0, np.nan))
    median_improvement = np.nanmedian(improvement) * 100
    
    return target_col, shrunk, shrunk_sem, gp_mean, median_improvement

def refine_dataset(
    input_file="pinn_training_tensor_phase2.csv",
    output_file="pinn_training_tensor_phase2_refined.csv",
):
    print("🚀 Spawning parallel Gaussian Process threads for Monte Carlo smoothing...")
    df = pd.read_csv(input_file)

    targets = [
        ("target_dose_sv_per_particle", "sem_dose"),
        ("target_secondary_neutrons", "sem_neutrons"),
    ]

    # Map the isolated GP tasks to independent CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Pass a copy of the dataframe to avoid memory locking
        futures = {
            executor.submit(_process_single_target, df.copy(), t_col, s_col): t_col 
            for t_col, s_col in targets
        }
        
        for future in concurrent.futures.as_completed(futures):
            target_col, shrunk, shrunk_sem, gp_mean, med_imp = future.result()
            
            # Map threaded results back to main dataframe
            df[f"{target_col}_shrunk"] = shrunk
            df[f"{target_col}_shrunk_sem"] = shrunk_sem
            df[f"{target_col}_gp_only"] = gp_mean
            
            print(f"✅ [{target_col}] median SEM reduction from GP shrinkage: {med_imp:.1f}%")

    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved refined dataset -> {output_file}")
    print(
        "\nUse the *_shrunk columns as your PINN training targets, and "
        "*_shrunk_sem as your loss weights, in place of the raw data."
    )
    return df

if __name__ == "__main__":
    refine_dataset()
