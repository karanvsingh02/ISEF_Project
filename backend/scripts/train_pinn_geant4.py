import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.models.pinn import ShieldingPINN

RANDOM_STATE = 42   # controls the train/test split
NN_SEED = 42        # controls weight init + training stochasticity
TEST_SIZE = 0.2
DOSE_SCALAR = 1e14
N_STRATA_BINS = 3   # quantile bins per input dimension for the design-aware split

# Physically-motivated dosimetric floor (Sv/particle). See prior notes:
# rows below this get their dose TARGET clamped to the floor (not
# excluded), so the network learns a concrete answer for "very small
# dose" instead of extrapolating into an untrained regime. Keep in sync
# with evaluate_pinn.py.
DOSE_FLOOR_SV = 1e-20


def set_seeds(seed):
    """Make weight initialization and training stochasticity reproducible
    run-to-run. The train/test split was already deterministic
    (random_state=42); this covers everything else."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_design_stratification_labels(df, n_bins=N_STRATA_BINS):
    """
    LHS already gives good coverage of the input design space across the
    full dataset. A plain random 80/20 split can still, by chance, pull a
    disproportionate share of any one region (e.g. thin-shielding or
    low-energy rows) into the test set, leaving the training set with an
    uneven picture of the space it's meant to cover. Binning each input
    dimension into quantile groups and stratifying the split on the
    combination keeps BOTH the training and test sets representative
    "sub-designs" of the original space, rather than lumpy random draws.
    """
    log_energy = np.log10(df["incident_energy_mev"].values)
    b_thick = pd.qcut(df["thickness_cm"], n_bins, labels=False, duplicates="drop")
    b_wreg = pd.qcut(df["w_regolith"], n_bins, labels=False, duplicates="drop")
    b_energy = pd.qcut(pd.Series(log_energy), n_bins, labels=False, duplicates="drop")
    # Combine into a single integer label (safe as long as n_bins <= 10 per axis).
    return (b_thick.values * 100) + (b_wreg.values * 10) + b_energy.values


def split_train_test(df):
    strata = build_design_stratification_labels(df)
    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
        )
    except ValueError as e:
        # Can happen if some stratum ends up with too few rows to split at all.
        print(f"⚠️  Design-aware stratified split failed ({e}); falling back to a plain random split.")
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    return idx_train, idx_test


def train_pinn():
    set_seeds(NN_SEED)
    print("🚀 Initializing Phase 2 Balanced Multi-Task PINN Training...")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    idx_train, idx_test = split_train_test(df)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    print(f"📐 Training on {len(df_train)} / {len(df)} rows "
          f"({len(idx_test)} held out for evaluation, design-aware split)")

    # 1. Inputs
    X = np.column_stack([
        df_train["thickness_cm"].values,
        df_train["w_regolith"].values,
        np.log10(df_train["incident_energy_mev"].values)
    ])

    # 2. Targets
    Y_dose_lin = df_train["target_dose_sv_per_particle_shrunk"].values * DOSE_SCALAR
    Y_neutrons = df_train["target_secondary_neutrons_shrunk"].values
    Y_protons = df_train["target_secondary_protons"].values
    Y_pions = df_train["target_secondary_pions"].values
    Y_ions = df_train["target_secondary_light_ions"].values

    # DOSE FLOOR: clamp rather than exclude (see prior notes).
    dose_floor_scaled = DOSE_FLOOR_SV * DOSE_SCALAR
    n_floored = int((Y_dose_lin < dose_floor_scaled).sum())
    if n_floored > 0:
        print(f"ℹ️  {n_floored} training rows have dose below the {DOSE_FLOOR_SV:.0e} Sv/particle "
              f"floor — their target is clamped to the floor (not excluded).")

    Y_dose_lin_clamped = np.maximum(Y_dose_lin, dose_floor_scaled)
    Y_dose = np.log10(Y_dose_lin_clamped)

    # 3. Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    Y_target = torch.tensor(np.column_stack([Y_dose, Y_neutrons, Y_protons, Y_pions, Y_ions]), dtype=torch.float32)

    # Per-channel variance scale so all 5 tasks contribute on a comparable footing.
    dose_var = np.var(Y_dose)
    hadron_vars = np.var(np.column_stack([Y_neutrons, Y_protons, Y_pions, Y_ions]), axis=0)
    target_variances = torch.tensor(np.concatenate([[dose_var], hadron_vars]) + 1e-6, dtype=torch.float32)

    # 4. Batch-wise Mass Fractions for NIST Tables
    w_reg = torch.tensor(df_train["w_regolith"].values, dtype=torch.float32).unsqueeze(1)
    w_hdpe = 1.0 - w_reg

    mass_fractions_batch = {
        "H": w_hdpe * (4.032 / 28.05),
        "C": w_hdpe * (24.022 / 28.05),
        "Si": w_reg * 0.2295,
        "Al": w_reg * 0.1392,
        "Ca": w_reg * 0.0967,
        "Fe": w_reg * 0.0248,
        "Mg": w_reg * 0.0172,
        "Na": w_reg * 0.0189,
        "K": w_reg * 0.0028,
        "Ti": w_reg * 0.0038,
        "P": w_reg * 0.0007,
        "O": w_reg * 0.4664
    }

    # 5. Model & Optimization
    model = ShieldingPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    epochs = 4000
    physics_weight_max = 1e-2
    physics_warmup_epochs = 1000
    max_grad_norm = 5.0

    print("🧠 Initiating balanced multi-task training loop...\n")

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)

        sq_err = (predictions - Y_target) ** 2
        normalized_task_losses = sq_err.mean(dim=0) / target_variances
        loss_data = normalized_task_losses.sum()

        loss_physics = model.compute_physics_loss(X_tensor, mass_fractions_batch)
        physics_weight = physics_weight_max * min(1.0, epoch / physics_warmup_epochs)

        total_loss = loss_data + (physics_weight * loss_physics)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        scheduler.step(total_loss.detach())

        if (epoch + 1) % 500 == 0:
            d_loss = normalized_task_losses[0].item()
            n_loss = normalized_task_losses[1].item()
            print(f"Epoch {epoch+1:04d}/{epochs} | Total: {total_loss.item():.4e} "
                  f"[Dose Loss: {d_loss:.4e} | Neut Loss: {n_loss:.4e} | "
                  f"Phys: {loss_physics.item():.4e} (w={physics_weight:.1e})]")

    weights_dir = os.path.join(base_dir, "app", "models")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "pinn_phase2.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"\n🎉 Balanced Training Complete! Saved to: {weight_path}")


if __name__ == "__main__":
    train_pinn()