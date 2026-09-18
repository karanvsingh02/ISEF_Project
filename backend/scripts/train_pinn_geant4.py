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

RANDOM_STATE = 42
NN_SEED = 42
TEST_SIZE = 0.2
DOSE_SCALAR = 1e14
N_STRATA_BINS = 3
DOSE_FLOOR_SV = 1e-20


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_design_stratification_labels(df, n_bins=N_STRATA_BINS):
    log_energy = np.log10(df["incident_energy_mev"].values)
    b_thick = pd.qcut(df["thickness_cm"], n_bins, labels=False, duplicates="drop")
    b_wreg = pd.qcut(df["w_regolith"], n_bins, labels=False, duplicates="drop")
    b_energy = pd.qcut(pd.Series(log_energy), n_bins, labels=False, duplicates="drop")
    return (b_thick.values * 100) + (b_wreg.values * 10) + b_energy.values


def split_train_test(df):
    strata = build_design_stratification_labels(df)
    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
        )
    except ValueError as e:
        print(f"⚠️  Design-aware stratified split failed ({e}); falling back to a plain random split.")
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    return idx_train, idx_test


def build_mass_fractions(w_reg_tensor):
    w_hdpe = 1.0 - w_reg_tensor
    return {
        "H": w_hdpe * (4.032 / 28.05), "C": w_hdpe * (24.022 / 28.05),
        "Si": w_reg_tensor * 0.2295, "Al": w_reg_tensor * 0.1392, "Ca": w_reg_tensor * 0.0967,
        "Fe": w_reg_tensor * 0.0248, "Mg": w_reg_tensor * 0.0172, "Na": w_reg_tensor * 0.0189,
        "K": w_reg_tensor * 0.0028, "Ti": w_reg_tensor * 0.0038, "P": w_reg_tensor * 0.0007,
        "O": w_reg_tensor * 0.4664
    }


def run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, mass_fractions_batch,
                          physics_weight, max_iter=100, history_size=50, n_calls=1):
    """
    Short L-BFGS fine-tuning pass after Adam. Adam makes fast, robust
    progress early on; L-BFGS uses curvature (second-order) information to
    polish the final approach to a minimum, which tends to help PINN loss
    landscapes specifically -- the physics-residual term adds sharp,
    non-quadratic structure that a fixed-step first-order method handles
    less precisely once already close to a good solution. Always run this
    AFTER Adam has converged, never from a random initialization.

    max_iter/n_calls were reduced from an earlier 200/3 default after
    k-fold CV showed that setting improved training loss substantially
    (~25%) but came at a real generalization cost on the light-ion
    channel specifically (ion R^2 dropped from 0.975 to 0.955, and its
    fold-to-fold variance more than tripled) while dose/neutron/proton/
    pion stayed flat. Light ions are the sparsest, noisiest secondary
    channel in this dataset, so they're the most vulnerable to a
    curvature-informed optimizer fitting training-set idiosyncrasies
    rather than a generalizable pattern. This gentler setting is a first
    adjustment, not a validated final answer -- re-run kfold_cv_pinn.py
    after any further change here to confirm the ion channel actually
    recovers before trusting it.
    """
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        history_size=history_size, line_search_fn='strong_wolfe'
    )

    def closure():
        lbfgs.zero_grad()
        predictions = model(X_tensor)
        sq_err = (predictions - Y_target) ** 2
        normalized_task_losses = sq_err.mean(dim=0) / target_variances
        loss_data = normalized_task_losses.sum()
        loss_physics = model.compute_physics_loss(X_tensor, mass_fractions_batch)
        total_loss = loss_data + (physics_weight * loss_physics)
        total_loss.backward()
        return total_loss

    print(f"\n🔧 Running L-BFGS refinement ({n_calls} x up to {max_iter} internal iterations)...")
    for call_idx in range(n_calls):
        loss_before = closure().item()
        lbfgs.step(closure)
        loss_after = closure().item()
        print(f"  L-BFGS pass {call_idx + 1}/{n_calls}: loss {loss_before:.6e} -> {loss_after:.6e}")

    return model


def train_pinn():
    set_seeds(NN_SEED)
    print("🚀 Initializing Phase 2 Balanced Multi-Task PINN Training...")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    idx_train, idx_test = split_train_test(df)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    print(f"📐 Training on {len(df_train)} / {len(df)} rows "
          f"({len(idx_test)} held out for evaluation, design-aware split)")

    X = np.column_stack([
        df_train["thickness_cm"].values,
        df_train["w_regolith"].values,
        np.log10(df_train["incident_energy_mev"].values)
    ])

    Y_dose_lin = df_train["target_dose_sv_per_particle_shrunk"].values * DOSE_SCALAR
    Y_neutrons = df_train["target_secondary_neutrons_shrunk"].values
    Y_protons = df_train["target_secondary_protons"].values
    Y_pions = df_train["target_secondary_pions"].values
    Y_ions = df_train["target_secondary_light_ions"].values

    dose_floor_scaled = DOSE_FLOOR_SV * DOSE_SCALAR
    n_floored = int((Y_dose_lin < dose_floor_scaled).sum())
    if n_floored > 0:
        print(f"ℹ️  {n_floored} training rows have dose below the {DOSE_FLOOR_SV:.0e} Sv/particle "
              f"floor — their target is clamped to the floor (not excluded).")

    Y_dose_lin_clamped = np.maximum(Y_dose_lin, dose_floor_scaled)
    Y_dose = np.log10(Y_dose_lin_clamped)

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    Y_target = torch.tensor(np.column_stack([Y_dose, Y_neutrons, Y_protons, Y_pions, Y_ions]), dtype=torch.float32)

    dose_var = np.var(Y_dose)
    hadron_vars = np.var(np.column_stack([Y_neutrons, Y_protons, Y_pions, Y_ions]), axis=0)
    target_variances = torch.tensor(np.concatenate([[dose_var], hadron_vars]) + 1e-6, dtype=torch.float32)

    w_reg = torch.tensor(df_train["w_regolith"].values, dtype=torch.float32).unsqueeze(1)
    mass_fractions_batch = build_mass_fractions(w_reg)

    model = ShieldingPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    epochs = 4000
    physics_weight_max = 1e-2
    physics_warmup_epochs = 1000
    max_grad_norm = 5.0

    best_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    print("🧠 Initiating balanced multi-task training loop (Adam)...\n")

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)

        sq_err = (predictions - Y_target) ** 2
        normalized_task_losses = sq_err.mean(dim=0) / target_variances
        loss_data = normalized_task_losses.sum()

        current_data_loss = loss_data.item()
        if current_data_loss < best_loss:
            best_loss = current_data_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

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

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nℹ️  Restored best Adam checkpoint from epoch {best_epoch} (data loss = {best_loss:.4e}).")

    run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, mass_fractions_batch,
                          physics_weight=physics_weight_max)

    weights_dir = os.path.join(base_dir, "app", "models")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "pinn_phase2.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"\n🎉 Balanced Training Complete! Saved to: {weight_path}")


if __name__ == "__main__":
    train_pinn()