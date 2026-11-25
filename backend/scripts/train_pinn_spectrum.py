import os
import sys
import random
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)
from app.models.pinn_spectrum import SpectrumPINN

RANDOM_STATE = 42
NN_SEED = 42
TEST_SIZE = 0.2
DOSE_SCALAR = 1e14
N_STRATA_BINS = 3
PHYSICS_WEIGHT_MAX = 1e-2
PHYSICS_WARMUP_EPOCHS = 1000
MAX_GRAD_NORM = 5.0
EPOCHS = 4000


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_design_stratification_labels(df, n_bins=N_STRATA_BINS):
    labels = np.empty(len(df), dtype=object)
    gcr_mask = (df["environment"] == "GCR").to_numpy()
    spe_mask = ~gcr_mask

    if gcr_mask.sum() > 0:
        b_thick = pd.qcut(df.loc[gcr_mask, "thickness_cm"], n_bins, labels=False, duplicates="drop")
        b_reg = pd.qcut(df.loc[gcr_mask, "w_regolith"], n_bins, labels=False, duplicates="drop")
        labels[gcr_mask] = ["GCR_%d_%d" % (t, r) for t, r in zip(b_thick.values, b_reg.values)]

    if spe_mask.sum() > 0:
        b_thick = pd.qcut(df.loc[spe_mask, "thickness_cm"], n_bins, labels=False, duplicates="drop")
        b_reg = pd.qcut(df.loc[spe_mask, "w_regolith"], n_bins, labels=False, duplicates="drop")
        b_sev = pd.qcut(df.loc[spe_mask, "spe_severity"], n_bins, labels=False, duplicates="drop")
        labels[spe_mask] = ["SPE_%d_%d_%d" % (t, r, s) for t, r, s in
                             zip(b_thick.values, b_reg.values, b_sev.values)]
    return labels


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


def build_inputs(df):
    is_spe = (df["environment"] == "SPE").astype(float).values
    severity = df["spe_severity"].fillna(0.0).values
    X = np.column_stack([
        df["thickness_cm"].values,
        df["w_regolith"].values,
        is_spe,
        severity,
    ])
    return X


def build_environment_row_weights(df):
    """
    SPE outnumbers GCR 2:1 in this dataset (1000 vs 500 rows overall).
    Left unweighted, the loss naturally prioritizes fitting SPE well over
    GCR, since SPE simply contributes more terms to the mean. This
    computes a per-row weight (GCR rows upweighted, SPE rows at 1.0) so
    the two environments contribute EQUALLY to the loss regardless of
    their row-count imbalance -- based on the ACTUAL environment counts
    in whatever split is passed in (not a hardcoded 1:2), so it stays
    correct even if the train/test split proportions shift slightly.
    """
    env = df["environment"].values
    n_gcr = int((env == "GCR").sum())
    n_spe = int((env == "SPE").sum())
    gcr_weight = (n_spe / n_gcr) if n_gcr > 0 else 1.0
    weights = np.where(env == "GCR", gcr_weight, 1.0)
    print(f"ℹ️  Environment row weights: GCR x{gcr_weight:.3f} (n={n_gcr}), SPE x1.0 (n={n_spe}) "
          f"-- balances the 2:1 SPE:GCR row-count imbalance in the loss.")
    return weights


def run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, row_weights, physics_weight,
                          max_iter=100, history_size=50, n_calls=1):
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        history_size=history_size, line_search_fn='strong_wolfe'
    )
    weight_sum = row_weights.sum()

    def closure():
        lbfgs.zero_grad()
        predictions = model(X_tensor)
        sq_err = (predictions - Y_target) ** 2
        weighted_sq_err = sq_err * row_weights.unsqueeze(1)
        normalized_task_losses = (weighted_sq_err.sum(dim=0) / weight_sum) / target_variances
        loss_data = normalized_task_losses.sum()
        loss_physics = model.compute_monotonicity_loss(X_tensor)
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


def train_pinn(data_path="pinn_training_tensor_spectrum.csv"):
    set_seeds(NN_SEED)
    print("🚀 Initializing Spectrum PINN Training...")

    df = pd.read_csv(data_path)
    idx_train, idx_test = split_train_test(df)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    print(f"📐 Training on {len(df_train)} / {len(df)} rows "
          f"({len(idx_test)} held out for evaluation, design-aware split)")
    print(f"   Train split: {(df_train.environment=='GCR').sum()} GCR / "
          f"{(df_train.environment=='SPE').sum()} SPE")

    row_weights_np = build_environment_row_weights(df_train)

    X = build_inputs(df_train)

    Y_dose_lin = df_train["target_dose_sv_per_particle"].values * DOSE_SCALAR
    Y_dose = np.log10(np.clip(Y_dose_lin, 1e-30, None))
    n_clipped = int((Y_dose_lin <= 0).sum())
    if n_clipped > 0:
        print(f"⚠️  {n_clipped} rows had non-positive dose -- unexpected for this dataset, investigate.")

    Y_neutrons = df_train["target_secondary_neutrons"].values
    Y_protons = df_train["target_secondary_protons"].values
    Y_pions = df_train["target_secondary_pions"].values
    Y_ions = df_train["target_secondary_light_ions"].values

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    Y_target = torch.tensor(np.column_stack([Y_dose, Y_neutrons, Y_protons, Y_pions, Y_ions]), dtype=torch.float32)
    row_weights = torch.tensor(row_weights_np, dtype=torch.float32)
    weight_sum = row_weights.sum()

    def weighted_var(values, weights):
        mean = np.average(values, weights=weights)
        return np.average((values - mean) ** 2, weights=weights)

    dose_var = weighted_var(Y_dose, row_weights_np)
    hadron_vars = np.array([
        weighted_var(Y_neutrons, row_weights_np),
        weighted_var(Y_protons, row_weights_np),
        weighted_var(Y_pions, row_weights_np),
        weighted_var(Y_ions, row_weights_np),
    ])
    target_variances = torch.tensor(np.concatenate([[dose_var], hadron_vars]) + 1e-6, dtype=torch.float32)

    model = SpectrumPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    best_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    print("🧠 Training (Adam)...\n")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_tensor)

        sq_err = (predictions - Y_target) ** 2
        weighted_sq_err = sq_err * row_weights.unsqueeze(1)
        normalized_task_losses = (weighted_sq_err.sum(dim=0) / weight_sum) / target_variances
        loss_data = normalized_task_losses.sum()

        current_data_loss = loss_data.item()
        if current_data_loss < best_loss:
            best_loss = current_data_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        loss_physics = model.compute_monotonicity_loss(X_tensor)
        physics_weight = PHYSICS_WEIGHT_MAX * min(1.0, epoch / PHYSICS_WARMUP_EPOCHS)

        total_loss = loss_data + (physics_weight * loss_physics)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step(total_loss.detach())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:04d}/{EPOCHS} | Total: {total_loss.item():.4e} "
                  f"[Dose: {normalized_task_losses[0].item():.4e} | "
                  f"Neut: {normalized_task_losses[1].item():.4e} | "
                  f"Phys(monotonicity): {loss_physics.item():.4e} (w={physics_weight:.1e})]")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nℹ️  Restored best Adam checkpoint from epoch {best_epoch} (data loss = {best_loss:.4e}).")

    run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, row_weights,
                          physics_weight=PHYSICS_WEIGHT_MAX)

    weight_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "pinn_spectrum.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"\n🎉 Training Complete! Saved to: {weight_path}")
    return model


if __name__ == "__main__":
    train_pinn()