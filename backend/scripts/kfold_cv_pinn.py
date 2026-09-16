import os
import sys
import random
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.models.pinn import ShieldingPINN

NN_SEED = 42
N_FOLDS = 5
DOSE_SCALAR = 1e14
DOSE_FLOOR_SV = 1e-20
N_STRATA_BINS = 3
EPOCHS = 4000
PHYSICS_WEIGHT_MAX = 1e-2
PHYSICS_WARMUP_EPOCHS = 1000
MAX_GRAD_NORM = 5.0


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
                          physics_weight, max_iter=200, history_size=50, n_calls=3):
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

    for _ in range(n_calls):
        lbfgs.step(closure)

    return model


def train_one_fold(df_train):
    """Same training recipe as train_pinn_geant4.py (Adam -> best-checkpoint
    restore -> L-BFGS refinement), applied to one fold's training rows."""
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

    best_loss = float("inf")
    best_state_dict = None

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        predictions = model(X_tensor)

        sq_err = (predictions - Y_target) ** 2
        normalized_task_losses = sq_err.mean(dim=0) / target_variances
        loss_data = normalized_task_losses.sum()

        current_data_loss = loss_data.item()
        if current_data_loss < best_loss:
            best_loss = current_data_loss
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        loss_physics = model.compute_physics_loss(X_tensor, mass_fractions_batch)
        physics_weight = PHYSICS_WEIGHT_MAX * min(1.0, epoch / PHYSICS_WARMUP_EPOCHS)

        total_loss = loss_data + (physics_weight * loss_physics)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step(total_loss.detach())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, mass_fractions_batch,
                          physics_weight=PHYSICS_WEIGHT_MAX)

    return model


def percentage_errors(y_true, y_pred, eps=1e-18):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100


def evaluate_fold(model, df_test):
    X = np.column_stack([
        df_test["thickness_cm"].values,
        df_test["w_regolith"].values,
        np.log10(df_test["incident_energy_mev"].values)
    ])
    y_true_dose = df_test["target_dose_sv_per_particle_shrunk"].values
    y_true_neutrons = df_test["target_secondary_neutrons_shrunk"].values
    y_true_protons = df_test["target_secondary_protons"].values
    y_true_pions = df_test["target_secondary_pions"].values
    y_true_ions = df_test["target_secondary_light_ions"].values

    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    y_pred_dose = (10.0 ** predictions[:, 0]) / DOSE_SCALAR
    y_pred_neutrons = predictions[:, 1]
    y_pred_protons = predictions[:, 2]
    y_pred_pions = predictions[:, 3]
    y_pred_ions = predictions[:, 4]

    r2_dose_lin = r2_score(y_true_dose, y_pred_dose)
    log_mask = y_true_dose > 1e-30
    r2_dose_log = (
        r2_score(np.log10(y_true_dose[log_mask]), np.log10(np.clip(y_pred_dose[log_mask], 1e-30, None)))
        if log_mask.sum() > 1 else float("nan")
    )
    pct_err = percentage_errors(y_true_dose, y_pred_dose)
    mape = pct_err.mean() if len(pct_err) > 0 else float("nan")
    mdape = np.median(pct_err) if len(pct_err) > 0 else float("nan")

    return {
        "dose_r2_lin": r2_dose_lin,
        "dose_r2_log": r2_dose_log,
        "dose_mape_pct": mape,
        "dose_mdape_pct": mdape,
        "neutron_r2": r2_score(y_true_neutrons, y_pred_neutrons),
        "proton_r2": r2_score(y_true_protons, y_pred_protons),
        "pion_r2": r2_score(y_true_pions, y_pred_pions),
        "ion_r2": r2_score(y_true_ions, y_pred_ions),
    }


def run_kfold_cv():
    print(f"🚀 Running {N_FOLDS}-fold cross-validation (design-aware stratified folds)...\n")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    strata = build_design_stratification_labels(df)
    try:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splits = list(skf.split(np.arange(len(df)), strata))
    except ValueError as e:
        print(f"⚠️  Stratified K-fold failed ({e}); falling back to a plain K-fold split.\n")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splits = list(kf.split(np.arange(len(df))))

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        set_seeds(NN_SEED)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        print(f"--- Fold {fold_idx + 1}/{N_FOLDS} (train={len(df_train)}, test={len(df_test)}) ---")
        model = train_one_fold(df_train)
        metrics = evaluate_fold(model, df_test)
        fold_metrics.append(metrics)

        print(f"  Dose R^2(log)={metrics['dose_r2_log']:.4f}  MdAPE={metrics['dose_mdape_pct']:.2f}%  "
              f"Neutron R^2={metrics['neutron_r2']:.4f}  Proton R^2={metrics['proton_r2']:.4f}  "
              f"Pion R^2={metrics['pion_r2']:.4f}  Ion R^2={metrics['ion_r2']:.4f}\n")

    print("========================================")
    print(f"🏆 {N_FOLDS}-FOLD CROSS-VALIDATION SUMMARY (mean ± std across folds)")
    print("========================================")
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics])
        print(f"{key:16s}: {values.mean():.4f} ± {values.std():.4f}")
    print("========================================\n")

    return fold_metrics


if __name__ == "__main__":
    run_kfold_cv()