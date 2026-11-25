import os
import sys
import random
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)
from app.models.pinn_spectrum import SpectrumPINN
from train_pinn_spectrum import (
    build_design_stratification_labels, build_inputs, build_environment_row_weights,
    DOSE_SCALAR, PHYSICS_WEIGHT_MAX, PHYSICS_WARMUP_EPOCHS, MAX_GRAD_NORM,
)

NN_SEED = 42
N_FOLDS = 5
EPOCHS = 4000


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    for _ in range(n_calls):
        lbfgs.step(closure)

    return model


def train_one_fold(df_train):
    """Same recipe as train_pinn_spectrum.py's train_pinn() (Adam ->
    best-checkpoint -> L-BFGS, with environment reweighting), reusing its
    shared utility functions so the two never drift apart."""
    row_weights_np = build_environment_row_weights(df_train)
    X = build_inputs(df_train)

    Y_dose_lin = df_train["target_dose_sv_per_particle"].values * DOSE_SCALAR
    Y_dose = np.log10(np.clip(Y_dose_lin, 1e-30, None))
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
    best_state_dict = None

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
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        loss_physics = model.compute_monotonicity_loss(X_tensor)
        physics_weight = PHYSICS_WEIGHT_MAX * min(1.0, epoch / PHYSICS_WARMUP_EPOCHS)

        total_loss = loss_data + (physics_weight * loss_physics)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step(total_loss.detach())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    run_lbfgs_refinement(model, X_tensor, Y_target, target_variances, row_weights,
                          physics_weight=PHYSICS_WEIGHT_MAX)

    return model


def percentage_errors(y_true, y_pred, eps=1e-18):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100


def evaluate_fold(model, df_test):
    X = build_inputs(df_test)
    y_true_dose = df_test["target_dose_sv_per_particle"].values
    y_true_neutrons = df_test["target_secondary_neutrons"].values
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
    r2_dose_log = r2_score(np.log10(y_true_dose), np.log10(np.clip(y_pred_dose, 1e-30, None)))
    pct_err = percentage_errors(y_true_dose, y_pred_dose)
    mdape = np.median(pct_err) if len(pct_err) > 0 else float("nan")

    result = {
        "dose_r2_lin": r2_dose_lin,
        "dose_r2_log": r2_dose_log,
        "dose_mdape": mdape,
        "neutron_r2": r2_score(y_true_neutrons, y_pred_neutrons),
        "proton_r2": r2_score(y_true_protons, y_pred_protons),
        "pion_r2": r2_score(y_true_pions, y_pred_pions),
        "ion_r2": r2_score(y_true_ions, y_pred_ions),
    }

    for env in ["GCR", "SPE"]:
        mask = (df_test["environment"] == env).values
        if mask.sum() < 2:
            result[f"{env.lower()}_dose_r2"] = float("nan")
            result[f"{env.lower()}_dose_mdape"] = float("nan")
            continue
        result[f"{env.lower()}_dose_r2"] = r2_score(y_true_dose[mask], y_pred_dose[mask])
        pct_err_env = percentage_errors(y_true_dose[mask], y_pred_dose[mask])
        result[f"{env.lower()}_dose_mdape"] = np.median(pct_err_env) if len(pct_err_env) > 0 else float("nan")

    return result


def run_kfold_cv(data_path="pinn_training_tensor_spectrum.csv"):
    print(f"🚀 Running {N_FOLDS}-fold cross-validation on the spectrum PINN "
          f"(GCR/SPE-stratified folds, environment reweighting per fold)...\n")

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

        print(f"--- Fold {fold_idx + 1}/{N_FOLDS} (train={len(df_train)}, test={len(df_test)}, "
              f"test GCR/SPE = {(df_test.environment=='GCR').sum()}/{(df_test.environment=='SPE').sum()}) ---")
        model = train_one_fold(df_train)
        metrics = evaluate_fold(model, df_test)
        fold_metrics.append(metrics)

        print(f"  Overall: R^2(log)={metrics['dose_r2_log']:.4f}  MdAPE={metrics['dose_mdape']:.2f}%  |  "
              f"GCR: R^2={metrics['gcr_dose_r2']:.4f} MdAPE={metrics['gcr_dose_mdape']:.2f}%  |  "
              f"SPE: R^2={metrics['spe_dose_r2']:.4f} MdAPE={metrics['spe_dose_mdape']:.2f}%")
        print(f"  Hadrons: Neutron R^2={metrics['neutron_r2']:.4f}  Proton R^2={metrics['proton_r2']:.4f}  "
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