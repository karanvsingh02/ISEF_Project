import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.models.pinn import ShieldingPINN

RANDOM_STATE = 42
TEST_SIZE = 0.2
DOSE_SCALAR = 1e14
N_STRATA_BINS = 3   # must match train_pinn_geant4.py

# Must match the floor used in train_pinn_geant4.py.
DOSE_FLOOR_SV = 1e-20


def build_design_stratification_labels(df, n_bins=N_STRATA_BINS):
    """Identical to train_pinn_geant4.py's version -- must stay in sync so
    both scripts land on the exact same train/test split."""
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


def percentage_errors(y_true, y_pred, eps=1e-18):
    """
    Per-row absolute percentage error, restricted to rows where the true
    value is meaningfully nonzero (mean APE on a heavy-tailed quantity is
    otherwise dominated by a handful of near-zero rows).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    pct_err = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return pct_err, int(mask.sum()), int((~mask).sum())


def report_dose_metrics(y_true_dose, y_pred_dose, label):
    r2_lin = r2_score(y_true_dose, y_pred_dose)

    log_mask = y_true_dose > 1e-30
    if log_mask.sum() > 1:
        r2_log = r2_score(
            np.log10(y_true_dose[log_mask]),
            np.log10(np.clip(y_pred_dose[log_mask], 1e-30, None))
        )
    else:
        r2_log = float("nan")

    pct_err, n_included, n_excluded = percentage_errors(y_true_dose, y_pred_dose)
    mape = pct_err.mean() if n_included > 0 else float("nan")
    mdape = np.median(pct_err) if n_included > 0 else float("nan")

    print(f"--- {label} (n={len(y_true_dose)}) ---")
    print(f"Dose R^2 (linear):  {r2_lin:.4f}")
    print(f"Dose R^2 (log10):   {r2_log:.4f}")
    print(f"Dose Mean APE:      {mape:.2f}%  (n={n_included}, {n_excluded} near-zero rows excluded)")
    print(f"Dose Median APE:    {mdape:.2f}%")
    print()

    return r2_log


def evaluate_model():
    print("📊 Evaluating PINN Accuracy against Geant4 Ground Truth (held-out test set)...")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    idx_train, idx_test = split_train_test(df)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    print(f"📐 Evaluating on {len(df_test)} / {len(df)} held-out rows (design-aware split)\n")

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

    model = ShieldingPINN()
    weight_path = os.path.join(base_dir, "app", "models", "pinn_phase2.pt")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    y_pred_dose = (10.0 ** predictions[:, 0]) / DOSE_SCALAR
    y_pred_neutrons = predictions[:, 1]
    y_pred_protons = predictions[:, 2]
    y_pred_pions = predictions[:, 3]
    y_pred_ions = predictions[:, 4]

    r2_neutrons = r2_score(y_true_neutrons, y_pred_neutrons)
    r2_protons = r2_score(y_true_protons, y_pred_protons)
    r2_pions = r2_score(y_true_pions, y_pred_pions)
    r2_ions = r2_score(y_true_ions, y_pred_ions)

    print("========================================")
    print("🏆 PINN ACCURACY METRICS (held-out test set)")
    print("========================================")
    r2_log_full = report_dose_metrics(y_true_dose, y_pred_dose, "Dose — full test set")

    valid_regime_mask = y_true_dose > DOSE_FLOOR_SV
    n_below_floor = int((~valid_regime_mask).sum())
    if valid_regime_mask.sum() > 1:
        report_dose_metrics(
            y_true_dose[valid_regime_mask], y_pred_dose[valid_regime_mask],
            f"Dose — physically valid regime only (> {DOSE_FLOOR_SV:.0e} Sv/particle, "
            f"{n_below_floor} below-floor rows excluded from this view)"
        )

    print(f"Neutron R^2 Score:    {r2_neutrons:.4f} (1.0 is perfect)")
    print(f"Proton R^2 Score:     {r2_protons:.4f} (1.0 is perfect)")
    print(f"Pion R^2 Score:       {r2_pions:.4f} (1.0 is perfect)")
    print(f"Light Ion R^2 Score:  {r2_ions:.4f} (1.0 is perfect)")
    print("========================================\n")

    # Graph (log-log axes, since dose spans many decades)
    plt.figure(figsize=(8, 8))
    plot_mask = (y_true_dose > 0) & (y_pred_dose > 0)
    plt.scatter(y_true_dose[plot_mask], y_pred_dose[plot_mask], alpha=0.5, color='blue', edgecolor='k', s=40)

    lo = min(y_true_dose[plot_mask].min(), y_pred_dose[plot_mask].min())
    hi = max(y_true_dose[plot_mask].max(), y_pred_dose[plot_mask].max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.axvline(DOSE_FLOOR_SV, color='gray', linestyle=':', lw=1.5,
                label=f'Training floor ({DOSE_FLOOR_SV:.0e} Sv)')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"PINN vs Geant4: Absorbed Dose (held-out set)\n$R^2_{{log}}$ = {r2_log_full:.4f}", fontsize=14)
    plt.xlabel("Geant4 Monte Carlo True Dose (Sv/particle)", fontsize=12)
    plt.ylabel("PINN Predicted Dose (Sv/particle)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    plot_path = os.path.join(os.path.dirname(__file__), "pinn_accuracy_dose.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved Accuracy Graph to: {plot_path}")


if __name__ == "__main__":
    evaluate_model()