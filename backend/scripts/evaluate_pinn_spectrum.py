import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)
from app.models.pinn_spectrum import SpectrumPINN
from train_pinn_spectrum import split_train_test, build_inputs, DOSE_SCALAR


def percentage_errors(y_true, y_pred, eps=1e-18):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    pct_err = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return pct_err, int(mask.sum()), int((~mask).sum())


def evaluate_model(data_path="pinn_training_tensor_spectrum.csv", weight_path="pinn_spectrum.pt"):
    print("📊 Evaluating Spectrum PINN on held-out test set...")

    df = pd.read_csv(data_path)
    idx_train, idx_test = split_train_test(df)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    print(f"📐 Evaluating on {len(df_test)} / {len(df)} held-out rows "
          f"({(df_test.environment=='GCR').sum()} GCR / {(df_test.environment=='SPE').sum()} SPE)\n")

    X = build_inputs(df_test)
    y_true_dose = df_test["target_dose_sv_per_particle"].values
    y_true_neutrons = df_test["target_secondary_neutrons"].values
    y_true_protons = df_test["target_secondary_protons"].values
    y_true_pions = df_test["target_secondary_pions"].values
    y_true_ions = df_test["target_secondary_light_ions"].values

    model = SpectrumPINN()
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

    r2_dose_lin = r2_score(y_true_dose, y_pred_dose)
    r2_dose_log = r2_score(np.log10(y_true_dose), np.log10(np.clip(y_pred_dose, 1e-30, None)))
    pct_err, n_included, n_excluded = percentage_errors(y_true_dose, y_pred_dose)
    mape = pct_err.mean() if n_included > 0 else float("nan")
    mdape = np.median(pct_err) if n_included > 0 else float("nan")

    r2_neutrons = r2_score(y_true_neutrons, y_pred_neutrons)
    r2_protons = r2_score(y_true_protons, y_pred_protons)
    r2_pions = r2_score(y_true_pions, y_pred_pions)
    r2_ions = r2_score(y_true_ions, y_pred_ions)

    print("========================================")
    print("🏆 SPECTRUM PINN ACCURACY METRICS (held-out test set)")
    print("========================================")
    print(f"Dose R^2 (linear):    {r2_dose_lin:.4f}")
    print(f"Dose R^2 (log10):     {r2_dose_log:.4f}")
    print(f"Dose Mean APE:        {mape:.2f}%  (n={n_included})")
    print(f"Dose Median APE:      {mdape:.2f}%")
    print(f"Neutron R^2 Score:    {r2_neutrons:.4f}")
    print(f"Proton R^2 Score:     {r2_protons:.4f}")
    print(f"Pion R^2 Score:       {r2_pions:.4f}")
    print(f"Light Ion R^2 Score:  {r2_ions:.4f}")
    print("========================================\n")

    for env in ["GCR", "SPE"]:
        mask = (df_test["environment"] == env).values
        if mask.sum() < 2:
            continue
        r2_env = r2_score(y_true_dose[mask], y_pred_dose[mask])
        pct_err_env, n_inc, _ = percentage_errors(y_true_dose[mask], y_pred_dose[mask])
        mdape_env = np.median(pct_err_env) if n_inc > 0 else float("nan")
        print(f"  {env}: n={mask.sum()}, Dose R^2={r2_env:.4f}, Dose Median APE={mdape_env:.2f}%")

    return {
        "r2_dose_lin": r2_dose_lin, "r2_dose_log": r2_dose_log,
        "mape_dose": mape, "mdape_dose": mdape,
        "r2_neutrons": r2_neutrons, "r2_protons": r2_protons,
        "r2_pions": r2_pions, "r2_ions": r2_ions,
    }


if __name__ == "__main__":
    evaluate_model()