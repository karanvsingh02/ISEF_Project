import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.models.pinn import ShieldingPINN

def evaluate_model():
    print("📊 Evaluating PINN Accuracy against Geant4 Ground Truth...")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    X = np.column_stack([
        df["thickness_cm"].values,
        df["w_regolith"].values,
        np.log10(df["incident_energy_mev"].values)
    ])
    
    # 1. Extract True Values (UNSCALED)
    y_true_dose = df["target_dose_sv_per_particle_shrunk"].values
    y_true_neutrons = df["target_secondary_neutrons_shrunk"].values

    # 2. Load Model
    model = ShieldingPINN()
    weight_path = os.path.join(base_dir, "app", "models", "pinn_phase2.pt")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # 3. Run Inference
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
        
    # -------------------------------------------------------------
    # THE FIX: UNSCALE THE DOSE PREDICTION TO MATCH GEANT4
    # -------------------------------------------------------------
    DOSE_SCALAR = 1e14
    y_pred_dose = predictions[:, 0] / DOSE_SCALAR
    y_pred_neutrons = predictions[:, 1]

    # 4. Calculate Metrics
    r2_dose = r2_score(y_true_dose, y_pred_dose)
    r2_neutrons = r2_score(y_true_neutrons, y_pred_neutrons)

    safe_dose_mask = y_true_dose > 1e-18
    mape_dose = np.mean(np.abs((y_true_dose[safe_dose_mask] - y_pred_dose[safe_dose_mask]) / y_true_dose[safe_dose_mask])) * 100

    print("\n========================================")
    print("🏆 PINN ACCURACY METRICS")
    print("========================================")
    print(f"Dose R^2 Score:       {r2_dose:.4f} (1.0 is perfect)")
    print(f"Dose Mean Error:      {mape_dose:.2f}%")
    print(f"Neutron R^2 Score:    {r2_neutrons:.4f} (1.0 is perfect)")
    print("========================================\n")

    # 5. Graph
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_dose, y_pred_dose, alpha=0.5, color='blue', edgecolor='k', s=40)
    
    max_val = max(y_true_dose.max(), y_pred_dose.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    
    plt.title(f"PINN vs Geant4: Absorbed Dose\n$R^2$ = {r2_dose:.4f}", fontsize=14)
    plt.xlabel("Geant4 Monte Carlo True Dose (Sv/particle)", fontsize=12)
    plt.ylabel("PINN Predicted Dose (Sv/particle)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(os.path.dirname(__file__), "pinn_accuracy_dose.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved Accuracy Graph to: {plot_path}")

if __name__ == "__main__":
    evaluate_model()