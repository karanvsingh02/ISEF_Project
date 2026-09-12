import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.models.pinn import ShieldingPINN

def train_pinn():
    print("🚀 Initializing Phase 2 Branched PINN Training Sequence...")

    data_path = os.path.join(os.path.dirname(__file__), "pinn_training_tensor_phase2_refined.csv")
    df = pd.read_csv(data_path)

    # 1. Inputs & Targets
    X = np.column_stack([
        df["thickness_cm"].values,
        df["w_regolith"].values,
        np.log10(df["incident_energy_mev"].values) 
    ])
    
    DOSE_SCALAR = 1e14
    Y_dose = df["target_dose_sv_per_particle_shrunk"].values * DOSE_SCALAR
    Y_neutrons = df["target_secondary_neutrons_shrunk"].values
    Y_protons = df["target_secondary_protons"].values
    Y_pions = df["target_secondary_pions"].values
    Y_ions = df["target_secondary_light_ions"].values

    # 2. Uncertainties & Weights
    Sem_dose = df["target_dose_sv_per_particle_shrunk_sem"].values * DOSE_SCALAR
    Sem_neutrons = df["target_secondary_neutrons_shrunk_sem"].values
    n_events = 10000.0
    Sem_protons = np.where(Y_protons > 0, np.sqrt(Y_protons * n_events) / n_events, 1e-4)
    Sem_pions = np.where(Y_pions > 0, np.sqrt(Y_pions * n_events) / n_events, 1e-4)
    Sem_ions = np.where(Y_ions > 0, np.sqrt(Y_ions * n_events) / n_events, 1e-4)

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    Y_target = torch.tensor(np.column_stack([Y_dose, Y_neutrons, Y_protons, Y_pions, Y_ions]), dtype=torch.float32)
    
    epsilon = 1e-12
    W_tensor = torch.stack([
        torch.tensor(1.0 / (Sem_dose**2 + epsilon), dtype=torch.float32),
        torch.tensor(1.0 / (Sem_neutrons**2 + epsilon), dtype=torch.float32),
        torch.tensor(1.0 / (Sem_protons**2 + epsilon), dtype=torch.float32),
        torch.tensor(1.0 / (Sem_pions**2 + epsilon), dtype=torch.float32),
        torch.tensor(1.0 / (Sem_ions**2 + epsilon), dtype=torch.float32)
    ], dim=1)
    
    # Normalize weights
    W_tensor = W_tensor / W_tensor.mean(dim=0, keepdim=True)

    # 3. Construct Batch-wise Mass Fractions for NIST Tables
    # LHS-1 elemental breakdown combined dynamically with HDPE (C2H4)
    w_reg = torch.tensor(df["w_regolith"].values, dtype=torch.float32).unsqueeze(1)
    w_hdpe = 1.0 - w_reg
    
    # Elemental fractions derived from your stoichiometry loop
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

    model = ShieldingPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150)
    epochs = 4000

    print("🧠 Training Branched Architecture with NIST Table Constraints...\n")

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        
        squared_errors = (predictions - Y_target) ** 2
        loss_data = torch.mean(squared_errors * W_tensor)
        
        loss_physics = model.compute_physics_loss(X_tensor, mass_fractions_batch)

        total_loss = loss_data + (1e-5 * loss_physics)
        total_loss.backward()
        optimizer.step()
        
        scheduler.step(total_loss.detach())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:04d}/{epochs} | Total: {total_loss.item():.4e} "
                  f"(Data MSE: {loss_data.item():.4e} | NIST Phys: {loss_physics.item():.4e})")

    weights_dir = os.path.join(base_dir, "app", "models")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "pinn_phase2.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"\n🎉 Training Complete! Saved to: {weight_path}")

if __name__ == "__main__":
    train_pinn()