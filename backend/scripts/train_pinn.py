import os
import sys
import torch
import torch.optim as optim

# Add the backend directory to the Python path so 'app' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.pinn import ShieldingPINN

def train():
    print("🚀 Initializing PINN Training Sequence...")
    
    # 1. Setup Model and Optimizer
    model = ShieldingPINN()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 2. Ensure weights directory exists
    weights_dir = os.path.join("app", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    epochs = 1000
    batch_size = 500

    print("Starting training loop (Target: 1000 Epochs)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- A. Generate Synthetic Collocation Points ---
        # Random Thickness: 0.5 cm to 20.0 cm
        # Random Energy: 10.0 MeV to 500.0 MeV
        thickness = (20.0 - 0.5) * torch.rand(batch_size, 1) + 0.5
        energy = (500.0 - 10.0) * torch.rand(batch_size, 1) + 10.0
        x_collocation = torch.cat([thickness, energy], dim=1)
        
        # --- B. Compute Physics Residual Loss ---
        # Enforces d(Dose)/dx = -Stopping Power
        loss_physics = model.compute_physics_loss(x_collocation)
        
        # --- C. Compute Boundary Condition Loss ---
        # At thickness = 0, the transmitted dose equals the incident dose.
        # For this synthetic baseline, we estimate incident dose (mGy) as a factor of energy.
        thickness_zero = torch.zeros(batch_size, 1)
        x_boundary = torch.cat([thickness_zero, energy], dim=1)
        
        predicted_boundary_dose = model(x_boundary)
        target_boundary_dose = energy * 0.15  # Synthetic assumption: 100 MeV -> 15 mGy dose
        
        loss_bc = torch.mean((predicted_boundary_dose - target_boundary_dose) ** 2)
        
        # --- D. Backpropagation ---
        loss_total = loss_physics + (10.0 * loss_bc)  # Weight the BC heavily so the curve anchors correctly
        loss_total.backward()
        optimizer.step()
        
        # Logging
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:04d}/{epochs} | Total Loss: {loss_total.item():.4f} "
                  f"(Physics: {loss_physics.item():.4f}, BC: {loss_bc.item():.4f})")

    # 3. Save the Trained Weights
    weight_path = os.path.join(weights_dir, "pinn_v1.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"\n✅ Training Complete! Weights successfully saved to: {weight_path}")

if __name__ == "__main__":
    train()