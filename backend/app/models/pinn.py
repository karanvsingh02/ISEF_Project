import torch
import torch.nn as nn
from app.core.physics import bethe_bloch_stopping_power

class ShieldingPINN(nn.Module):
    """
    PINN mapping [Thickness (cm), Incident Energy (MeV)] -> Transmitted Dose (mGy)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs: x = [thickness_cm, energy_MeV]
        return self.net(x)

    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates d(Dose)/d(Thickness) residual against stopping power.
        """
        # 1. Separate thickness and energy, enabling grad directly on thickness
        thickness = x[:, 0:1].clone().detach().requires_grad_(True)
        energy = x[:, 1:2].clone().detach()
        
        # 2. Re-combine into a single input tensor for forward pass
        x_input = torch.cat([thickness, energy], dim=1)
        
        # 3. Forward pass
        predicted_dose = self.forward(x_input)
        
        # 4. Calculate d(Dose)/d(Thickness) via Autograd
        d_dose_d_x = torch.autograd.grad(
            outputs=predicted_dose,
            inputs=thickness,
            grad_outputs=torch.ones_like(predicted_dose),
            create_graph=True
        )[0]
        
        # 5. Physics target gradient derived from Bethe-Bloch
        theoretical_stopping = bethe_bloch_stopping_power(energy)
        
        # 6. Differential residual loss
        physics_residual = d_dose_d_x + theoretical_stopping
        return torch.mean(physics_residual ** 2)