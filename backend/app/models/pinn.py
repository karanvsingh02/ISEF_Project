import torch
import torch.nn as nn
from app.core.physics import bethe_bloch_stopping_power

class ResidualBlock(nn.Module):
    """A standard skip-connection block using SiLU."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        # Skip connection: add identity before final activation
        return self.activation(out + identity)

class ShieldingPINN(nn.Module):
    """
    Residual PINN mapping [Thickness, w_regolith, Density, Topology_Index] -> [Dose, Neutron Flux]
    """
    def __init__(self):
        super().__init__()
        
        # Input layer: Upgraded to 4 inputs -> 64 hidden
        self.input_layer = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU()
        )
        
        # 3 Residual Blocks (Equivalent to 6 hidden layers, but with gradient highways)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        # Output compression and final layer
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2) # Outputs: Dose, Neutron Flux
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x

    def compute_physics_loss(self, x: torch.Tensor, effective_energy_mev: torch.Tensor) -> torch.Tensor:
        """
        Calculates d(Dose)/d(Thickness) residual against stopping power.
        Includes dynamic Z/A and mean excitation energy mapping based on regolith %.
        """
        # 1. Separate all 4 inputs (only thickness requires grad)
        thickness = x[:, 0:1].clone().detach().requires_grad_(True)
        w_regolith = x[:, 1:2].clone().detach()
        density_g_cm3 = x[:, 2:3].clone().detach()
        topology_idx = x[:, 3:4].clone().detach()
        
        # 2. Re-combine into a single input tensor for forward pass
        x_input = torch.cat([thickness, w_regolith, density_g_cm3, topology_idx], dim=1)
        predictions = self.forward(x_input)
        
        # 3. Isolate Dose (index 0) from Neutron Flux (index 1)
        predicted_dose = predictions[:, 0:1]
        
        # 4. Calculate d(Dose)/d(Thickness) via Autograd
        d_dose_d_x = torch.autograd.grad(
            outputs=predicted_dose,
            inputs=thickness,
            grad_outputs=torch.ones_like(predicted_dose),
            create_graph=True
        )[0]
        
        # 5. Dynamic Composition Mapping (Rule of Mixtures for PyTorch Tensors)
        # HDPE: Z/A = 0.556, I = 57.4 | LHS-1 Regolith: Z/A = 0.498, I = 135.0
        z_over_a_mix = (1.0 - w_regolith) * 0.556 + (w_regolith * 0.498)
        
        # Logarithmic averaging for Mean Excitation Energy (Bragg Rule)
        ln_I_mix = (1.0 - w_regolith) * torch.log(torch.tensor(57.4)) + (w_regolith * torch.log(torch.tensor(135.0)))
        mean_excitation_ev_mix = torch.exp(ln_I_mix)
        
        # 6. Physics target gradient derived from Bethe-Bloch
        # We explicitly pass the calculated tensors, overriding physics.py's defaults.
        # PyTorch will broadcast these automatically through the physics.py math.
        theoretical_stopping = bethe_bloch_stopping_power(
            kinetic_energy_MeV=effective_energy_mev,
            density_g_cm3=density_g_cm3,
            z_over_a=z_over_a_mix,
            mean_excitation_eV=mean_excitation_ev_mix
        )
        
        # 7. Differential residual loss
        physics_residual = d_dose_d_x + theoretical_stopping
        return torch.mean(physics_residual ** 2)