from app.core.physics_tables import table_based_stopping_power
import torch
import torch.nn as nn
from app.core.physics import bethe_bloch_stopping_power

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + identity)

class ShieldingPINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. SHARED TRUNK: Learns the basic 3D geometry and energy scales
        self.shared_trunk = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            ResidualBlock(128)
        )
        
        # 2. DOSE BRANCH: Dedicated to learning smooth Bethe-Bloch Coulomb physics
        self.dose_branch = nn.Sequential(
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # 3. HADRONIC BRANCH: Dedicated to learning sharp Spallation thresholds
        self.hadron_branch = nn.Sequential(
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4), # Outputs: Neutrons, Protons, Pions, Light Ions
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_trunk(x)
        dose_out = self.dose_branch(shared_features)
        hadrons_out = self.hadron_branch(shared_features)
        
        # Recombine them into the [Dose, Neutrons, Protons, Pions, Ions] format
        return torch.cat([dose_out, hadrons_out], dim=1)

    def compute_physics_loss(self, x: torch.Tensor, mass_fractions_dict: dict) -> torch.Tensor:
        """
        Calculates d(Dose)/d(Thickness) residual against NIST experimental tables.
        Includes Bragg additivity, molecular binding, and Sternheimer density effects.
        """
        thickness = x[:, 0:1].clone().detach().requires_grad_(True)
        w_regolith = x[:, 1:2].clone().detach()
        log_energy = x[:, 2:3].clone().detach()
        incident_energy_mev = 10.0 ** log_energy
        
        x_input = torch.cat([thickness, w_regolith, log_energy], dim=1)
        predictions = self.forward(x_input)
        predicted_dose = predictions[:, 0:1]
        
        # Calculate network derivative
        d_dose_d_x = torch.autograd.grad(
            outputs=predicted_dose, inputs=thickness,
            grad_outputs=torch.ones_like(predicted_dose), create_graph=True
        )[0]
        
        # Calculate dynamic density
        rho_hdpe, rho_regolith = 0.95, 2.75
        w_hdpe = 1.0 - w_regolith
        density_g_cm3 = 1.0 / ((w_hdpe / rho_hdpe) + (w_regolith / rho_regolith))
        
        # Calculate dynamic mean excitation energy
        ln_I_mix = w_hdpe * torch.log(torch.tensor(57.4)) + (w_regolith * torch.log(torch.tensor(135.0)))
        mean_excitation_ev_mix = torch.exp(ln_I_mix)
        
        # NIST Table query with Sternheimer-Peierls density effect
        # Note: mass_fractions_dict must contain PyTorch tensors matching the batch size
        theoretical_stopping = table_based_stopping_power(
            kinetic_energy_MeV=incident_energy_mev,
            mass_fractions=mass_fractions_dict,
            density_g_cm3=density_g_cm3,
            mean_excitation_eV=mean_excitation_ev_mix,
            apply_density_effect_correction=True
        )
        
        dose_scaling_factor = 1e-13 
        physics_residual = d_dose_d_x + (theoretical_stopping * dose_scaling_factor)
        
        return torch.mean(physics_residual ** 2)