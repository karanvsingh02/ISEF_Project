from app.core.physics_tables import table_based_stopping_power
import math
import torch
import torch.nn as nn
from app.core.physics import bethe_bloch_stopping_power

LN10 = math.log(10.0)


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

        # SEPARATE trunks for dose vs hadronics. These used to share one
        # trunk. The physics-residual loss only ever touches the dose
        # branch, but because it backpropagates into a SHARED trunk, its
        # (often noisy) gradients were also reshaping the features the
        # hadron branch depends on -- and the hadron branch's training
        # loss went completely flat as a result. Giving each branch its
        # own trunk removes that interference entirely.
        self.dose_trunk = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            ResidualBlock(128)
        )
        self.hadron_trunk = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            ResidualBlock(128)
        )

        # DOSE BRANCH: predicts log10(scaled_dose). Linear output (no
        # Softplus) since a log-quantity can legitimately be negative.
        self.dose_branch = nn.Sequential(
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # HADRONIC BRANCH: Neutrons, Protons, Pions, Light Ions (physical
        # counts, always >= 0).
        self.hadron_branch = nn.Sequential(
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dose_features = self.dose_trunk(x)
        hadron_features = self.hadron_trunk(x)

        dose_out = self.dose_branch(dose_features)  # log10(scaled dose)
        # Defensive clamp: unbounded Linear output is later exponentiated
        # (10**x). +/-20 leaves an enormous safety margin (scaled dose from
        # 1e-20 to 1e20) without touching any physically plausible value.
        dose_out = torch.clamp(dose_out, min=-20.0, max=20.0)

        hadrons_out = self.hadron_branch(hadron_features)
        return torch.cat([dose_out, hadrons_out], dim=1)

    def compute_physics_loss(self, x: torch.Tensor, mass_fractions_dict: dict) -> torch.Tensor:
        """
        Calculates d(Dose)/d(Thickness) residual against NIST experimental tables.
        Only routes through the DOSE trunk/branch -- the hadron trunk plays no
        part in this loss and is never touched by it.
        """
        thickness = x[:, 0:1].clone().detach().requires_grad_(True)
        w_regolith = x[:, 1:2].clone().detach()
        log_energy = x[:, 2:3].clone().detach()
        incident_energy_mev = 10.0 ** log_energy

        x_input = torch.cat([thickness, w_regolith, log_energy], dim=1)

        dose_features = self.dose_trunk(x_input)
        predicted_log_dose = self.dose_branch(dose_features)
        predicted_log_dose = torch.clamp(predicted_log_dose, min=-20.0, max=20.0)
        predicted_scaled_dose = 10.0 ** predicted_log_dose

        d_log_dose_d_x = torch.autograd.grad(
            outputs=predicted_log_dose, inputs=thickness,
            grad_outputs=torch.ones_like(predicted_log_dose), create_graph=True
        )[0]

        # Chain rule: d(scaled_dose)/dx = ln(10) * scaled_dose * d(log10 scaled_dose)/dx
        d_dose_d_x = LN10 * predicted_scaled_dose * d_log_dose_d_x

        rho_hdpe, rho_regolith = 0.95, 2.75
        w_hdpe = 1.0 - w_regolith
        density_g_cm3 = 1.0 / ((w_hdpe / rho_hdpe) + (w_regolith / rho_regolith))

        ln_I_mix = w_hdpe * torch.log(torch.tensor(57.4)) + (w_regolith * torch.log(torch.tensor(135.0)))
        mean_excitation_ev_mix = torch.exp(ln_I_mix)

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