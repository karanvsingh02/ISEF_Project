"""
PINN for the GCR/SPE spectrum-integrated dataset.

Deliberately a SEPARATE, smaller model from the mono-energetic pinn.py --
see the earlier discussion on why: the mono-energetic model's physics
loss (Bethe-Bloch-derived stopping power vs energy) doesn't directly
apply here, since energy has already been integrated out of the design
space (each row's source is a FIXED environment -- GCR at solar
minimum, or SPE at some severity -- not a free continuous energy input).
Forcing a shared architecture would mean feeding this model a
meaningless placeholder energy value, exactly the mistake flagged and
avoided earlier in this project.

Inputs (4): thickness_cm, w_regolith, is_spe (0=GCR, 1=SPE), spe_severity
  (0 for GCR rows -- semantically fine here, unlike a placeholder energy,
  because is_spe correctly gates whether severity is meaningful; the
  network only ever needs to learn "ignore severity when is_spe=0").
Outputs (5): log10(scaled dose), neutrons, protons, pions, light_ions --
  same 5 channels as the mono-energetic model, same DOSE_SCALAR/log-space
  treatment for consistency (dose here has a narrower dynamic range than
  the mono-energetic case, but log-space is still a safe, low-risk
  choice and costs nothing).

Physics regularizer: a much simpler one than the mono-energetic model's
NIST-table term, and self-contained (no external physics tables needed).
Shielding should attenuate monotonically with thickness -- d(dose)/d
(thickness) should never be positive. This is weaker than a quantitative
stopping-power match, but it's a real, physically-motivated constraint
that costs nothing extra to compute (pure autograd, no lookup tables).
"""

import torch
import torch.nn as nn


class SpectrumPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.dose_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.hadron_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 4),  # neutrons, protons, pions, light_ions
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        dose_out = self.dose_head(features)
        dose_out = torch.clamp(dose_out, min=-20.0, max=20.0)  # same overflow guard as pinn.py
        hadrons_out = self.hadron_head(features)
        return torch.cat([dose_out, hadrons_out], dim=1)

    def compute_monotonicity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Penalizes d(dose)/d(thickness) > 0 -- shielding must attenuate,
        never amplify, dose as thickness increases. x[:, 0] is thickness.
        """
        thickness = x[:, 0:1].clone().detach().requires_grad_(True)
        rest = x[:, 1:].clone().detach()
        x_input = torch.cat([thickness, rest], dim=1)

        predictions = self.forward(x_input)
        predicted_log_dose = predictions[:, 0:1]

        d_log_dose_d_thickness = torch.autograd.grad(
            outputs=predicted_log_dose, inputs=thickness,
            grad_outputs=torch.ones_like(predicted_log_dose), create_graph=True
        )[0]

        # Only penalize the WRONG-SIGN case (positive derivative); a
        # correctly negative derivative contributes zero loss.
        violation = torch.clamp(d_log_dose_d_thickness, min=0.0)
        return torch.mean(violation ** 2)