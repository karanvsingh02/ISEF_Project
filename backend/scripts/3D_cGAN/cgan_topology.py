"""
3D conditional GAN -- architecture/mechanics validation stage.

Per the project plan (Month 3 Week 12 / Month 4 Week 13): build the
3D-cGAN generator/discriminator and validate it trains on a small
synthetic voxel dataset FIRST, before wiring it to any real physics
evaluation. That's what this script does.

NOT included here: any connection to the PINN or Geant4 for scoring a
generated topology's actual shielding performance -- that's a separate,
larger decision (full free-form voxel + topology-aware PINN, vs. a
parametric/layered representation -- see the accompanying discussion)
and deliberately out of scope for this mechanics-validation step.

Conditioning: generator/discriminator are conditioned on a single
scalar -- target regolith volume fraction -- consistent with w_regolith
elsewhere in this project. Pattern FAMILY is not an explicit condition;
the generator should learn to produce diverse structure from the noise
vector on its own. Family labels in the dataset exist only so you can
sanity-check afterward that the generator hasn't mode-collapsed onto
one family.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

LATENT_DIM = 128
GRID_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 200
LR = 2e-4
SEED = 42


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Generator(nn.Module):
    """latent noise + target volume fraction -> GxGxG voxel probability map."""
    def __init__(self, latent_dim=LATENT_DIM, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        self.init_size = grid_size // 8  # 3 upsampling stages: x2, x2, x2
        self.fc = nn.Linear(latent_dim + 1, 128 * self.init_size ** 3)

        self.net = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, target_fraction):
        x = torch.cat([z, target_fraction], dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, self.init_size, self.init_size, self.init_size)
        return self.net(x)  # (B, 1, G, G, G), values in [0,1]


class Discriminator(nn.Module):
    """GxGxG voxel map + target volume fraction -> real/fake logit."""
    def __init__(self, grid_size=GRID_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        final_size = grid_size // 8
        self.fc = nn.Linear(128 * final_size ** 3 + 1, 1)

    def forward(self, vol, target_fraction):
        x = self.net(vol)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, target_fraction], dim=1)
        return self.fc(x)  # raw logit (BCEWithLogitsLoss expects this)


def load_dataset(path="synthetic_topology_dataset.npz"):
    data = np.load(path, allow_pickle=True)
    return data["volumes"], data["fractions"]


def save_sample_slices(G, device, epoch, latent_dim, n_samples=4, target_fraction=0.35, out_dir="gan_samples"):
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        frac = torch.full((n_samples, 1), target_fraction, device=device)
        vols = G(z, frac).cpu().numpy()

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    for i in range(n_samples):
        mid_slice = vols[i, 0, vols.shape[2] // 2, :, :]
        ax = axes[i] if n_samples > 1 else axes
        ax.imshow(mid_slice, cmap="viridis", vmin=0, vmax=1)
        achieved = vols[i, 0].mean()
        ax.set_title(f"sample {i}, achieved frac={achieved:.2f}")
        ax.axis("off")
    plt.suptitle(f"Generator mid-slice samples, epoch {epoch} (target fraction={target_fraction})")
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    G.train()


def train_gan(dataset_path="synthetic_topology_dataset.npz", epochs=EPOCHS,
              batch_size=BATCH_SIZE, latent_dim=LATENT_DIM, grid_size=GRID_SIZE,
              out_dir="gan_samples", checkpoint_prefix="cgan"):
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    volumes, fractions = load_dataset(dataset_path)
    volumes_t = torch.tensor(volumes, dtype=torch.float32).unsqueeze(1)
    fractions_t = torch.tensor(fractions, dtype=torch.float32).unsqueeze(1)
    n = volumes_t.shape[0]
    print(f"Loaded {n} synthetic topologies for GAN training.")

    G = Generator(latent_dim=latent_dim, grid_size=grid_size).to(device)
    D = Discriminator(grid_size=grid_size).to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        g_losses, d_losses = [], []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            real_vol = volumes_t[idx].to(device)
            real_frac = fractions_t[idx].to(device)
            b = real_vol.size(0)

            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            # --- Discriminator ---
            opt_D.zero_grad()
            z = torch.randn(b, latent_dim, device=device)
            fake_vol = G(z, real_frac).detach()

            d_real = D(real_vol, real_frac)
            d_fake = D(fake_vol, real_frac)
            d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
            d_loss.backward()
            opt_D.step()

            # --- Generator ---
            opt_G.zero_grad()
            z = torch.randn(b, latent_dim, device=device)
            gen_vol = G(z, real_frac)
            d_on_gen = D(gen_vol, real_frac)
            g_adv_loss = bce(d_on_gen, real_labels)

            # Auxiliary term: pure adversarial loss doesn't guarantee the
            # generated volume's OWN achieved fraction matches what was
            # requested -- nudge it explicitly.
            achieved_frac = gen_vol.mean(dim=[1, 2, 3, 4]).unsqueeze(1)
            frac_loss = torch.mean((achieved_frac - real_frac) ** 2)

            g_loss = g_adv_loss + 5.0 * frac_loss
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"Epoch {epoch+1:04d}/{epochs} | D loss: {np.mean(d_losses):.4f} | "
                  f"G loss: {np.mean(g_losses):.4f}")
            save_sample_slices(G, device, epoch + 1, latent_dim, out_dir=out_dir)

    torch.save(G.state_dict(), f"{checkpoint_prefix}_generator.pt")
    torch.save(D.state_dict(), f"{checkpoint_prefix}_discriminator.pt")
    print(f"Saved {checkpoint_prefix}_generator.pt / {checkpoint_prefix}_discriminator.pt")


if __name__ == "__main__":
    train_gan()