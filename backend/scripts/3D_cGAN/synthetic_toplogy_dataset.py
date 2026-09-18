"""
Synthetic 3D voxel topology dataset generator.

Per the project plan (Month 3, Week 12): "Begin 3D-cGAN generator/
discriminator architecture ... build and test on a small synthetic voxel
dataset first." This builds that dataset: several distinct FAMILIES of
structured binary voxel patterns (regolith=1, HDPE=0), each at a range
of target regolith volume fractions, so the GAN has something non-
trivial and structurally diverse to learn before it's ever wired to
real physics data.

This is a MECHANICS validation step only -- not yet tied to Geant4/PINN
performance. Families included:
  - random:      matches the CURRENT Geant4 pipeline's shuffle (no
                  spatial structure -- the baseline to beat)
  - layered:     regolith concentrated in bands along the thickness axis
  - gradient:    regolith probability increases/decreases smoothly with
                  depth (a functionally-graded material, common in real
                  shielding design literature)
  - checkerboard: periodic 3D blocks
  - clustered:   smoothed-noise "blob" clusters
  - core_shell:  regolith concentrated toward (or away from) the center
"""

import numpy as np
from scipy.ndimage import gaussian_filter

GRID_SIZE = 32  # smaller than the physics grid (64^3) on purpose, for
                 # fast local iteration while validating GAN mechanics.
                 # Scale up once training is stable and the topology-
                 # representation decision (see accompanying note) is made.


def make_random(grid_size, target_fraction, rng):
    n_total = grid_size ** 3
    n_regolith = int(round(n_total * target_fraction))
    flat = np.zeros(n_total, dtype=np.float32)
    flat[:n_regolith] = 1.0
    rng.shuffle(flat)
    return flat.reshape(grid_size, grid_size, grid_size)


def make_layered(grid_size, target_fraction, rng, n_layers=None):
    if n_layers is None:
        n_layers = int(rng.integers(2, 6))
    z = np.linspace(0, 1, grid_size)
    band_width = 1.0 / n_layers
    layer_id = np.clip(np.floor(z / band_width).astype(int), 0, n_layers - 1)

    n_regolith_layers = max(1, int(round(n_layers * target_fraction)))
    regolith_layers = set(rng.choice(n_layers, size=min(n_regolith_layers, n_layers), replace=False))
    z_mask = np.array([1.0 if l in regolith_layers else 0.0 for l in layer_id], dtype=np.float32)

    vol = np.repeat(z_mask[np.newaxis, np.newaxis, :], grid_size, axis=0)
    vol = np.repeat(vol, grid_size, axis=1)
    return vol


def make_gradient(grid_size, target_fraction, rng, slope=None):
    """slope: how strongly regolith probability varies with depth
    (z-axis). Positive = denser toward the far/exit face, negative =
    denser toward the near/entrance face, 0 = no gradient (flat).
    Explicit slope enables LHS sampling; default (None) keeps the
    original random-positive-slope behavior used by the already-tested
    GAN synthetic dataset."""
    if slope is None:
        slope = rng.uniform(0.5, 1.5)
    z = np.linspace(0, 1, grid_size)
    prob_z = np.clip(target_fraction + slope * (z - 0.5), 0.0, 1.0)
    noise = rng.random((grid_size, grid_size, grid_size))
    return (noise < prob_z[np.newaxis, np.newaxis, :]).astype(np.float32)


def make_checkerboard(grid_size, target_fraction, rng, block=None):
    if block is None:
        block = int(rng.integers(2, 6))
    idx = np.indices((grid_size, grid_size, grid_size))
    parity = ((idx[0] // block) + (idx[1] // block) + (idx[2] // block)) % 2
    vol = parity.astype(np.float32)
    current_frac = vol.mean()
    if current_frac > 0:
        flip_prob = np.clip(1.0 - target_fraction / current_frac, 0.0, 1.0)
        flips = rng.random(vol.shape) < flip_prob
        vol[flips & (vol == 1)] = 0.0
    return vol


def make_clustered(grid_size, target_fraction, rng, smoothness=None):
    if smoothness is None:
        smoothness = rng.uniform(1.5, 4.0)
    noise = rng.standard_normal((grid_size, grid_size, grid_size))
    smoothed = gaussian_filter(noise, sigma=smoothness)
    threshold = np.quantile(smoothed, 1.0 - target_fraction)
    return (smoothed >= threshold).astype(np.float32)


def make_core_shell(grid_size, target_fraction, rng, radial_bias=None):
    """radial_bias in [-1, 1]: +1 = regolith concentrated toward the
    center, -1 = concentrated toward the edges, 0 = radially flat
    (roughly uniform). Explicit radial_bias enables LHS sampling;
    default (None) keeps the original random +/-1 behavior used by the
    already-tested GAN synthetic dataset."""
    if radial_bias is None:
        radial_bias = 1.0 if rng.random() < 0.5 else -1.0
    c = (grid_size - 1) / 2.0
    idx = np.indices((grid_size, grid_size, grid_size)).astype(np.float32)
    r = np.sqrt(((idx[0] - c) ** 2) + ((idx[1] - c) ** 2) + ((idx[2] - c) ** 2))
    r_norm = r / r.max()
    weight = 0.5 + 0.5 * radial_bias * (1.0 - 2.0 * r_norm)
    weight = np.clip(weight, 1e-6, None)
    prob = weight * (target_fraction / max(weight.mean(), 1e-6))
    prob = np.clip(prob, 0.0, 1.0)
    noise = rng.random((grid_size, grid_size, grid_size))
    return (noise < prob).astype(np.float32)


FAMILIES = {
    "random": make_random,
    "layered": make_layered,
    "gradient": make_gradient,
    "checkerboard": make_checkerboard,
    "clustered": make_clustered,
    "core_shell": make_core_shell,
}


def generate_synthetic_dataset(n_per_family=200, grid_size=GRID_SIZE,
                                fraction_bounds=(0.05, 0.65), seed=7,
                                out_path="synthetic_topology_dataset.npz"):
    rng = np.random.default_rng(seed)
    volumes, fractions, family_labels = [], [], []

    for family_name, fn in FAMILIES.items():
        for _ in range(n_per_family):
            target_fraction = float(rng.uniform(*fraction_bounds))
            vol = fn(grid_size, target_fraction, rng)
            volumes.append(vol.astype(np.float32))
            fractions.append(float(vol.mean()))  # ACTUAL achieved fraction
            family_labels.append(family_name)

    volumes = np.stack(volumes, axis=0)
    fractions = np.array(fractions, dtype=np.float32)
    family_labels = np.array(family_labels)

    print(f"Generated {len(volumes)} synthetic topologies "
          f"({len(FAMILIES)} families x {n_per_family} each) at grid_size={grid_size}")
    print(f"Achieved volume fraction range: {fractions.min():.3f} - {fractions.max():.3f}")

    np.savez_compressed(out_path, volumes=volumes, fractions=fractions, family_labels=family_labels)
    print(f"Saved {out_path}")
    return volumes, fractions, family_labels


if __name__ == "__main__":
    generate_synthetic_dataset()