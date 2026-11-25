import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ============================================================
# LHS design for the GCR/SPE spectrum-based dataset.
#
# This is a SEPARATE, lower-dimensional design space from the
# mono-energetic pipeline (lhs_sample_phase2.py, untouched). Once energy
# is sampled per-event from a spectrum (inside Geant4, via GPS), it stops
# being a free per-run input -- so this design varies:
#   - thickness_cm, w_regolith          (shared with the mono-energetic
#                                         design's bounds, for comparability)
#   - environment ("GCR" or "SPE")
#   - spe_severity (0-1, only meaningful for SPE rows; NaN for GCR rows)
#   - topology_family + its two parameters        <-- ADDED
#
# Two independent LHS sub-designs are generated (for GCR and SPE) and
# concatenated, rather than one combined design across a categorical
# "environment" axis -- LHS doesn't handle mixed continuous/categorical
# spaces cleanly, and GCR (solar minimum only, per the project's plan)
# genuinely doesn't need a severity-like axis the way SPE does.
#
# ---------------------------------------------------------------------
# WHY TOPOLOGY IS NOW SAMPLED
# ---------------------------------------------------------------------
# Two controlled experiments settled this. A seed-sensitivity test over
# the uncorrelated 64^3 shuffle found NO effect at all (omega^2 < 0 on all
# 12 metric/test-point combinations, p >= 0.55) -- a random shuffle
# saturates interface density, so every realization looks identical to a
# traversing particle. A structured-topology test over eight families at
# IDENTICAL volume fraction then found an enormous effect: omega^2 = 0.98,
# family spread 18% against a 1% Monte Carlo noise floor, p = 4e-31.
#
# Two descriptors carry that signal, and both survive dropping the
# high-leverage channeled outlier:
#
#   areal_density_cv   sigma/mu of per-column areal density.
#                      Predicts TRANSMISSION, r^2 = 0.97.
#   z_centroid         centre of mass of regolith along the beam.
#                      Predicts NEUTRON YIELD, r^2 = 0.98 at 5 GeV.
#                      Regolith-first shields produce 15-24% fewer
#                      neutrons than regolith-last at matched volume
#                      fraction AND matched interface density.
#
# They are orthogonal, so each is sampled independently below via the
# graded_correlated family's two parameters. Interface density and
# anisotropy are NOT sampled: the first is weak (r^2 <= 0.26) and the
# second was a leverage artifact (r^2 0.85 -> 0.15 once channeled is
# removed) measured in voxel units on 10:1 anisotropic voxels.
# ============================================================

VOXEL_SEED_BASE = 20260907 + 1_000_000  # offset from the mono-energetic
                                        # pipeline's base so geometry
                                        # seeds never collide across
                                        # the two datasets

THICKNESS_BOUNDS = (1.0, 15.0)   # cm -- matches the mono-energetic design
W_REGOLITH_BOUNDS = (0.0, 0.70)  # matches the mono-energetic design

# graded_correlated parameter ranges.
#
# Correlation length is sampled LOG-uniformly: areal_density_cv responds
# roughly logarithmically to it (ell 1 -> cv 0.10, ell 4 -> 0.20,
# ell 10 -> 0.35), so linear sampling would crowd the high-cv end and
# starve the low-cv end where the measured dose optimum sits (cv ~ 0.16).
CORR_LENGTH_BOUNDS = (0.5, 12.0)   # voxels, log-uniform

# Gradient is sampled linearly and symmetrically about zero so that
# regolith-first and regolith-last are equally represented. That symmetry
# is what makes the ordering effect cleanly estimable rather than
# confounded with overall composition.
GRADIENT_BOUNDS = (-3.0, 3.0)

# Family mix. graded_correlated is the workhorse because it is the only
# family that varies both descriptors independently; the rest are included
# for coverage and as controls, each at a modest share.
#
#   random      baseline, and the statistical equivalent of the old C++
#               shuffle -- keeps the null case inside the dataset
#   layered     the areal_density_cv = 0 extreme (every column identical)
#   gyroid      3D-printable from ISRU feedstock; the manufacturability case
#   channeled   deliberate NEGATIVE CONTROL. Beam-aligned low-density
#               channels are the classic streaming failure mode, measured
#               at +20.7% dose. An optimizer that learns to avoid these is
#               demonstrating physics, not curve-fitting -- but they must
#               be IN the training set for it to learn that.
FAMILY_MIX = {
    "graded_correlated": 0.60,
    "random":            0.10,
    "layered":           0.10,
    "gyroid":            0.10,
    "channeled":         0.10,
}

# For the single-parameter families, the LHS "param A" column is remapped
# onto that family's own parameter, so stratification is preserved rather
# than discarded.
SINGLE_PARAM_RANGE = {
    "random":    None,            # no parameter
    "layered":   (2.0, 32.0),     # n_layers (rounded to int downstream)
    "gyroid":    (4.0, 32.0),     # cell_size_voxels
    "channeled": (2.0, 8.0),      # channel_radius_voxels
}


def _shared_columns(thickness_cm, w_regolith, run_id):
    w_hdpe = 1.0 - w_regolith
    rho_hdpe, rho_regolith = 0.95, 2.75
    regolith_volume_fraction = (w_regolith / rho_regolith) / (
        (w_regolith / rho_regolith) + (w_hdpe / rho_hdpe)
    )
    return {
        "run_id": run_id,
        "thickness_cm": thickness_cm,
        "w_regolith": w_regolith,
        "w_hdpe": w_hdpe,
        "regolith_volume_fraction_target": regolith_volume_fraction,
        "voxel_seed": VOXEL_SEED_BASE + run_id,
    }


def _assign_families(n, seed):
    """
    Deterministic stratified family assignment.

    Drawing families at random would leave the exact proportions to chance,
    and with channeled at 10% a bad draw could leave the negative control
    badly under-represented. Building the list explicitly and shuffling it
    guarantees the mix while keeping the order uncorrelated with the LHS
    rows.
    """
    counts = {f: int(round(p * n)) for f, p in FAMILY_MIX.items()}
    # Absorb any rounding remainder into the workhorse family.
    counts["graded_correlated"] += n - sum(counts.values())

    fams = []
    for f, c in counts.items():
        fams.extend([f] * c)
    fams = np.array(fams, dtype=object)
    np.random.default_rng(seed).shuffle(fams)
    return fams


def _topology_columns(family, u_param_a, u_param_b):
    """
    Map two LHS uniforms onto one family's parameters.

    topology_param_a is the family's primary knob, topology_param_b is the
    gradient (graded_correlated only, NaN elsewhere). Keeping a fixed
    two-column schema means make_topology_maps.py reads the same columns
    for every row regardless of family.
    """
    if family == "graded_correlated":
        lo, hi = np.log10(CORR_LENGTH_BOUNDS[0]), np.log10(CORR_LENGTH_BOUNDS[1])
        a = float(10.0 ** (lo + u_param_a * (hi - lo)))
        b = float(GRADIENT_BOUNDS[0]
                  + u_param_b * (GRADIENT_BOUNDS[1] - GRADIENT_BOUNDS[0]))
        return a, b

    rng = SINGLE_PARAM_RANGE[family]
    if rng is None:
        return float("nan"), float("nan")
    a = float(rng[0] + u_param_a * (rng[1] - rng[0]))
    return a, float("nan")


def _build(n_samples, environment, seed, run_id_offset=0):
    """
    One LHS sub-design.

    Dimensions: thickness, w_regolith, topo_param_a, topo_param_b, and for
    SPE an additional severity axis. Topology parameters go INSIDE the LHS
    rather than being drawn separately, so the joint coverage of
    (thickness, composition, topology) is stratified -- otherwise thin
    shields and high-variance topologies could end up correlated by
    accident and the surrogate would learn a spurious interaction.
    """
    is_spe = environment == "SPE"
    d = 5 if is_spe else 4
    sample = qmc.LatinHypercube(d=d, seed=seed).random(n=n_samples)

    lo = [THICKNESS_BOUNDS[0], W_REGOLITH_BOUNDS[0], 0.0, 0.0]
    hi = [THICKNESS_BOUNDS[1], W_REGOLITH_BOUNDS[1], 1.0, 1.0]
    if is_spe:
        lo.append(0.0)
        hi.append(1.0)
    scaled = qmc.scale(sample, lo, hi)

    families = _assign_families(n_samples, seed=seed + 7919)

    rows = []
    for i, row in enumerate(scaled):
        run_id = run_id_offset + i
        rec = _shared_columns(float(row[0]), float(row[1]), run_id=run_id)
        rec["environment"] = environment
        rec["spe_severity"] = float(row[4]) if is_spe else np.nan

        fam = families[i]
        a, b = _topology_columns(fam, float(row[2]), float(row[3]))
        rec["topology_family"] = fam
        rec["topology_param_a"] = a
        rec["topology_param_b"] = b
        rows.append(rec)

    return pd.DataFrame(rows)


def generate_gcr_design(n_samples=150, seed=101):
    print(f"Generating {n_samples} GCR design points "
          f"(thickness x w_regolith x topology)...")
    return _build(n_samples, "GCR", seed)


def generate_spe_design(n_samples=350, seed=202, run_id_offset=0):
    print(f"Generating {n_samples} SPE design points "
          f"(thickness x w_regolith x severity x topology)...")
    return _build(n_samples, "SPE", seed, run_id_offset=run_id_offset)


def generate_spectrum_lhs_design(n_gcr=150, n_spe=350):
    gcr_df = generate_gcr_design(n_samples=n_gcr)
    spe_df = generate_spe_design(n_samples=n_spe, run_id_offset=n_gcr)

    df = pd.concat([gcr_df, spe_df], ignore_index=True)

    # Re-derive voxel_seed after concatenation so every run_id (now
    # spanning both sub-designs) gets a unique geometry seed.
    df["voxel_seed"] = VOXEL_SEED_BASE + df["run_id"]

    assert df["run_id"].is_unique
    assert df["voxel_seed"].is_unique
    assert df["thickness_cm"].between(*THICKNESS_BOUNDS).all()
    assert df["w_regolith"].between(*W_REGOLITH_BOUNDS).all()
    assert df["environment"].isin(["GCR", "SPE"]).all()
    assert df.loc[df["environment"] == "SPE", "spe_severity"].between(0.0, 1.0).all()
    assert df.loc[df["environment"] == "GCR", "spe_severity"].isna().all()
    assert df["topology_family"].isin(FAMILY_MIX).all()

    gc = df[df.topology_family == "graded_correlated"]
    assert gc["topology_param_a"].between(*CORR_LENGTH_BOUNDS).all()
    assert gc["topology_param_b"].between(*GRADIENT_BOUNDS).all()

    print("\nSpectrum design summary:")
    print(df.groupby("environment")[["thickness_cm", "w_regolith"]].describe())
    print("\nTopology family mix:")
    print(df["topology_family"].value_counts().to_string())

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "lhs_design_space_spectrum.csv")
    df.to_csv(output_path, index=False, float_format="%.17g")
    print(f"\n✅ Spectrum LHS dataset saved to {output_path} ({len(df)} total runs: "
          f"{n_gcr} GCR + {n_spe} SPE)")
    print("\nNEXT: python make_topology_maps.py   (writes one .bin per run "
          "plus topology_descriptors.csv)")
    return df


if __name__ == "__main__":
    generate_spectrum_lhs_design(n_gcr=500, n_spe=1000)