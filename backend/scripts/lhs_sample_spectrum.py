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
# being a free per-run input -- so this design only varies:
#   - thickness_cm, w_regolith          (shared with the mono-energetic
#                                         design's bounds, for comparability)
#   - environment ("GCR" or "SPE")
#   - spe_severity (0-1, only meaningful for SPE rows; NaN for GCR rows)
#
# Two independent LHS sub-designs are generated (2D for GCR, 3D for SPE)
# and concatenated, rather than one combined design across a categorical
# "environment" axis -- LHS doesn't handle mixed continuous/categorical
# spaces cleanly, and GCR (solar minimum only, per the project's plan)
# genuinely doesn't need a severity-like axis the way SPE does.
#
# Sample counts default to far fewer than the mono-energetic set's 1500:
# this design space has 2-3 dimensions instead of 3, and (thickness,
# w_regolith) coverage doesn't need to be re-earned per environment --
# only the NEW axis (environment, severity) needs its own density.
# Time a small batch (see run_pinn_batch_spectrum.py's guidance) before
# committing to these counts; they are a starting point, not validated.
# ============================================================

VOXEL_SEED_BASE = 20260907 + 1_000_000  # offset from the mono-energetic
                                          # pipeline's base so geometry
                                          # seeds never collide across
                                          # the two datasets

THICKNESS_BOUNDS = (1.0, 15.0)   # cm -- matches the mono-energetic design
W_REGOLITH_BOUNDS = (0.0, 0.70)  # matches the mono-energetic design


def _shared_columns(thickness_cm, w_regolith, run_id, voxel_seed):
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


def generate_gcr_design(n_samples=150, seed=101):
    print(f"Generating {n_samples} GCR design points (thickness x w_regolith)...")
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    sample = sampler.random(n=n_samples)
    lo = [THICKNESS_BOUNDS[0], W_REGOLITH_BOUNDS[0]]
    hi = [THICKNESS_BOUNDS[1], W_REGOLITH_BOUNDS[1]]
    scaled = qmc.scale(sample, lo, hi)

    rows = []
    for i, row in enumerate(scaled):
        rec = _shared_columns(float(row[0]), float(row[1]), run_id=i, voxel_seed=None)
        rec["environment"] = "GCR"
        rec["spe_severity"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_spe_design(n_samples=350, seed=202, run_id_offset=0):
    print(f"Generating {n_samples} SPE design points (thickness x w_regolith x severity)...")
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    sample = sampler.random(n=n_samples)
    lo = [THICKNESS_BOUNDS[0], W_REGOLITH_BOUNDS[0], 0.0]
    hi = [THICKNESS_BOUNDS[1], W_REGOLITH_BOUNDS[1], 1.0]
    scaled = qmc.scale(sample, lo, hi)

    rows = []
    for i, row in enumerate(scaled):
        run_id = run_id_offset + i
        rec = _shared_columns(float(row[0]), float(row[1]), run_id=run_id, voxel_seed=None)
        rec["environment"] = "SPE"
        rec["spe_severity"] = float(row[2])
        rows.append(rec)
    return pd.DataFrame(rows)


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

    print("\nSpectrum design summary:")
    print(df.groupby("environment")[["thickness_cm", "w_regolith"]].describe())

    output_path = os.path.join(os.path.dirname(__file__), "lhs_design_space_spectrum.csv")
    df.to_csv(output_path, index=False, float_format="%.17g")
    print(f"\n✅ Spectrum LHS dataset saved to {output_path} ({len(df)} total runs: "
          f"{n_gcr} GCR + {n_spe} SPE)")
    return df


if __name__ == "__main__":
    generate_spectrum_lhs_design(n_gcr=500, n_spe=1000)