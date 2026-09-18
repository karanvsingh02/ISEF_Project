import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

# ============================================================
# LHS design for the TOPOLOGY-focused Geant4 campaign (Option A groundwork).
#
# Deliberately SEPARATE from both the mono-energetic and GCR/SPE
# pipelines -- nothing here touches those files or their design spaces.
#
# SCOPING DECISION: the source environment is held FIXED at GCR/solar
# minimum (reusing spectrum_models.py's already-validated machinery
# directly), rather than varying topology AND spectrum simultaneously.
# This isolates "does spatial arrangement matter?" as its own clean
# question, keeps the design space tractable, and avoids diluting the
# coverage of the GCR/SPE campaign that's already running. Extending to
# vary source too is a reasonable follow-on once this simpler campaign
# validates the concept -- not a scope cut, a sequencing choice.
#
# Each topology FAMILY gets its own LHS sub-design (same reasoning as
# the GCR/SPE split: LHS doesn't handle mixed continuous/categorical
# spaces cleanly), varying (thickness, w_regolith, <family parameter>)
# where that family has one, or just (thickness, w_regolith) for
# "random" (the current baseline, no extra structure to parameterize).
# ============================================================

GRID_SIZE = 64  # MUST match DetectorConstruction.cc's physics grid (nx=ny=nz=64) --
                 # NOT the 32 used for GAN mechanics testing.

VOXEL_SEED_BASE = 20260907 + 2_000_000  # offset from BOTH the mono-energetic
                                          # AND GCR/SPE pipelines' bases, so
                                          # geometry seeds never collide
                                          # across any of the three datasets.

THICKNESS_BOUNDS = (1.0, 15.0)
W_REGOLITH_BOUNDS = (0.0, 0.70)

# Family-specific parameter bounds (see synthetic_topology_dataset.py for
# what each parameter controls).
FAMILY_PARAM_BOUNDS = {
    "layered": ("n_layers", (2.0, 10.0)),
    "gradient": ("slope", (-2.0, 2.0)),
    "checkerboard": ("block", (1.0, 8.0)),
    "clustered": ("smoothness", (0.5, 6.0)),
    "core_shell": ("radial_bias", (-1.0, 1.0)),
}

# Starting sample counts per family -- a first pass, not validated.
# "random" (2D) needs less density than the 3D families. Time a small
# batch before committing (same discipline as the GCR/SPE campaign).
DEFAULT_N_PER_FAMILY = {
    "random": 100,
    "layered": 150,
    "gradient": 200,
    "checkerboard": 150,
    "clustered": 200,
    "core_shell": 200,
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


def generate_family_design(family, n_samples, run_id_offset, seed):
    rng_seed = seed
    if family == "random":
        sampler = qmc.LatinHypercube(d=2, seed=rng_seed)
        sample = sampler.random(n=n_samples)
        lo = [THICKNESS_BOUNDS[0], W_REGOLITH_BOUNDS[0]]
        hi = [THICKNESS_BOUNDS[1], W_REGOLITH_BOUNDS[1]]
        scaled = qmc.scale(sample, lo, hi)

        rows = []
        for i, row in enumerate(scaled):
            run_id = run_id_offset + i
            rec = _shared_columns(float(row[0]), float(row[1]), run_id)
            rec["topology_family"] = "random"
            rec["family_param"] = np.nan
            rows.append(rec)
        return pd.DataFrame(rows)

    param_name, (lo_p, hi_p) = FAMILY_PARAM_BOUNDS[family]
    sampler = qmc.LatinHypercube(d=3, seed=rng_seed)
    sample = sampler.random(n=n_samples)
    lo = [THICKNESS_BOUNDS[0], W_REGOLITH_BOUNDS[0], lo_p]
    hi = [THICKNESS_BOUNDS[1], W_REGOLITH_BOUNDS[1], hi_p]
    scaled = qmc.scale(sample, lo, hi)

    rows = []
    for i, row in enumerate(scaled):
        run_id = run_id_offset + i
        rec = _shared_columns(float(row[0]), float(row[1]), run_id)
        rec["topology_family"] = family
        rec["family_param"] = float(row[2])
        rows.append(rec)
    return pd.DataFrame(rows)


def generate_topology_lhs_design(n_per_family=None, out_path=None):
    if n_per_family is None:
        n_per_family = DEFAULT_N_PER_FAMILY

    families = ["random", "layered", "gradient", "checkerboard", "clustered", "core_shell"]
    dfs = []
    run_id_offset = 0
    for i, family in enumerate(families):
        n = n_per_family[family]
        print(f"Generating {n} '{family}' design points...")
        df_fam = generate_family_design(family, n, run_id_offset, seed=300 + i)
        dfs.append(df_fam)
        run_id_offset += n

    df = pd.concat(dfs, ignore_index=True)
    df["voxel_seed"] = VOXEL_SEED_BASE + df["run_id"]

    assert df["run_id"].is_unique
    assert df["voxel_seed"].is_unique
    assert df["thickness_cm"].between(*THICKNESS_BOUNDS).all()
    assert df["w_regolith"].between(*W_REGOLITH_BOUNDS).all()
    assert df["topology_family"].isin(families).all()
    assert df.loc[df["topology_family"] == "random", "family_param"].isna().all()
    for family, (param_name, (lo_p, hi_p)) in FAMILY_PARAM_BOUNDS.items():
        vals = df.loc[df["topology_family"] == family, "family_param"]
        assert vals.between(lo_p, hi_p).all(), f"{family} family_param out of bounds"

    print("\nTopology design summary:")
    print(df.groupby("topology_family")[["thickness_cm", "w_regolith", "family_param"]].describe())

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "lhs_design_space_topology.csv")
    df.to_csv(out_path, index=False, float_format="%.17g")
    total = len(df)
    print(f"\n✅ Topology LHS dataset saved to {out_path} ({total} total runs across {len(families)} families)")
    return df


if __name__ == "__main__":
    generate_topology_lhs_design()