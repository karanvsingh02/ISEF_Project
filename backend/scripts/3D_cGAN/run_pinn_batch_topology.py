import concurrent.futures
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from synthetic_topology_dataset import FAMILIES
from topology_io import write_topology_file
from run_pinn_batch_spectrum import build_gcr_source_macro_lines  # reuse, don't reimplement

# ============================================================
# Batch runner for the TOPOLOGY campaign. Deliberately a SEPARATE script
# from run_pinn_batch.py AND run_pinn_batch_spectrum.py -- nothing here
# touches either of those, or their output folders/macro naming, so
# there is zero risk to the (already-validated, currently running)
# GCR/SPE pipeline.
#
# NOTE: this drives Geant4 through a NEW capability
# (/shield/setTopologyFile) added to DetectorConstruction.cc that has
# NOT been verified against a real Geant4 run -- only the Python-side
# file write/read round-trip has been checked (topology_io.py). Run
# run_topology_ordering_check() (below) FIRST, on a single trivially-
# verifiable topology, before trusting this at any scale.
# ============================================================

OUTPUT_DIR = "pinn_training_data_topology"
TOPOLOGY_FILE_DIR = "topology_files"
GRID_SIZE = 64  # must match DetectorConstruction.cc's nx=ny=nz=64
BEAM_ON_EVENTS = 3000  # starting point -- re-time before trusting at scale

_FAMILY_PARAM_NAME = {
    "layered": "n_layers", "gradient": "slope",
    "checkerboard": "block", "clustered": "smoothness",
    "core_shell": "radial_bias",
}


def build_topology_volume(row):
    family = row["topology_family"]
    fn = FAMILIES[family]
    rng = np.random.default_rng(int(row["voxel_seed"]))  # topology's OWN
    # deterministic seed, independent of Geant4's physics-event seed
    # (set separately via /random/setSeeds, same convention as the other
    # two pipelines).

    target_fraction = row["w_regolith"]  # reuse w_regolith as the target
    # volume fraction for topology construction, keeping this column's
    # meaning consistent with the mono-energetic/GCR/SPE pipelines even
    # though the ACHIEVED fraction (what actually matters physically) can
    # differ slightly -- build_pinn_dataset_topology.py records the
    # achieved fraction separately.

    if family == "random":
        vol = fn(GRID_SIZE, target_fraction, rng)
    else:
        param_name = _FAMILY_PARAM_NAME[family]
        raw_param = row["family_param"]
        if family in ("layered", "checkerboard"):
            raw_param = int(round(raw_param))
        kwargs = {param_name: raw_param}
        vol = fn(GRID_SIZE, target_fraction, rng, **kwargs)

    return vol


def merge_thread_outputs(run_id, output_dir=OUTPUT_DIR):
    pattern = os.path.join(output_dir, f"output_run_{run_id}_nt_PINN_Data_t*.csv")
    thread_files = sorted(glob.glob(pattern))
    if not thread_files:
        raise RuntimeError(f"No thread output files found for run {run_id} (pattern: {pattern})")

    merged_path = os.path.join(output_dir, f"output_run_{run_id}.csv")
    all_data_lines = []
    for tf in thread_files:
        with open(tf, "r") as infile:
            lines = infile.readlines()
        data_lines = [l.rstrip("\n") for l in lines if not l.startswith("#") and l.strip()]
        all_data_lines.extend(data_lines)

    with open(merged_path, "w") as outfile:
        outfile.write("\n".join(all_data_lines) + "\n")

    for tf in thread_files:
        os.remove(tf)

    return merged_path


def worker_simulation(row, beam_on_events=BEAM_ON_EVENTS, verbose=False):
    run_id = int(row["run_id"])
    mac_filename = f"run_topology_{run_id}.mac"
    geant4_exe = "./build/ShieldSim"

    os.makedirs(TOPOLOGY_FILE_DIR, exist_ok=True)
    vol = build_topology_volume(row)
    achieved_fraction = float(vol.mean())
    topo_path = os.path.join(TOPOLOGY_FILE_DIR, f"topology_{run_id}.txt")
    write_topology_file(vol, topo_path)

    with open(mac_filename, "w") as f:
        f.write("/control/verbose 0\n")
        f.write("/run/verbose 0\n")
        f.write(f"/random/setSeeds {run_id} {run_id + 10000}\n")

        f.write(f"/shield/setThickness {row['thickness_cm']} cm\n")
        f.write(f"/shield/setTopologyFile {os.path.abspath(topo_path)}\n")
        # setRegolithVolumeFraction is harmless/unused when a topology file
        # is given (see DetectorConstruction.cc), kept only for logging
        # consistency with the other two pipelines.
        f.write(f"/shield/setRegolithVolumeFraction {achieved_fraction}\n")

        f.write("/run/initialize\n")

        for line in build_gcr_source_macro_lines():  # fixed GCR-solar-min source, all 3 species
            f.write(line + "\n")

        f.write(f"/analysis/setFileName {OUTPUT_DIR}/output_run_{run_id}.csv\n")
        f.write(f"/run/beamOn {beam_on_events}\n")

    if verbose:
        print(f"--- macro for run_id {run_id} ({row['topology_family']}, "
              f"achieved_fraction={achieved_fraction:.3f}) ---")
        with open(mac_filename) as f:
            print(f.read()[:2000], "... [truncated]")

    try:
        subprocess.run(
            [geant4_exe, mac_filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
        merge_thread_outputs(run_id)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Geant4 crashed: {e.stderr.decode('utf-8').strip()}")
    finally:
        if os.path.exists(mac_filename):
            os.remove(mac_filename)

    return run_id, achieved_fraction


def run_topology_ordering_check():
    """
    RUN THIS FIRST -- before any real batch. Builds one trivially
    human-verifiable topology (the front HALF of the shield along the
    beam/thickness axis z is pure regolith; the back half is pure HDPE)
    and writes a single Geant4 job for it. This doesn't need a PINN or
    any statistics -- just eyeball the resulting dose relative to a
    same-thickness all-HDPE or all-regolith run you already have from
    the mono-energetic dataset. An x/y/z ordering mismatch between the
    Python file writer and Geant4's reader would put the "dense" half on
    the WRONG side of the beam path -- this specific asymmetric-along-z
    test is the one most likely to expose that.
    """
    vol = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    vol[:, :, :GRID_SIZE // 2] = 1.0  # front half (low z, beam entrance side) = regolith

    os.makedirs(TOPOLOGY_FILE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    topo_path = os.path.join(TOPOLOGY_FILE_DIR, "topology_ORDERING_CHECK.txt")
    write_topology_file(vol, topo_path)

    mac_filename = "run_topology_ORDERING_CHECK.mac"
    with open(mac_filename, "w") as f:
        f.write("/control/verbose 0\n/run/verbose 0\n")
        f.write("/random/setSeeds 999999 1009999\n")
        f.write("/shield/setThickness 10.0 cm\n")
        f.write(f"/shield/setTopologyFile {os.path.abspath(topo_path)}\n")
        f.write("/shield/setRegolithVolumeFraction 0.5\n")
        f.write("/run/initialize\n")
        for line in build_gcr_source_macro_lines():
            f.write(line + "\n")
        f.write(f"/analysis/setFileName {OUTPUT_DIR}/output_run_999999.csv\n")
        f.write("/run/beamOn 5000\n")

    print(f"Wrote {mac_filename} and {topo_path}.")
    print("Run manually: ./build/ShieldSim run_topology_ORDERING_CHECK.mac")
    print("Then compare the resulting dose to a same-thickness pure-regolith and")
    print("pure-HDPE run from your mono-energetic dataset -- it should sit BETWEEN")
    print("them. If it instead matches one of the pure-material extremes exactly,")
    print("or looks physically nonsensical, the file ordering convention is")
    print("mismatched between Python and Geant4 -- stop and debug before any batch.")


def run_small_timing_test(csv_path="lhs_design_space_topology.csv", n=10):
    df = pd.read_csv(csv_path)
    sample = df.sample(min(n, len(df)), random_state=1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    times = []
    for _, row in sample.iterrows():
        t0 = time.time()
        worker_simulation(row.to_dict())
        dt = time.time() - t0
        times.append(dt)
        print(f"  run_id {int(row['run_id'])} ({row['topology_family']}): {dt:.1f}s")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f}s/run. Full design ({len(df)} runs) estimate: "
          f"{avg * len(df) / 3600:.1f} hours on this hardware.")


def run_parallel_batch(csv_path="lhs_design_space_topology.csv"):
    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOPOLOGY_FILE_DIR, exist_ok=True)

    run_configs = df.to_dict(orient="records")
    max_workers = max(1, os.cpu_count() - 2)

    print(f"🚀 Spawning parallel worker pool across {max_workers} cores for topology runs...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_simulation, config): config["run_id"] for config in run_configs}

        for future in concurrent.futures.as_completed(futures):
            run_id = int(futures[future])
            try:
                future.result()
                print(f"✅ Completed Run ID {run_id}")
            except Exception as e:
                print(f"❌ Error in Run ID {run_id}: {e}")


if __name__ == "__main__":
    run_parallel_batch()