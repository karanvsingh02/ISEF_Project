"""
Build a local XCOM photon mass-attenuation database WITHOUT scraping NIST's website.

Why not scrape?
----------------
physics.nist.gov/cgi-bin/Xcom/xcom3_1 is a second-stage CGI endpoint. It only
accepts requests that carry the exact hidden fields / state produced by
NIST's own two-step form flow, which isn't documented and can change without
notice. That's why you were getting "XCOM: Error / Invalid User Input" even
with a plausible-looking payload -- it's not an IP-ban or firewall issue,
it's that xcom3_1 doesn't consider the request well-formed.

What this does instead
-----------------------
The `nist-calculators` PyPI package ships NIST's own tabulated XCOM dataset
as a bundled offline HDF5 file (NIST_XCOM.hdf5), digitized directly from the
same source NIST's website reads from. It includes the native per-element
energy grid, which is denser right around each element's absorption edges
(K, L1, L2, L3, ...) -- the same behavior you'd see in the website's output
table. No network calls, no sessions, no rate limits, and it's fully
reproducible (anyone re-running this script gets identical numbers).

Sanity check performed: Pb (Z=82) at 1 MeV -> 0.07102 cm^2/g, matching the
widely-published NIST XCOM reference value.

Known dataset gap -- actinide K-edges
--------------------------------------
The bundled dataset tabulates only the PRE-edge point for the K-edges of
Th (Z=90), Pa (Z=91), and U (Z=92) -- confirmed by cross-checking against
Pb and Au, whose K-edges DO carry both a pre- and a post-edge point spaced
~0.1 eV apart. For these three elements only, the code below reconstructs
the missing post-edge point from the K-shell jump ratio J_K, applied to the
photoelectric term ONLY (coherent, incoherent, and pair production vary
smoothly through the edge and are carried over unchanged into the new point).
The J_K values below are placeholders -- source them from a citable
reference (e.g. Hubbell & Seltzer, or Daoudi et al. 2020, NIMB 479, which
measures jump ratios specifically across Z=76-92) before relying on them.

Install:
    pip install nist-calculators mendeleev --break-system-packages
"""

import json
import os

import numpy as np
import tables
from mendeleev import element as mendeleev_element

# Barn/atom -> cm^2/g conversion factor:
#   1 barn = 1e-24 cm^2 ;  N_A = 6.02214076e23 /mol
#   mu/rho [cm^2/g] = sigma[barn/atom] * (1e-24 * N_A) / A[g/mol]
BARN_TO_CM2G_NUMERATOR = 0.602214076

MIN_Z = 1
MAX_Z = 92  # inclusive, matches your original range

# K-shell jump ratio J_K = tau(just above edge) / tau(just below edge),
# applied to the photoelectric cross section only. SOURCE THESE before
# trusting them for anything you plan to defend at ISEF.
K_EDGE_JUMPS = {
    90: {"E_K_eV": 109651.0, "J_K": 4.88},  # Thorium -- SOURCE THIS
    91: {"E_K_eV": 112601.0, "J_K": 4.80},  # Protactinium -- SOURCE THIS
    92: {"E_K_eV": 115606.0, "J_K": 4.74},  # Uranium -- SOURCE THIS
}


def get_hdf5_path():
    """Locate the HDF5 data file bundled inside the installed xcom package."""
    import xcom
    path = os.path.join(os.path.dirname(xcom.__file__), "data", "NIST_XCOM.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find bundled NIST_XCOM.hdf5 at {path}. "
            "Try: pip install --force-reinstall nist-calculators"
        )
    return path


def build_database():
    print("Loading bundled NIST XCOM dataset (offline, no network calls)...")
    database = {"elements": {}}

    hdf5_path = get_hdf5_path()
    with tables.open_file(hdf5_path, mode="r") as f:
        for z in range(MIN_Z, MAX_Z + 1):
            node_path = f"/Z{z:03d}/data"
            try:
                table = f.get_node(node_path).read()
            except tables.NoSuchNodeError:
                print(f"  Skipping Z={z}: no data node in file.")
                continue

            el = mendeleev_element(z)
            symbol = el.symbol
            atomic_weight = el.atomic_weight

            energy_ev = table["energy"]
            coherent = table["coherent"]
            incoherent = table["incoherent"]
            photoelectric = table["photoelectric"]
            pair_atom = table["pair_atom"]
            pair_electron = table["pair_electron"]

            total_barn = (
                coherent + incoherent + photoelectric + pair_atom + pair_electron
            )
            mu_rho = total_barn * BARN_TO_CM2G_NUMERATOR / atomic_weight  # cm^2/g

            # grid entries as [energy_MeV, total_attenuation_cm2_per_g]
            grid = [
                [round(float(e) / 1e6, 8), float(m)]
                for e, m in zip(energy_ev, mu_rho)
            ]

            # ==========================================
            # ISEF PATCH: ACTINIDE K-EDGES (corrected)
            # ==========================================
            if z in K_EDGE_JUMPS:
                E_K_eV = K_EDGE_JUMPS[z]["E_K_eV"]
                J_K = K_EDGE_JUMPS[z]["J_K"]

                idx = int(np.argmin(np.abs(energy_ev - E_K_eV)))
                if energy_ev[idx] == E_K_eV:  # confirm exact tabulated edge point
                    photo_below = photoelectric[idx]
                    photo_above = photo_below * J_K  # jump ratio: photoelectric ONLY

                    total_barn_above = (
                        coherent[idx] + incoherent[idx] + photo_above
                        + pair_atom[idx] + pair_electron[idx]
                    )
                    mu_above = total_barn_above * BARN_TO_CM2G_NUMERATOR / atomic_weight

                    # insert the missing point 0.1 eV AFTER the edge, matching
                    # the NIST convention seen elsewhere in this dataset
                    # (e.g. Pb's 88004.4/88004.5 eV pair) -- not before it
                    e_above_mev = round(E_K_eV / 1e6 + 0.0000001, 8)
                    grid.insert(idx + 1, [e_above_mev, float(mu_above)])
                    print(f"    -> patched missing post-K-edge point for {symbol}")

            database["elements"][symbol] = {
                "Z": z,
                "A": round(atomic_weight, 4),
                "grid": grid,
            }
            print(f"  {symbol:>2} (Z={z:>3}): {len(grid)} energy points loaded")

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "app", "data", "xcom_data.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    print(f"\nDone. Wrote {len(database['elements'])} elements to {output_path}")


if __name__ == "__main__":
    build_database()