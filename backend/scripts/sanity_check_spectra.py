"""
Run this FIRST, before touching Geant4 at all. Pure Python, checks that
the GCR and SPE spectra from spectrum_models.py look physically sensible
(smooth, peaked somewhere reasonable, no NaNs/all-zero/garbage curves)
before you trust them inside a Geant4 macro.

Usage:
    python3 sanity_check_spectra.py

Produces:
    gcr_sanity_check.png   -- all three GCR species overlaid
    spe_sanity_check.png   -- SPE at three severities overlaid
Also prints a few numeric checks directly to the console.
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrum_models import (
    gcr_flux_vs_energy_per_nucleon,
    spe_flux_vs_kinetic_energy,
    energy_grid_for_environment,
    SPECIES,
)


def check_gcr():
    print("=== GCR spectra ===")
    E_total = energy_grid_for_environment("GCR")  # total kinetic energy, MeV

    plt.figure(figsize=(7, 5))
    for species in ["proton", "helium", "iron"]:
        A = SPECIES[species]["A"]
        flux = gcr_flux_vs_energy_per_nucleon(species, E_total / A)

        n_nan = np.isnan(flux).sum()
        n_zero = (flux == 0).sum()
        peak_E = E_total[np.argmax(flux)]

        print(f"  {species:8s}: NaNs={n_nan}, zeros={n_zero}/{len(flux)}, "
              f"peak at ~{peak_E:.1f} MeV total KE, max={flux.max():.3e}")

        plt.loglog(E_total, flux, label=species)

    plt.xlabel("Total kinetic energy (MeV)")
    plt.ylabel("Relative differential flux")
    plt.title("GCR spectra (solar minimum, force-field approximation)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("gcr_sanity_check.png", dpi=150, bbox_inches="tight")
    print("  Saved gcr_sanity_check.png\n")


def check_spe():
    print("=== SPE spectra ===")
    E = energy_grid_for_environment("SPE")

    plt.figure(figsize=(7, 5))
    for sev in [0.0, 0.5, 1.0]:
        flux = spe_flux_vs_kinetic_energy(E, severity=sev)
        n_nan = np.isnan(flux).sum()
        peak_E = E[np.argmax(flux)]
        print(f"  severity={sev:.1f}: NaNs={n_nan}, peak at ~{peak_E:.1f} MeV, max={flux.max():.3e}")
        plt.loglog(E, flux, label=f"severity={sev:.1f}")

    plt.xlabel("Kinetic energy (MeV)")
    plt.ylabel("Relative differential flux")
    plt.title("SPE spectra (Xapsos Weibull form)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("spe_sanity_check.png", dpi=150, bbox_inches="tight")
    print("  Saved spe_sanity_check.png\n")


if __name__ == "__main__":
    check_gcr()
    check_spe()
    print("Done. Open the two PNGs and confirm: smooth curves, sensible peak")
    print("locations, no NaNs, no all-zero species -- before running any Geant4.")