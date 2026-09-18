"""
Run this FIRST, before touching Geant4 at all. Pure Python.

Usage:
    python3 sanity_check_spectra.py

Produces:
    gcr_sanity_check_per_nucleon.png   -- shape comparison (per-nucleon energy)
    gcr_sanity_check_total_energy.png  -- what GPS actually samples (each
                                            species' own total-energy range)
    spe_sanity_check.png               -- SPE at three severities overlaid
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrum_models import (
    gcr_flux_vs_energy_per_nucleon,
    gcr_species_energy_table,
    per_nucleon_energy_grid_gcr,
    spe_flux_vs_kinetic_energy,
    energy_grid_for_environment,
)


def check_gcr():
    print("=== GCR spectra ===")

    # (a) SHAPE comparison, in per-nucleon energy. All three species share
    # the same underlying shape function, so on THIS axis their peaks
    # should land in roughly the same region -- if iron's peak is missing
    # here too, something is still wrong upstream of the grid fix.
    T_pn_grid = per_nucleon_energy_grid_gcr()
    plt.figure(figsize=(7, 5))
    for species in ["proton", "helium", "iron"]:
        flux = gcr_flux_vs_energy_per_nucleon(species, T_pn_grid)
        peak_T_pn = T_pn_grid[np.argmax(flux)]
        print(f"  {species:8s} (per-nucleon): peak at ~{peak_T_pn:.1f} MeV/n")
        plt.loglog(T_pn_grid, flux, label=species)
    plt.xlabel("Kinetic energy per nucleon (MeV/n)")
    plt.ylabel("Relative differential flux")
    plt.title("GCR spectra -- SHAPE comparison (per-nucleon energy)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("gcr_sanity_check_per_nucleon.png", dpi=150, bbox_inches="tight")
    print("  Saved gcr_sanity_check_per_nucleon.png")

    # (b) What GPS actually samples: each species' OWN total-kinetic-
    # energy grid. These will span very different total-energy ranges on
    # purpose (iron's grid reaches ~56x higher total energy than
    # proton's) -- that's the fix, not a bug.
    plt.figure(figsize=(7, 5))
    for species in ["proton", "helium", "iron"]:
        T_total_grid, flux = gcr_species_energy_table(species)
        n_nan = np.isnan(flux).sum()
        peak_T = T_total_grid[np.argmax(flux)]
        print(f"  {species:8s} (total KE): NaNs={n_nan}, peak at ~{peak_T:.1f} MeV total, "
              f"grid spans {T_total_grid.min():.1f}-{T_total_grid.max():.1f} MeV")
        plt.loglog(T_total_grid, flux, label=species)
    plt.xlabel("Total kinetic energy (MeV) -- NOTE: each species' own range")
    plt.ylabel("Relative differential flux")
    plt.title("GCR spectra -- what GPS actually samples (per-species grid)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("gcr_sanity_check_total_energy.png", dpi=150, bbox_inches="tight")
    print("  Saved gcr_sanity_check_total_energy.png\n")


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
    print("Done. For GCR, confirm: (a) all three species' peaks land in a")
    print("similar per-nucleon-energy region in the FIRST plot, and (b) the")
    print("SECOND plot shows each species spanning its own (different, wider")
    print("for heavier ions) total-energy range with a real peak visible --")
    print("iron should no longer look like a monotonically rising curve.")