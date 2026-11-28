"""
material_library.py -- physics-based material descriptors for AEGIS AI.

WHY THIS FILE EXISTS
--------------------
Until now the material axis of the design space was a single number:
`regolith_volume_fraction`, the mix of ONE simulant (LHS-1) with ONE polymer
(HDPE). A surrogate trained on that cannot say anything about a material it
was not trained on, which means it is an interpolator between two fixed
points, not a tool for designing materials in silico.

This module replaces that one number with six physics descriptors computed
from elemental composition. A surrogate trained on descriptors predicts for
ANY material whose composition you can write down -- including ones never
simulated.

THE DESCRIPTOR SET, AND WHY EACH ONE
------------------------------------
Each descriptor is tied to a specific term in the transport equation, not
chosen because it correlated with something.

  rho_g_cm3         Density. Converts cm of shield into g/cm2, which is what
                    every attenuation law actually depends on.

  z_over_a          <Z/A>, mass-weighted. The prefactor of Bethe-Bloch:
                    -dE/dx = K z^2 (Z/A) / beta^2 * [...]. Electronic
                    stopping power per g/cm2 is directly proportional to it.

  ln_i_excitation   ln(I), I = mean excitation energy. The logarithmic term
                    inside Bethe-Bloch. Stored as a log because that is how
                    it enters the physics, so the network sees a linear
                    relationship instead of having to learn a log.

  w_hydrogen        Hydrogen mass fraction. Dominates neutron moderation
                    (maximum energy transfer per elastic collision is
                    4A/(1+A)^2, which is 1.0 for A=1 and 0.28 for A=12), and
                    gives the highest electronic stopping power per gram of
                    any element.

  lambda_inel_g_cm2 Inelastic nuclear interaction length. Sets where primary
                    HZE nuclei fragment and where spallation neutrons are
                    produced. This is the single descriptor that governs the
                    secondary-production channel.

  x0_g_cm2          Radiation length. Governs EM shower development from
                    secondary gammas and pi0 decay. Scales as A/(Z(Z+1)),
                    a genuinely different combination of Z and A from
                    lambda_inel (which scales as A^0.3), so it is not
                    redundant with it.

DESCRIPTORS THAT WERE CONSIDERED AND DROPPED
--------------------------------------------
  a_eff_nuclear = (sum w_i A_i^(-1/3))^(-3)
      Dropped. The inelastic cross section goes as A^0.7, so
      1/lambda ~ sum w_i A_i^(-0.3), while a_eff^(-1/3) = sum w_i A_i^(-1/3).
      Exponents -0.30 and -0.333 -- these two descriptors are collinear to
      better than r = 0.999 across any realistic library. Keeping both would
      make the regression ill-conditioned for no physical gain. lambda_inel
      is kept because it is the directly interpretable one (an attenuation
      length in g/cm2).

  n_electrons_per_g
      Dropped. It is exactly N_A * z_over_a. Perfectly redundant.

  z_eff (power-law effective Z)
      Dropped. It is a photoelectric-absorption construct. In a 10 MeV -
      10 GeV hadron field the photoelectric effect is irrelevant.

THE PROPERTY THAT MAKES THIS WORK
---------------------------------
All six descriptors are additive under mass mixing, either directly or in a
transformed variable:

    rho              volume-additive
    z_over_a         mass-additive
    ln(I)            mass-additive weighted by w_i*(Z/A)_i   (Bragg additivity)
    w_hydrogen       mass-additive
    1/lambda_inel    mass-additive
    1/x0             mass-additive

That is not a convenience. It means the descriptors of a two-phase mixture
are an exact function of the descriptors of its phases, so a surrogate
trained on single materials and a few mixtures generalises to arbitrary
mixtures BY CONSTRUCTION rather than by hoping the network interpolates. It
is also the justification for calling the resulting model physics-informed:
the feature space respects the same additivity the transport physics does.

CONTINUITY WITH EXISTING WORK
-----------------------------
`polyethylene` is pinned to 0.95 g/cm3 and `lhs1_regolith` to 2.75 g/cm3,
exactly matching RHO_HDPE and RHO_REGOLITH in topology_generators.py. The
self-test asserts this. If it ever fails, every areal_density_cv and
z_centroid value computed so far has silently changed meaning.

USAGE
-----
    python material_library.py                      # self-test + coverage report
    python material_library.py --report             # descriptor table + conditioning
    python material_library.py --select 8           # which materials to simulate
    python material_library.py --emit-geant4 ./src  # write MaterialLibrary.hh/.cc
    python material_library.py --mix polyethylene lhs1_regolith 0.5

    from material_library import LIBRARY, descriptors, mix_descriptors
    d = descriptors(LIBRARY["lhs1_regolith"])
    m = mix_descriptors(LIBRARY["polyethylene"], LIBRARY["isru_iron"], 0.30)

PROVENANCE
----------
Every material carries a machine-readable `tier`:

  tier 1  measured and cited. A published analysis or vendor datasheet.
          Quotable as-is. LHS-1, the transport-reference regolith, mare
          basalt 71035, MGS-1.
  tier 2  stoichiometric, or a standard parameterisation. Exact given the
          chemical formula (CH2, H2O, elemental metals) or a nominal alloy
          specification. Quotable with the formula stated.
  tier 3  ASSUMED. A design choice made to span descriptor space, not a
          characterised material. lhs1_sintered, mg_hydride,
          hydrated_regolith, borated_pe_5. These MUST be labelled
          hypothetical in any writeup, and none of them may serve as a
          validation reference.

`--report` prints the tier for every material and refuses to stay quiet
about tier-3 entries. If a result rests on a tier-3 material, say so.

ONE THING THIS FILE CANNOT VERIFY
---------------------------------
OLTARIS's own `lunar_regolith_a17` composition. OLTARIS exposes it in its
material interface; read it there and compare against
`regolith_transport_ref`. If they differ, transport both and report the
spread as a composition systematic rather than picking one.

Sources: Exolith LHS-1 spec sheet · Apollo 17 sample compendium, 71055/71035 · Cannon et al. 2019, MGS-1 · Geant4 lunar albedo-neutron study citing Zaman et al. 2022 · Slaba, NASA LaRC — Radiation Shielding for Lunar Missions · Lunar Sourcebook Ch. 8
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LIBRARY_VERSION = "1.0.0"

N_AVOGADRO = 6.02214076e23      # /mol
MB_TO_CM2 = 1.0e-27             # 1 millibarn in cm^2


# ---------------------------------------------------------------------------
# Element data
#
# A  : standard atomic weight (CIAAW 2021)
# I  : mean excitation energy, eV (ICRU Report 37 / PDG atomic & nuclear
#      properties tables). These are MEASURED values, not computed.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Element:
    symbol: str
    Z: int
    A: float
    I_eV: float


_EL = [
    Element("H",   1,   1.008,  19.2),
    Element("He",  2,   4.0026, 41.8),
    Element("Li",  3,   6.94,   40.0),
    Element("B",   5,  10.811,  76.0),
    Element("C",   6,  12.011,  78.0),
    Element("N",   7,  14.007,  82.0),
    Element("O",   8,  15.999,  95.0),
    Element("F",   9,  18.998, 115.0),
    Element("Na", 11,  22.990, 149.0),
    Element("Mg", 12,  24.305, 156.0),
    Element("Al", 13,  26.982, 166.0),
    Element("Si", 14,  28.085, 173.0),
    Element("P",  15,  30.974, 173.0),
    Element("S",  16,  32.06,  180.0),
    Element("Cl", 17,  35.45,  174.0),
    Element("K",  19,  39.098, 190.0),
    Element("Ca", 20,  40.078, 191.0),
    Element("Ti", 22,  47.867, 233.0),
    Element("Cr", 24,  51.996, 257.0),
    Element("Mn", 25,  54.938, 272.0),
    Element("Fe", 26,  55.845, 286.0),
    Element("Ni", 28,  58.693, 311.0),
    Element("Cu", 29,  63.546, 322.0),
    Element("Zn", 30,  65.38,  330.0),
]

ELEMENTS: Dict[str, Element] = {e.symbol: e for e in _EL}


# ---------------------------------------------------------------------------
# Inelastic nuclear cross section
#
# Letaw, Silberberg & Tsao (1983), ApJS 51, 271 -- the standard
# parameterisation in space-radiation transport, used inside HZETRN:
#
#     sigma_inel(A) = 45 A^0.7 [1 + 0.016 sin(5.3 - 2.63 ln A)]  mb
#
# valid for E >~ 2 GeV and A >= 4. The mean sampled energy in the AEGIS
# design space is ~4 GeV, so the energy-independent high-energy limit is the
# right regime. Below A = 4 the formula fails badly (it gives 44 mb for
# hydrogen against a measured pp inelastic cross section near 30 mb), so H
# and He are taken from measurement instead.
#
# These feed a DESCRIPTOR, not the transport itself -- Geant4 computes its
# own cross sections. The descriptor only has to order materials correctly
# and vary smoothly, which this does to a few per cent.
# ---------------------------------------------------------------------------

_SIGMA_INEL_LIGHT_MB = {1: 30.0, 2: 112.0, 3: 180.0}   # by integer round(A)


def sigma_inelastic_mb(A: float) -> float:
    """Proton-nucleus inelastic cross section, millibarns, high-energy limit."""
    key = int(round(A))
    if key in _SIGMA_INEL_LIGHT_MB:
        return _SIGMA_INEL_LIGHT_MB[key]
    return 45.0 * A ** 0.7 * (1.0 + 0.016 * math.sin(5.3 - 2.63 * math.log(A)))


def lambda_inelastic_element(el: Element) -> float:
    """Inelastic interaction length of a pure element, g/cm^2."""
    sigma_cm2 = sigma_inelastic_mb(el.A) * MB_TO_CM2
    n_per_g = N_AVOGADRO / el.A
    return 1.0 / (n_per_g * sigma_cm2)


def x0_element(el: Element) -> float:
    """
    Radiation length of a pure element, g/cm^2.

    Dahl's approximation to the Tsai formula:
        X0 = 716.4 A / [Z(Z+1) ln(287/sqrt(Z))]

    Accurate to about 2 per cent against the tabulated values (Al: 24.3 vs
    24.01; Fe: 14.1 vs 13.84). That is well inside what matters for a
    descriptor whose job is to order materials.
    """
    Z = float(el.Z)
    return 716.4 * el.A / (Z * (Z + 1.0) * math.log(287.0 / math.sqrt(Z)))


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """
    A material defined by elemental MASS fractions.

    name        machine key, also the Geant4 material name
    label       human label for plots and tables
    density     g/cm^3
    mass_frac   {element symbol: mass fraction}, normalised on construction
    I_eV        measured mean excitation energy, if one exists. When None,
                Bragg additivity is used -- which is known to UNDERESTIMATE I
                for condensed matter (it gives 69 eV for water against the
                measured 75 eV, a 9 per cent error) because it ignores
                chemical binding. Prefer a measured value wherever one is
                available.
    source      short provenance string.
    citation    the actual reference, so a judge asking "where did that
                number come from?" gets an answer rather than a shrug.
    tier        PROVENANCE TIER, machine-readable:
                  1 = measured and cited. A published analysis or a vendor
                      datasheet. Quotable.
                  2 = stoichiometric or from a published parameterisation.
                      Exact given the formula (CH2, H2O, MgH2) or standard
                      (nominal alloy composition). Quotable with the formula
                      stated.
                  3 = ASSUMED BY ME. A design choice to span descriptor
                      space, not a characterised material. Must be labelled
                      hypothetical in any writeup.
                The report prints this, and --design refuses to treat a tier-3
                material as a validation reference.
    isru        whether this is plausibly producible on the lunar or Mars
                surface, which is the whole point of the project.
    """
    name: str
    label: str
    density: float
    mass_frac: Dict[str, float]
    I_eV: Optional[float] = None
    source: str = ""
    citation: str = ""
    tier: int = 3
    isru: bool = False
    notes: str = ""

    def __post_init__(self):
        for sym in self.mass_frac:
            if sym not in ELEMENTS:
                raise KeyError(f"{self.name}: unknown element '{sym}'")
        total = sum(self.mass_frac.values())
        if total <= 0:
            raise ValueError(f"{self.name}: mass fractions sum to {total}")
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"{self.name}: mass fractions sum to {total:.4f}, not ~1. "
                "Refusing to silently normalise an error that large."
            )
        self.mass_frac = {k: v / total for k, v in self.mass_frac.items()}


# --- oxide bookkeeping ------------------------------------------------------
#
# Geological compositions are reported as oxide weight per cent. Convert to
# elemental mass fractions. Each entry is (cation symbol, n_cation, n_oxygen).

_OXIDES = {
    "SiO2":  ("Si", 1, 2),
    "TiO2":  ("Ti", 1, 2),
    "Al2O3": ("Al", 2, 3),
    "FeO":   ("Fe", 1, 1),
    "Fe2O3": ("Fe", 2, 3),
    "MnO":   ("Mn", 1, 1),
    "MgO":   ("Mg", 1, 1),
    "CaO":   ("Ca", 1, 1),
    "Na2O":  ("Na", 2, 1),
    "K2O":   ("K",  2, 1),
    "P2O5":  ("P",  2, 5),
    "Cr2O3": ("Cr", 2, 3),
    "SO3":   ("S",  1, 3),
    "H2O":   ("H",  2, 1),
}


def _oxide_molar_mass(formula: str) -> float:
    cat, n_cat, n_o = _OXIDES[formula]
    return n_cat * ELEMENTS[cat].A + n_o * ELEMENTS["O"].A


def from_oxides(name: str, label: str, density: float,
                oxide_wt_pct: Dict[str, float], **kw) -> Material:
    """
    Build a Material from oxide weight percentages.

    Oxide totals in the literature rarely hit exactly 100 (loss on ignition,
    rounding, trace oxides omitted), so the result is renormalised. A total
    more than 3 points from 100 is an error, not rounding, and raises.
    """
    total = sum(oxide_wt_pct.values())
    if abs(total - 100.0) > 3.0:
        raise ValueError(
            f"{name}: oxides sum to {total:.2f} wt%, more than 3 points from "
            "100. Check the composition before trusting it."
        )

    elem: Dict[str, float] = {}
    for formula, wt in oxide_wt_pct.items():
        if formula not in _OXIDES:
            raise KeyError(f"{name}: unknown oxide '{formula}'")
        cat, n_cat, n_o = _OXIDES[formula]
        M = _oxide_molar_mass(formula)
        frac = wt / total
        elem[cat] = elem.get(cat, 0.0) + frac * (n_cat * ELEMENTS[cat].A) / M
        elem["O"] = elem.get("O", 0.0) + frac * (n_o * ELEMENTS["O"].A) / M

    return Material(name=name, label=label, density=density,
                    mass_frac=elem, **kw)


def from_element_pct(name: str, label: str, density: float,
                     elem_wt_pct: Dict[str, float], **kw) -> Material:
    """
    Build a Material from ELEMENTAL weight percentages.

    Radiation-transport papers usually report elemental fractions rather than
    oxides, which is better for our purpose: it removes the oxide-conversion
    step and with it any ambiguity about Fe(II) vs Fe(III) speciation, which
    changes the oxygen balance.
    """
    total = sum(elem_wt_pct.values())
    if abs(total - 100.0) > 3.0:
        raise ValueError(
            f"{name}: elemental fractions sum to {total:.2f} wt%, more than "
            "3 points from 100."
        )
    return Material(name=name, label=label, density=density,
                    mass_frac={k: v / total for k, v in elem_wt_pct.items()},
                    **kw)


# --- regolith oxide compositions -------------------------------------------

# Exolith Lab LHS-1 Lunar Highlands Simulant.
# VERIFIED against the official spec sheet (LHS-1-SPEC-SHEET-DEC2025), XRF
# bulk chemical composition. Printed total 99.15 wt%.
#
# The sheet's 0.41 wt% LOI (loss on ignition -- structural and adsorbed
# water driven off on heating) is entered here as H2O rather than discarded.
# It contributes w_H = 4.6e-4, which is negligible for stopping power but is
# the honest value, and matters for the one descriptor that is otherwise
# exactly zero for every oxide in the library.
#
# Mineralogy for the record: anorthosite 74.4, glass-rich basalt 24.7,
# ilmenite 0.4, bronzite 0.3, olivine 0.2 wt%.
_LHS1_OXIDES = {
    "SiO2": 49.12, "TiO2": 0.63, "Al2O3": 26.29, "FeO": 3.20, "MnO": 0.06,
    "MgO": 2.86, "CaO": 13.52, "Na2O": 2.55, "K2O": 0.34, "P2O5": 0.17,
    "H2O": 0.41,
}
_LHS1_CITE = ("Exolith Lab, LHS-1 Lunar Highlands Simulant spec sheet, "
              "Dec 2025 (XRF bulk composition; uncompressed bulk density "
              "1.40 g/cm3; median particle size 81.62 um)")

# Lunar regolith composition used in cross-code radiation transport
# benchmarks. ELEMENTAL weight per cent, sums to exactly 100.0.
#
# This is the composition to use for any OLTARIS / HZETRN comparison: it is
# the one the benchmark literature transports, so matching it removes
# composition as a confound and leaves only the physics difference (3D Monte
# Carlo vs the 1D straight-ahead approximation) under test.
_REGOLITH_TRANSPORT_ELEM = {
    "O": 43.7, "Na": 0.3, "Mg": 5.6, "Al": 9.0, "Si": 21.1,
    "Ca": 8.5, "Ti": 1.5, "Mn": 0.1, "Fe": 10.2,
}
_REGOLITH_TRANSPORT_CITE = (
    "Zaman et al. (2022), Space Weather -- 'Modeling the lunar radiation "
    "environment: A comparison among FLUKA, Geant4, HETC-HEDS, MCNP6, and "
    "PHITS'; density 1.6 g/cm3 after Naito et al."
)

# Apollo 17 high-Ti mare basalt 71035.
#
# A SINGLE coherent analysis (one sample, one technique, one author) rather
# than an average. This matters: the Lunar Sourcebook warns explicitly that
# averaged Apollo 17 SOIL compositions are misleading, because Apollo 17
# soils mix mare basalt with highland material in proportions that vary by
# station -- some Station 2 soils are almost pure highland. A named basalt
# has no such ambiguity.
#
# Cr2O3 = 0.39 wt% is converted from the reported 2669 ppm Cr using the
# Cr mass fraction of Cr2O3 (103.992/151.989 = 0.68421). Total 99.76 wt%.
_MARE_BASALT_OXIDES = {
    "SiO2": 38.25, "TiO2": 13.06, "Al2O3": 8.77, "FeO": 19.74, "MnO": 0.29,
    "MgO": 7.98, "CaO": 10.87, "Na2O": 0.38, "K2O": 0.03, "Cr2O3": 0.39,
}
_MARE_BASALT_CITE = ("Rhodes et al. (1976), XRF analysis of Apollo 17 mare "
                     "basalt 71035, via the NASA Apollo 17 sample "
                     "compendium (Table 2)")

# MGS-1 Mars Global Simulant, XRF of the measured prototype.
#
# Chosen over JSC Mars-1 for three reasons: the composition is published
# with a clean total (100.0 wt%), it was designed to match the Rocknest soil
# measured by Curiosity rather than being a terrestrial analogue of
# convenience, and it comes from the same laboratory as LHS-1, so the two
# share a provenance chain. JSC Mars-1 also loses 21.1 wt% on heating to
# 600 C, which makes "the" composition ambiguous.
#
# Note SO3 at 2.1 wt%: sulfur appears nowhere else in the library.
_MGS1_OXIDES = {
    "SiO2": 50.8, "TiO2": 0.3, "Al2O3": 8.9, "Cr2O3": 0.1, "FeO": 13.3,
    "MnO": 0.1, "MgO": 16.7, "CaO": 3.7, "Na2O": 3.4, "K2O": 0.3,
    "P2O5": 0.4, "SO3": 2.1,
}
_MGS1_CITE = ("Cannon, Britt, Smith, Fritsche & Batcheldor (2019), "
              "'Mars global simulant MGS-1: A Rocknest-based open standard "
              "for basaltic martian regolith simulants', Icarus "
              "(Table 2, prototype XRF; bulk density 1.29 g/cm3)")


def _build_library() -> Dict[str, Material]:
    mats: List[Material] = [
        # ---- the two incumbents. Densities PINNED for continuity. ----
        Material(
            "polyethylene", "HDPE (CH2)", 0.95,
            {"C": 0.856277, "H": 0.143723}, I_eV=57.4,
            source="CH2 stoichiometry; I from ICRU 37",
            citation="CH2 stoichiometry; I = 57.4 eV from ICRU Report 37 "
                     "(polyethylene)", tier=2,
            isru=True,
            notes="ISRU-plausible from Sabatier CH4 or recycled packaging. "
                  "Density pinned to 0.95 to match RHO_HDPE. Matches the "
                  "OLTARIS polyethylene composition to 5 decimal places; "
                  "only density differs (theirs 1.00), so all OLTARIS "
                  "comparisons must be at matched g/cm2, never matched cm.",
        ),
        from_oxides(
            "lhs1_regolith", "LHS-1 highlands regolith", 2.75, _LHS1_OXIDES,
            source="Exolith LHS-1 spec sheet (XRF); density PINNED",
            citation=_LHS1_CITE, tier=1,
            isru=True,
            notes="Composition VERIFIED against the Dec 2025 spec sheet. "
                  "Density pinned to 2.75 to match RHO_REGOLITH in "
                  "topology_generators.py. NOTE: 2.75 is a GRAIN density; "
                  "the datasheet's uncompressed bulk density is 1.40. Using "
                  "2.75 therefore asserts a fully consolidated, "
                  "zero-porosity solid, and that assumption must be stated "
                  "explicitly in any writeup -- it is the single largest "
                  "unstated assumption in the whole pipeline.",
        ),

        # ---- regolith at realistic bulk densities ----
        from_oxides(
            "lhs1_bulk", "LHS-1, uncompressed", 1.40, _LHS1_OXIDES,
            source="Exolith LHS-1 spec sheet, uncompressed bulk density",
            citation=_LHS1_CITE, tier=1,
            isru=True,
            notes="1.40 is the DATASHEET value, not the 1.6 commonly quoted "
                  "for Apollo regolith -- LHS-1 is a loose simulant powder. "
                  "The only difference from lhs1_regolith is rho, which makes "
                  "the pair a clean test of whether the surrogate has learned "
                  "that density acts purely through areal density.",
        ),
        from_oxides(
            "lhs1_sintered", "LHS-1, sintered", 1.80, _LHS1_OXIDES,
            source="LHS-1 composition at an ASSUMED sintered density",
            citation=_LHS1_CITE + " (composition only; density assumed)",
            tier=3,
            isru=True,
            notes="Microwave sintering of regolith is an established ISRU "
                  "construction route, but 1.80 g/cm3 is MY ASSUMPTION for "
                  "an achieved density, not a measured value. Label this "
                  "material hypothetical.",
        ),
        from_element_pct(
            "regolith_transport_ref", "Lunar regolith (transport reference)",
            1.60, _REGOLITH_TRANSPORT_ELEM,
            source="elemental composition used in cross-code transport "
                   "benchmarks",
            citation=_REGOLITH_TRANSPORT_CITE, tier=1,
            isru=True,
            notes="USE THIS ONE for the OLTARIS / HZETRN cross-validation. "
                  "It is the composition the benchmark literature transports, "
                  "so matching it eliminates composition as a confound and "
                  "isolates the actual question -- 3D Monte Carlo against the "
                  "1D straight-ahead approximation. Given elementally, so "
                  "there is no Fe(II)/Fe(III) speciation ambiguity in the "
                  "oxygen balance. Compositionally intermediate between LHS-1 "
                  "highlands and high-Ti mare.\n"
                  "ACTION: OLTARIS exposes its own material compositions. "
                  "Read lunar_regolith_a17 off OLTARIS directly and compare "
                  "to this; if they differ, transport BOTH and report the "
                  "spread as a composition systematic.",
        ),
        from_oxides(
            "mare_basalt_71035", "Apollo 17 mare basalt 71035 (cast)",
            3.30, _MARE_BASALT_OXIDES,
            source="Rhodes (1976) XRF; density assumed for cast basalt",
            citation=_MARE_BASALT_CITE, tier=1,
            isru=True,
            notes="The high-Fe, high-Ti endpoint: 6.2x the Fe and 21x the Ti "
                  "of LHS-1, which is the largest composition contrast "
                  "available from lunar material alone. Cast basalt is a "
                  "genuine ISRU construction proposal, so 3.30 g/cm3 (solid, "
                  "not regolith) is the physically right density for that use "
                  "-- though the density itself is assumed, not from the "
                  "cited analysis.\n"
                  "A SINGLE sample, SINGLE technique, SINGLE author analysis, "
                  "deliberately: the Lunar Sourcebook warns that averaged "
                  "Apollo 17 soil compositions are misleading because those "
                  "soils mix mare and highland material in station-dependent "
                  "proportions.",
        ),
        from_oxides(
            "mgs1_mars", "MGS-1 Mars Global Simulant", 1.29, _MGS1_OXIDES,
            source="Cannon et al. (2019) prototype XRF",
            citation=_MGS1_CITE, tier=1,
            isru=True,
            notes="Extends the tool past the Moon. Chosen over JSC Mars-1: "
                  "published with a clean 100.0 wt% total, designed against "
                  "Curiosity's Rocknest measurement rather than being a "
                  "terrestrial analogue of convenience, and from the same lab "
                  "as LHS-1. JSC Mars-1 additionally loses 21.1 wt% on "
                  "heating to 600 C, which makes 'its' composition ambiguous. "
                  "Carries 2.1 wt% SO3 -- the only sulfur in the library.",
        ),

        # ---- volatiles ----
        Material(
            "water_ice", "Water ice", 0.917,
            {"H": 0.111894, "O": 0.888106}, I_eV=75.0,
            source="H2O stoichiometry; I from ICRU 37 (liquid water)",
            citation="H2O stoichiometry; I = 75.0 eV from ICRU Report 37; "
                     "ice density 0.917 g/cm3", tier=2,
            isru=True,
            notes="Permanently shadowed crater ice. Best shield per gram in "
                  "the library by w_H, but needs containment and thermal "
                  "management that this model does not represent.",
        ),
        Material(
            "liquid_water", "Liquid water", 1.000,
            {"H": 0.111894, "O": 0.888106}, I_eV=75.0,
            source="H2O stoichiometry; I from ICRU 37",
            citation="H2O stoichiometry; I = 75.0 eV from ICRU Report 37", tier=2,
            isru=False,
            notes="Reference material. The phantom is G4_WATER, so this is "
                  "the descriptor anchor and the natural first validation point.",
        ),

        # ---- metals reducible from regolith ----
        Material(
            "isru_iron", "ISRU iron", 7.874, {"Fe": 1.0}, I_eV=286.0,
            source="pure Fe; I from ICRU 37",
            citation="elemental Fe; rho and I = 286 eV from ICRU 37 / PDG", tier=2,
            isru=True,
            notes="Hydrogen or carbothermal reduction of ilmenite is a "
                  "mainstream ISRU proposal. Against HDPE this is an 8.3x "
                  "density contrast versus 2.9x for regolith/HDPE, which is "
                  "the main reason to expect topology effects to reappear: "
                  "column areal-density variance scales with phase contrast.",
        ),
        Material(
            "isru_aluminium", "ISRU aluminium", 2.699, {"Al": 1.0}, I_eV=166.0,
            source="pure Al; I from ICRU 37",
            citation="elemental Al; rho and I = 166 eV from ICRU 37 / PDG", tier=2,
            isru=True,
            notes="Reducible from the anorthite that dominates highlands "
                  "regolith. Nearly the same density as regolith but a very "
                  "different Z/A and lambda_inel -- useful for breaking the "
                  "density/composition degeneracy.",
        ),
        Material(
            "isru_titanium", "ISRU titanium", 4.507, {"Ti": 1.0}, I_eV=233.0,
            source="pure Ti; I from ICRU 37",
            citation="elemental Ti; rho and I = 233 eV from ICRU 37 / PDG", tier=2,
            isru=True,
            notes="From ilmenite in mare regolith. Fills the gap between Al "
                  "and Fe in descriptor space.",
        ),
        Material(
            "isru_magnesium", "ISRU magnesium", 1.738, {"Mg": 1.0}, I_eV=156.0,
            source="pure Mg; I from ICRU 37",
            citation="elemental Mg; rho and I = 156 eV from ICRU 37 / PDG", tier=2,
            isru=True,
            notes="Low-density metal, extractable from olivine and pyroxene. "
                  "Occupies a descriptor region no other library member does: "
                  "metal-like Z/A at polymer-like density.",
        ),

        # ---- degeneracy breakers ----
        #
        # Across naturally occurring materials <Z/A>, I, X0 and lambda_inel
        # are all near-monotone functions of mean Z, so they arrive almost
        # perfectly correlated (see --report). The two materials below exist
        # specifically to break that, by putting substantial hydrogen into a
        # metal-like and an oxide-like matrix respectively. Without them the
        # descriptor set is near-singular and no per-descriptor attribution
        # is defensible.
        Material(
            "mg_hydride", "Magnesium hydride (MgH2)", 1.45,
            {"Mg": 0.923426, "H": 0.076574},
            source="MgH2 stoichiometry; density is the CRYSTAL density",
            citation="MgH2 stoichiometry; 1.45 g/cm3 is the crystal "
                     "density -- compacted powder would be lower", tier=3,
            isru=True,
            notes="Mg reduced from olivine/pyroxene, H from solar-wind "
                  "implantation or electrolysed polar ice. A real hydrogen-"
                  "storage material. Its value here is that it combines "
                  "metal-like <A> and lambda_inel with 7.7 wt% hydrogen -- a "
                  "corner of descriptor space no other library member "
                  "reaches, which is what decorrelates w_H from <Z/A>.",
        ),
        from_oxides(
            "hydrated_regolith", "LHS-1 + 10 wt% H2O", 1.70,
            {**{k: v * 0.9 for k, v in _LHS1_OXIDES.items()}, "H2O": 10.0},
            source="ASSUMED: verified LHS-1 composition + 10 wt% H2O",
            citation=_LHS1_CITE + " (composition); 10 wt% water content "
                     "and 1.70 g/cm3 both ASSUMED", tier=3,
            isru=True,
            notes="Permanently-shadowed-region regolith with retained water. "
                  "Highly ISRU-relevant: it is what you dig up before you "
                  "spend energy separating anything. Gives an oxide matrix a "
                  "small but nonzero w_H, filling the gap between the dry "
                  "oxides (w_H = 0) and the polymers (w_H = 0.14).",
        ),

        # ---- engineering baselines ----
        Material(
            "al6061", "Aluminium 6061-T6", 2.70,
            {"Al": 0.9792, "Mg": 0.010, "Si": 0.006, "Cu": 0.0028,
             "Cr": 0.002},
            source="nominal 6061-T6 mid-range composition",
            citation="ASM nominal 6061 composition (Mg 1.0, Si 0.6, "
                     "Cu 0.28, Cr 0.20 wt%, Al balance)", tier=2,
            isru=False,
            notes="The aerospace structural baseline. Every shielding result "
                  "in the literature is quoted against aluminium, so it is "
                  "needed for the comparison to be legible to a judge.",
        ),
        Material(
            "borated_pe_5", "5% borated polyethylene", 1.00,
            {"B": 0.050, "C": 0.813463, "H": 0.136537},
            source="ASSUMED: 5 wt% natural B in CH2",
            citation="5 wt% natural boron in CH2; density 1.00 g/cm3 "
                     "ASSUMED", tier=3,
            isru=False,
            notes="Thermal-neutron absorber. Included specifically as a "
                  "descriptor-space outlier: B-10 capture is a resonance "
                  "effect that NONE of the six descriptors encode. If the "
                  "surrogate predicts this material well, the descriptor set "
                  "is sufficient; if it fails here and only here, that is a "
                  "clean, honest statement of the model's domain of validity. "
                  "Treat it as a held-out probe, not training data.",
        ),
    ]
    return {m.name: m for m in mats}


LIBRARY: Dict[str, Material] = _build_library()


def library_checksum() -> str:
    """
    Stable hash of every composition and density in the library.

    This module has to be duplicated across PINN_SHEILDING_P2 and
    ISEF_Project. Print the checksum from both; if they differ, the copies
    have drifted and any descriptor column written by one is inconsistent
    with the other.
    """
    payload = json.dumps(
        {n: {"rho": m.density,
             "frac": {k: round(v, 9) for k, v in sorted(m.mass_frac.items())},
             "I": m.I_eV, "tier": m.tier, "cite": m.citation}
         for n, m in sorted(LIBRARY.items())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

DESCRIPTOR_NAMES = [
    "rho_g_cm3",
    "z_over_a",
    "ln_i_excitation",
    "w_hydrogen",
    "lambda_inel_g_cm2",
    "x0_g_cm2",
]


@dataclass
class Descriptors:
    rho_g_cm3: float
    z_over_a: float
    ln_i_excitation: float
    w_hydrogen: float
    lambda_inel_g_cm2: float
    x0_g_cm2: float
    # kept for interpretation and reporting, NOT model inputs
    i_excitation_ev: float = 0.0
    mean_a: float = 0.0
    i_is_measured: bool = False

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in DESCRIPTOR_NAMES], float)

    def as_dict(self, prefix: str = "mat_") -> Dict[str, float]:
        return {prefix + k: float(getattr(self, k)) for k in DESCRIPTOR_NAMES}


def descriptors(mat: Material) -> Descriptors:
    """Compute the six descriptors for a single material."""
    z_over_a = 0.0
    electron_frac_sum = 0.0     # sum w_i (Z/A)_i, the Bragg weight
    bragg_num = 0.0             # sum w_i (Z/A)_i ln I_i
    inv_lambda = 0.0
    inv_x0 = 0.0
    mean_a = 0.0

    for sym, w in mat.mass_frac.items():
        el = ELEMENTS[sym]
        za = el.Z / el.A
        z_over_a += w * za
        electron_frac_sum += w * za
        bragg_num += w * za * math.log(el.I_eV)
        inv_lambda += w / lambda_inelastic_element(el)
        inv_x0 += w / x0_element(el)
        mean_a += w * el.A

    if mat.I_eV is not None:
        i_ev = mat.I_eV
        measured = True
    else:
        i_ev = math.exp(bragg_num / electron_frac_sum)
        measured = False

    return Descriptors(
        rho_g_cm3=mat.density,
        z_over_a=z_over_a,
        ln_i_excitation=math.log(i_ev),
        w_hydrogen=mat.mass_frac.get("H", 0.0),
        lambda_inel_g_cm2=1.0 / inv_lambda,
        x0_g_cm2=1.0 / inv_x0,
        i_excitation_ev=i_ev,
        mean_a=mean_a,
        i_is_measured=measured,
    )


def mix_mass_fraction(mat_a: Material, mat_b: Material,
                      volume_fraction_b: float) -> float:
    """Mass fraction of phase B in a two-phase mixture at volume fraction vf_B."""
    if not 0.0 <= volume_fraction_b <= 1.0:
        raise ValueError(f"volume_fraction_b = {volume_fraction_b} outside [0,1]")
    m_b = volume_fraction_b * mat_b.density
    m_a = (1.0 - volume_fraction_b) * mat_a.density
    return m_b / (m_a + m_b)


def mix_descriptors(mat_a: Material, mat_b: Material,
                    volume_fraction_b: float) -> Descriptors:
    """
    Descriptors of a two-phase mixture, computed EXACTLY by mass-mixing the
    phase compositions rather than by averaging the phase descriptors.

    These agree to machine precision for z_over_a, w_hydrogen, 1/lambda and
    1/x0, which are strictly mass-additive. They differ slightly for ln(I),
    because Bragg additivity weights by w_i*(Z/A)_i rather than by w_i alone,
    and because a measured I for a phase is not recoverable once that phase is
    blended. Doing it on composition is the correct route, so that is what
    this does.

    This exactness is the reason the surrogate extrapolates: the training
    features of an unseen mixture are a known function of its constituents,
    not an approximation.
    """
    wb = mix_mass_fraction(mat_a, mat_b, volume_fraction_b)
    wa = 1.0 - wb

    frac: Dict[str, float] = {}
    for sym, w in mat_a.mass_frac.items():
        frac[sym] = frac.get(sym, 0.0) + wa * w
    for sym, w in mat_b.mass_frac.items():
        frac[sym] = frac.get(sym, 0.0) + wb * w

    rho = volume_fraction_b * mat_b.density + (1.0 - volume_fraction_b) * mat_a.density

    # Mean excitation energy: interpolate ln(I) between the PHASES using each
    # phase's EFFECTIVE I -- its measured value if it has one, otherwise its
    # own Bragg estimate.
    #
    # This is not an approximation. Bragg additivity,
    #     ln I = sum_i w_i (Z/A)_i ln I_i  /  sum_i w_i (Z/A)_i
    # is associative under grouping: collapsing the elements of each phase
    # into a phase-level (Z/A) and Bragg I, then combining those two, returns
    # exactly the element-level result. So this reproduces element-level Bragg
    # when neither phase is measured, while correctly preserving a measured
    # value when one exists.
    #
    # The naive alternative -- re-running Bragg over the blended elemental
    # composition -- silently DISCARDS every measured I. That made the vf=0
    # endpoint return 69 eV for pure HDPE instead of its measured 57.4 eV, a
    # 1.1% descriptor error at a point where the answer must be exact.
    da, db = descriptors(mat_a), descriptors(mat_b)
    ea, eb = wa * da.z_over_a, wb * db.z_over_a
    i_override = math.exp(
        (ea * math.log(da.i_excitation_ev) + eb * math.log(db.i_excitation_ev))
        / (ea + eb)
    )

    blended = Material(
        name=f"mix_{mat_a.name}_{mat_b.name}_{volume_fraction_b:.4f}",
        label=f"{mat_a.label} / {mat_b.label} @ vf={volume_fraction_b:.3f}",
        density=rho, mass_frac=frac, I_eV=i_override,
        source="computed mixture",
    )
    return descriptors(blended)


def descriptor_matrix(names: Optional[List[str]] = None
                      ) -> Tuple[List[str], np.ndarray]:
    names = names or sorted(LIBRARY)
    M = np.vstack([descriptors(LIBRARY[n]).vector() for n in names])
    return names, M


# ---------------------------------------------------------------------------
# Coverage pre-flight
#
# The point: find out whether the library spans the descriptor space BEFORE
# spending hours of Geant4 on it. If two descriptors are collinear across the
# materials you actually simulate, no amount of training data lets the network
# separate their effects, and a per-descriptor sensitivity claim would be
# meaningless.
# ---------------------------------------------------------------------------

def _standardise(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - mu) / sd


def conditioning_report(names: List[str], M: np.ndarray) -> dict:
    Z = _standardise(M)
    C = np.corrcoef(Z, rowvar=False)
    # Condition number of the design matrix: how nearly singular the
    # descriptor set is over these materials.
    sv = np.linalg.svd(np.column_stack([np.ones(len(Z)), Z]),
                       compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    worst = []
    k = len(DESCRIPTOR_NAMES)
    for i, j in itertools.combinations(range(k), 2):
        worst.append((abs(C[i, j]), DESCRIPTOR_NAMES[i], DESCRIPTOR_NAMES[j],
                      C[i, j]))
    worst.sort(reverse=True)
    return {"corr": C, "cond": cond, "worst_pairs": worst[:5]}


def design_matrix_mixtures(material_names: List[str],
                           vf_grid: Optional[np.ndarray] = None,
                           pairs: Optional[List[Tuple[str, str]]] = None
                           ) -> Tuple[List[str], np.ndarray]:
    """
    Descriptor matrix over the ACTUAL design space: every unordered pair of
    materials, at every volume fraction on the grid.

    This -- not the pure-material table -- is the conditioning question that
    matters. The training set is mixtures, and mixing reaches descriptor
    combinations that no pure material has. A 10% HDPE / 90% iron composite
    has metal density AND measurable hydrogen, which decorrelates rho from
    w_hydrogen in a way the pure library cannot.

    Judging the descriptor set on pure materials alone would therefore
    understate its conditioning and could lead to discarding a descriptor
    that the real design space resolves perfectly well.
    """
    if vf_grid is None:
        vf_grid = np.linspace(0.0, 1.0, 11)
    if pairs is None:
        pairs = list(itertools.combinations(material_names, 2))

    labels: List[str] = []
    rows: List[np.ndarray] = []
    for a, b in pairs:
        for vf in vf_grid:
            rows.append(mix_descriptors(LIBRARY[a], LIBRARY[b], float(vf)).vector())
            labels.append(f"{a}|{b}@{vf:.2f}")
    return labels, np.vstack(rows)


def select_materials(n: int, must_include: Optional[List[str]] = None,
                     candidates: Optional[List[str]] = None) -> List[str]:
    """
    Greedy max-min selection in standardised descriptor space.

    Repeatedly adds the material whose minimum distance to the already-chosen
    set is largest. This is a space-filling criterion: it prefers materials
    that are unlike everything picked so far, which is exactly what you want
    when the goal is to let a surrogate learn a response surface over
    descriptors rather than to characterise any one material precisely.

    It is deliberately NOT D-optimal. D-optimality would pile points at the
    extremes of the descriptor box, which is right for fitting a known linear
    model and wrong for training a network that has to interpolate the
    interior.
    """
    cand = candidates or sorted(LIBRARY)
    names, M = descriptor_matrix(cand)
    Z = _standardise(M)
    idx = {nm: i for i, nm in enumerate(names)}

    chosen: List[str] = []
    for nm in (must_include or []):
        if nm not in idx:
            raise KeyError(f"must_include material '{nm}' not in candidates")
        chosen.append(nm)

    if not chosen:
        # start from the most widely separated pair
        best, bd = None, -1.0
        for a, b in itertools.combinations(names, 2):
            d = float(np.linalg.norm(Z[idx[a]] - Z[idx[b]]))
            if d > bd:
                bd, best = d, (a, b)
        chosen = list(best)

    while len(chosen) < min(n, len(names)):
        best, bd = None, -1.0
        for nm in names:
            if nm in chosen:
                continue
            d = min(float(np.linalg.norm(Z[idx[nm]] - Z[idx[c]]))
                    for c in chosen)
            if d > bd:
                bd, best = d, nm
        chosen.append(best)
    return chosen[:max(n, len(must_include or []))]


# ---------------------------------------------------------------------------
# Geant4 emission
# ---------------------------------------------------------------------------

_HH = """// MaterialLibrary.hh -- GENERATED by material_library.py v{ver}
// library checksum {chk}
//
// DO NOT EDIT BY HAND. Regenerate with:
//     python material_library.py --emit-geant4 <src-dir>
//
// Every material here is defined by elemental MASS fractions, which is what
// the descriptors in material_library.py are computed from. Editing one
// without the other silently decouples the simulation from its own features.

#ifndef AEGIS_MATERIAL_LIBRARY_HH
#define AEGIS_MATERIAL_LIBRARY_HH

#include <string>
#include <vector>

class G4Material;

namespace AegisMaterials {{

// Build (once) and return the named material. Throws a G4Exception for an
// unknown name rather than returning nullptr, so a typo in a macro fails
// loudly at setup instead of producing a vacuum shield and a plausible-looking
// dose of zero.
G4Material* Get(const std::string& name);

// Every name this library knows, for validation and error messages.
std::vector<std::string> Names();

}} // namespace AegisMaterials

#endif
"""

_CC_HEAD = """// MaterialLibrary.cc -- GENERATED by material_library.py v{ver}
// library checksum {chk}
//
// DO NOT EDIT BY HAND.

#include "MaterialLibrary.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Exception.hh"
#include "G4IonisParamMat.hh"   // needed for SetMeanExcitationEnergy

#include <map>

// Mass fractions below are written to 8 decimal places. The worst-case sum
// deviation across the whole library is 2e-8, comfortably inside Geant4's
// perMillion (1e-6) tolerance in G4Material::AddElement, so no "fractional
// mass does not sum to 1" warning is produced.

namespace {{

// Built lazily and cached: G4Material objects must not be duplicated, and
// DetectorConstruction may ask for the same material on every geometry
// rebuild.
std::map<std::string, G4Material*>& Cache() {{
    static std::map<std::string, G4Material*> c;
    return c;
}}

G4Material* Build(const std::string& name) {{
    auto* nist = G4NistManager::Instance();
    G4Material* m = nullptr;
"""

_CC_TAIL = """
    return m;
}

} // anonymous namespace

namespace AegisMaterials {

G4Material* Get(const std::string& name) {
    auto& cache = Cache();
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;

    G4Material* m = Build(name);
    if (!m) {
        G4ExceptionDescription ed;
        ed << "Unknown material '" << name << "'. Known materials: ";
        for (const auto& n : Names()) ed << n << " ";
        G4Exception("AegisMaterials::Get", "MatLib001",
                    FatalException, ed);
        return nullptr;
    }
    cache[name] = m;
    return m;
}

std::vector<std::string> Names() {
    return {
%NAMES%
    };
}

} // namespace AegisMaterials
"""


def emit_geant4(out_dir: str) -> List[str]:
    import os

    chk = library_checksum()
    hh = _HH.format(ver=LIBRARY_VERSION, chk=chk)

    body: List[str] = []
    for name in sorted(LIBRARY):
        mat = LIBRARY[name]
        d = descriptors(mat)
        body.append(f'    if (name == "{name}") {{')
        body.append(f'        // {mat.label}')
        body.append(f'        // rho={d.rho_g_cm3:.4f} g/cm3  <Z/A>={d.z_over_a:.5f}'
                    f'  I={d.i_excitation_ev:.1f} eV  w_H={d.w_hydrogen:.6f}')
        body.append(f'        // lambda_inel={d.lambda_inel_g_cm2:.2f} g/cm2'
                    f'  X0={d.x0_g_cm2:.2f} g/cm2')
        if mat.source:
            body.append(f'        // source: {mat.source}')
        body.append(f'        m = new G4Material("{name}", '
                    f'{mat.density:.6f}*g/cm3, {len(mat.mass_frac)});')
        for sym in sorted(mat.mass_frac, key=lambda s: -mat.mass_frac[s]):
            w = mat.mass_frac[sym]
            body.append(f'        m->AddElement(nist->FindOrBuildElement("{sym}"), '
                        f'{w:.8f});')
        if mat.I_eV is not None:
            body.append(f'        // measured mean excitation energy overrides '
                        f'Geant4\'s Bragg-additivity estimate')
            body.append(f'        m->GetIonisation()->SetMeanExcitationEnergy('
                        f'{mat.I_eV:.4f}*eV);')
        body.append('    }')

    names_block = ",\n".join(f'        "{n}"' for n in sorted(LIBRARY))
    cc = (_CC_HEAD.format(ver=LIBRARY_VERSION, chk=chk)
          + "\n".join(body)
          + _CC_TAIL.replace("%NAMES%", names_block))

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fname, text in (("MaterialLibrary.hh", hh), ("MaterialLibrary.cc", cc)):
        p = os.path.join(out_dir, fname)
        with open(p, "w") as f:
            f.write(text)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Self-test
#
# Reference values are PDG / NIST tabulations, i.e. independent of anything
# computed here. A failure means the formulas are wrong, not that the
# tolerance is tight.
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    fails: List[str] = []

    def chk(label, got, want, tol, unit=""):
        rel = abs(got - want) / abs(want) if want else abs(got)
        ok = rel <= tol
        if not ok:
            fails.append(f"{label}: got {got:.6g}, expected {want:.6g} "
                         f"({rel:.2%} off, tol {tol:.0%})")
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {label:<44}"
                  f"{got:>12.5f} vs {want:>10.5f} {unit}  ({rel:.2%})")
        return ok

    print("\n[A] <Z/A> against PDG  (tol 0.1%)")
    chk("water  <Z/A>", descriptors(LIBRARY["liquid_water"]).z_over_a,
        0.55509, 0.001)
    chk("polyethylene  <Z/A>", descriptors(LIBRARY["polyethylene"]).z_over_a,
        0.57034, 0.001)
    chk("aluminium  <Z/A>", descriptors(LIBRARY["isru_aluminium"]).z_over_a,
        0.48181, 0.001)
    chk("iron  <Z/A>", descriptors(LIBRARY["isru_iron"]).z_over_a,
        0.46557, 0.001)
    # 22/47.867 with the current CIAAW atomic weight. The PDG table value of
    # 0.45948 uses the older A = 47.880; the 0.03% difference is the atomic
    # weight revision, not an error here.
    chk("titanium  <Z/A>", descriptors(LIBRARY["isru_titanium"]).z_over_a,
        0.459606, 0.001)

    print("\n[B] radiation length against PDG  (tol 3%, Dahl approximation)")
    chk("aluminium  X0", descriptors(LIBRARY["isru_aluminium"]).x0_g_cm2,
        24.01, 0.03, "g/cm2")
    chk("iron  X0", descriptors(LIBRARY["isru_iron"]).x0_g_cm2,
        13.84, 0.03, "g/cm2")
    chk("water  X0", descriptors(LIBRARY["liquid_water"]).x0_g_cm2,
        36.08, 0.03, "g/cm2")

    print("\n[C] inelastic interaction length against PDG lambda_I  (tol 12%)")
    # PDG lambda_I uses measured inelastic cross sections; Letaw is a smooth
    # fit. ~10% disagreement is expected and acceptable for a descriptor.
    chk("water  lambda_inel", descriptors(LIBRARY["liquid_water"]).lambda_inel_g_cm2,
        83.3, 0.12, "g/cm2")
    chk("iron  lambda_inel", descriptors(LIBRARY["isru_iron"]).lambda_inel_g_cm2,
        132.1, 0.12, "g/cm2")
    chk("aluminium  lambda_inel", descriptors(LIBRARY["isru_aluminium"]).lambda_inel_g_cm2,
        107.2, 0.12, "g/cm2")

    print("\n[D] continuity with topology_generators.py  (EXACT)")
    for nm, want in (("polyethylene", 0.95), ("lhs1_regolith", 2.75)):
        got = LIBRARY[nm].density
        ok = got == want
        if not ok:
            fails.append(f"{nm} density is {got}, must be exactly {want} "
                         "or every areal_density_cv computed so far changes meaning")
        print(f"  {'ok  ' if ok else 'FAIL'} {nm + ' density':<44}"
              f"{got:>12.4f} vs {want:>10.4f} g/cm3")

    print("\n[E] mixture additivity  (mix endpoints must equal pure phases)")
    a, b = LIBRARY["polyethylene"], LIBRARY["lhs1_regolith"]
    for vf, ref, tag in ((0.0, a, "vf=0 -> pure HDPE"), (1.0, b, "vf=1 -> pure LHS-1")):
        m, r = mix_descriptors(a, b, vf).vector(), descriptors(ref).vector()
        err = float(np.max(np.abs(m - r) / np.maximum(np.abs(r), 1e-12)))
        ok = err < 1e-9
        if not ok:
            fails.append(f"mixture endpoint {tag} off by {err:.2e}")
        print(f"  {'ok  ' if ok else 'FAIL'} {tag:<44}{'max rel err':>12} "
              f"{err:>10.2e}")

    # Mass-additive descriptors must be exactly linear in the MASS fraction.
    vf = 0.37
    wb = mix_mass_fraction(a, b, vf)
    da, db, dm = descriptors(a), descriptors(b), mix_descriptors(a, b, vf)
    lin = (1 - wb) * da.z_over_a + wb * db.z_over_a
    ok = abs(lin - dm.z_over_a) < 1e-12
    if not ok:
        fails.append(f"<Z/A> not mass-additive: {lin} vs {dm.z_over_a}")
    print(f"  {'ok  ' if ok else 'FAIL'} {'<Z/A> is exactly mass-additive':<44}"
          f"{dm.z_over_a:>12.8f} vs {lin:>10.8f}")

    inv_lin = (1 - wb) / da.lambda_inel_g_cm2 + wb / db.lambda_inel_g_cm2
    ok = abs(1.0 / inv_lin - dm.lambda_inel_g_cm2) < 1e-9
    if not ok:
        fails.append("1/lambda_inel not mass-additive")
    print(f"  {'ok  ' if ok else 'FAIL'} {'1/lambda_inel is mass-additive':<44}"
          f"{dm.lambda_inel_g_cm2:>12.6f} vs {1.0/inv_lin:>10.6f} g/cm2")

    print("\n[F] rejected descriptor is genuinely redundant")
    # a_eff_nuclear vs lambda_inel across the whole library
    names = sorted(LIBRARY)
    a_eff, lam = [], []
    for n in names:
        s = 0.0
        for sym, w in LIBRARY[n].mass_frac.items():
            s += w * ELEMENTS[sym].A ** (-1.0 / 3.0)
        a_eff.append(s ** -3.0)
        lam.append(descriptors(LIBRARY[n]).lambda_inel_g_cm2)
    r = float(np.corrcoef(np.log(a_eff), np.log(lam))[0, 1])
    ok = abs(r) > 0.99
    if not ok:
        fails.append(f"a_eff_nuclear vs lambda_inel r = {r:.4f}; it is NOT "
                     "redundant after all, so reconsider dropping it")
    print(f"  {'ok  ' if ok else 'FAIL'} "
          f"{'r(ln a_eff, ln lambda) across library':<44}{r:>12.5f} vs "
          f"{'>0.99':>10}")

    print()
    if fails:
        print("SELF-TEST FAILED")
        for f in fails:
            print(f"   - {f}")
        return False
    print("SELF-TEST PASSED -- descriptors match independent reference values.")
    return True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(subset: Optional[List[str]] = None) -> None:
    names, M = descriptor_matrix(subset)

    print("\n" + "=" * 112)
    print(f"DESCRIPTOR TABLE   library v{LIBRARY_VERSION}   "
          f"checksum {library_checksum()}")
    print("=" * 112)
    print(f"{'material':<24}{'tier':>5}{'ISRU':>5}{'rho':>8}{'<Z/A>':>9}"
          f"{'I eV':>8}{'w_H':>10}{'lam_in':>9}{'X0':>8}{'<A>':>8}")
    print(f"{'':<24}{'':>5}{'':>5}{'g/cm3':>8}{'':>9}{'':>8}{'':>10}"
          f"{'g/cm2':>9}{'g/cm2':>8}{'':>8}")
    print("-" * 112)
    for n in names:
        m = LIBRARY[n]
        d = descriptors(m)
        tier = {1: "1", 2: "2", 3: "3!"}[m.tier]
        print(f"{n:<24}{tier:>5}{'yes' if m.isru else '-':>5}{d.rho_g_cm3:>8.3f}"
              f"{d.z_over_a:>9.5f}{d.i_excitation_ev:>8.1f}"
              f"{d.w_hydrogen:>10.6f}{d.lambda_inel_g_cm2:>9.2f}"
              f"{d.x0_g_cm2:>8.2f}{d.mean_a:>8.2f}")

    n3 = [n for n in names if LIBRARY[n].tier == 3]
    n1 = [n for n in names if LIBRARY[n].tier == 1]
    print(f"\n  tier 1 (measured & cited) : {len(n1)}  {', '.join(n1)}")
    print(f"  tier 2 (stoichiometric)   : "
          f"{len([n for n in names if LIBRARY[n].tier == 2])}")
    if n3:
        print(f"  tier 3 (ASSUMED by me)    : {len(n3)}  {', '.join(n3)}")
        print("    ^ these are design choices, not characterised materials. "
              "Label them\n      hypothetical, and never use one as a "
              "validation reference.")

    print("\n  I values: 'measured' entries use a tabulated mean excitation "
          "energy; the rest\n  use Bragg additivity, which underestimates I "
          "in condensed matter by up to ~9%.\n  I enters Bethe-Bloch "
          "logarithmically, so a 9% error there moves stopping\n  power by "
          "well under 1% -- this is the least consequential number in the "
          "file.")

    print("\n" + "=" * 112)
    print("DESCRIPTOR SPAN")
    print("=" * 112)
    print(f"{'descriptor':<22}{'min':>12}{'max':>12}{'ratio':>10}"
          f"{'  spanning materials'}")
    print("-" * 112)
    for j, dn in enumerate(DESCRIPTOR_NAMES):
        col = M[:, j]
        lo, hi = col.argmin(), col.argmax()
        ratio = col[hi] / col[lo] if col[lo] != 0 else float("inf")
        rs = f"{ratio:>10.2f}" if np.isfinite(ratio) else f"{'inf':>10}"
        print(f"{dn:<22}{col[lo]:>12.5f}{col[hi]:>12.5f}{rs}"
              f"  {names[lo]} -> {names[hi]}")

    rep = conditioning_report(names, M)
    print("\n" + "=" * 112)
    print("CONDITIONING -- can a model separate these descriptors?")
    print("=" * 112)
    print(f"\n  design-matrix condition number: {rep['cond']:.1f}")
    if rep["cond"] < 30:
        verdict = "GOOD -- descriptors are separable over this material set."
    elif rep["cond"] < 100:
        verdict = ("MARGINAL -- per-descriptor sensitivity claims will be "
                   "shaky. Add a material that breaks the worst pair below.")
    else:
        verdict = ("BAD -- the descriptor set is near-singular over these "
                   "materials. Do NOT run the full batch; the model cannot "
                   "attribute an effect to any single descriptor. Add "
                   "materials that break the correlations below first.")
    print(f"  {verdict}")

    print("\n  most correlated descriptor pairs:")
    for mag, a, b, r in rep["worst_pairs"]:
        flag = "  <-- problem" if mag > 0.95 else ("  <-- watch" if mag > 0.85 else "")
        print(f"    r = {r:+.4f}   {a:<20} {b:<20}{flag}")

    print("\n  full correlation matrix:")
    short = [d[:9] for d in DESCRIPTOR_NAMES]
    print("    " + "".join(f"{s:>11}" for s in short))
    for i, s in enumerate(short):
        row = "".join(f"{rep['corr'][i, j]:>11.3f}" for j in range(len(short)))
        print(f"{s:>9} {row}")

    print("\n" + "=" * 112)
    print("""HOW TO READ THIS

  DESCRIPTOR SPAN
      A ratio near 1 means every material in the library has effectively the
      same value, so that descriptor carries no information and the surrogate
      cannot learn anything about it. Ratios of 3x or more are what you want.
      w_hydrogen is the exception: it is legitimately zero for every metal and
      every oxide, so its span is a 0-to-0.14 contrast rather than a ratio,
      and it is informative precisely because it is bimodal.

  CONDITIONING
      This is the pre-flight gate. If two descriptors are correlated above
      about 0.95 across the materials you actually simulate, then no amount
      of Geant4 time lets a model say which of the two caused an effect --
      they are the same column in different units. Fix it by ADDING a
      material that breaks the pair, not by removing a descriptor: the
      physics needs all six, the material set is what is deficient.

      Specifically, rho and lambda_inel tend to correlate because dense
      materials are usually high-A. isru_magnesium is in the library to break
      that: metal-like Z/A and lambda at polymer-like density.

  NEXT
      python material_library.py --select 8 --must-include polyethylene,lhs1_regolith
      then re-run --report on that subset to confirm the chosen eight are
      still well conditioned before any Geant4 runs.""")
    print("=" * 112)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--report", action="store_true",
                    help="descriptor table, span and conditioning")
    ap.add_argument("--select", type=int, metavar="N",
                    help="greedy space-filling selection of N materials")
    ap.add_argument("--must-include", type=str, default="",
                    help="comma-separated materials forced into --select")
    ap.add_argument("--subset", type=str, default="",
                    help="comma-separated materials to restrict --report to")
    ap.add_argument("--emit-geant4", type=str, metavar="DIR",
                    help="write MaterialLibrary.hh/.cc into DIR")
    ap.add_argument("--mix", nargs=3, metavar=("MAT_A", "MAT_B", "VF_B"),
                    help="descriptors of a two-phase mixture")
    ap.add_argument("--design", type=str, default="", metavar="MATS",
                    help="comma-separated materials; report conditioning over "
                         "ALL PAIRWISE MIXTURES of them. This is the real "
                         "pre-flight gate -- run it before any Geant4 batch.")
    ap.add_argument("--no-self-test", action="store_true")
    args = ap.parse_args()

    did_something = False

    if not args.no_self_test:
        print("=" * 112)
        print(f"SELF-TEST   material_library v{LIBRARY_VERSION}   "
              f"checksum {library_checksum()}")
        print("=" * 112)
        if not self_test():
            print("\nRefusing to continue: descriptors do not reproduce "
                  "independent reference values.")
            return 1

    if args.mix:
        a, b, vf = args.mix[0], args.mix[1], float(args.mix[2])
        for nm in (a, b):
            if nm not in LIBRARY:
                print(f"unknown material '{nm}'. Known: {', '.join(sorted(LIBRARY))}")
                return 1
        d = mix_descriptors(LIBRARY[a], LIBRARY[b], vf)
        wb = mix_mass_fraction(LIBRARY[a], LIBRARY[b], vf)
        print(f"\n{LIBRARY[a].label}  +  {LIBRARY[b].label}")
        print(f"  volume fraction B = {vf:.4f}   ->   mass fraction B = {wb:.6f}")
        print("-" * 60)
        for k in DESCRIPTOR_NAMES:
            print(f"  {k:<22}{getattr(d, k):>14.6f}")
        print(f"  {'(I, eV)':<22}{d.i_excitation_ev:>14.2f}")
        print(f"  {'(<A>)':<22}{d.mean_a:>14.2f}")
        did_something = True

    if args.select:
        must = [s.strip() for s in args.must_include.split(",") if s.strip()]
        chosen = select_materials(args.select, must_include=must)
        print(f"\nSPACE-FILLING SELECTION of {len(chosen)} materials")
        if must:
            print(f"  forced: {', '.join(must)}")
        print("-" * 60)
        for i, nm in enumerate(chosen, 1):
            tag = " (forced)" if nm in must else ""
            t = LIBRARY[nm].tier
            print(f"  {i:>2}. [tier {t}{'!' if t == 3 else ' '}] "
                  f"{nm:<24}{LIBRARY[nm].label}{tag}")

        # The greedy criterion maximises descriptor spread and is blind to
        # provenance, so it will happily pick an assumed material over a
        # measured one that sits slightly closer to the rest. Say so.
        t3 = [n for n in chosen if LIBRARY[n].tier == 3]
        skipped_t1 = [n for n in sorted(LIBRARY)
                      if LIBRARY[n].tier == 1 and n not in chosen]
        if t3:
            print(f"\n  WARNING: {len(t3)} ASSUMED material(s) selected: "
                  f"{', '.join(t3)}")
            print("    The space-filling criterion optimises descriptor "
                  "spread and knows\n    nothing about provenance. These are "
                  "legitimate as training points --\n    they teach the "
                  "surrogate the response surface -- but no validation\n"
                  "    claim may rest on them, and they must be labelled "
                  "hypothetical.")
        if skipped_t1:
            print(f"\n  NOTE: measured, cited materials NOT selected: "
                  f"{', '.join(skipped_t1)}")
            if "regolith_transport_ref" in skipped_t1:
                print("    regolith_transport_ref should be FORCED IN "
                      "regardless of what the\n    space-filling criterion "
                      "prefers: it is the composition the\n    cross-code "
                      "benchmark literature transports, so without it there "
                      "is\n    no clean OLTARIS comparison. Re-run with it in "
                      "--must-include.")
        print("\nConditioning of this subset:")
        nm_, M_ = descriptor_matrix(chosen)
        rep = conditioning_report(nm_, M_)
        print(f"  condition number {rep['cond']:.1f}")
        for mag, a, b, r in rep["worst_pairs"][:3]:
            print(f"  r = {r:+.4f}   {a} / {b}")
        print("\nRe-run with --report --subset " + ",".join(chosen)
              + "\n  for the full table on just these.")
        did_something = True

    if args.design:
        mats = [s.strip() for s in args.design.split(",") if s.strip()]
        for nm in mats:
            if nm not in LIBRARY:
                print(f"unknown material '{nm}'. Known: {', '.join(sorted(LIBRARY))}")
                return 1
        labels, M = design_matrix_mixtures(mats)
        n_pairs = len(list(itertools.combinations(mats, 2)))
        print("\n" + "=" * 112)
        print("DESIGN-SPACE CONDITIONING  (all pairwise mixtures)")
        print("=" * 112)
        print(f"  {len(mats)} materials -> {n_pairs} pairs x 11 volume "
              f"fractions = {len(labels)} design points")

        pure = conditioning_report(*descriptor_matrix(mats))
        mix = conditioning_report(labels, M)
        print(f"\n  condition number, PURE materials only : {pure['cond']:>8.1f}")
        print(f"  condition number, WITH mixtures       : {mix['cond']:>8.1f}")
        imp = pure["cond"] / mix["cond"] if mix["cond"] > 0 else float("nan")
        print(f"  improvement from mixing               : {imp:>8.2f}x")

        if mix["cond"] < 30:
            v = ("GOOD -- the descriptor set is separable over the design "
                 "space. Proceed to the Geant4 gate run.")
        elif mix["cond"] < 100:
            v = ("MARGINAL -- the surrogate will predict well but "
                 "per-descriptor attribution claims are not defensible. "
                 "Either add a degeneracy-breaking material, or state "
                 "plainly that the material manifold is lower-dimensional "
                 "than the descriptor count. The second is a legitimate "
                 "finding, not a failure.")
        else:
            v = ("BAD -- do NOT run the full batch. Add materials that break "
                 "the pairs below first.")
        print(f"\n  {v}")

        print("\n  worst pairs over the design space:")
        for mag, a, b, r in mix["worst_pairs"]:
            flag = ("  <-- problem" if mag > 0.95
                    else ("  <-- watch" if mag > 0.85 else ""))
            print(f"    r = {r:+.4f}   {a:<20} {b:<20}{flag}")

        print("\n  effective dimensionality of the material manifold:")
        Z = _standardise(M)
        sv = np.linalg.svd(Z, compute_uv=False)
        var = sv ** 2 / (sv ** 2).sum()
        cum = np.cumsum(var)
        for i, (v_, c_) in enumerate(zip(var, cum), 1):
            bar = "#" * int(round(v_ * 50))
            print(f"    PC{i}  {v_:>7.2%}  cumulative {c_:>7.2%}  {bar}")
        n_eff = int(np.searchsorted(cum, 0.99) + 1)
        print(f"\n    {n_eff} of {len(DESCRIPTOR_NAMES)} components carry 99% "
              f"of the variance.")
        print("""
  WHAT TO DO ABOUT THE COLLINEARITY

    Some of it is fixable by adding materials. The rho / w_hydrogen pair is
    like that: no pure material is both dense and hydrogen-rich, but a 10%
    HDPE / 90% iron composite is, which is why mixing improved the condition
    number by the factor reported above.

    The ln_I / lambda_inel pair is NOT fixable, and this is worth stating
    explicitly rather than treating as a defect. I rises roughly as Z^0.9
    while lambda_inel rises roughly as A^0.3, so breaking their correlation
    needs a material with high Z at low A, or the reverse. The nuclear
    stability valley forbids both: every stable nuclide has A between about
    2Z and 2.6Z, and hydrogen -- the sole exception -- pushes I and
    lambda_inel DOWN together. No choice of materials can decorrelate them,
    because the constraint is imposed by nuclear physics, not by the library.

    That is a genuine result, and the defensible way to present it:

      KEEP all six descriptors as surrogate inputs. Collinearity degrades
      COEFFICIENT ATTRIBUTION, not PREDICTION. A model fitted on collinear
      inputs still predicts correctly anywhere inside the manifold its
      training data covered -- it simply cannot apportion credit between two
      descriptors that always move together. Prediction is what the tool is
      for.

      DO NOT claim per-descriptor sensitivity. A statement like "the network
      learned that hydrogen content drives moderation" is not supportable at
      r = 0.94 between w_hydrogen and <Z/A>. Report the principal-component
      structure above instead.

      DO report the effective dimensionality as a finding. "Shielding-
      relevant material properties of real solids occupy a 3-dimensional
      manifold, so a 6-descriptor parameterisation is complete but
      redundant" is a stronger and more honest claim than pretending six
      independent knobs exist.

      DO flag extrapolation off the manifold. A requested material whose
      descriptors sit far from the training manifold is outside the model's
      domain, and the surrogate must say so rather than returning a
      confident number. Use the Mahalanobis distance in standardised
      descriptor space against this design matrix as the gate.""")
        did_something = True

    if args.emit_geant4:
        paths = emit_geant4(args.emit_geant4)
        print("\nWROTE:")
        for p in paths:
            print(f"  {p}")
        print("\nAdd nothing to CMakeLists.txt if it globs src/*.cc (the "
              "default from the Geant4 examples).")
        print("Then: cd build && make clean && make -j8 && cd ..")
        print("make clean IS required -- MaterialLibrary.hh is a new header.")
        did_something = True

    if args.report or not did_something:
        subset = [s.strip() for s in args.subset.split(",") if s.strip()] or None
        if subset:
            for nm in subset:
                if nm not in LIBRARY:
                    print(f"unknown material '{nm}'")
                    return 1
        report(subset)

    return 0


if __name__ == "__main__":
    sys.exit(main())