"""
Utilities for writing/reading a 3D binary voxel topology to/from the flat
file format DetectorConstruction.cc's new /shield/setTopologyFile command
expects.

CRITICAL -- ORDERING. Geant4's VoxelParam::ComputeTransformation maps a
flat copyNo to (ix, iy, iz) via:
    ix = copyNo % nx
    iy = (copyNo / nx) % ny
    iz = copyNo / (nx * ny)
i.e. x varies FASTEST, then y, then z. That is FORTRAN order (first axis
fastest), NOT numpy's default C order (last axis fastest). Topology
arrays in this project have shape (nx, ny, nz) with axis 0 = x, axis 1 =
y, axis 2 = z (thickness direction) -- matching the convention already
used throughout synthetic_topology_dataset.py, where z-dependence is
built along axis 2. Flattening must use order='F' to match Geant4's
copyNo convention.

Getting this wrong produces NO error and NO crash -- just a silently
scrambled topology that isn't the pattern you intended. There is no way
to catch this from Python alone (Python can only verify its own
write-then-read round-trip is self-consistent, which is what
verify_roundtrip() below does) -- the C++ SIDE of this convention has
NOT been verified against a real Geant4 run. Before trusting this at
scale, run the physical sanity check described in
topology_ordering_geant4_check.md (generate a trivially verifiable
topology -- e.g. front half pure regolith, back half pure HDPE -- and
confirm the resulting dose response matches physical intuition given
the beam direction).
"""

import numpy as np


def write_topology_file(volume, path):
    """
    volume: 3D numpy array, shape (nx, ny, nz), values 0/1, axis 0=x,
    axis 1=y, axis 2=z (thickness direction).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")
    flat = volume.astype(np.int32).flatten(order='F')  # x fastest -- see module docstring
    np.savetxt(path, flat, fmt='%d')
    return path


def read_topology_file(path, nx, ny, nz):
    """Inverse of write_topology_file -- for verification/debugging only.
    Geant4 does its own independent reading; this only confirms Python's
    own write+read round-trip is self-consistent."""
    flat = np.loadtxt(path, dtype=np.int32)
    if flat.size != nx * ny * nz:
        raise ValueError(f"File has {flat.size} values, expected {nx * ny * nz} (nx*ny*nz)")
    return flat.reshape((nx, ny, nz), order='F')


def verify_roundtrip(volume, path="_roundtrip_test.txt"):
    """Write then read back a volume, confirm an EXACT match. Run this on
    every family at least once before trusting the pipeline at scale."""
    nx, ny, nz = volume.shape
    write_topology_file(volume, path)
    recovered = read_topology_file(path, nx, ny, nz)
    matches = np.array_equal(volume, recovered)
    if not matches:
        raise AssertionError("Round-trip mismatch -- topology file I/O is NOT self-consistent!")
    return matches