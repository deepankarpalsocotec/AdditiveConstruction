# -*- coding: utf-8 -*-
"""
stl_layered_concrete_fea.py

Updates in this version:
1) --first_octant (with --auto_fit): shifts STL so padded bounding box lies entirely in +x,+y,+z
   (domain origin becomes (0,0,0)).
2) Robust voxelization for coarse grids: mark a voxel solid if ANY of 27 (3x3x3) sample points
   inside the cell are inside the STL volume (ray-casting). This prevents "0 solid cells" when
   the object is small relative to voxel size and padding is large.

Notes:
- Internal units are SI (meters). STL assumed mm by default (mm -> m scale = 1e-3).
- Padding (--pad) is in meters.
"""

import os
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


# =========================================================
# STL reader (binary + ascii) - returns triangles [ntri,3,3]
# =========================================================
def load_stl_triangles(path: str) -> np.ndarray:
    """
    Minimal STL reader: tries binary first, falls back to ASCII.
    Returns: triangles float64 array of shape (ntri, 3, 3).
    """
    with open(path, "rb") as f:
        data = f.read()

    # Try binary parse
    try:
        if len(data) < 84:
            raise ValueError("Too small for binary STL")

        ntri = int.from_bytes(data[80:84], byteorder="little", signed=False)
        expected = 84 + 50 * ntri
        if expected != len(data):
            raise ValueError("Binary length mismatch")

        tris = np.zeros((ntri, 3, 3), dtype=np.float64)
        off = 84
        for i in range(ntri):
            off += 12  # normal
            v0 = np.frombuffer(data, dtype=np.float32, count=3, offset=off).astype(np.float64)
            off += 12
            v1 = np.frombuffer(data, dtype=np.float32, count=3, offset=off).astype(np.float64)
            off += 12
            v2 = np.frombuffer(data, dtype=np.float32, count=3, offset=off).astype(np.float64)
            off += 12
            off += 2  # attr

            tris[i, 0, :] = v0
            tris[i, 1, :] = v1
            tris[i, 2, :] = v2
        return tris

    except Exception:
        # ASCII parse
        text = data.decode("utf-8", errors="ignore").splitlines()
        verts = []
        for line in text:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(verts) == 0 or (len(verts) % 3 != 0):
            raise ValueError("Could not parse STL as binary or ASCII.")

        return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


# =========================================================
# Structured mesh
# =========================================================
@dataclass
class StructuredHexMesh:
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny, nz must be > 0")
        if self.Lx <= 0 or self.Ly <= 0 or self.Lz <= 0:
            raise ValueError("Lx, Ly, Lz must be > 0")

        self.ox, self.oy, self.oz = self.origin
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def cell_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        return (
            self.ox + (i + 0.5) * self.dx,
            self.oy + (j + 0.5) * self.dy,
            self.oz + (k + 0.5) * self.dz,
        )

    def cell_min_corner(self, i: int, j: int, k: int) -> np.ndarray:
        return np.array(
            [self.ox + i * self.dx, self.oy + j * self.dy, self.oz + k * self.dz],
            dtype=np.float64,
        )


# =========================================================
# Geometry helpers
# =========================================================
def compute_bbox(tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = tris.reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def ray_intersect_triangle(p: np.ndarray, d: np.ndarray, tri: np.ndarray) -> bool:
    """
    Möller–Trumbore ray-triangle intersection.
    Ray: p + t d, t > 0
    Returns True if intersects.
    """
    eps = 1e-12
    v0, v1, v2 = tri
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if -eps < a < eps:
        return False
    f = 1.0 / a
    s = p - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(e2, q)
    return t > eps


def point_in_mesh_ray_cast(point: np.ndarray, tris: np.ndarray) -> bool:
    """
    Point-in-solid test via ray casting along +x.
    Odd intersections => inside.

    NOTE: For very large triangle counts, this is slow. This script keeps it simple.
    """
    d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p = point.copy()
    # tiny jitter to avoid grazing edges/vertices
    p[1] += 1e-10
    p[2] += 2e-10

    count = 0
    for tri in tris:
        if ray_intersect_triangle(p, d, tri):
            count += 1
    return (count % 2) == 1


# =========================================================
# Auto-fit domain to STL bounds (SI)
# =========================================================
def auto_fit_domain_to_stl(tris_m: np.ndarray, pad_m: float, first_octant: bool = False):
    """
    Auto-fit domain to STL bounds in meters with padding.

    If first_octant=True:
      - Translate STL so padded min corner (mn - pad) maps to (0,0,0)
      - Domain origin becomes (0,0,0)
      - Ensures padded object is entirely in the +x,+y,+z octant.

    Returns:
      tris_m_out, origin, Lx, Ly, Lz
    """
    mn, mx = compute_bbox(tris_m)

    Lx = float((mx[0] - mn[0]) + 2.0 * pad_m)
    Ly = float((mx[1] - mn[1]) + 2.0 * pad_m)
    Lz = float((mx[2] - mn[2]) + 2.0 * pad_m)

    if first_octant:
        shift = -(mn - pad_m)     # so (mn - pad) -> 0
        tris_out = tris_m + shift
        origin = (0.0, 0.0, 0.0)
        return tris_out, origin, Lx, Ly, Lz

    origin = (float(mn[0] - pad_m), float(mn[1] - pad_m), float(mn[2] - pad_m))
    return tris_m, origin, Lx, Ly, Lz


# =========================================================
# Robust voxelization (3x3x3 samples per cell)
# =========================================================
def voxelize_stl(mesh: StructuredHexMesh, tris: np.ndarray, samples_1d: int = 3) -> np.ndarray:
    """
    Returns solid[k,j,i] boolean array.

    Strategy:
    - Bounding-box cull quickly.
    - For each cell, test multiple interior sample points (default 27 points).
      Mark solid if ANY sample is inside the STL volume.

    This prevents 0-solid issues when the object is small relative to coarse voxel sizes.
    """
    solid = np.zeros((mesh.nz, mesh.ny, mesh.nx), dtype=bool)
    mn, mx = compute_bbox(tris)

    # Sample offsets within the cell (fractions in [0,1])
    # For samples_1d=3 -> {1/6, 3/6, 5/6} (avoids faces/edges)
    fracs = (np.arange(samples_1d, dtype=np.float64) * 2.0 + 1.0) / (2.0 * samples_1d)
    offsets = np.array(np.meshgrid(fracs, fracs, fracs, indexing="ij")).reshape(3, -1).T  # (ns,3)

    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    for k in range(mesh.nz):
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                cell_min = mesh.cell_min_corner(i, j, k)

                # quick reject: if cell's AABB doesn't overlap STL bbox, skip
                cell_max = cell_min + np.array([dx, dy, dz], dtype=np.float64)
                if (cell_max[0] < mn[0]) or (cell_min[0] > mx[0]) or \
                   (cell_max[1] < mn[1]) or (cell_min[1] > mx[1]) or \
                   (cell_max[2] < mn[2]) or (cell_min[2] > mx[2]):
                    continue

                # test sample points
                is_solid = False
                for (fx, fy, fz) in offsets:
                    p = cell_min + np.array([fx * dx, fy * dy, fz * dz], dtype=np.float64)
                    # additional bbox cull for point
                    if p[0] < mn[0] or p[0] > mx[0] or p[1] < mn[1] or p[1] > mx[1] or p[2] < mn[2] or p[2] > mx[2]:
                        continue
                    if point_in_mesh_ray_cast(p, tris):
                        is_solid = True
                        break

                solid[k, j, i] = is_solid

    return solid


# =========================================================
# Layer scheduling (kept simple)
# =========================================================
def build_layer_activation(nz: int, layers_per_output: int) -> List[Tuple[int, int]]:
    steps = []
    k = 0
    while k < nz:
        k2 = min(nz, k + layers_per_output)
        steps.append((k, k2))
        k = k2
    return steps


# =========================================================
# Dummy solver / outputs (stub)
# =========================================================
def solve_linear_elasticity_dummy(mesh: StructuredHexMesh, solid: np.ndarray):
    n = mesh.n_cells
    u = np.zeros((n, 3), dtype=np.float64)
    s = np.zeros((n,), dtype=np.float64)
    return u, s


def write_vti_stub(out_dir: str, step: int, mesh: StructuredHexMesh, solid: np.ndarray, stress: np.ndarray):
    """
    Minimal VTI writer (ASCII).
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"step_{step:04d}.vti")

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    ox, oy, oz = mesh.origin

    solid_flat = solid.astype(np.uint8).ravel(order="C")
    stress_flat = stress.reshape(nz, ny, nx).ravel(order="C")

    with open(fname, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n')
        f.write(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">\n')
        f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n')
        f.write('      <CellData Scalars="scalars">\n')

        f.write('        <DataArray type="UInt8" Name="solid" format="ascii">\n')
        for v in solid_flat:
            f.write(f"{int(v)} ")
        f.write('\n        </DataArray>\n')

        f.write('        <DataArray type="Float64" Name="stress" format="ascii">\n')
        for v in stress_flat:
            f.write(f"{float(v)} ")
        f.write('\n        </DataArray>\n')

        f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </ImageData>\n')
        f.write('</VTKFile>\n')


# =========================================================
# Main simulation
# =========================================================
def run_sim(
    stl_path: str,
    nx: int, ny: int, nz: int,
    Lx: float, Ly: float, Lz: float,
    origin=(0.0, 0.0, 0.0),
    layers_per_output: int = 2,
    out_dir: str = "out",
    auto_fit: bool = False,
    pad_m: float = 0.02,
    stl_units: str = "mm",
    first_octant: bool = False,
    voxel_samples_1d: int = 3,
):
    # STL unit scaling to meters
    stl_units = stl_units.lower()
    if stl_units == "mm":
        stl_scale_to_m = 1e-3
    elif stl_units == "cm":
        stl_scale_to_m = 1e-2
    elif stl_units == "m":
        stl_scale_to_m = 1.0
    else:
        raise ValueError(f"Unsupported stl_units: {stl_units}")

    print("[1/5] Loading STL...")
    tris = load_stl_triangles(stl_path)
    print(f"  STL triangles: {tris.shape[0]} | assumed units: {stl_units}")

    tris_m = tris * stl_scale_to_m

    if auto_fit:
        tris_m, origin, Lx, Ly, Lz = auto_fit_domain_to_stl(tris_m, pad_m=pad_m, first_octant=first_octant)
        print(f"  Auto-fit domain (meters): origin={origin}, L=({Lx:.4f},{Ly:.4f},{Lz:.4f})")
    else:
        print(f"  Domain (meters): origin={origin}, L=({Lx:.4f},{Ly:.4f},{Lz:.4f})")

    print("[2/5] Building structured mesh...")
    mesh = StructuredHexMesh(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, origin=origin)

    print("[3/5] Voxelizing STL into solid cells (can be slow)...")
    solid = voxelize_stl(mesh, tris_m, samples_1d=voxel_samples_1d)
    nsolid = int(solid.sum())
    print(f"  Solid cells: {nsolid}/{mesh.n_cells} ({100.0*nsolid/mesh.n_cells:.3f}%)")

    if nsolid == 0:
        mn, mx = compute_bbox(tris_m)
        raise ValueError(
            "Voxelization produced 0 solid cells.\n"
            f"STL bounds in meters: min={mn}, max={mx}\n"
            "Fix by using --auto_fit OR adjusting --ox/--oy/--oz/--Lx/--Ly/--Lz OR confirming STL units.\n"
            "Also ensure mesh resolution is fine enough relative to geometry and padding.\n"
            "Try increasing --nx/--ny (or reduce --pad), or raise --voxel_samples_1d."
        )

    print("[4/5] Running layer-by-layer simulation (placeholder elasticity kernel)...")
    steps = build_layer_activation(mesh.nz, layers_per_output)
    stress = np.zeros(mesh.n_cells, dtype=np.float64)

    for step_id, (_k0, _k1) in enumerate(steps):
        _u, s = solve_linear_elasticity_dummy(mesh, solid)
        stress[:] = s
        write_vti_stub(out_dir, step_id, mesh, solid, stress)

    print("[5/5] Done.")
    print(f"Outputs in: {out_dir}")


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True, help="Input STL path")
    ap.add_argument("--stl_units", default="mm", choices=["mm", "cm", "m"],
                    help="Units of STL coordinates. Internal solver uses meters. Default: mm")

    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz", type=int, default=60)

    # Domain in meters (ignored if --auto_fit)
    ap.add_argument("--Lx", type=float, default=0.8)
    ap.add_argument("--Ly", type=float, default=0.4)
    ap.add_argument("--Lz", type=float, default=0.3)
    ap.add_argument("--ox", type=float, default=0.0)
    ap.add_argument("--oy", type=float, default=0.0)
    ap.add_argument("--oz", type=float, default=0.0)

    ap.add_argument("--auto_fit", action="store_true",
                    help="Auto-fit domain to STL bounds (after converting STL to meters)")
    ap.add_argument("--pad", type=float, default=0.02,
                    help="Padding (meters) used with --auto_fit")

    ap.add_argument("--first_octant", action="store_true",
                    help="With --auto_fit, shift STL so padded bounds lie in +x,+y,+z (origin becomes 0,0,0)")

    ap.add_argument("--layers_per_output", type=int, default=2)
    ap.add_argument("--out", type=str, default="out")

    ap.add_argument("--voxel_samples_1d", type=int, default=3,
                    help="Voxelization samples per axis per cell (3 => 27 samples). Increase if still missing.")

    args = ap.parse_args()

    run_sim(
        stl_path=args.stl,
        nx=args.nx, ny=args.ny, nz=args.nz,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        origin=(args.ox, args.oy, args.oz),
        layers_per_output=args.layers_per_output,
        out_dir=args.out,
        auto_fit=args.auto_fit,
        pad_m=args.pad,
        stl_units=args.stl_units,
        first_octant=args.first_octant,
        voxel_samples_1d=args.voxel_samples_1d,
    )


if __name__ == "__main__":
    main()