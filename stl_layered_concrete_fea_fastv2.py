# -*- coding: utf-8 -*-
"""
STL-layered concrete FEA (fast v2) — ACTIVE-SUBMESH MECHANICAL SOLVE
+ THERMAL/HYDRATION SUBSTEPPING (explicit stability)
+ ARRHENIUS OVERFLOW PROTECTION
+ STRONG AMBIENT (DIRICHLET) BOUNDARY FOR THERMAL (debug-stable)

Run example:
python stl_layered_concrete_fea_fastv2.py --stl statueofliberty2.stl --nx 20 --ny 20 --nz 80 --auto_fit
"""

import os
import struct
import argparse
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numba import njit, prange


# =========================================================
# STL loader
# =========================================================
def load_stl_triangles(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        _ = f.read(80)
        ntri_bytes = f.read(4)
        if len(ntri_bytes) < 4:
            raise ValueError("Not a valid STL file.")
        ntri = struct.unpack("<I", ntri_bytes)[0]
        f.seek(0, 2)
        fsize = f.tell()
        expected = 84 + ntri * 50
        is_binary = (fsize == expected)

    if is_binary:
        tris = np.zeros((ntri, 3, 3), dtype=np.float64)
        with open(path, "rb") as f:
            f.read(80)
            ntri = struct.unpack("<I", f.read(4))[0]
            for i in range(ntri):
                f.read(12)  # normal
                v = struct.unpack("<9f", f.read(36))
                tris[i, 0] = v[0:3]
                tris[i, 1] = v[3:6]
                tris[i, 2] = v[6:9]
                f.read(2)
        return tris
    else:
        verts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip().split()
                if len(s) == 4 and s[0].lower() == "vertex":
                    verts.append([float(s[1]), float(s[2]), float(s[3])])
        if len(verts) == 0 or (len(verts) % 3 != 0):
            raise ValueError("Failed to parse ASCII STL triangles.")
        return np.array(verts, dtype=np.float64).reshape((-1, 3, 3))


# =========================================================
# Numba voxelization (+Z ray, odd-even)
# =========================================================
@njit(inline="always")
def _cross(ax, ay, az, bx, by, bz):
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

@njit(inline="always")
def _dot(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz

@njit(inline="always")
def _ray_intersects_triangle_plus_z(x, y, z0,
                                   v0x, v0y, v0z,
                                   v1x, v1y, v1z,
                                   v2x, v2y, v2z,
                                   eps):
    e1x = v1x - v0x; e1y = v1y - v0y; e1z = v1z - v0z
    e2x = v2x - v0x; e2y = v2y - v0y; e2z = v2z - v0z

    # pvec = d x e2, d=(0,0,1) => (-e2y, e2x, 0)
    pvecx = -e2y
    pvecy =  e2x
    pvecz =  0.0

    det = _dot(e1x, e1y, e1z, pvecx, pvecy, pvecz)
    if det > -eps and det < eps:
        return False
    inv_det = 1.0 / det

    tvecx = x - v0x
    tvecy = y - v0y
    tvecz = z0 - v0z

    u = _dot(tvecx, tvecy, tvecz, pvecx, pvecy, pvecz) * inv_det
    if u < 0.0 or u > 1.0:
        return False

    qx, qy, qz = _cross(tvecx, tvecy, tvecz, e1x, e1y, e1z)

    v = qz * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = _dot(e2x, e2y, e2z, qx, qy, qz) * inv_det
    return t > 0.0

@njit(parallel=True)
def voxelize_stl_to_cells_numba(tris, tri_min, tri_max, nx, ny, nz, ox, oy, oz, dx, dy, dz):
    ne = nx * ny * nz
    solid = np.zeros(ne, dtype=np.uint8)
    eps = 1e-12

    for idx in prange(ne):
        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)

        x = ox + (i + 0.5) * dx
        y = oy + (j + 0.5) * dy
        zc = oz + (k + 0.5) * dz
        z0 = zc - 1e-9

        count = 0
        for t in range(tris.shape[0]):
            if tri_min[t, 0] > x or tri_max[t, 0] < x:
                continue
            if tri_min[t, 1] > y or tri_max[t, 1] < y:
                continue
            if tri_max[t, 2] < z0:
                continue

            v0x = tris[t, 0, 0]; v0y = tris[t, 0, 1]; v0z = tris[t, 0, 2]
            v1x = tris[t, 1, 0]; v1y = tris[t, 1, 1]; v1z = tris[t, 1, 2]
            v2x = tris[t, 2, 0]; v2y = tris[t, 2, 1]; v2z = tris[t, 2, 2]

            if _ray_intersects_triangle_plus_z(x, y, z0, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, eps):
                count += 1

        solid[idx] = 1 if (count % 2 == 1) else 0

    return solid


# =========================================================
# Connectivity filter: keep only active cells connected to base (k=0)
# =========================================================
@njit
def filter_connected_to_base(active_u8, nx, ny, nz):
    ne = nx * ny * nz
    keep = np.zeros(ne, dtype=np.uint8)

    q = np.empty(ne, dtype=np.int64)
    head = 0
    tail = 0

    # seed from base layer k=0
    k = 0
    base_offset = k * nx * ny
    for j in range(ny):
        row_offset = base_offset + j * nx
        for i in range(nx):
            idx = row_offset + i
            if active_u8[idx] == 1 and keep[idx] == 0:
                keep[idx] = 1
                q[tail] = idx
                tail += 1

    while head < tail:
        idx = q[head]
        head += 1

        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)

        # +/-x
        if i > 0:
            n = idx - 1
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1
        if i < nx - 1:
            n = idx + 1
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1
        # +/-y
        if j > 0:
            n = idx - nx
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1
        if j < ny - 1:
            n = idx + nx
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1
        # +/-z
        if k > 0:
            n = idx - nx * ny
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1
        if k < nz - 1:
            n = idx + nx * ny
            if active_u8[n] == 1 and keep[n] == 0:
                keep[n] = 1
                q[tail] = n
                tail += 1

    return keep


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
    origin: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.nodes, self.conn, self.elem_layer = self._build()

    def _build(self):
        ox, oy, oz = self.origin
        xs = ox + np.linspace(0, self.Lx, self.nx + 1)
        ys = oy + np.linspace(0, self.Ly, self.ny + 1)
        zs = oz + np.linspace(0, self.Lz, self.nz + 1)

        node_id = lambda i, j, k: (k * (self.ny + 1) + j) * (self.nx + 1) + i

        nodes = np.zeros(((self.nx + 1) * (self.ny + 1) * (self.nz + 1), 3), float)
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    nodes[node_id(i, j, k)] = [x, y, z]

        conn = []
        layers = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n000 = node_id(i, j, k)
                    n100 = node_id(i + 1, j, k)
                    n110 = node_id(i + 1, j + 1, k)
                    n010 = node_id(i, j + 1, k)
                    n001 = node_id(i, j, k + 1)
                    n101 = node_id(i + 1, j, k + 1)
                    n111 = node_id(i + 1, j + 1, k + 1)
                    n011 = node_id(i, j + 1, k + 1)
                    conn.append([n000, n100, n110, n010, n001, n101, n111, n011])
                    layers.append(k)
        return nodes, np.array(conn, dtype=np.int64), np.array(layers, dtype=np.int64)

    @property
    def nnodes(self):
        return self.nodes.shape[0]

    @property
    def nelems(self):
        return self.conn.shape[0]


# =========================================================
# Material & process
# =========================================================
@dataclass
class MaterialParams:
    nu: float = 0.2
    E0: float = 1e9
    E_inf: float = 30e9
    alpha_T: float = 1.0e-5
    beta_ch: float = 4.0e-4
    rho: float = 2400.0
    cp: float = 900.0
    kappa: float = 1.6
    Q_total: float = 3.5e5
    binder_frac: float = 0.18
    A_h: float = 2.5e-4
    E_a: float = 40000.0
    R: float = 8.314
    T_ref: float = 293.15
    alpha0: float = 0.0
    alpha_max: float = 0.85

@dataclass
class ProcessParams:
    layer_time: float = 90.0
    dt: float = 5.0                  # outer/global step (printing schedule + outputs)
    t_end: float = 90.0 * 10
    T_amb: float = 293.15
    T_print: float = 303.15
    g: float = 9.81


# =========================================================
# FEM helpers
# =========================================================
def gauss_points_2():
    a = 1.0 / np.sqrt(3.0)
    gps = []
    w = []
    for xi in (-a, a):
        for eta in (-a, a):
            for zeta in (-a, a):
                gps.append((xi, eta, zeta))
                w.append(1.0)
    return np.array(gps), np.array(w)

def hex8_dN_dxi(xi, eta, zeta):
    s = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], float)
    dN = np.zeros((8, 3), float)
    for a in range(8):
        sx, sy, sz = s[a]
        dN[a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
        dN[a, 1] = 0.125 * sy * (1 + sx * xi)  * (1 + sz * zeta)
        dN[a, 2] = 0.125 * sz * (1 + sx * xi)  * (1 + sy * eta)
    return dN

def B_matrix(dN_dx):
    B = np.zeros((6, 24), float)
    for a in range(8):
        ix = 3*a
        dNx, dNy, dNz = dN_dx[a]
        B[0, ix+0] = dNx
        B[1, ix+1] = dNy
        B[2, ix+2] = dNz
        B[3, ix+0] = dNy
        B[3, ix+1] = dNx
        B[4, ix+1] = dNz
        B[4, ix+2] = dNy
        B[5, ix+0] = dNz
        B[5, ix+2] = dNx
    return B

def C_iso(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    C = np.zeros((6,6), float)
    C[:3,:3] = lam
    np.fill_diagonal(C[:3,:3], lam + 2*mu)
    C[3,3] = mu
    C[4,4] = mu
    C[5,5] = mu
    return C

def voigt_I():
    return np.array([1,1,1,0,0,0], float)


# =========================================================
# Hydration + modulus (overflow-safe Arrhenius)
# =========================================================
def hydration_rate(alpha_h, T, mat: MaterialParams):
    alpha_h = np.clip(alpha_h, 0.0, mat.alpha_max)

    # Clamp temperature used in kinetics (prevents exp overflow)
    T_eff = np.clip(T, 273.15, 373.15)

    expo = (-mat.E_a/mat.R) * (1.0/T_eff - 1.0/mat.T_ref)
    expo = np.clip(expo, -60.0, 60.0)
    arr = np.exp(expo)

    return mat.A_h * arr * (1.0 - alpha_h / mat.alpha_max)

def modulus_evolution(alpha_h, mat: MaterialParams):
    n = 2.0
    r = np.clip(alpha_h / mat.alpha_max, 0.0, 1.0)
    return mat.E0 + (mat.E_inf - mat.E0) * (r ** n)


# =========================================================
# ACTIVE-NODE SUBMESH (Numba mapping)
# =========================================================
@njit
def build_active_node_map(conn: np.ndarray, active_elem: np.ndarray, nnodes: int):
    used = np.zeros(nnodes, dtype=np.uint8)
    ne = conn.shape[0]

    for e in range(ne):
        if not active_elem[e]:
            continue
        for a in range(8):
            used[conn[e, a]] = 1

    n_active = 0
    for n in range(nnodes):
        if used[n] == 1:
            n_active += 1

    active_nodes = np.empty(n_active, dtype=np.int64)
    node_map = np.empty(nnodes, dtype=np.int64)

    idx = 0
    for n in range(nnodes):
        if used[n] == 1:
            node_map[n] = idx
            active_nodes[idx] = n
            idx += 1
        else:
            node_map[n] = -1

    return node_map, active_nodes, n_active


# =========================================================
# Assembly (ACTIVE SUBMESH)
# =========================================================
def assemble_mech_active_submesh(mesh: StructuredHexMesh,
                                 active_elem: np.ndarray,
                                 node_map: np.ndarray,
                                 n_active_nodes: int,
                                 E_elem: np.ndarray,
                                 eigenstrain_voigt_elem: np.ndarray,
                                 mat: MaterialParams,
                                 body_force=np.array([0.0, 0.0, 0.0])):

    ndof = n_active_nodes * 3
    K = lil_matrix((ndof, ndof), dtype=float)
    f = np.zeros(ndof, float)

    gps, w = gauss_points_2()

    for e in range(mesh.nelems):
        if not active_elem[e]:
            continue

        conn_g = mesh.conn[e]
        conn_l = node_map[conn_g].astype(int)
        if np.any(conn_l < 0):
            continue

        Xe = mesh.nodes[conn_g]
        C = C_iso(E_elem[e], mat.nu)
        eps_star = eigenstrain_voigt_elem[e]

        Ke = np.zeros((24, 24), float)
        fe = np.zeros(24, float)

        for (xi, eta, zeta), wi in zip(gps, w):
            dN_dxi = hex8_dN_dxi(xi, eta, zeta)
            J = Xe.T @ dN_dxi
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Non-positive Jacobian det in element {e}: detJ={detJ}")
            invJ = np.linalg.inv(J)
            dN_dx = dN_dxi @ invJ.T
            B = B_matrix(dN_dx)
            dV = wi * detJ

            Ke += B.T @ C @ B * dV
            fe += B.T @ (C @ eps_star) * dV

            # body force
            s = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
            N = 0.125 * (1 + s[:,0]*xi) * (1 + s[:,1]*eta) * (1 + s[:,2]*zeta)
            for a in range(8):
                fe[3*a:3*a+3] += N[a] * body_force * dV

        dofs = np.zeros(24, dtype=int)
        for a in range(8):
            na = conn_l[a]
            dofs[3*a:3*a+3] = [3*na, 3*na+1, 3*na+2]

        for i in range(24):
            f[dofs[i]] += fe[i]
            row = dofs[i]
            for j in range(24):
                K[row, dofs[j]] += Ke[i, j]

    return K.tocsr(), f


# =========================================================
# Strong Dirichlet: properly adjusts RHS (-K[:,d]*u_d)
# =========================================================
def apply_dirichlet_strong(K: csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, values: np.ndarray):
    K = K.tolil()
    for dof, val in zip(fixed_dofs, values):
        if val != 0.0:
            col = K[:, dof].toarray().ravel()
            f -= col * val
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        f[dof] = val
    return K.tocsr(), f


# =========================================================
# Heat diffusion (explicit Laplacian)
# =========================================================
def laplacian_structured_node(T, nx, ny, nz, dx, dy, dz):
    T3 = T.reshape((nz+1, ny+1, nx+1))

    def get(k, j, i):
        k2 = min(max(k, 0), nz)
        j2 = min(max(j, 0), ny)
        i2 = min(max(i, 0), nx)
        return T3[k2, j2, i2]

    lap = np.zeros_like(T3)
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                cx = (get(k, j, i+1) - 2*get(k, j, i) + get(k, j, i-1)) / (dx*dx)
                cy = (get(k, j+1, i) - 2*get(k, j, i) + get(k, j-1, i)) / (dy*dy)
                cz = (get(k+1, j, i) - 2*get(k, j, i) + get(k-1, j, i)) / (dz*dz)
                lap[k, j, i] = cx + cy + cz
    return lap.reshape(-1)


# =========================================================
# VTI writer
# =========================================================
def _vtk_dataarray_ascii(name, ncomp, data_flat, dtype="Float32"):
    txt = " ".join(f"{x:.7e}" for x in data_flat)
    return (
        f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
        f'{txt}\n'
        f'</DataArray>\n'
    )

def write_vti(filename, nx, ny, nz, dx, dy, dz, origin,
              point_data=None, cell_data=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    point_data = {} if point_data is None else point_data
    cell_data  = {} if cell_data  is None else cell_data

    extent = f"0 {nx} 0 {ny} 0 {nz}"
    ox, oy, oz = origin

    parts = []
    parts.append('<?xml version="1.0"?>\n')
    parts.append('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
    parts.append(f'  <ImageData WholeExtent="{extent}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">\n')
    parts.append(f'    <Piece Extent="{extent}">\n')

    parts.append('      <PointData>\n')
    for name, arr in point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            parts.append(_vtk_dataarray_ascii(name, 1, arr.reshape(-1)))
        else:
            parts.append(_vtk_dataarray_ascii(name, arr.shape[1], arr.reshape(-1)))
    parts.append('      </PointData>\n')

    parts.append('      <CellData>\n')
    for name, arr in cell_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            parts.append(_vtk_dataarray_ascii(name, 1, arr.reshape(-1)))
        else:
            parts.append(_vtk_dataarray_ascii(name, arr.shape[1], arr.reshape(-1)))
    parts.append('      </CellData>\n')

    parts.append('    </Piece>\n')
    parts.append('  </ImageData>\n')
    parts.append('</VTKFile>\n')

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(parts)


# =========================================================
# Domain fit: XY first octant, zmin->0
# =========================================================
def compute_bbox(tris: np.ndarray):
    mn = tris.reshape(-1, 3).min(axis=0)
    mx = tris.reshape(-1, 3).max(axis=0)
    return mn, mx

def fit_domain_xy_octant_z0(tris_m: np.ndarray, pad_m: float):
    mn, _ = compute_bbox(tris_m)
    sx = -(mn[0] - pad_m)
    sy = -(mn[1] - pad_m)
    sz = -mn[2]
    shift = np.array([sx, sy, sz], dtype=np.float64)
    tris_s = tris_m + shift[None, None, :]

    _, mx2 = compute_bbox(tris_s)
    origin = (0.0, 0.0, 0.0)
    Lx = float(mx2[0] + pad_m)
    Ly = float(mx2[1] + pad_m)
    Lz = float(mx2[2] + pad_m)
    return tris_s, origin, Lx, Ly, Lz, shift


# =========================================================
# Main simulation
# =========================================================
def run_sim(stl_path: str,
            nx: int, ny: int, nz: int,
            layers_per_output: int,
            out_dir: str,
            auto_fit: bool,
            pad_m: float,
            stl_units: str):

    # STL units -> meters
    if stl_units.lower() == "mm":
        scale = 1e-3
    elif stl_units.lower() == "cm":
        scale = 1e-2
    elif stl_units.lower() == "m":
        scale = 1.0
    else:
        raise ValueError("stl_units must be one of: mm, cm, m")

    print("[1/5] Loading STL...")
    tris = load_stl_triangles(stl_path)
    print(f"STL triangles: {tris.shape[0]} | assumed units: {stl_units}")
    tris_m = tris * scale

    if not auto_fit:
        auto_fit = True

    tris_m, origin, Lx, Ly, Lz, shift = fit_domain_xy_octant_z0(tris_m, pad_m=pad_m)
    print(f"Auto-fit domain: origin={origin}, L=({Lx:.4f},{Ly:.4f},{Lz:.4f}) m | pad={pad_m} m")
    print(f"Applied STL shift (m): dx={shift[0]:.6f}, dy={shift[1]:.6f}, dz={shift[2]:.6f}")

    print("[2/5] Building structured mesh...")
    mesh = StructuredHexMesh(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, origin=origin)

    print("[3/5] Voxelizing STL into solid cells (Numba accelerated)...")
    tri_min = tris_m.min(axis=1)
    tri_max = tris_m.max(axis=1)
    ox, oy, oz = origin
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    solid_u8 = voxelize_stl_to_cells_numba(tris_m, tri_min, tri_max, nx, ny, nz, ox, oy, oz, dx, dy, dz)
    solid = solid_u8.astype(np.bool_)
    solid_cells = int(solid.sum())
    print(f"Solid cells: {solid_cells}/{mesh.nelems} ({solid_cells/mesh.nelems:.3%})")
    if solid_cells == 0:
        mn, mx = compute_bbox(tris_m)
        raise ValueError(f"Voxelization produced 0 solid cells. STL bounds (m): min={mn}, max={mx}")

    # BC candidates at z=0 (global)
    bottom_nodes = np.where(np.isclose(mesh.nodes[:, 2], 0.0))[0]
    print(f"[BC] Build plate candidates at z=0: {bottom_nodes.size} nodes (global).")

    mat = MaterialParams()
    proc = ProcessParams()

    nn = mesh.nnodes
    ne = mesh.nelems

    # Thermal stability + substepping parameters
    D = mat.kappa / (mat.rho * mat.cp)
    inv_h2 = (1.0/mesh.dx**2) + (1.0/mesh.dy**2) + (1.0/mesh.dz**2)
    dtcrit = 1.0 / (2.0 * D * inv_h2)
    dt_safe = 0.45 * dtcrit
    print(f"[thermal] dx={mesh.dx:.6f} dy={mesh.dy:.6f} dz={mesh.dz:.6f} m | D={D:.3e} m^2/s")
    print(f"[thermal] dtcrit={dtcrit:.3f} s | dt_safe={dt_safe:.3f} s | outer dt={proc.dt:.3f} s")

    # Strong ambient boundary (Dirichlet) while debugging stability
    boundary = (
        np.isclose(mesh.nodes[:, 0], 0.0) | np.isclose(mesh.nodes[:, 0], Lx) |
        np.isclose(mesh.nodes[:, 1], 0.0) | np.isclose(mesh.nodes[:, 1], Ly) |
        np.isclose(mesh.nodes[:, 2], Lz)
    )

    T = np.full(nn, proc.T_amb, float)
    alpha_h = np.full(ne, mat.alpha0, float)
    E_elem = np.full(ne, mat.E0, float)

    u_full = np.zeros(nn * 3, float)

    elem_print_time = mesh.elem_layer * proc.layer_time
    body_force = np.array([0.0, 0.0, -mat.rho * proc.g])

    times = np.arange(0.0, proc.t_end + 1e-12, proc.dt)

    last_output = -1
    step_out = 0

    print("[4/5] Time marching...")
    for step, t in enumerate(times):
        # printed-by-time and solid
        active_raw = ((t >= elem_print_time) & solid).astype(np.uint8)

        # keep only base-connected (prevents floating islands)
        keep_u8 = filter_connected_to_base(active_raw, nx, ny, nz)
        active = (keep_u8 == 1)

        # newly activated -> print temperature
        newly_active = active & ((t - proc.dt) < elem_print_time)
        if np.any(newly_active):
            for e in np.where(newly_active)[0]:
                T[mesh.conn[e]] = proc.T_print

        # Update modulus on OUTER step based on current alpha_h (will be updated in substeps)
        E_active = modulus_evolution(alpha_h, mat)
        E_elem = np.where(active, E_active, mat.E0)

        # ---- THERMAL + HYDRATION SUBSTEPPING (stable explicit) ----
        if proc.dt <= dt_safe:
            nsub = 1
        else:
            nsub = int(np.ceil(proc.dt / dt_safe))
        dt = proc.dt / nsub

        for _ in range(nsub):
            # element-average temperature from current T
            Te = np.zeros(ne, float)
            for e in range(ne):
                Te[e] = T[mesh.conn[e]].mean()

            # hydration update
            dadt = hydration_rate(alpha_h, Te, mat)
            alpha_h = alpha_h + dt * dadt * active
            alpha_h = np.clip(alpha_h, 0.0, mat.alpha_max)

            # update modulus inside substeps (helps when alpha changes quickly)
            E_active = modulus_evolution(alpha_h, mat)
            E_elem = np.where(active, E_active, mat.E0)

            # heat source -> nodes (smoothed nodal volumetric source)
            q_node = np.zeros(nn, float)
            q_count = np.zeros(nn, float)
            for e in range(ne):
                if not active[e]:
                    continue
                q_vol = (mat.binder_frac * mat.rho) * mat.Q_total * dadt[e]  # W/m^3
                nodes_e = mesh.conn[e]
                q_node[nodes_e] += q_vol
                q_count[nodes_e] += 1.0
            q_node = np.divide(q_node, np.maximum(q_count, 1.0))

            # diffusion + source
            lapT = laplacian_structured_node(T, nx, ny, nz, dx, dy, dz)
            T = T + dt * (D * lapT + (q_node/(mat.rho*mat.cp)))

            # strong ambient boundary (debug-stable)
            T[boundary] = proc.T_amb

            # sanity clamp to avoid a single bad step poisoning everything
            T = np.clip(T, 250.0, 500.0)

        # recompute Te on OUTER step for eigenstrain
        Te = np.zeros(ne, float)
        for e in range(ne):
            Te[e] = T[mesh.conn[e]].mean()

        # eigenstrain
        eps_th = mat.alpha_T * (Te - mat.T_ref)
        eps_ch = mat.beta_ch * (alpha_h - mat.alpha0)
        eps_iso = eps_th + eps_ch
        I6 = voigt_I()
        eigenstrain = (eps_iso[:, None] * I6[None, :]).astype(float)

        # -------------------------
        # Mechanical solve (ACTIVE SUBMESH ONLY)
        # -------------------------
        if not active.any():
            u_full[:] = 0.0
        else:
            node_map, active_nodes, n_active_nodes = build_active_node_map(mesh.conn, active, nn)

            bottom_active = bottom_nodes[node_map[bottom_nodes] >= 0]
            bottom_local = node_map[bottom_active].astype(np.int64)

            fixed_dofs = np.empty(bottom_local.size * 3, dtype=np.int64)
            for i, ln in enumerate(bottom_local):
                fixed_dofs[3*i + 0] = 3*ln + 0
                fixed_dofs[3*i + 1] = 3*ln + 1
                fixed_dofs[3*i + 2] = 3*ln + 2
            fixed_vals = np.zeros_like(fixed_dofs, float)

            K, f = assemble_mech_active_submesh(
                mesh, active,
                node_map=node_map, n_active_nodes=n_active_nodes,
                E_elem=E_elem,
                eigenstrain_voigt_elem=eigenstrain,
                mat=mat,
                body_force=body_force
            )
            Kbc, fbc = apply_dirichlet_strong(K, f, fixed_dofs, fixed_vals)
            u_sub = spsolve(Kbc, fbc)

            u_full[:] = 0.0
            for li in range(n_active_nodes):
                g = active_nodes[li]
                u_full[3*g:3*g+3] = u_sub[3*li:3*li+3]

        # output trigger
        if active.any():
            active_layers = int(mesh.elem_layer[active].max()) + 1
        else:
            active_layers = 0
        numerical_layer = active_layers // max(int(layers_per_output), 1)

        if numerical_layer > last_output and active_layers > 0:
            last_output = numerical_layer
            U = u_full.reshape((nn, 3))
            Umag = np.linalg.norm(U, axis=1)

            vti_path = os.path.join(out_dir, f"step_{step_out:04d}_layers{active_layers:03d}.vti")
            write_vti(
                vti_path,
                nx=nx, ny=ny, nz=nz,
                dx=dx, dy=dy, dz=dz,
                origin=origin,
                point_data={"T": T, "U": U, "Umag": Umag},
                cell_data={"alpha_h": alpha_h, "active": active.astype(np.float32), "solid": solid.astype(np.float32)}
            )
            print(f"[VTI] {vti_path} (active_layers={active_layers}, numerical_layer={numerical_layer})")
            step_out += 1

        if step % 10 == 0 or step == len(times) - 1:
            umax = float(np.max(np.abs(u_full))) if np.all(np.isfinite(u_full)) else float(np.nanmax(np.abs(u_full)))
            ah = float(alpha_h[active].mean()) if active.any() else 0.0
            print(f"t={t:8.1f}s | active_layers={active_layers:3d}/{nz} | max|u|={umax:.3e} m | "
                  f"Tavg={T.mean():.2f} K | alpha_avg(active)={ah:.3f}")

    print("Done. Load .vti in ParaView. Use Warp By Vector on U.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True, help="Input STL path")
    ap.add_argument("--stl_units", default="mm", choices=["mm", "cm", "m"])
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--auto_fit", action="store_true", help="Auto-fit domain with XY first octant and zmin->0")
    ap.add_argument("--pad", type=float, default=0.002, help="Padding in meters")
    ap.add_argument("--layers_per_output", type=int, default=2)
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()

    run_sim(
        stl_path=args.stl,
        nx=args.nx, ny=args.ny, nz=args.nz,
        layers_per_output=args.layers_per_output,
        out_dir=args.out,
        auto_fit=args.auto_fit,
        pad_m=args.pad,
        stl_units=args.stl_units,
    )


if __name__ == "__main__":
    main()