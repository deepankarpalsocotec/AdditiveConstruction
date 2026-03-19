# -*- coding: utf-8 -*-
"""
stl_layered_concrete_fea_fastv12_microplane.py  (V12)

V12 = V11 + optional microplane-style scalar damage model with a constitutive
model flag and VTI damage output.

What changed from V11:
- Added --mech_model with choices: linear_elastic, microplane_damage
- Added a fast microplane-inspired damage driver using element-center strains,
  irreversible history variable kappa_eq, and secant stiffness degradation
- VTI output now also includes cell damage
- Stress recovery uses the effective (damage-reduced) modulus when the damage
  model is selected

Notes:
- The microplane option implemented here is a computationally light,
  microplane-inspired scalar damage extension for this structured-mesh solver.
  It is not a full Bažant M4/M7 constitutive update.
- The damage update is handled in a staggered fixed-point manner to preserve the
  overall structure and speed of the original code.

Run:
python stl_layered_concrete_fea_fastv12_microplane.py --stl statueofliberty2.stl --nx 20 --ny 20 --nz 80 --auto_fit --mech_model microplane_damage
"""

import os
import json
import struct
import argparse
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
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
                f.read(12)
                v = struct.unpack("<9f", f.read(36))
                tris[i, 0] = v[0:3]
                tris[i, 1] = v[3:6]
                tris[i, 2] = v[6:9]
                f.read(2)
        return tris

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
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


@njit(inline="always")
def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(inline="always")
def _ray_intersects_triangle_plus_z(x, y, z0,
                                   v0x, v0y, v0z,
                                   v1x, v1y, v1z,
                                   v2x, v2y, v2z,
                                   eps):
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    pvecx = -e2y
    pvecy = e2x
    pvecz = 0.0

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

            v0x = tris[t, 0, 0]
            v0y = tris[t, 0, 1]
            v0z = tris[t, 0, 2]
            v1x = tris[t, 1, 0]
            v1y = tris[t, 1, 1]
            v1z = tris[t, 1, 2]
            v2x = tris[t, 2, 0]
            v2y = tris[t, 2, 1]
            v2z = tris[t, 2, 2]

            if _ray_intersects_triangle_plus_z(
                x, y, z0,
                v0x, v0y, v0z,
                v1x, v1y, v1z,
                v2x, v2y, v2z,
                eps,
            ):
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

    base_offset = 0
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
        xs = ox + np.linspace(0.0, self.Lx, self.nx + 1)
        ys = oy + np.linspace(0.0, self.Ly, self.ny + 1)
        zs = oz + np.linspace(0.0, self.Lz, self.nz + 1)

        def node_id(i, j, k):
            return (k * (self.ny + 1) + j) * (self.nx + 1) + i

        nodes = np.zeros(((self.nx + 1) * (self.ny + 1) * (self.nz + 1), 3), dtype=np.float64)
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    nodes[node_id(i, j, k)] = [x, y, z]

        conn = []
        layers = []
        # Eight corners of the elements have crystalline symmetry
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

        return (
            np.array(nodes, dtype=np.float64),
            np.array(conn, dtype=np.int64),
            np.array(layers, dtype=np.int64),
        )

    @property
    def nnodes(self):
        return self.nodes.shape[0]

    @property
    def nelems(self):
        return self.conn.shape[0]


# =========================================================
# Material and process
# =========================================================
@dataclass
class MaterialParams:
    nu: float = 0.2
    E0: float = 1.0e9
    E_inf: float = 30.0e9
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
    damage_eps0: float = 1.0e-4
    damage_epsf: float = 8.0e-4
    damage_exp: float = 2.0
    damage_max: float = 0.995
    damage_residual_stiffness: float = 5.0e-3
    microplane_shear_c: float = 0.25
    microplane_max_iters: int = 2
    microplane_tol: float = 1.0e-5


@dataclass
class ProcessParams:
    layer_time: float = 90.0
    dt: float = 5.0
    t_end: float = 90.0 * 80
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
    return np.array(gps, dtype=np.float64), np.array(w, dtype=np.float64)


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
    ], dtype=np.float64)
    dN = np.zeros((8, 3), dtype=np.float64)
    for a in range(8):
        sx, sy, sz = s[a]
        dN[a, 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN[a, 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        dN[a, 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return dN


def C_iso(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], lam + 2.0 * mu)
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def voigt_I():
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def B_matrix(dN_dx):
    B = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        ix = 3 * a
        dNx, dNy, dNz = dN_dx[a]
        B[0, ix + 0] = dNx
        B[1, ix + 1] = dNy
        B[2, ix + 2] = dNz
        B[3, ix + 0] = dNy
        B[3, ix + 1] = dNx
        B[4, ix + 1] = dNz
        B[4, ix + 2] = dNy
        B[5, ix + 0] = dNz
        B[5, ix + 2] = dNx
    return B


# =========================================================
# Precomputed reference HEX8 mechanics templates
# =========================================================
_GPS_2, _GW_2 = gauss_points_2()
_DN_DXI_2 = np.array([hex8_dN_dxi(xi, eta, zeta) for (xi, eta, zeta) in _GPS_2], dtype=np.float64)
_DN_DXI_CENTER = hex8_dN_dxi(0.0, 0.0, 0.0)
_HEX8_S = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=np.float64)

_N_2 = np.empty((len(_GPS_2), 8), dtype=np.float64)
for igp, (xi, eta, zeta) in enumerate(_GPS_2):
    _N_2[igp, :] = 0.125 * (1.0 + _HEX8_S[:, 0] * xi) * (1.0 + _HEX8_S[:, 1] * eta) * (1.0 + _HEX8_S[:, 2] * zeta)


def build_hex8_reference_templates(Xe_ref: np.ndarray, mat: MaterialParams, body_force: np.ndarray):
    if Xe_ref.shape != (8, 3):
        raise ValueError("Xe_ref must be shape (8,3).")

    C_unit = C_iso(1.0, mat.nu)
    I6 = voigt_I()

    J = Xe_ref.T @ _DN_DXI_CENTER
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(f"Non-positive reference element Jacobian detJ={detJ}")
    invJ = np.linalg.inv(J)

    dN_dx_center = _DN_DXI_CENTER @ invJ.T
    B_center = B_matrix(dN_dx_center)

    Ke_unit = np.zeros((24, 24), dtype=np.float64)
    fe_iso_unit = np.zeros(24, dtype=np.float64)
    fe_body = np.zeros(24, dtype=np.float64)

    for igp in range(8):
        dN_dx = _DN_DXI_2[igp] @ invJ.T
        B = B_matrix(dN_dx)
        dV = _GW_2[igp] * detJ

        Ke_unit += (B.T @ (C_unit @ B)) * dV
        fe_iso_unit += (B.T @ (C_unit @ I6)) * dV

        if body_force is not None and np.any(body_force != 0.0):
            N = _N_2[igp]
            for a in range(8):
                ia = 3 * a
                fe_body[ia:ia + 3] += N[a] * body_force * dV

    return Ke_unit, fe_iso_unit, fe_body, B_center, C_unit


# =========================================================
# Hydration and modulus
# =========================================================
def hydration_rate(alpha_h, T, mat: MaterialParams):
    alpha_h = np.clip(alpha_h, 0.0, mat.alpha_max)
    T_eff = np.clip(T, 273.15, 373.15)
    expo = (-mat.E_a / mat.R) * (1.0 / T_eff - 1.0 / mat.T_ref)
    expo = np.clip(expo, -60.0, 60.0)
    arr = np.exp(expo)
    return mat.A_h * arr * (1.0 - alpha_h / mat.alpha_max)


def modulus_evolution(alpha_h, mat: MaterialParams):
    n = 2.0
    r = np.clip(alpha_h / mat.alpha_max, 0.0, 1.0)
    return mat.E0 + (mat.E_inf - mat.E0) * (r ** n)


def voigt_to_strain_tensor(strain_voigt: np.ndarray):
    strain_voigt = np.asarray(strain_voigt, dtype=np.float64)
    eps = np.zeros(strain_voigt.shape[:-1] + (3, 3), dtype=np.float64)
    eps[..., 0, 0] = strain_voigt[..., 0]
    eps[..., 1, 1] = strain_voigt[..., 1]
    eps[..., 2, 2] = strain_voigt[..., 2]
    eps[..., 0, 1] = eps[..., 1, 0] = 0.5 * strain_voigt[..., 3]
    eps[..., 1, 2] = eps[..., 2, 1] = 0.5 * strain_voigt[..., 4]
    eps[..., 0, 2] = eps[..., 2, 0] = 0.5 * strain_voigt[..., 5]
    return eps


def build_microplane_normals():
    normals = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
    ], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals


_MICROPLANE_NORMALS = build_microplane_normals()
_MICROPLANE_WEIGHTS = np.full(_MICROPLANE_NORMALS.shape[0], 1.0 / _MICROPLANE_NORMALS.shape[0], dtype=np.float64)


def compute_microplane_equivalent_strain_active(cache: "MechTopologyCache",
                                                u_sub: np.ndarray,
                                                ne_total: int,
                                                eps_iso_elem: np.ndarray,
                                                B_center: np.ndarray,
                                                microplane_normals: np.ndarray,
                                                microplane_weights: np.ndarray,
                                                shear_c: float):
    eq = np.zeros(ne_total, dtype=np.float64)
    if cache.active_elems.size == 0 or u_sub is None or u_sub.size == 0:
        return eq

    u_elem = u_sub[cache.edofs]
    strain = u_elem @ B_center.T

    I6 = voigt_I()
    strain_mech = strain - eps_iso_elem[cache.active_elems, None] * I6[None, :]
    eps_tensor = voigt_to_strain_tensor(strain_mech)

    eq2_accum = np.zeros(eps_tensor.shape[0], dtype=np.float64)
    for w, n in zip(microplane_weights, microplane_normals):
        eps_vec = eps_tensor @ n
        eps_n = eps_vec @ n
        eps_t = eps_vec - eps_n[:, None] * n[None, :]
        eq2_plane = np.maximum(eps_n, 0.0) ** 2 + shear_c * np.sum(eps_t * eps_t, axis=1)
        eq2_accum += w * eq2_plane

    eq[cache.active_elems] = np.sqrt(np.maximum(eq2_accum, 0.0))
    return eq


def damage_from_kappa(kappa_eq: np.ndarray, mat: MaterialParams):
    d = np.zeros_like(kappa_eq, dtype=np.float64)
    mask = kappa_eq > mat.damage_eps0
    if np.any(mask):
        xi = np.maximum((kappa_eq[mask] - mat.damage_eps0) / max(mat.damage_epsf, 1.0e-16), 0.0)
        d[mask] = mat.damage_max * (1.0 - np.exp(-(xi ** mat.damage_exp)))
    return np.clip(d, 0.0, mat.damage_max)


def effective_modulus_with_damage(E_base: np.ndarray, damage: np.ndarray, mat: MaterialParams):
    reduction = np.clip(1.0 - damage, mat.damage_residual_stiffness, 1.0)
    return E_base * reduction


# =========================================================
# Active-node submesh map
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
# Cached active mechanics topology
# =========================================================
class MechTopologyCache:
    def __init__(self):
        self.active_signature = None
        self.active_elems = None
        self.node_map = None
        self.active_nodes = None
        self.n_active_nodes = 0
        self.rows = None
        self.cols = None
        self.edofs = None
        self.fixed_dofs = None
        self.free_dofs = None
        self.ndof = 0
        self.data = None
        self.elem_offsets = None

    def build(self, mesh: StructuredHexMesh, active: np.ndarray, bottom_nodes: np.ndarray):
        active_elems = np.where(active)[0].astype(np.int64)
        sig = tuple(active_elems.tolist())

        if sig == self.active_signature:
            return False

        self.active_signature = sig
        self.active_elems = active_elems

        if active_elems.size == 0:
            self.node_map = np.full(mesh.nnodes, -1, dtype=np.int64)
            self.active_nodes = np.zeros(0, dtype=np.int64)
            self.n_active_nodes = 0
            self.rows = np.zeros(0, dtype=np.int64)
            self.cols = np.zeros(0, dtype=np.int64)
            self.edofs = np.zeros((0, 24), dtype=np.int64)
            self.fixed_dofs = np.zeros(0, dtype=np.int64)
            self.free_dofs = np.zeros(0, dtype=np.int64)
            self.ndof = 0
            self.data = np.zeros(0, dtype=np.float64)
            self.elem_offsets = np.zeros(0, dtype=np.int64)
            return True

        node_map, active_nodes, n_active_nodes = build_active_node_map(mesh.conn, active, mesh.nnodes)
        self.node_map = node_map
        self.active_nodes = active_nodes
        self.n_active_nodes = int(n_active_nodes)
        self.ndof = 3 * self.n_active_nodes

        ne_valid = active_elems.size
        block_nnz = 24 * 24
        edofs = np.empty((ne_valid, 24), dtype=np.int64)
        rows = np.empty(ne_valid * block_nnz, dtype=np.int64)
        cols = np.empty(ne_valid * block_nnz, dtype=np.int64)
        elem_offsets = np.empty(ne_valid, dtype=np.int64)

        p = 0
        for idx, e in enumerate(active_elems):
            conn_g = mesh.conn[e]
            conn_l = node_map[conn_g]
            if np.any(conn_l < 0):
                raise ValueError(f"Inactive local node found in active element {e}")

            dofs = np.empty(24, dtype=np.int64)
            for a in range(8):
                na = conn_l[a]
                base = 3 * na
                ia = 3 * a
                dofs[ia + 0] = base + 0
                dofs[ia + 1] = base + 1
                dofs[ia + 2] = base + 2
            edofs[idx] = dofs

            ii = np.repeat(dofs, 24)
            jj = np.tile(dofs, 24)
            rows[p:p + block_nnz] = ii
            cols[p:p + block_nnz] = jj
            elem_offsets[idx] = p
            p += block_nnz

        self.edofs = edofs
        self.rows = rows
        self.cols = cols
        self.elem_offsets = elem_offsets
        self.data = np.empty(ne_valid * block_nnz, dtype=np.float64)

        bottom_active = bottom_nodes[node_map[bottom_nodes] >= 0]
        bottom_local = node_map[bottom_active].astype(np.int64)

        fixed_dofs = np.empty(bottom_local.size * 3, dtype=np.int64)
        for i, ln in enumerate(bottom_local):
            fixed_dofs[3 * i + 0] = 3 * ln + 0
            fixed_dofs[3 * i + 1] = 3 * ln + 1
            fixed_dofs[3 * i + 2] = 3 * ln + 2
        self.fixed_dofs = fixed_dofs

        fixed_mask = np.zeros(self.ndof, dtype=bool)
        fixed_mask[fixed_dofs] = True
        self.free_dofs = np.where(~fixed_mask)[0].astype(np.int64)

        return True


# =========================================================
# Mechanics assembly using templates + cached topology + cached data buffer
# =========================================================
def assemble_mech_active_submesh_v10(cache: MechTopologyCache,
                                     E_elem: np.ndarray,
                                     eps_iso_elem: np.ndarray,
                                     Ke_unit_flat: np.ndarray,
                                     fe_iso_unit: np.ndarray,
                                     fe_body_unit: np.ndarray):
    ndof = cache.ndof
    if ndof == 0 or cache.active_elems.size == 0:
        K = sp.csr_matrix((ndof, ndof), dtype=np.float64)
        f = np.zeros(ndof, dtype=np.float64)
        return K, f

    block_nnz = 24 * 24
    data = cache.data
    f = np.zeros(ndof, dtype=np.float64)

    for idx, e in enumerate(cache.active_elems):
        start = cache.elem_offsets[idx]
        stop = start + block_nnz
        scale = E_elem[e]
        data[start:stop] = scale * Ke_unit_flat

        dofs = cache.edofs[idx]
        f[dofs] += (-scale * eps_iso_elem[e]) * fe_iso_unit + fe_body_unit

    K = sp.coo_matrix((data, (cache.rows, cache.cols)), shape=(ndof, ndof), dtype=np.float64).tocsr()
    return K, f


# =========================================================
# Solve on free DOFs
# =========================================================
def solve_free_dofs(K: sp.csr_matrix, f: np.ndarray, free_dofs: np.ndarray):
    ndof = K.shape[0]
    u = np.zeros(ndof, dtype=np.float64)
    if ndof == 0 or free_dofs.size == 0:
        return u

    Kff = K[free_dofs][:, free_dofs]
    ff = f[free_dofs]
    u[free_dofs] = spsolve(Kff, ff)
    return u


# =========================================================
# Fast stress recovery at output time only
# =========================================================
def compute_cell_stress_active(cache: MechTopologyCache,
                               u_sub: np.ndarray,
                               ne_total: int,
                               E_eff_elem: np.ndarray,
                               eps_iso_elem: np.ndarray,
                               B_center: np.ndarray,
                               C_unit: np.ndarray):
    sigma = np.zeros((ne_total, 6), dtype=np.float64)
    sigma_vm = np.zeros(ne_total, dtype=np.float64)

    if cache.active_elems.size == 0 or u_sub is None or u_sub.size == 0:
        return sigma, sigma_vm

    u_elem = u_sub[cache.edofs]                      # (n_active, 24)
    strain = u_elem @ B_center.T                    # (n_active, 6)

    I6 = voigt_I()
    strain_mech = strain - eps_iso_elem[cache.active_elems, None] * I6[None, :]
    sigma_act = (strain_mech @ C_unit.T) * E_eff_elem[cache.active_elems, None]

    sx = sigma_act[:, 0]
    sy = sigma_act[:, 1]
    sz = sigma_act[:, 2]
    txy = sigma_act[:, 3]
    tyz = sigma_act[:, 4]
    txz = sigma_act[:, 5]
    sigma_vm_act = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) +
        3.0 * (txy ** 2 + tyz ** 2 + txz ** 2)
    )

    sigma[cache.active_elems, :] = sigma_act
    sigma_vm[cache.active_elems] = sigma_vm_act
    return sigma, sigma_vm


# =========================================================
# Heat diffusion (explicit Laplacian)
# =========================================================
def laplacian_structured_node(T, nx, ny, nz, dx, dy, dz):
    T3 = T.reshape((nz + 1, ny + 1, nx + 1))

    def get(k, j, i):
        k2 = min(max(k, 0), nz)
        j2 = min(max(j, 0), ny)
        i2 = min(max(i, 0), nx)
        return T3[k2, j2, i2]

    lap = np.zeros_like(T3)
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                cx = (get(k, j, i + 1) - 2.0 * get(k, j, i) + get(k, j, i - 1)) / (dx * dx)
                cy = (get(k, j + 1, i) - 2.0 * get(k, j, i) + get(k, j - 1, i)) / (dy * dy)
                cz = (get(k + 1, j, i) - 2.0 * get(k, j, i) + get(k - 1, j, i)) / (dz * dz)
                lap[k, j, i] = cx + cy + cz
    return lap.reshape(-1)


# =========================================================
# VTI writer (ASCII) + series writer
# =========================================================
def _vtk_dataarray_ascii(name, ncomp, data_flat, dtype="Float32"):
    txt = " ".join(f"{float(x):.7e}" for x in data_flat)
    return (
        f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
        f'{txt}\n'
        f'</DataArray>\n'
    )


def write_vti(filename, nx, ny, nz, dx, dy, dz, origin, point_data=None, cell_data=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data

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


def write_vti_series(series_path: str, entries):
    payload = {"file-series-version": "1.0", "files": entries}
    os.makedirs(os.path.dirname(series_path), exist_ok=True)
    with open(series_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# Domain fit: XY first octant, zmin -> 0
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
            nx: int,
            ny: int,
            nz: int,
            layers_per_output: int,
            out_dir: str,
            auto_fit: bool,
            pad_m: float,
            stl_units: str,
            mech_model: str):

    if stl_units.lower() == "mm":
        scale = 1.0e-3
    elif stl_units.lower() == "cm":
        scale = 1.0e-2
    elif stl_units.lower() == "m":
        scale = 1.0
    else:
        raise ValueError("stl_units must be one of: mm, cm, m")

    mech_model = mech_model.lower()
    if mech_model not in ("linear_elastic", "microplane_damage"):
        raise ValueError("mech_model must be one of: linear_elastic, microplane_damage")

    print(f"[model] Mechanics model: {mech_model}")
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
    # STL cells to Numba conversion-number of divisions in x, y and z. Origin in x, y and z and teh grid spacing in x, y and z
    solid = solid_u8.astype(np.bool_)
    solid_cells = int(solid.sum())
    print(f"Solid cells: {solid_cells}/{mesh.nelems} ({solid_cells / mesh.nelems:.3%})")
    if solid_cells == 0:
        mn, mx = compute_bbox(tris_m)
        raise ValueError(f"Voxelization produced 0 solid cells. STL bounds (m): min={mn}, max={mx}")

    bottom_nodes = np.where(np.isclose(mesh.nodes[:, 2], 0.0))[0]
    print(f"[BC] Build plate candidates at z=0: {bottom_nodes.size} nodes (global).")

    mat = MaterialParams()
    proc = ProcessParams()

    nn = mesh.nnodes
    ne = mesh.nelems

    D = mat.kappa / (mat.rho * mat.cp)
    inv_h2 = (1.0 / mesh.dx ** 2) + (1.0 / mesh.dy ** 2) + (1.0 / mesh.dz ** 2)
    dtcrit = 1.0 / (2.0 * D * inv_h2)
    dt_safe = 0.45 * dtcrit
    print(f"[thermal] dx={mesh.dx:.6f} dy={mesh.dy:.6f} dz={mesh.dz:.6f} m | D={D:.3e} m^2/s")
    print(f"[thermal] dtcrit={dtcrit:.3f} s | dt_safe={dt_safe:.3f} s | outer dt={proc.dt:.3f} s")

    boundary = (
        np.isclose(mesh.nodes[:, 0], 0.0) | np.isclose(mesh.nodes[:, 0], Lx) |
        np.isclose(mesh.nodes[:, 1], 0.0) | np.isclose(mesh.nodes[:, 1], Ly) |
        np.isclose(mesh.nodes[:, 2], Lz)
    )

    T = np.full(nn, proc.T_amb, dtype=np.float64)
    alpha_h = np.full(ne, mat.alpha0, dtype=np.float64)
    E_base = np.full(ne, mat.E0, dtype=np.float64)
    E_eff = E_base.copy()
    damage = np.zeros(ne, dtype=np.float64)
    kappa_eq = np.zeros(ne, dtype=np.float64)
    u_full = np.zeros(nn * 3, dtype=np.float64)
    u_sub = None

    elem_print_time = mesh.elem_layer * proc.layer_time
    body_force = np.array([0.0, 0.0, -mat.rho * proc.g], dtype=np.float64)

    Xe_ref = mesh.nodes[mesh.conn[0]]
    Ke_unit, fe_iso_unit, fe_body_unit, B_center, C_unit = build_hex8_reference_templates(Xe_ref, mat, body_force)
    Ke_unit_flat = Ke_unit.reshape(-1).copy()
    print("[mechanics] Precomputed reference HEX8 templates and center-stress operator.")

    mech_cache = MechTopologyCache()

    times = np.arange(0.0, proc.t_end + 1.0e-12, proc.dt)
    last_output = -1
    step_out = 0
    series_entries = []
    series_path = os.path.join(out_dir, "sim.vti.series")
    prev_active = np.zeros(ne, dtype=bool)

    print("[4/5] Time marching...")
    for step, t in enumerate(times):
        active_raw = ((t >= elem_print_time) & solid).astype(np.uint8)
        keep_u8 = filter_connected_to_base(active_raw, nx, ny, nz)
        active = (keep_u8 == 1)

        newly_active = active & (~prev_active)
        cache_rebuilt = mech_cache.build(mesh, active, bottom_nodes)

        node_in_active = np.zeros(nn, dtype=bool)
        if mech_cache.n_active_nodes > 0:
            node_in_active[mech_cache.active_nodes] = True

        if np.any(newly_active):
            for e in np.where(newly_active)[0]:
                T[mesh.conn[e]] = proc.T_print

        T[~node_in_active] = proc.T_amb
        T[boundary] = proc.T_amb

        nsub = 1 if (proc.dt <= dt_safe) else int(np.ceil(proc.dt / dt_safe))
        dt = proc.dt / nsub

        for _ in range(nsub):
            Te = np.full(ne, proc.T_amb, dtype=np.float64)
            if active.any():
                active_elems = np.where(active)[0]
                for e in active_elems:
                    Te[e] = T[mesh.conn[e]].mean()

            dadt = np.zeros(ne, dtype=np.float64)
            if active.any():
                dadt_active = hydration_rate(alpha_h[active], Te[active], mat)
                dadt[active] = dadt_active
                alpha_h[active] = np.clip(alpha_h[active] + dt * dadt_active, 0.0, mat.alpha_max)

            E_active = modulus_evolution(alpha_h, mat)
            E_base = np.where(active, E_active, mat.E0)

            q_node = np.zeros(nn, dtype=np.float64)
            q_count = np.zeros(nn, dtype=np.float64)
            if active.any():
                active_elems = np.where(active)[0]
                for e in active_elems:
                    q_vol = (mat.binder_frac * mat.rho) * mat.Q_total * dadt[e]
                    nodes_e = mesh.conn[e]
                    q_node[nodes_e] += q_vol
                    q_count[nodes_e] += 1.0
                q_node = np.divide(q_node, np.maximum(q_count, 1.0))

            lapT = laplacian_structured_node(T, nx, ny, nz, dx, dy, dz)
            T_new = T + dt * (D * lapT + (q_node / (mat.rho * mat.cp)))

            if node_in_active.any():
                T[node_in_active] = T_new[node_in_active]

            T[~node_in_active] = proc.T_amb
            T[boundary] = proc.T_amb
            T = np.clip(T, 250.0, 500.0)

        Te = np.full(ne, proc.T_amb, dtype=np.float64)
        if active.any():
            active_elems = np.where(active)[0]
            for e in active_elems:
                Te[e] = T[mesh.conn[e]].mean()

        eps_th = mat.alpha_T * (Te - mat.T_ref)
        eps_ch = mat.beta_ch * (mat.alpha0 - alpha_h)
        eps_iso = eps_th + eps_ch

        if mech_model == "linear_elastic":
            damage[:] = 0.0
            kappa_eq[:] = 0.0

        if not active.any():
            E_eff = E_base.copy()
            u_full[:] = 0.0
            u_sub = None
        else:
            if mech_model == "linear_elastic":
                E_eff = E_base.copy()
                K, f = assemble_mech_active_submesh_v10(
                    cache=mech_cache,
                    E_elem=E_eff,
                    eps_iso_elem=eps_iso,
                    Ke_unit_flat=Ke_unit_flat,
                    fe_iso_unit=fe_iso_unit,
                    fe_body_unit=fe_body_unit,
                )
                u_sub = solve_free_dofs(K, f, mech_cache.free_dofs)
            else:
                damage_iter = damage.copy()
                kappa_iter = kappa_eq.copy()
                max_iters = max(int(mat.microplane_max_iters), 1)

                for _iter in range(max_iters):
                    E_eff_iter = effective_modulus_with_damage(E_base, damage_iter, mat)
                    K, f = assemble_mech_active_submesh_v10(
                        cache=mech_cache,
                        E_elem=E_eff_iter,
                        eps_iso_elem=eps_iso,
                        Ke_unit_flat=Ke_unit_flat,
                        fe_iso_unit=fe_iso_unit,
                        fe_body_unit=fe_body_unit,
                    )
                    u_sub = solve_free_dofs(K, f, mech_cache.free_dofs)

                    eq_trial = compute_microplane_equivalent_strain_active(
                        cache=mech_cache,
                        u_sub=u_sub,
                        ne_total=mesh.nelems,
                        eps_iso_elem=eps_iso,
                        B_center=B_center,
                        microplane_normals=_MICROPLANE_NORMALS,
                        microplane_weights=_MICROPLANE_WEIGHTS,
                        shear_c=mat.microplane_shear_c,
                    )
                    kappa_candidate = np.maximum(kappa_eq, eq_trial)
                    damage_candidate = np.maximum(damage, damage_from_kappa(kappa_candidate, mat))

                    delta_d = 0.0
                    if active.any():
                        delta_d = float(np.max(np.abs(damage_candidate[active] - damage_iter[active])))

                    damage_iter = damage_candidate
                    kappa_iter = kappa_candidate
                    if delta_d < mat.microplane_tol:
                        break

                damage[active] = damage_iter[active]
                kappa_eq[active] = kappa_iter[active]
                E_eff = effective_modulus_with_damage(E_base, damage, mat)

            u_full[:] = 0.0
            for li in range(mech_cache.n_active_nodes):
                g = mech_cache.active_nodes[li]
                u_full[3 * g:3 * g + 3] = u_sub[3 * li:3 * li + 3]

        if active.any():
            active_layers = int(mesh.elem_layer[active].max()) + 1
        else:
            active_layers = 0
        numerical_layer = active_layers // max(int(layers_per_output), 1)

        if numerical_layer > last_output and active_layers > 0:
            last_output = numerical_layer

            U = u_full.reshape((nn, 3))
            Umag = np.linalg.norm(U, axis=1)
            sigma, sigma_vm = compute_cell_stress_active(
                cache=mech_cache,
                u_sub=u_sub,
                ne_total=mesh.nelems,
                E_eff_elem=E_eff,
                eps_iso_elem=eps_iso,
                B_center=B_center,
                C_unit=C_unit,
            )

            vti_name = f"step_{step_out:04d}_layers{active_layers:03d}.vti"
            vti_path = os.path.join(out_dir, vti_name)

            write_vti(
                vti_path,
                nx=nx,
                ny=ny,
                nz=nz,
                dx=dx,
                dy=dy,
                dz=dz,
                origin=origin,
                point_data={"T": T, "U": U, "Umag": Umag},
                cell_data={
                    "alpha_h": alpha_h,
                    "active": active.astype(np.float32),
                    "solid": solid.astype(np.float32),
                    "active_layers": np.full(mesh.nelems, float(active_layers), dtype=np.float32),
                    "damage": damage.astype(np.float32),
                    "sigma_xx": sigma[:, 0].astype(np.float32),
                    "sigma_yy": sigma[:, 1].astype(np.float32),
                    "sigma_zz": sigma[:, 2].astype(np.float32),
                    "sigma_xy": sigma[:, 3].astype(np.float32),
                    "sigma_yz": sigma[:, 4].astype(np.float32),
                    "sigma_xz": sigma[:, 5].astype(np.float32),
                    "sigma_vm": sigma_vm.astype(np.float32),
                },
            )

            series_entries.append({"name": vti_name, "time": float(t)})
            write_vti_series(series_path, series_entries)

            print(f"[VTI] {vti_path} (t={t:.1f}s, active_layers={active_layers}, numerical_layer={numerical_layer})")
            step_out += 1

        if step % 10 == 0 or step == len(times) - 1:
            umax = float(np.max(np.abs(u_full))) if u_full.size else 0.0
            ah = float(alpha_h[active].mean()) if active.any() else 0.0
            dmax = float(damage[active].max()) if active.any() else 0.0
            rebuilt_msg = "rebuild" if cache_rebuilt else "reuse"
            print(
                f"t={t:8.1f}s | active_layers={active_layers:3d}/{nz} | max|u|={umax:.3e} m | "
                f"Tavg={T.mean():.2f} K | alpha_avg(active)={ah:.3f} | dmax(active)={dmax:.3f} | topo={rebuilt_msg}"
            )

        prev_active = active.copy()

    print("Done.")
    print(f"Open in ParaView: {series_path}")


# =========================================================
# CLI
# =========================================================
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
    ap.add_argument("--mech_model", type=str, default="linear_elastic",
                    choices=["linear_elastic", "microplane_damage"],
                    help="Mechanical constitutive model")
    args = ap.parse_args()

    run_sim(
        stl_path=args.stl,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        layers_per_output=args.layers_per_output,
        out_dir=args.out,
        auto_fit=args.auto_fit,
        pad_m=args.pad,
        stl_units=args.stl_units,
        mech_model=args.mech_model,
    )


if __name__ == "__main__":
    main()