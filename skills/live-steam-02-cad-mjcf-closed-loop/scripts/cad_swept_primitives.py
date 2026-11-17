#!/usr/bin/env python3
"""
Reusable build123d Craftsmanship Primitives & Exact STL Metal Mass/Inertia Tensor Integrator.
Prevents Mistakes 2, 12, 15, and 20.
"""
import math
import struct
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

RHO_COPPER = 8940.0
RHO_BRASS = 8500.0
RHO_STAINLESS = 7930.0
RHO_CAST_IRON = 7350.0


def make_swept_tube(waypoints: Sequence[Tuple[float, float, float]], r_tube: float, bend_r: float = 3.5):
    """Sweep a circular cross-section along a G1 filleted 3D polyline (zero Box elbows!)."""
    from build123d import Circle, Plane, Vector, Wire, sweep

    pts = [Vector(*p) for p in waypoints]
    wire = Wire.make_filleted_polyline(pts, radius=bend_r)
    p0 = wire @ 0.0
    t0 = wire % 0.0
    prof = Plane(origin=p0, z_dir=t0) * Circle(radius=r_tube)
    return sweep(prof, path=wire)


def make_hex_nut(center_xyz: Tuple[float, float, float], axis_xyz: Tuple[float, float, float], hex_r: float, length: float):
    """Create an oriented 6-sided union nut / gland boss."""
    from build123d import Plane, RegularPolygon, Vector, extrude

    ax = Vector(*axis_xyz).normalized()
    pl = Plane(origin=Vector(*center_xyz) - ax * (0.5 * length), z_dir=ax)
    return extrude(pl * RegularPolygon(radius=hex_r, side_count=6), amount=length)


def make_banjo_eye(
    center_xyz: Tuple[float, float, float],
    bolt_axis_xyz: Tuple[float, float, float],
    r_eye: float = 2.0,
    h_eye: float = 1.8,
    hex_r: float = 1.5,
):
    """Silver-brazed cylindrical Banjo eye + coaxial hollow hex bolt head (4-15 / 13-5 / 13-12)."""
    from build123d import Cylinder, Plane, Vector

    ax = Vector(*bolt_axis_xyz).normalized()
    pl = Plane(origin=Vector(*center_xyz), z_dir=ax)
    eye = pl * Cylinder(radius=r_eye, height=h_eye)
    bolt_head = make_hex_nut(
        tuple((Vector(*center_xyz) + ax * (0.5 * h_eye + 0.45)))[:3],  # type: ignore[arg-type]
        bolt_axis_xyz,
        hex_r=hex_r,
        length=1.1,
    )
    return eye + bolt_head


def make_stepped_pin_L(x: float, z: float, y_in: float, y_out: float, r_shank: float, r_head: float, r_nut: float):
    """Slotted stepped crank/valve-gear pin aligned along Y with inner nut & outer slotted cap."""
    from build123d import Box, Cylinder, Pos, Rot

    ym = 0.5 * (y_in + y_out)
    span = abs(y_out - y_in)
    shank = Pos(x, ym, z) * Rot(90, 0, 0) * Cylinder(radius=r_shank, height=span)
    nut = Pos(x, y_in + 0.35, z) * Rot(90, 0, 0) * Cylinder(radius=r_nut, height=0.70)
    head = (
        Pos(x, y_out - 0.40, z) * Rot(90, 0, 0) * Cylinder(radius=r_head, height=0.80)
        - Pos(x, y_out - 0.12, z) * Box(2.2 * r_head, 0.45, 0.38)
    )
    return shank + nut + head


def load_binary_stl_mm(stl_path: Path) -> np.ndarray:
    data = stl_path.read_bytes()
    n_tri = struct.unpack("<I", data[80:84])[0]
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec = np.frombuffer(data, dtype=dt, offset=84, count=n_tri)
    return rec["v"].astype(np.float64)


def compute_stl_mass_props(stl_path: Path, density_kg_m3: float, scale_to_m: float = 1e-3) -> Dict[str, object]:
    """Divergence-theorem exact polyhedral volume, COM, and MuJoCo fullinertia [Ixx,Iyy,Izz,Ixy,Ixz,Iyz]."""
    tris = load_binary_stl_mm(stl_path) * scale_to_m
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    det = np.einsum("ij,ij->i", v0, cross)
    vol = float(np.sum(det) / 6.0)
    com = np.sum((v0 + v1 + v2) * det[:, None], axis=0) / (24.0 * max(vol, 1e-18))

    u0, u1, u2 = v0 - com, v1 - com, v2 - com
    c_u = np.cross(u1 - u0, u2 - u0)
    d_u = np.einsum("ij,ij->i", u0, c_u)

    def _int_sq(a: np.ndarray) -> float:
        return float(np.sum(d_u * (a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2 + a[:, 0] * a[:, 1] + a[:, 1] * a[:, 2] + a[:, 2] * a[:, 0])) / 60.0)

    def _int_prod(a: np.ndarray, b: np.ndarray) -> float:
        s = (
            2.0 * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2])
            + a[:, 0] * (b[:, 1] + b[:, 2])
            + a[:, 1] * (b[:, 0] + b[:, 2])
            + a[:, 2] * (b[:, 0] + b[:, 1])
        )
        return float(np.sum(d_u * s) / 120.0)

    x = np.stack([u0[:, 0], u1[:, 0], u2[:, 0]], axis=1)
    y = np.stack([u0[:, 1], u1[:, 1], u2[:, 1]], axis=1)
    z = np.stack([u0[:, 2], u1[:, 2], u2[:, 2]], axis=1)
    ix2, iy2, iz2 = _int_sq(x), _int_sq(y), _int_sq(z)
    ixy, ixz, iyz = _int_prod(x, y), _int_prod(x, z), _int_prod(y, z)

    rho = float(density_kg_m3)
    mass = max(1e-7, vol * rho)
    Ixx = max(1e-12, rho * (iy2 + iz2))
    Iyy = max(1e-12, rho * (ix2 + iz2))
    Izz = max(1e-12, rho * (ix2 + iy2))
    return {
        "volume_m3": vol,
        "mass_kg": mass,
        "com_m": [float(com[0]), float(com[1]), float(com[2])],
        "fullinertia": [Ixx, Iyy, Izz, -rho * ixy, -rho * ixz, -rho * iyz],
    }
