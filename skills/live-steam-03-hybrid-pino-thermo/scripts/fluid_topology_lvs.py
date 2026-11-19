#!/usr/bin/env python3
"""
sim/fluid_topology_lvs.py
================================================================================
Full-Vehicle 3D Fluid-LVS (Layout-Versus-Schematic) Verification Engine
(Vehicle-Agnostic Reusable Operator for Precision Live-Steam Digital Twins)

借鉴芯片 EDA 版图与原理图一致性验证 (LVS: Open / Short / Necking DRC) 与航空液压阀块三维流道着色验证方法，
直接从编译后的 MuJoCo 多体装配模型 (MjModel / MjData) 提取全车所有真实固体零件 (逐件 STEP->STL 网格 + 结构几何体)，
在 GPU (CUDA) 或 CPU 上通过逐件布尔并集 (torch.maximum，严禁多网格拼接奇偶抵消!) 与 6-邻域面连通漫水填充，
对声明式全车流体网表 (fluid_topology.json) 执行四大物理拓扑验证：

  1. Zero-Open (端到端内腔连通性): 限制在管壁内侧套筒 (r_sleeve < r_od，严防外部空气绕流假阳性!) 内从上游端口向
     下游端口做 6-邻域面连通漫水，精确定位任何盲板、穿墙未打孔或错位断口的三维毫米坐标；
  2. Zero-Missing-Wall (管壁闭合度): 检验流道外周环形区 [r_id, r_od] 的真实金属/橡胶管壁包裹率，防止删管悬空假通过；
  3. Zero-Necking (最小水力截面直径): 沿连通流道逐轴向切片积分真实过流面积 A_open(s) 与等效水力直径 d_eq(s)，
     捕获车架板、横销或弯头塌陷导致的局部缩颈与流阻激增，并直接导出物理流导系数 G_edge ∈ [0, 1]；
  4. Zero-Short & Dynamic Valve Topology (多腔室防串腔隔离与可动阀态切换): 检验互斥流体腔室之间的零串腔泄漏，
     以及滑阀/排水栓/安全阀在不同关节位移 qpos 下的真实三维通断拓扑切换。
================================================================================
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

try:
    from sim.thermo_3d_gpu import voxelize_stl_cuda
except ImportError:
    # Standalone fallback when invoked from live-steam-twin plugin directory
    import hashlib

    _LOCAL_VOXEL_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}

    @torch.no_grad()
    def voxelize_stl_cuda(
        tris_mm: np.ndarray,
        x_range_mm: Tuple[float, float],
        y_range_mm: Tuple[float, float],
        z_range_mm: Tuple[float, float],
        shape: Tuple[int, int, int],
        device: torch.device,
        ray_batch: int = 4096,
    ) -> torch.Tensor:
        mesh_sig = hashlib.blake2b(np.ascontiguousarray(tris_mm, dtype=np.float32).view(np.uint8), digest_size=12).hexdigest()
        cache_key = (mesh_sig, x_range_mm, y_range_mm, z_range_mm, shape, str(device))
        cached = _LOCAL_VOXEL_CACHE.get(cache_key)
        if cached is not None:
            return cached.clone()
        nx, ny, nz = shape
        xs = torch.linspace(x_range_mm[0], x_range_mm[1], nx, device=device, dtype=torch.float32) + 1.37e-4
        ys = torch.linspace(y_range_mm[0], y_range_mm[1], ny, device=device, dtype=torch.float32) - 1.91e-4
        zs = torch.linspace(z_range_mm[0], z_range_mm[1], nz, device=device, dtype=torch.float32)
        t_gpu = torch.as_tensor(tris_mm, device=device, dtype=torch.float32)
        if t_gpu.numel() == 0:
            return torch.zeros(shape, device=device, dtype=torch.float32)
        v0, v1, v2 = t_gpu[:, 0, :], t_gpu[:, 1, :], t_gpu[:, 2, :]
        e1x, e1y = v1[:, 0] - v0[:, 0], v1[:, 1] - v0[:, 1]
        e2x, e2y = v2[:, 0] - v0[:, 0], v2[:, 1] - v0[:, 1]
        det = e1x * e2y - e1y * e2x
        valid = det.abs() > 1e-9
        v0, v1, v2 = v0[valid], v1[valid], v2[valid]
        e1x, e1y, e2x, e2y, inv_det = e1x[valid], e1y[valid], e2x[valid], e2y[valid], 1.0 / det[valid]
        tx_min = torch.minimum(torch.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
        tx_max = torch.maximum(torch.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
        ty_min = torch.minimum(torch.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
        ty_max = torch.maximum(torch.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        rx_all, ry_all = gx.reshape(-1), gy.reshape(-1)
        counts_all = torch.zeros((rx_all.numel(), nz), device=device, dtype=torch.int32)
        for r0 in range(0, rx_all.numel(), ray_batch):
            r1 = min(rx_all.numel(), r0 + ray_batch)
            rx, ry = rx_all[r0:r1], ry_all[r0:r1]
            sub = (tx_max >= rx.min()) & (tx_min <= rx.max()) & (ty_max >= ry.min()) & (ty_min <= ry.max())
            if not torch.any(sub):
                continue
            sv0, sv1, sv2 = v0[sub], v1[sub], v2[sub]
            se1x, se1y, se2x, se2y, sinv = e1x[sub], e1y[sub], e2x[sub], e2y[sub], inv_det[sub]
            dx = rx[:, None] - sv0[None, :, 0]
            dy = ry[:, None] - sv0[None, :, 1]
            u = (dx * se2y[None, :] - dy * se2x[None, :]) * sinv[None, :]
            v = (se1x[None, :] * dy - se1y[None, :] * dx) * sinv[None, :]
            hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
            if not torch.any(hit):
                continue
            z_hit = (1.0 - u - v) * sv0[None, :, 2] + u * sv1[None, :, 2] + v * sv2[None, :, 2]
            hr, ht = torch.nonzero(hit, as_tuple=True)
            counts_all[r0:r1].index_add_(0, hr, (z_hit[hr, ht][:, None] > zs[None, :]).to(torch.int32))
        out_vox = (counts_all & 1).to(torch.float32).reshape(nx, ny, nz)
        _LOCAL_VOXEL_CACHE[cache_key] = out_vox
        return out_vox.clone()


@dataclass
class EdgeLVSResult:
    edge_id: str
    subnet: str
    part_no: str
    title: str
    from_node: str
    to_node: str
    connected: bool
    necking_ok: bool
    wall_ok: bool
    passed: bool
    conductance_ratio: float
    spec_id_mm: float
    min_allowed_dia_mm: float
    min_measured_dia_mm: float
    median_measured_dia_mm: float
    wall_coverage_ratio: float
    min_wall_coverage: float
    total_length_mm: float
    max_s_reached_mm: float
    blocked_at_loco_mm: Optional[List[float]] = None
    pinch_at_loco_mm: Optional[List[float]] = None
    blocking_geoms: List[str] = field(default_factory=list)


@dataclass
class IsolationLVSResult:
    pair_id: str
    title: str
    compartment_a: str
    compartment_b: str
    isolated: bool
    short_voxels: int
    flooded_voxels_a: int
    leak_at_loco_mm: Optional[List[float]] = None


@dataclass
class DynamicValveLVSResult:
    test_id: str
    title: str
    passed: bool
    expected_connected: bool
    actual_connected: bool
    conductance_ratio: float
    qpos_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class FluidLVSReport:
    xml_path: str
    topology_path: str
    device: str
    elapsed_ms: float
    total_edges: int
    open_edges: int
    neck_free_edges: int
    wall_valid_edges: int
    passed_edges: int
    total_isolation_pairs: int
    passed_isolation_pairs: int
    total_dynamic_tests: int
    passed_dynamic_tests: int
    all_passed: bool
    edge_conductances: Dict[str, float]
    edges: List[EdgeLVSResult]
    isolation_pairs: List[IsolationLVSResult]
    dynamic_valve_tests: List[DynamicValveLVSResult]


@torch.no_grad()
def flood_fill_6face_3d(passable: torch.Tensor, seed: torch.Tensor, max_steps: int = 800) -> torch.Tensor:
    """
    严格 6-邻域共面有限体积漫水填充 (禁止 26-邻域对角顶点零面积穿透薄壁!)。
    passable, seed: (Nx, Ny, Nz) bool or float32 tensor.
    """
    pass_b = passable > 0.5
    reach = (seed > 0.5) & pass_b
    if not torch.any(reach):
        return reach.to(torch.float32)

    for step in range(max_steps):
        nxt = reach.clone()
        nxt[1:, :, :] |= reach[:-1, :, :]
        nxt[:-1, :, :] |= reach[1:, :, :]
        nxt[:, 1:, :] |= reach[:, :-1, :]
        nxt[:, :-1, :] |= reach[:, 1:, :]
        nxt[:, :, 1:] |= reach[:, :, :-1]
        nxt[:, :, :-1] |= reach[:, :, 1:]
        nxt &= pass_b
        if step % 12 == 11 and torch.equal(nxt, reach):
            break
        reach = nxt
    return reach.to(torch.float32)


def _apply_qpos_overrides(model: mujoco.MjModel, data: mujoco.MjData, overrides: Optional[Dict[str, float]]) -> None:
    mujoco.mj_resetData(model, data)
    if overrides:
        for jname, val in overrides.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"MuJoCo joint not found: {jname}")
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = float(val)
    mujoco.mj_forward(model, data)


def _body_point_to_world_mm(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pt_mm: Sequence[float]) -> np.ndarray:
    p_local_m = np.asarray(pt_mm, dtype=np.float64) * 1e-3
    if body_name == "world":
        return p_local_m * 1000.0
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"MuJoCo body not found: {body_name}")
    rot = data.xmat[bid].reshape(3, 3)
    return (data.xpos[bid] + rot @ p_local_m) * 1000.0


def _world_mm_to_loco_mm(model: mujoco.MjModel, data: mujoco.MjData, pt_world_mm: Sequence[float]) -> List[float]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "loco_frame")
    if bid < 0:
        return [round(float(v), 2) for v in pt_world_mm]
    rot = data.xmat[bid].reshape(3, 3)
    p_loco = rot.T @ (np.asarray(pt_world_mm, dtype=np.float64) * 1e-3 - data.xpos[bid]) * 1000.0
    return [round(float(v), 2) for v in p_loco]


def _fillet_polyline_3d(pts_mm: np.ndarray, bend_r_mm: float, arc_subdiv: int = 6) -> np.ndarray:
    """将折线各拐角按真实圆弧弯管半径 bend_r_mm 离散为 C^1 光滑圆弧，消除 90° 尖角套筒裁切与 Voronoi 楔形面积失真。"""
    if bend_r_mm <= 0.05 or len(pts_mm) < 3:
        return pts_mm
    out: List[np.ndarray] = [pts_mm[0].astype(np.float64)]
    for i in range(1, len(pts_mm) - 1):
        p_prev = pts_mm[i - 1].astype(np.float64)
        p_curr = pts_mm[i].astype(np.float64)
        p_next = pts_mm[i + 1].astype(np.float64)
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 < 1e-5 or l2 < 1e-5:
            continue
        e1 = v1 / l1
        e2 = v2 / l2
        cos_alpha = float(np.clip(np.dot(e1, e2), -1.0, 1.0))
        alpha = math.acos(cos_alpha)  # interior angle in (0, pi]
        theta = math.pi - alpha       # deflection angle
        if theta < 0.06 or alpha < 0.06:
            out.append(p_curr)
            continue
        tan_half_t = math.tan(0.5 * theta)
        dt = min(bend_r_mm * tan_half_t, 0.44 * l1, 0.44 * l2)
        r_eff = dt / max(1e-8, tan_half_t)
        a_pt = p_curr + dt * e1
        b_pt = p_curr + dt * e2
        bis = e1 + e2
        bis_n = float(np.linalg.norm(bis))
        if bis_n < 1e-6:
            out.append(p_curr)
            continue
        bis /= bis_n
        center = p_curr + (r_eff / max(1e-6, math.sin(0.5 * alpha))) * bis
        va = a_pt - center
        vb = b_pt - center
        for k in range(arc_subdiv + 1):
            tk = k / float(arc_subdiv)
            vm = (1.0 - tk) * va + tk * vb
            vn = float(np.linalg.norm(vm))
            out.append(center + (r_eff / max(1e-8, vn)) * vm)
    out.append(pts_mm[-1].astype(np.float64))
    return np.asarray(out, dtype=np.float32)


def _resolve_edge_polyline_world_mm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    edge_spec: Dict[str, Any],
) -> np.ndarray:
    """将局部系或跨刚体混合系 (如煤水车-牵引杆软管-机车) 折线统一变换到当前 qpos 下的世界系毫米坐标并做圆弧导角。"""
    bend_r = float(edge_spec.get("bend_r_mm", 2.2))
    if "mixed_pts" in edge_spec:
        raw_w = [
            _body_point_to_world_mm(model, data, item["body"], item["pt_mm"])
            for item in edge_spec["mixed_pts"]
        ]
        if "catenary_sag_mm" in edge_spec and len(raw_w) >= 2:
            sag = float(edge_spec["catenary_sag_mm"])
            p_start, p_end = raw_w[0], raw_w[-1]
            p_mid = 0.5 * (p_start + p_end)
            p_mid[2] = min(float(p_start[2]), float(p_end[2])) - sag
            pts_w = np.asarray([p_start, p_mid, p_end], dtype=np.float32)
            return _fillet_polyline_3d(pts_w, max(bend_r, 6.0), arc_subdiv=8)
        pts_w = np.asarray(raw_w, dtype=np.float32)
        return _fillet_polyline_3d(pts_w, bend_r)

    frame_map = {
        "loco": "loco_frame",
        "tender": "tender_body",
        "drawbar": "drawbar_8_8",
        "world": "world",
    }
    raw_frame = edge_spec.get("frame", "loco")
    body_name = frame_map.get(raw_frame, raw_frame)
    pts_w = np.asarray(
        [_body_point_to_world_mm(model, data, body_name, p) for p in edge_spec["pts_mm"]],
        dtype=np.float32,
    )
    return _fillet_polyline_3d(pts_w, bend_r)


@torch.no_grad()
def build_roi_solid_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x_range_mm: Tuple[float, float],
    y_range_mm: Tuple[float, float],
    z_range_mm: Tuple[float, float],
    shape: Tuple[int, int, int],
    device: torch.device,
    exclude_geoms: Optional[Sequence[str]] = None,
    only_geoms: Optional[Sequence[str]] = None,
    return_per_geom: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    在世界系 ROI 包围盒内，逐件体素化 MuJoCo 所有实体几何 (mjGEOM_MESH + 可见金属 Box/Cylinder/Sphere/Capsule)
    并以 torch.maximum 严格求并集，彻底免疫多网格奇偶射线相消伪影!
    """
    excl = set(exclude_geoms or ())
    only = set(only_geoms) if only_geoms is not None else None
    solid = torch.zeros(shape, device=device, dtype=torch.float32)
    per_geom: Dict[str, torch.Tensor] = {}

    nx, ny, nz = shape
    xs = torch.linspace(x_range_mm[0], x_range_mm[1], nx, device=device, dtype=torch.float32)
    ys = torch.linspace(y_range_mm[0], y_range_mm[1], ny, device=device, dtype=torch.float32)
    zs = torch.linspace(z_range_mm[0], z_range_mm[1], nz, device=device, dtype=torch.float32)
    lazy_grid: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    roi_min = np.array([x_range_mm[0], y_range_mm[0], z_range_mm[0]], dtype=np.float64)
    roi_max = np.array([x_range_mm[1], y_range_mm[1], z_range_mm[1]], dtype=np.float64)

    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if bname == "world":
            continue
        if int(model.geom_group[gid]) == 3 or float(model.geom_rgba[gid, 3]) <= 0.05:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        if gname in excl:
            continue
        if only is not None and gname not in only:
            continue

        gtype = int(model.geom_type[gid])
        gpos_mm = data.geom_xpos[gid] * 1000.0
        gmat = data.geom_xmat[gid].reshape(3, 3)

        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[gid])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            verts_mj = model.mesh_vert[va : va + vn]
            verts_w_mm = gpos_mm + (verts_mj @ gmat.T) * 1000.0
            vmin = verts_w_mm.min(axis=0)
            vmax = verts_w_mm.max(axis=0)
            if np.any(vmax < roi_min - 0.6) or np.any(vmin > roi_max + 0.6):
                continue

            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            faces = model.mesh_face[fa : fa + fn]
            tris_w_mm = verts_w_mm[faces].astype(np.float32, copy=False)
            vox = voxelize_stl_cuda(tris_w_mm, x_range_mm, y_range_mm, z_range_mm, shape, device)
            if torch.any(vox > 0.5):
                solid = torch.maximum(solid, vox)
                if return_per_geom:
                    per_geom[gname] = vox
        elif gtype in (
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
        ):
            sz_mm = model.geom_size[gid] * 1000.0
            bound_r = float(np.linalg.norm(sz_mm[:3]) + sz_mm[0] + 1.0)
            if np.any(gpos_mm + bound_r < roi_min) or np.any(gpos_mm - bound_r > roi_max):
                continue
            if lazy_grid is None:
                lazy_grid = torch.meshgrid(xs, ys, zs, indexing="ij")
            gx, gy, gz = lazy_grid
            dx = gx - float(gpos_mm[0])
            dy = gy - float(gpos_mm[1])
            dz = gz - float(gpos_mm[2])
            r_t = torch.as_tensor(gmat, device=device, dtype=torch.float32)
            lx = dx * r_t[0, 0] + dy * r_t[1, 0] + dz * r_t[2, 0]
            ly = dx * r_t[0, 1] + dy * r_t[1, 1] + dz * r_t[2, 1]
            lz = dx * r_t[0, 2] + dy * r_t[1, 2] + dz * r_t[2, 2]

            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                pm = ((lx.abs() <= float(sz_mm[0])) & (ly.abs() <= float(sz_mm[1])) & (lz.abs() <= float(sz_mm[2]))).to(torch.float32)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                pm = ((lx * lx + ly * ly <= float(sz_mm[0] ** 2)) & (lz.abs() <= float(sz_mm[1]))).to(torch.float32)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                zc = lz.clamp(-float(sz_mm[1]), float(sz_mm[1]))
                pm = (lx * lx + ly * ly + (lz - zc) ** 2 <= float(sz_mm[0] ** 2)).to(torch.float32)
            else:
                pm = (lx * lx + ly * ly + lz * lz <= float(sz_mm[0] ** 2)).to(torch.float32)

            if torch.any(pm > 0.5):
                solid = torch.maximum(solid, pm)
                if return_per_geom:
                    per_geom[gname] = pm

    return solid, per_geom


@torch.no_grad()
def _compute_polyline_fields_gpu(
    pts_w_mm: np.ndarray,
    x_range_mm: Tuple[float, float],
    y_range_mm: Tuple[float, float],
    z_range_mm: Tuple[float, float],
    shape: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, np.ndarray]:
    """计算每个体素到三维折线的垂直径向距离 dist_perp (mm) 与沿程弧长坐标 s_coord (mm)。"""
    nx, ny, nz = shape
    xs = torch.linspace(x_range_mm[0], x_range_mm[1], nx, device=device, dtype=torch.float32)
    ys = torch.linspace(y_range_mm[0], y_range_mm[1], ny, device=device, dtype=torch.float32)
    zs = torch.linspace(z_range_mm[0], z_range_mm[1], nz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")

    dist_sq_min = torch.full(shape, 1e12, device=device, dtype=torch.float32)
    s_best = torch.zeros(shape, device=device, dtype=torch.float32)

    cum_s = [0.0]
    for i in range(len(pts_w_mm) - 1):
        seg = pts_w_mm[i + 1] - pts_w_mm[i]
        seg_len = float(np.linalg.norm(seg))
        cum_s.append(cum_s[-1] + seg_len)
        if seg_len < 1e-6:
            continue
        ax, ay, az = float(pts_w_mm[i, 0]), float(pts_w_mm[i, 1]), float(pts_w_mm[i, 2])
        ux, uy, uz = float(seg[0]) / seg_len, float(seg[1]) / seg_len, float(seg[2]) / seg_len
        proj = ((gx - ax) * ux + (gy - ay) * uy + (gz - az) * uz).clamp(0.0, seg_len)
        cx = ax + proj * ux
        cy = ay + proj * uy
        cz = az + proj * uz
        d2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
        better = d2 < dist_sq_min
        dist_sq_min = torch.where(better, d2, dist_sq_min)
        s_best = torch.where(better, float(cum_s[i]) + proj, s_best)

    return torch.sqrt(dist_sq_min.clamp_min(0.0)), s_best, cum_s[-1], np.asarray(cum_s, dtype=np.float64)


def _interp_polyline_mm(pts_w_mm: np.ndarray, cum_s: np.ndarray, s_target: float) -> np.ndarray:
    s_clamped = float(np.clip(s_target, 0.0, cum_s[-1]))
    idx = int(np.searchsorted(cum_s, s_clamped, side="right")) - 1
    idx = max(0, min(len(pts_w_mm) - 2, idx))
    ds = max(1e-8, float(cum_s[idx + 1] - cum_s[idx]))
    t = (s_clamped - float(cum_s[idx])) / ds
    return (1.0 - t) * pts_w_mm[idx] + t * pts_w_mm[idx + 1]


@torch.no_grad()
def evaluate_pipe_edge_lvs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    edge_spec: Dict[str, Any],
    device: torch.device,
    default_exclude_geoms: Sequence[str] = (),
) -> EdgeLVSResult:
    qpos_ov = edge_spec.get("open_qpos")
    _apply_qpos_overrides(model, data, qpos_ov)

    pts_w_mm = _resolve_edge_polyline_world_mm(model, data, edge_spec)
    spec_id_mm = float(edge_spec["spec_id_mm"])
    spec_od_mm = float(edge_spec["spec_od_mm"])
    r_id = 0.5 * spec_id_mm
    r_od = 0.5 * spec_od_mm
    # 关键防呆不变量 (Mistake 49): 搜索套筒半径严格小于外管壁半径 (或圆弧弯头允许的最大偏置)，杜绝管外空气绕流假连通!
    r_sleeve = float(edge_spec.get("sleeve_radius_mm", min(r_od - 0.18, r_id + 0.55)))
    r_wall_outer = float(edge_spec.get("wall_outer_radius_mm", r_od + 0.85))
    pitch_mm = float(edge_spec.get("voxel_pitch_mm", max(0.16, min(0.32, 0.22 * spec_id_mm))))

    pad = r_wall_outer + 2.0 * pitch_mm + 0.6
    pmin = pts_w_mm.min(axis=0) - pad
    pmax = pts_w_mm.max(axis=0) + pad
    x_rng = (float(pmin[0]), float(pmax[0]))
    y_rng = (float(pmin[1]), float(pmax[1]))
    z_rng = (float(pmin[2]), float(pmax[2]))

    nx = max(12, min(420, int(math.ceil((x_rng[1] - x_rng[0]) / pitch_mm)) + 1))
    ny = max(12, min(320, int(math.ceil((y_rng[1] - y_rng[0]) / pitch_mm)) + 1))
    nz = max(12, min(320, int(math.ceil((z_rng[1] - z_rng[0]) / pitch_mm)) + 1))
    shape = (nx, ny, nz)

    dx = (x_rng[1] - x_rng[0]) / max(1, nx - 1)
    dy = (y_rng[1] - y_rng[0]) / max(1, ny - 1)
    dz = (z_rng[1] - z_rng[0]) / max(1, nz - 1)
    voxel_vol_mm3 = dx * dy * dz

    excl = list(default_exclude_geoms) + list(edge_spec.get("exclude_geoms", []))
    solid, _ = build_roi_solid_mask(
        model, data, x_rng, y_rng, z_rng, shape, device, exclude_geoms=excl, return_per_geom=False
    )
    dist_perp, s_coord, total_len_mm, cum_s = _compute_polyline_fields_gpu(
        pts_w_mm, x_rng, y_rng, z_rng, shape, device
    )

    sleeve_mask = dist_perp <= r_sleeve
    fluid_mask = sleeve_mask & (solid < 0.5)

    seed_len = float(edge_spec.get("seed_window_mm", max(1.2, 2.8 * pitch_mm)))
    seed_up = fluid_mask & (s_coord <= seed_len)
    target_dn = fluid_mask & (s_coord >= max(0.0, total_len_mm - seed_len))

    max_flood_steps = int(math.ceil(total_len_mm / max(0.08, min(dx, dy, dz))) * 1.65) + 48
    reach = flood_fill_6face_3d(fluid_mask, seed_up, max_steps=max_flood_steps)
    connected = bool(torch.any((reach > 0.5) & target_dn).item())

    reached_s = torch.where(reach > 0.5, s_coord, torch.tensor(-1.0, device=device))
    max_s_reached = float(reached_s.max().item()) if torch.any(reach > 0.5) else 0.0

    blocked_loco_mm: Optional[List[float]] = None
    blocking_geoms: List[str] = []
    if not connected:
        probe_s = min(total_len_mm, max(0.0, max_s_reached + 0.6 * pitch_mm))
        pt_blk_w = _interp_polyline_mm(pts_w_mm, cum_s, probe_s)
        blocked_loco_mm = _world_mm_to_loco_mm(model, data, pt_blk_w)
        # Re-query per-geom masks only when a blockage is detected to name exact offending CAD parts!
        _, per_geom = build_roi_solid_mask(
            model, data, x_rng, y_rng, z_rng, shape, device, exclude_geoms=excl, return_per_geom=True
        )
        blk_zone = sleeve_mask & (s_coord >= max(0.0, max_s_reached - 1.2)) & (s_coord <= min(total_len_mm, max_s_reached + 2.5))
        for gname, gvox in per_geom.items():
            if bool(torch.any((gvox > 0.5) & blk_zone).item()):
                blocking_geoms.append(gname)

    # 2. 沿程逐切片积分真实过流截面积 A_open(s) 与等效水力直径 d_eq(s)
    neck_trim_start = float(edge_spec.get("neck_trim_start_mm", seed_len))
    neck_trim_end = float(edge_spec.get("neck_trim_end_mm", seed_len))
    s0_eval = min(0.35 * total_len_mm, neck_trim_start)
    s1_eval = max(0.65 * total_len_mm, total_len_mm - neck_trim_end)
    n_bins = max(6, min(64, int(math.ceil((s1_eval - s0_eval) / max(1.5, 3.5 * pitch_mm)))))
    bin_edges = np.linspace(s0_eval, s1_eval, n_bins + 1)

    active_lumen = (reach > 0.5) if connected else fluid_mask
    station_dias: List[float] = []
    station_s_mids: List[float] = []
    station_areas: List[float] = []
    for bi in range(n_bins):
        bs0, bs1 = float(bin_edges[bi]), float(bin_edges[bi + 1])
        ds_bin = max(1e-6, bs1 - bs0)
        in_bin = active_lumen & (s_coord >= bs0) & (s_coord < bs1)
        vol_bin = float(in_bin.sum().item()) * voxel_vol_mm3
        area_mm2 = vol_bin / ds_bin
        dia_eq = 2.0 * math.sqrt(max(0.0, area_mm2) / math.pi)
        station_areas.append(area_mm2)
        station_dias.append(dia_eq)
        station_s_mids.append(0.5 * (bs0 + bs1))

    min_measured_dia = float(min(station_dias)) if station_dias else 0.0
    median_measured_dia = float(np.median(station_dias)) if station_dias else 0.0
    min_allowed_dia = float(edge_spec.get("min_hydraulic_dia_mm", 0.65 * spec_id_mm))
    necking_ok = connected and (min_measured_dia >= min_allowed_dia)

    pinch_loco_mm: Optional[List[float]] = None
    if station_dias:
        idx_min = int(np.argmin(station_dias))
        pt_pinch_w = _interp_polyline_mm(pts_w_mm, cum_s, station_s_mids[idx_min])
        pinch_loco_mm = _world_mm_to_loco_mm(model, data, pt_pinch_w)

    # 3. 管壁包裹度核验 (Zero-Missing-Wall)
    annulus_mask = (dist_perp >= max(0.25, r_id - 0.10)) & (dist_perp <= r_wall_outer) & (solid > 0.5)
    wall_hits = 0
    for bi in range(n_bins):
        bs0, bs1 = float(bin_edges[bi]), float(bin_edges[bi + 1])
        if bool(torch.any(annulus_mask & (s_coord >= bs0) & (s_coord < bs1)).item()):
            wall_hits += 1
    wall_cov = float(wall_hits) / float(max(1, n_bins))
    min_wall_cov = float(edge_spec.get("min_wall_coverage", 0.75))
    wall_ok = wall_cov >= min_wall_cov

    # 4. 物理流导比 G_edge = R_nominal / ∫ (ds / A_open(s)^2)
    nom_area = math.pi * (r_id ** 2)
    if not connected or any(a <= 1e-6 for a in station_areas):
        cond_ratio = 0.0
    else:
        r_nom = (s1_eval - s0_eval) / max(1e-9, nom_area ** 2)
        ds_step = (s1_eval - s0_eval) / float(n_bins)
        r_3d = sum(ds_step / max(1e-9, a ** 2) for a in station_areas)
        cond_ratio = float(np.clip(r_nom / max(1e-12, r_3d), 0.0, 1.25))

    passed = bool(connected and necking_ok and wall_ok)
    return EdgeLVSResult(
        edge_id=str(edge_spec["id"]),
        subnet=str(edge_spec["subnet"]),
        part_no=str(edge_spec["part_no"]),
        title=str(edge_spec["title"]),
        from_node=str(edge_spec["from"]),
        to_node=str(edge_spec["to"]),
        connected=connected,
        necking_ok=necking_ok,
        wall_ok=wall_ok,
        passed=passed,
        conductance_ratio=round(cond_ratio, 4),
        spec_id_mm=round(spec_id_mm, 3),
        min_allowed_dia_mm=round(min_allowed_dia, 3),
        min_measured_dia_mm=round(min_measured_dia, 3),
        median_measured_dia_mm=round(median_measured_dia, 3),
        wall_coverage_ratio=round(wall_cov, 3),
        min_wall_coverage=round(min_wall_cov, 3),
        total_length_mm=round(total_len_mm, 2),
        max_s_reached_mm=round(max_s_reached, 2),
        blocked_at_loco_mm=blocked_loco_mm,
        pinch_at_loco_mm=pinch_loco_mm if not necking_ok else None,
        blocking_geoms=blocking_geoms,
    )


def _boxes_mask_in_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    boxes_spec: Sequence[Dict[str, Any]],
    gx: torch.Tensor,
    gy: torch.Tensor,
    gz: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    frame_map = {"loco": "loco_frame", "tender": "tender_body", "drawbar": "drawbar_8_8", "world": "world"}
    mask = torch.zeros_like(gx, dtype=torch.bool)
    for b in boxes_spec:
        bname = frame_map.get(b.get("frame", "loco"), b.get("frame", "loco"))
        c_w = _body_point_to_world_mm(model, data, bname, b["center_mm"])
        hs = [0.5 * float(v) for v in b["size_mm"]]
        if bname == "world":
            rot = np.eye(3, dtype=np.float64)
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            rot = data.xmat[bid].reshape(3, 3)
        r_t = torch.as_tensor(rot, device=device, dtype=torch.float32)
        dx, dy, dz = gx - float(c_w[0]), gy - float(c_w[1]), gz - float(c_w[2])
        lx = dx * r_t[0, 0] + dy * r_t[1, 0] + dz * r_t[2, 0]
        ly = dx * r_t[0, 1] + dy * r_t[1, 1] + dz * r_t[2, 1]
        lz = dx * r_t[0, 2] + dy * r_t[1, 2] + dz * r_t[2, 2]
        mask |= (lx.abs() <= hs[0]) & (ly.abs() <= hs[1]) & (lz.abs() <= hs[2])
    return mask


@torch.no_grad()
def evaluate_isolation_pair_lvs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    iso_spec: Dict[str, Any],
    device: torch.device,
    default_exclude_geoms: Sequence[str] = (),
) -> IsolationLVSResult:
    """
    多腔室防串腔隔离核验 (Zero-Short):
    对装配体施加可选端口闷盖 (port_caps，如同真实水压/气密试验堵头)，从腔室 A 种子区做 6-邻域漫水，
    检验是否与腔室 B 目标区 (或边界大气面) 发生任何单一体素串通!
    """
    _apply_qpos_overrides(model, data, iso_spec.get("qpos_overrides"))
    frame_map = {"loco": "loco_frame", "tender": "tender_body", "world": "world"}
    bname = frame_map.get(iso_spec.get("frame", "loco"), iso_spec.get("frame", "loco"))

    roi_local = iso_spec["roi_mm"]  # [[x0,x1], [y0,y1], [z0,z1]] in frame
    corners_l = [
        [rx, ry, rz]
        for rx in roi_local[0]
        for ry in roi_local[1]
        for rz in roi_local[2]
    ]
    corners_w = np.asarray([_body_point_to_world_mm(model, data, bname, c) for c in corners_l], dtype=np.float64)
    x_rng = (float(corners_w[:, 0].min()), float(corners_w[:, 0].max()))
    y_rng = (float(corners_w[:, 1].min()), float(corners_w[:, 1].max()))
    z_rng = (float(corners_w[:, 2].min()), float(corners_w[:, 2].max()))

    pitch_mm = float(iso_spec.get("voxel_pitch_mm", 0.35))
    nx = max(12, min(360, int(math.ceil((x_rng[1] - x_rng[0]) / pitch_mm)) + 1))
    ny = max(12, min(280, int(math.ceil((y_rng[1] - y_rng[0]) / pitch_mm)) + 1))
    nz = max(12, min(280, int(math.ceil((z_rng[1] - z_rng[0]) / pitch_mm)) + 1))
    shape = (nx, ny, nz)

    excl = list(default_exclude_geoms) + list(iso_spec.get("exclude_geoms", []))
    only = iso_spec.get("only_geoms")
    solid, _ = build_roi_solid_mask(
        model, data, x_rng, y_rng, z_rng, shape, device, exclude_geoms=excl, only_geoms=only, return_per_geom=False
    )

    xs = torch.linspace(x_rng[0], x_rng[1], nx, device=device, dtype=torch.float32)
    ys = torch.linspace(y_rng[0], y_rng[1], ny, device=device, dtype=torch.float32)
    zs = torch.linspace(z_rng[0], z_rng[1], nz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")

    if iso_spec.get("port_caps"):
        cap_mask = _boxes_mask_in_frame(model, data, iso_spec["port_caps"], gx, gy, gz, device)
        solid = torch.maximum(solid, cap_mask.to(torch.float32))

    passable = solid < 0.5
    seed_a = _boxes_mask_in_frame(model, data, iso_spec["seed_a_boxes"], gx, gy, gz, device) & passable
    if iso_spec.get("target_b_is_roi_boundary", False):
        target_b = torch.zeros(shape, device=device, dtype=torch.bool)
        target_b[0, :, :] = True
        target_b[-1, :, :] = True
        target_b[:, 0, :] = True
        target_b[:, -1, :] = True
        target_b[:, :, 0] = True
        target_b[:, :, -1] = True
        target_b &= passable
    else:
        target_b = _boxes_mask_in_frame(model, data, iso_spec["target_b_boxes"], gx, gy, gz, device) & passable

    max_steps = int(iso_spec.get("max_flood_steps", nx + ny + nz + 180))
    reach_a = flood_fill_6face_3d(passable, seed_a, max_steps=max_steps)
    flooded_a = int((reach_a > 0.5).sum().item())
    overlap = (reach_a > 0.5) & target_b
    short_voxels = int(overlap.sum().item())
    isolated = (flooded_a > 0) and (short_voxels == 0)

    leak_loco_mm: Optional[List[float]] = None
    if short_voxels > 0:
        idx = torch.nonzero(overlap, as_tuple=False)[0]
        lw = [float(xs[idx[0]].item()), float(ys[idx[1]].item()), float(zs[idx[2]].item())]
        leak_loco_mm = _world_mm_to_loco_mm(model, data, lw)

    return IsolationLVSResult(
        pair_id=str(iso_spec["id"]),
        title=str(iso_spec["title"]),
        compartment_a=str(iso_spec["compartment_a"]),
        compartment_b=str(iso_spec["compartment_b"]),
        isolated=isolated,
        short_voxels=short_voxels,
        flooded_voxels_a=flooded_a,
        leak_at_loco_mm=leak_loco_mm,
    )


@torch.no_grad()
def evaluate_dynamic_valve_lvs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dyn_spec: Dict[str, Any],
    device: torch.device,
    default_exclude_geoms: Sequence[str] = (),
) -> DynamicValveLVSResult:
    edge_like = dict(dyn_spec["probe_edge"])
    edge_like["open_qpos"] = dyn_spec.get("qpos_overrides", {})
    edge_like.setdefault("id", dyn_spec["id"])
    edge_like.setdefault("subnet", "dynamic_valve")
    edge_like.setdefault("part_no", dyn_spec.get("part_no", dyn_spec["id"]))
    edge_like.setdefault("title", dyn_spec["title"])
    edge_like.setdefault("from", dyn_spec.get("from", "upstream"))
    edge_like.setdefault("to", dyn_spec.get("to", "downstream"))

    res = evaluate_pipe_edge_lvs(model, data, edge_like, device, default_exclude_geoms=default_exclude_geoms)
    expected = bool(dyn_spec["expected_connected"])
    actual = bool(res.connected)
    passed = (actual == expected) and (not expected or res.conductance_ratio > 0.02)
    return DynamicValveLVSResult(
        test_id=str(dyn_spec["id"]),
        title=str(dyn_spec["title"]),
        passed=passed,
        expected_connected=expected,
        actual_connected=actual,
        conductance_ratio=res.conductance_ratio,
        qpos_overrides=dict(dyn_spec.get("qpos_overrides", {})),
    )


def run_full_vehicle_fluid_lvs(
    xml_path: Path | str,
    topology_path: Path | str,
    device: Optional[torch.device] = None,
    out_report_path: Optional[Path | str] = None,
) -> FluidLVSReport:
    t0 = time.perf_counter()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xml_p = Path(xml_path)
    topo_p = Path(topology_path)
    topo = json.loads(topo_p.read_text(encoding="utf-8"))

    model = mujoco.MjModel.from_xml_path(str(xml_p))
    data = mujoco.MjData(model)
    default_excl = topo.get("global_permeable_or_non_wall_geoms", [])

    edge_results: List[EdgeLVSResult] = []
    edge_conds: Dict[str, float] = {}
    for edge_spec in topo.get("edges", []):
        er = evaluate_pipe_edge_lvs(model, data, edge_spec, dev, default_exclude_geoms=default_excl)
        edge_results.append(er)
        edge_conds[er.edge_id] = er.conductance_ratio

    iso_results: List[IsolationLVSResult] = []
    for iso_spec in topo.get("isolation_pairs", []):
        ir = evaluate_isolation_pair_lvs(model, data, iso_spec, dev, default_exclude_geoms=default_excl)
        iso_results.append(ir)

    dyn_results: List[DynamicValveLVSResult] = []
    for dyn_spec in topo.get("dynamic_valve_tests", []):
        dr = evaluate_dynamic_valve_lvs(model, data, dyn_spec, dev, default_exclude_geoms=default_excl)
        dyn_results.append(dr)

    _apply_qpos_overrides(model, data, None)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    open_n = sum(1 for e in edge_results if e.connected)
    neck_n = sum(1 for e in edge_results if e.necking_ok)
    wall_n = sum(1 for e in edge_results if e.wall_ok)
    pass_e = sum(1 for e in edge_results if e.passed)
    pass_i = sum(1 for i in iso_results if i.isolated)
    pass_d = sum(1 for d in dyn_results if d.passed)
    all_ok = (
        pass_e == len(edge_results)
        and pass_i == len(iso_results)
        and pass_d == len(dyn_results)
    )

    report = FluidLVSReport(
        xml_path=str(xml_p),
        topology_path=str(topo_p),
        device=str(dev),
        elapsed_ms=round(elapsed_ms, 1),
        total_edges=len(edge_results),
        open_edges=open_n,
        neck_free_edges=neck_n,
        wall_valid_edges=wall_n,
        passed_edges=pass_e,
        total_isolation_pairs=len(iso_results),
        passed_isolation_pairs=pass_i,
        total_dynamic_tests=len(dyn_results),
        passed_dynamic_tests=pass_d,
        all_passed=all_ok,
        edge_conductances=edge_conds,
        edges=edge_results,
        isolation_pairs=iso_results,
        dynamic_valve_tests=dyn_results,
    )

    if out_report_path is not None:
        out_p = Path(out_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def load_lvs_edge_conductances(report_path: Path | str = "configs/d51_fluid_lvs_report.json") -> Dict[str, float]:
    p = Path(report_path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in raw.get("edge_conductances", {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-Vehicle 3D Fluid-LVS Topology & Conductance Verifier")
    ap.add_argument("--xml", default="models/d51_track.xml", help="Compiled MuJoCo MJCF XML path")
    ap.add_argument("--topology", default="configs/d51_fluid_topology.json", help="Declarative fluid netlist JSON")
    ap.add_argument("--out", default="configs/d51_fluid_lvs_report.json", help="Output Fluid-LVS JSON report")
    ap.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = ap.parse_args()

    dev = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = run_full_vehicle_fluid_lvs(args.xml, args.topology, device=dev, out_report_path=args.out)

    print(
        f"[Fluid-LVS] device={rep.device} elapsed={rep.elapsed_ms:.1f}ms | "
        f"Edges={rep.passed_edges}/{rep.total_edges} (Open={rep.open_edges}, NeckFree={rep.neck_free_edges}, Wall={rep.wall_valid_edges}) | "
        f"Isolation={rep.passed_isolation_pairs}/{rep.total_isolation_pairs} | "
        f"DynamicValves={rep.passed_dynamic_tests}/{rep.total_dynamic_tests}"
    )
    for e in rep.edges:
        if not e.passed:
            print(
                f"  [EDGE FAIL] {e.edge_id} ({e.part_no}): connected={e.connected} "
                f"d_min={e.min_measured_dia_mm:.2f}/{e.min_allowed_dia_mm:.2f}mm "
                f"wall={e.wall_coverage_ratio:.2f} s_reach={e.max_s_reached_mm:.1f}/{e.total_length_mm:.1f}mm "
                f"blk_loco={e.blocked_at_loco_mm} geoms={e.blocking_geoms}"
            )
    for iso in rep.isolation_pairs:
        if not iso.isolated:
            print(
                f"  [ISO FAIL] {iso.pair_id} ({iso.compartment_a} <-> {iso.compartment_b}): "
                f"short_voxels={iso.short_voxels} flooded_a={iso.flooded_voxels_a} leak_loco={iso.leak_at_loco_mm}"
            )
    for dv in rep.dynamic_valve_tests:
        if not dv.passed:
            print(
                f"  [VALVE FAIL] {dv.test_id}: expected={dv.expected_connected} actual={dv.actual_connected} G={dv.conductance_ratio:.3f}"
            )
    if not rep.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
