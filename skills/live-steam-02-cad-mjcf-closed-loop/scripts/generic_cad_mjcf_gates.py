#!/usr/bin/env python3
"""
Vehicle-agnostic Automated CAD Watertightness, Single-Solid (cc==1), Closed-Loop Residual & Cross-Body Port Mating Gate Suite.
Reusable across any locomotive project (D51, C62, D52, C57, Big Boy, etc.).
"""
import argparse
import json
import math
from pathlib import Path
import struct
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def load_binary_stl(stl_path: Path) -> np.ndarray:
    raw = stl_path.read_bytes()
    n_tri = struct.unpack("<I", raw[80:84])[0]
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=dt, offset=84, count=n_tri)["v"].astype(np.float64)


def verify_stl_manifold_and_single_solid(tris_mm: np.ndarray, weld_tol_mm: float = 1e-4) -> Tuple[int, int]:
    """Return (unmatched_directed_half_edges, connected_component_count)."""
    n_tri = len(tris_mm)
    scale = 1.0 / max(weld_tol_mm, 1e-7)
    pts = np.round(tris_mm.reshape(-1, 3) * scale).astype(np.int64)
    _, inv = np.unique(pts, axis=0, return_inverse=True)
    f = inv.reshape(n_tri, 3)

    # Directed half-edge exact opposite matching (handles multi-shell / thin-wall hollow tubes cleanly)
    directed = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    max_vid = int(inv.max()) + 1
    fwd_sorted = np.sort(directed[:, 0] * max_vid + directed[:, 1])
    rev_sorted = np.sort(directed[:, 1] * max_vid + directed[:, 0])
    unmatched = int(np.sum(fwd_sorted != rev_sorted))

    # Union-Find connected components
    parent = list(range(max_vid))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, w in f.tolist():
        ru, rv, rw = _find(u), _find(v), _find(w)
        if ru != rv:
            parent[rv] = ru
        if ru != rw:
            parent[rw] = ru
    cc = len({_find(v) for v in range(max_vid)})
    return unmatched, cc


def sample_triangle_surface(tris: np.ndarray, max_spacing: float = 0.85) -> np.ndarray:
    """Densify flat plates & straight swept tubes so ROI slices never hit zero-vertex blind spots (Mistake 20/24)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    max_edge = np.maximum(np.maximum(np.linalg.norm(v1 - v0, axis=1), np.linalg.norm(v2 - v1, axis=1)), np.linalg.norm(v0 - v2, axis=1))
    chunks = [tris.reshape(-1, 3)]
    for k_div, lo, hi in [(2, 1.0, 2.2), (4, 2.2, 4.5), (8, 4.5, 10.0), (14, 10.0, 1e9)]:
        mask = (max_edge > lo * max_spacing) & (max_edge <= hi * max_spacing)
        if not np.any(mask):
            continue
        sv0, sv1, sv2 = v0[mask], v1[mask], v2[mask]
        bary = []
        for i in range(k_div + 1):
            for j in range(k_div + 1 - i):
                if i in (0, k_div) and j in (0, k_div):
                    continue
                u, w = i / k_div, j / k_div
                bary.append((1.0 - u - w, u, w))
        if bary:
            b = np.asarray(bary, dtype=np.float64)
            pts_sub = b[:, 0:1, None] * sv0[None, :, :] + b[:, 1:2, None] * sv1[None, :, :] + b[:, 2:3, None] * sv2[None, :, :]
            chunks.append(pts_sub.reshape(-1, 3))
    return np.vstack(chunks)


def _max_connect_residual_mm(model, data) -> float:
    import mujoco

    max_res_m = 0.0
    for eid in range(model.neq):
        if int(model.eq_type[eid]) != int(mujoco.mjtEq.mjEQ_CONNECT) or not bool(model.eq_active0[eid]):
            continue
        id1, id2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
        if int(model.eq_objtype[eid]) == int(mujoco.mjtObj.mjOBJ_SITE):
            p1, p2 = data.site_xpos[id1], data.site_xpos[id2]
        else:
            p1 = data.xpos[id1] + data.xmat[id1].reshape(3, 3) @ model.eq_data[eid, 0:3]
            p2 = data.xpos[id2] + data.xmat[id2].reshape(3, 3) @ model.eq_data[eid, 3:6]
        max_res_m = max(max_res_m, float(np.linalg.norm(p1 - p2)))
    return max_res_m * 1000.0


def verify_mjcf_closed_loop_and_ports(xml_path: Path, mesh_dir: Path, port_spec_path: Path | None = None) -> Tuple[bool, List[str]]:
    import mujoco

    ok = True
    msgs: List[str] = []
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    res_mm = _max_connect_residual_mm(m, d)
    if res_mm > 0.02:
        ok = False
        msgs.append(f"FAIL [{xml_path.name}]: t=0 <connect> residual = {res_mm:.4f} mm > 0.02 mm!")
    else:
        msgs.append(f"PASS [{xml_path.name}]: t=0 <connect> residual = {res_mm:.5f} mm <= 0.02 mm")

    if port_spec_path and port_spec_path.exists():
        spec = json.loads(port_spec_path.read_text(encoding="utf-8"))
        contracts = spec.get("topology_and_port_mating_contracts", {}).get("cross_body_port_mating_contracts", {})
        max_gap = float(contracts.get("max_gap_mm", 0.50))
        ports = contracts.get("ports", [])
        stl_cache = {}

        def _geom_pts_world_mm(gname: str, roi_mm=None) -> np.ndarray:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid < 0:
                return np.zeros((0, 3))
            gtype = int(m.geom_type[gid])
            xpos, xmat = d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3)

            if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[gid])
                mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
                stem = mname[2:] if mname.startswith("m_") else mname
                if stem not in stl_cache:
                    stl_cache[stem] = sample_triangle_surface(load_binary_stl(mesh_dir / f"{stem}.stl"), max_spacing=0.80)
                pts_cad = stl_cache[stem]
                if roi_mm is not None:
                    lo, hi = np.array(roi_mm[0]), np.array(roi_mm[1])
                    pts_cad = pts_cad[np.all((pts_cad >= lo) & (pts_cad <= hi), axis=1)]
                if len(pts_cad) == 0:
                    return np.zeros((0, 3))
                R_m = np.zeros(9)
                mujoco.mju_quat2Mat(R_m, m.mesh_quat[mid])
                pts_m = (pts_cad * 1e-3 - m.mesh_pos[mid]) @ R_m.reshape(3, 3)
                return (xpos + pts_m @ xmat.T) * 1000.0

            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                rad, h_half = float(m.geom_size[gid][0]), float(m.geom_size[gid][1])
                thetas = np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False)
                zs = np.linspace(-h_half, +h_half, max(5, int(math.ceil(2.0 * h_half / 0.00045)) + 1))
                pts = [[rad * math.cos(th), rad * math.sin(th), z] for z in zs for th in thetas]
                for z_cap in (-h_half, +h_half):
                    pts.append([0.0, 0.0, z_cap])
                    for r_sub in np.linspace(rad * 0.25, rad, 4):
                        for th in thetas:
                            pts.append([r_sub * math.cos(th), r_sub * math.sin(th), z_cap])
                return (xpos + np.asarray(pts) @ xmat.T) * 1000.0

            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                sx, sy, sz = [float(v) for v in m.geom_size[gid][:3]]
                c = np.array([[-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz], [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz]])
                tris = np.asarray([[c[a], c[b], c[d_i]] for a, b, d_i, _ in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]] + [[c[a], c[d_i], c[e]] for a, _, d_i, e in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]])
                pts_loc = sample_triangle_surface(tris, max_spacing=0.00045)
                return (xpos + pts_loc @ xmat.T) * 1000.0

            return (xpos[None, :]) * 1000.0

        worst_gap, worst_id = 0.0, ""
        for p in ports:
            pa = _geom_pts_world_mm(p["geom_a"], p.get("roi_a_local_mm"))
            pb = _geom_pts_world_mm(p["geom_b"], p.get("roi_b_local_mm"))
            if len(pa) == 0 or len(pb) == 0:
                ok = False
                msgs.append(f"FAIL [{xml_path.name}]: port '{p['id']}' empty ROI ({len(pa)} vs {len(pb)})!")
                continue
            gap = float(cKDTree(pb).query(pa, k=1)[0].min())
            if gap > worst_gap:
                worst_gap, worst_id = gap, p["id"]
            if gap > max_gap:
                ok = False
                msgs.append(f"FAIL [{xml_path.name}]: port '{p['id']}' gap = {gap:.3f} mm > {max_gap:.2f} mm!")
        if ok and ports:
            msgs.append(f"PASS [{xml_path.name}]: All {len(ports)}/{len(ports)} ports mated (worst={worst_gap:.3f} mm at {worst_id})")
    return ok, msgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic CAD topology & MJCF closed-loop/port gate verifier")
    ap.add_argument("--mesh-dir", type=Path, required=True, help="Directory of binary .stl files")
    ap.add_argument("--xml", type=Path, nargs="*", default=[], help="MJCF .xml files to verify")
    ap.add_argument("--port-spec", type=Path, default=None, help="Optional *_boiler_piping.json with port contracts")
    args = ap.parse_args()

    all_ok = True
    stls = sorted(args.mesh_dir.glob("*.stl"))
    bad_b, bad_cc = [], []
    for p in stls:
        be, cc = verify_stl_manifold_and_single_solid(load_binary_stl(p))
        if be != 0:
            bad_b.append((p.name, be))
        if cc != 1:
            bad_cc.append((p.name, cc))
    if bad_b or bad_cc:
        all_ok = False
        print(f"FAIL [STL Topology]: boundary_violations={bad_b}, multi_solid_violations={bad_cc}")
    else:
        print(f"PASS [STL Topology]: All {len(stls)}/{len(stls)} STLs 100% watertight (be==0) & single-solid (cc==1)")

    for xp in args.xml:
        ok, msgs = verify_mjcf_closed_loop_and_ports(xp, args.mesh_dir, args.port_spec)
        all_ok &= ok
        for m in msgs:
            print(m)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
