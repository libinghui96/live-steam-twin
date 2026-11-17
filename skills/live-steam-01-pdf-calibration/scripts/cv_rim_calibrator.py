#!/usr/bin/env python3
"""
Reusable Shadow- & Camera-Pitch-Invariant Photogrammetry & Multi-Drawing Evidence Verifier.
Prevents Mistakes 1, 13, 19, 26, and 27.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def measure_circle_rim_aspect(gray_roi: np.ndarray, cx: float, cy: float, r_est: float) -> float:
    """
    Measure true outer metal rim D_vert / D_horiz via 4-quadrant radial gradient peaks.
    Immune to cab-roof / bezel-lip crescent shadows on white dial faces and central needle occlusion.
    """
    radii: Dict[int, float] = {}
    h, w = gray_roi.shape[:2]
    for deg in (0, 90, 180, 270):
        th = math.radians(deg)
        best_r, best_g = r_est, -1.0
        for rr in np.linspace(r_est * 0.85, r_est * 1.15, 37):
            px = int(round(cx + rr * math.cos(th)))
            py = int(round(cy + rr * math.sin(th)))
            if 1 <= px < w - 1 and 1 <= py < h - 1:
                gx = float(gray_roi[py, px + 1]) - float(gray_roi[py, px - 1])
                gy = float(gray_roi[py + 1, px]) - float(gray_roi[py - 1, px])
                grad = abs(gx * math.cos(th) + gy * math.sin(th))
                if grad > best_g:
                    best_g, best_r = grad, float(rr)
        radii[deg] = best_r
    d_horiz = max(1e-6, radii[0] + radii[180])
    d_vert = radii[90] + radii[270]
    return float(d_vert / d_horiz)


def calibrate_dial_tilt_differential(
    target_gray: np.ndarray,
    target_circle_xyr: Tuple[float, float, float],
    coplanar_ref_gray: np.ndarray,
    ref_circle_xyr: Tuple[float, float, float],
    stem_vertical_confirmed: bool = True,
    zero_lock_tol: float = 0.05,
) -> Dict[str, float]:
    """
    Cancel camera extrinsic pitch by dividing target rim aspect ratio by a known-vertical
    coplanar reference circle on the same wall (e.g., 11-13 handwheel or firehole ring).
    """
    ar_target = measure_circle_rim_aspect(target_gray, *target_circle_xyr)
    ar_ref = measure_circle_rim_aspect(coplanar_ref_gray, *ref_circle_xyr)
    rel_cos = ar_target / max(0.80, ar_ref)
    cam_pitch_est_deg = math.degrees(math.acos(max(-1.0, min(1.0, ar_ref))))
    if abs(rel_cos - 1.0) <= zero_lock_tol and stem_vertical_confirmed:
        tilt_deg = 0.0
    else:
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, rel_cos))))
    return {
        "target_rim_dvert_dhoriz": round(ar_target, 4),
        "coplanar_ref_dvert_dhoriz": round(ar_ref, 4),
        "estimated_camera_pitch_deg": round(cam_pitch_est_deg, 2),
        "differential_cos_rel": round(min(1.0, rel_cos), 4),
        "calibrated_tilt_deg": round(tilt_deg, 2),
    }


def verify_part_evidence_matrix(json_path: Path) -> Tuple[bool, List[str]]:
    """
    Gate check for configs/<loco>_part_evidence.json:
      - Every moving or behavior-affecting part MUST cite >= 2 distinct drawing/photo views;
      - Moving parts MUST declare joint_type in {'hinge','slide'} and a unit 3D axis_xyz;
      - Dummy parts MUST declare joint_type == 'static' and dummy == True.
    """
    if not json_path.exists():
        return False, [f"Missing evidence matrix file: {json_path}"]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    parts: List[Dict[str, Any]] = data.get("parts", [])
    if not parts:
        return False, ["Evidence matrix contains 0 parts"]

    ok = True
    msgs: List[str] = []
    for item in parts:
        pid = str(item.get("part_no", "UNKNOWN"))
        views = item.get("verified_views") or []
        jtype = str(item.get("joint_type", ""))
        is_dummy = bool(item.get("dummy", False))
        if len(set(views)) < 2:
            ok = False
            msgs.append(f"FAIL [{pid}]: only {len(set(views))} view(s) cited {views}; require >= 2 orthogonal/complementary views!")
        if is_dummy and jtype != "static":
            ok = False
            msgs.append(f"FAIL [{pid}]: marked dummy=True but joint_type='{jtype}' != 'static' (Mistake 17/26)!")
        if jtype in ("hinge", "slide"):
            ax = item.get("axis_xyz")
            if not (isinstance(ax, list) and len(ax) == 3 and abs(math.sqrt(sum(float(v) ** 2 for v in ax)) - 1.0) < 1e-2):
                ok = False
                msgs.append(f"FAIL [{pid}]: moving joint '{jtype}' missing normalized 3D axis_xyz={ax}!")
    if ok:
        msgs.append(f"PASS: All {len(parts)} parts in {json_path.name} verified with >=2 cross-referenced views & valid axes.")
    return ok, msgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-drawing evidence verifier & shadow-invariant rim calibrator")
    ap.add_argument("--verify-evidence", type=Path, help="Path to configs/<loco>_part_evidence.json")
    args = ap.parse_args()
    if args.verify_evidence:
        ok, msgs = verify_part_evidence_matrix(args.verify_evidence)
        for m in msgs:
            print(m)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
