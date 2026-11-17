# Precision Mechanical & Live-Steam Digital Twin Global Rules

When working on a locomotive, live-steam model, CAD-to-MuJoCo assembly, or coupled 1D/3D thermo-fluid simulation project, enforce these non-negotiable engineering invariants:

1. **Zero High-Res Vision Loop (`Vision Token Guard`)**:
   - NEVER loop-read raw `.pdf` files or full-page 300 DPI `.png` scans with file-reading tools. Extract BOM text, dimensions, and geometry via local `pymupdf` / `cv2` Python scripts first; when visual inspection is required, crop and downsample a single low-resolution composite ROI strip (`<= 1280 px`) and read it once.
2. **Multi-Drawing Cross-Verification & 1:1 Moving/Behavioral Part Fidelity**:
   - Every part that **actually moves** or **affects fluid, thermal, valve-timing, or clearance behavior** MUST be reconstructed 1:1 in size, true geometry, orientation, and kinematic chain (from human lever/crank through rods/cams/screw-nuts to terminal spool/poppet). Static decorative `ダミー` parts get static mass/mesh only (zero fake DOFs).
   - Never infer 3D depth layering, screw/valve axis orientation, fixed-vs-removable partition ownership, or dial tilt from a single exploded drawing or single photo. Cross-verify `>= 2` complementary views (main assembly + exploded sheet + detail section `図 x.y` + 1:1 templates + orthogonal photos + manual assembly text) and activate `live-steam-01-pdf-calibration`.
3. **Shadow- & Camera-Pitch-Invariant Photogrammetry**:
   - NEVER estimate circular dial/flange tilt via brightness-thresholded `cv2.boundingRect` + `arccos(h / w)` (roof/bezel shadows clip top pixels and camera tripod pitch masquerades as part tilt!). Always measure metal rim radial-gradient peaks and normalize by a coplanar known-vertical reference circle on the same plane.
4. **CAD Single-Solid & True Tube Craftsmanship (`live-steam-02-cad-mjcf-closed-loop`)**:
   - Every `.stl` must satisfy `boundary_edges == 0` AND `connected_components == 1` with `0.3–0.6 mm` boolean overlap (never invent non-existent bridge rods to fake `cc == 1`; split physically separate parts into separate `.stl` files).
   - Never use `Box` cubes as 90-deg pipe elbows; sweep smooth circular-arc bends (`Wire.make_filleted_polyline` + `Circle`) with cylindrical Banjo eyes and hex union nuts. All cross-body fluid/mechanical ports must mate within `<= 0.50 mm` in both straight and minimum-curve postures.
5. **3-Tier Physics-PINO-Copilot Golden Division (`live-steam-03-hybrid-pino-thermo` & `live-steam-04-webgl-fluid-ha-server`)**:
   - Never hardcode `if not airtight: *= 0.05` cheat multipliers in 3D thermo-fluid solvers, and never put LLMs inside high-frequency PDE loops. Use L1 analytical conservation (IAPWS-IF97 + full cylindrical chord-segment boil-off down to `0 ml`), L2 boundary-projected PINO/FVM operators, and L3 async local `bonsai-2:27b` diagnostics.
