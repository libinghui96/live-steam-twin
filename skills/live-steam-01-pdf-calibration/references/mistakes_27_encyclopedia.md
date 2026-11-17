# 27 Hard-Won Lessons Encyclopedia (Distilled from Aster JNR D51 Gates 1–20)

## A. PDF Scanning, Photogrammetry & Multi-Drawing Calibration (Mistakes 1, 6, 13, 19, 26, 27)
- **M1 (Raster vs Vector PDF)**: Scanned assembly PDFs have zero CAD vectors; `page.get_drawings()` fails. Render 300 DPI raster slices, deskew via Hough lines, and calibrate `px/mm` on 1:1 templates (e.g. boiler ID `ø53 mm`) and `45.0 mm` rail gauge.
- **M6 (Vision Token Snowball)**: Never loop-read full 300 DPI PNGs or raw PDFs in agent turns; extract text/contours via local Python and inspect at most one downsampled (`<=1280 px`) composite ROI strip.
- **M13 (Manual Text & Handedness)**: Always transcribe `試運転` & `ネジ・Oリング・その他` tables first to lock exact cross-car hose count/materials (`ST` silicone vs `BT` Viton), shared check-ball hydraulic loops, inline burner wick layout (`1x3` vs `2x2`), and JNR Left-Hand (`+Y`) vs Right-Hand (`-Y`) cab controls.
- **M19 (Exploded View Axial Stretch)**: Exploded drawings pull parts apart along assembly axes; never eyeball `(X, Y, Z)` from a single exploded sheet. Triangulate BOM millimeter dimensions + scaled orthogonal photos + detail section views (`図 x.y`).
- **M26 (Single-View Depth/Axis/Partition Ambiguity)**: Never guess front-vs-rear smokebox pipe order, under-cab blowdown screw axis (`+Y` outward screwdriver slot vs `-X`), coaxial straight valve spindle vs fake dogleg, flat-steel arched reach rod vs round rod, or fixed side bunker walls (`35-5`) vs narrow removable water lid (`36-2`) from one sheet — cross-verify `>= 2` views.
- **M27 (Shadow-Clipped Threshold + Camera Pitch Tilt Trap)**: Never compute dial/flange tilt via brightness threshold `boundingRect` + `arccos(h/w)`. Roof/bezel shadows clip top white pixels and camera tripod depression angle (`~24°`) mimics part tilt. Always use 4-quadrant radial rim gradients divided by a coplanar known-vertical reference circle (`cv_rim_calibrator.py`).

## B. Parametric CAD & Mesh Topology (Mistakes 2, 4, 7, 11, 12, 15, 20, 22, 24)
- **M2 (Coplanar Boolean Swallowing)**: OpenCascade union of kissing cylinders (`0.0 mm` overlap) drops shells silently. Enforce `0.3–0.6 mm` solid penetration overlap at every union joint and verify continuous X-bin vertex occupancy.
- **M4 / M7 (Mirror Sign & X-Ray Material Split)**: Verify `min_y + max_y == 0` on symmetric parts and mirror pairs; split transparent Pyrex glass (`11-7`), white porcelain gauge dial (`PG-ø20`), bright stainless slide bars, and internal copper flues/superheaters from matte-black outer shells so X-Ray mode reveals true internals.
- **M11 / M15 (Goodhart `cc == 1` Fake Bridge Trap)**: Every `.stl` must satisfy `boundary_edges == 0` and `connected_components == 1`, **WITHOUT EVER inventing non-existent bridge bars** (`tie_rod`). Physically separate parts (front sheaths vs rear slide bars) MUST be exported as separate `.stl` files.
- **M12 (Zero `Box` Pipe Elbows)**: Never join orthogonal cylinders with `Box` cubes; use `Wire.make_filleted_polyline(..., radius=R)` + `Circle` `sweep` (`make_swept_tube`) + cylindrical Banjo eyes + hex union nuts.
- **M20 (`build123d` Right-Hand Rule, `Face(Wire)`, Roof Overhang & Z-Fighting)**:
  - `extrude()` on a closed 3D planar `Wire` requires `Face(wire)` to create a `Solid` rather than a hollow `Shell`;
  - Verify right-hand rotation sign on angled levers with unit vectors;
  - Hollow cab roof overhangs completely and leave `>= 0.3 mm` normal step between overlapping dissimilar-material plates to prevent Z-fighting;
  - Densify long straight tubes/plates via `sample_triangle_surface()` before slicing (raw STL vertices only exist at tube bends/corners!).
- **M24 (Lever Sine Lift & Bolster Whitelist)**: Rotating shafts with nonzero `x` offset lift by `-x * sin(q)` at full travel — check clearance at extreme joint angles! Whitelist legitimate `Gate 9` load-bearing bolster center-plates inside `Gate 6` non-mating clearance scans.

## C. MuJoCo Closed-Loop Kinematics & Curve Dynamics (Mistakes 3, 5, 8, 9, 14)
- **M3 / M8 (3-Tier Closed-Loop Contract)**:
  - Tier 1: Keep `<worldbody>` tree-structured; close Walschaerts & 4-bar loops via `<equality><connect>` (never `weld` on rotational pins!) initialized at analytical `solve_side_pose` zero-residual (`< 0.02 mm`) posture, with curved expansion-link hinge matching visual slot radius `R = radius_rod_length` (`Δr < 0.02 mm`).
  - Tier 2: Release `±0.25–0.40 mm` crank-pin axial play across floating axles (`j_lat_y`, `j_susp_z`) so `<connect>` never locks curve lateral play.
  - Tier 3: Bind screw-reverser nuts, reach rods, and 8-stage drain-cock linkages via `<equality><joint>` polynomials (`Gate 14` full human-lever-to-spool chain).
- **M9 (Cross-Body Port Mating `Gate 9`)**: Verify all fluid & mechanical ports across rigid bodies have surface ROI gap `<= 0.50 mm` in both straight (`roller`) and minimum-curve (`R2m`) postures, with flexible umbilical hoses tracking drawbar yaw.

## D. 3-Tier Thermo-Fluid Physics, PINO & WebGL HA Deployment (Mistakes 10, 16, 17, 18, 21, 23, 25)
- **M10 / M17 (Zero-`if` 3-Tier Hybrid Physics)**: L1 `2000 Hz` IAPWS-IF97 mass/energy conservation + Domain 6 shallow-water tank; L2 `250 Hz` BF16 PINO operators where smokebox door breach, sight-glass purge, 5-ball pump competition, and Mariotte flood emerge naturally from $p|_{\Gamma_{\text{atm}}} = p_{\text{atm}}$ boundary conditions (zero `if not airtight: *= 0.05`); L3 `0.5 Hz` async `bonsai-2:27b` copilot with `P_boiler == 0` cold-start guard (`Gate 13`).
- **M23 (Real Exhaust Flow & Full Boil-Off to `0 ml`)**: Stream physical `exhaust_g_s / blower_g_s / safety_g_s` to volumetric steam shaders; map horizontal cylindrical boiler level via exact chord-segment volume inversion `water_volume_to_free_z_mm(ml)` down to `0.0 ml` (`Z_bottom`), decoupling lower-tube immersion from crown-sheet immersion so dry-firing boils all the way to `0 ml`.
- **M18 / M21 / M25 (WebGL UV/PBR, Closed-PTY `EIO` HA & SHA-256 CI Cache-Busting)**:
  - Register every MJCF material in Three.js PBR map and inject planar `uv` attributes on binary `.stl` gauge dials;
  - Wrap WebSocket server `stdout/stderr` in `_SafeTeeStream` + `SIGHUP` ignore (prevents `Errno 5 EIO` instant crash when launching terminal closes), sanitize `NaN/Inf` (`allow_nan=False`), auto-replace stale server instances, and proxy `/ws` + `/healthz` in Vite;
  - Install CPU `torch` in GitHub Actions CI, test gates locally with `CUDA_VISIBLE_DEVICES=""`, and append `?v=<sha256>` from `manifest.json` to every `.xml/.stl/.json` fetch URL.
