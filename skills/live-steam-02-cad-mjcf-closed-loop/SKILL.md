---
name: live-steam-02-cad-mjcf-closed-loop
description: >-
  Use this skill at Stage 3 & Stage 4 when building parametric 3D CAD meshes (build123d / OpenCascade STL), integrating metal mass/inertia tensors, compiling closed-loop MuJoCo MJCF valve-gear/rod/pump kinematics, or running watertightness, single-solid (cc==1), 360-deg clearance, and cross-body port mating gates.
---

# Stage 3–4: Parametric CAD Craftsmanship, 3-Tier MJCF Closed-Loop & Topology/Clearance Gates

Activate this skill before authoring `02_build_cad_meshes.py`, `03_generate_mjcf.py`, or running CAD/MJCF verification gates.

## 1. Parametric CAD Invariants ([cad_swept_primitives.py](./scripts/cad_swept_primitives.py))
- **G1 Circular-Arc Swept Tubing Only**: Import `make_swept_tube`, `make_banjo_eye`, `make_hex_nut`, and `make_stepped_pin_L` from [cad_swept_primitives.py](./scripts/cad_swept_primitives.py). **Never** use `Box` cubes as 90-deg pipe elbows.
- **`0.3–0.6 mm` Boolean Overlap & `Face(Wire)` Extrusions**:
  - Kissing OpenCascade solids (`0.0 mm` overlap) silently drop shells or split into `connected_components >= 2`. Always embed mating tube ends/clevises by `0.4–0.6 mm`.
  - When extruding a closed 3D planar `Wire` (e.g., spline bunker walls or arched flat-steel reach rods), ALWAYS wrap `extrude(Face(wire), amount=..., dir=...)` so OpenCascade produces a manifold `Solid` instead of an open `Shell`.
- **Optical & X-Ray Mesh Splitting**:
  - Export transparent Pyrex sight-glass tubes (`mat_pyrex_glass`), white porcelain gauge dials (`mat_gauge_dial`), bright stainless slide bars (`mat_machined_steel`), copper flues/firebox (`mat_copper`), internal superheaters/nozzles, and independent cylinder front sheaths as **separate single-solid `.stl` files** — never boolean-union glass/dials into opaque brass or invent fake bridge bars (`tie_rod`) to force `cc == 1`.
- **Metal Density Inertia Integration**:
  - Integrate volume, eccentric COM, and full inertia tensor from every `.stl` using real metal densities (`RHO_COPPER = 8940`, `RHO_BRASS = 8500`, `RHO_STAINLESS = 7930`, `RHO_CAST_IRON = 7350` kg/m³) via `compute_stl_mass_props()` in [cad_swept_primitives.py](./scripts/cad_swept_primitives.py).

## 2. 3-Tier MuJoCo MJCF Closed-Loop Rules
- **Never hardcode symmetric coordinates in `.xml`**; generate `.xml` deterministically from a Python compiler equipped with an analytical `t = 0` root solver (`solve_side_pose`, quartering $\Delta\theta = -90^\circ$, initial `<connect>` residual `< 0.001 mm`).
- **Tier 1 (Pin-Joint Loops)**: Close crosshead, coupling-rod, Walschaerts valve-gear, and 4-bar hand-pump loops with `<equality><connect>` (never `weld`!). Match expansion-link curved hinge center to visual slot radius $R = L_{\text{radius\_rod}}$.
- **Tier 2 (Floating-Axle Crank-Pin Play)**: Across floating wheelsets (`j_lat_y`, `j_susp_z`) ↔ rigid frame rods, provide `±0.25–0.40 mm` crank-pin axial play so `<connect>` lateral forces never lock curve tracking on minimum radius (`R2m`).
- **Tier 3 (Geared/Screw/Push-Pull Controls)**: Couple screw-reverser nuts, reach rods, and multi-stage cylinder drain linkages via `<equality><joint polycoef="..."/>`.

## 3. Mandatory Automated Verification ([generic_cad_mjcf_gates.py](./scripts/generic_cad_mjcf_gates.py))
Run the vehicle-agnostic topology, symmetry, and cross-body port gate suite before proceeding to thermo-fluid coupling:
`python3 ~/.gemini/config/plugins/live-steam-twin/skills/live-steam-02-cad-mjcf-closed-loop/scripts/generic_cad_mjcf_gates.py --mesh-dir models/meshes --xml models/<loco>_full.xml --port-spec configs/<loco>_boiler_piping.json`
- Also render a single multi-angle headless EGL sheet (`MUJOCO_GL=egl`, `opt.flags[mjVIS_CONSTRAINT] = 0`) and inspect once for Z-fighting, roof overhang clipping, or boolean slivers.
