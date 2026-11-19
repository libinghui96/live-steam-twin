---
name: live-steam-03-hybrid-pino-thermo
description: >-
  Use this skill at Stage 5 when implementing 1D/3D boiler & valve-gear thermo-fluid physics, IAPWS-IF97 two-phase evaporation, dry-firing boil-off to 0 ml, CUDA STL voxelization, zero-if PINO surrogate training, cold-start chimney-fan draft calibration, or CPU-only CI verification.
---

# Stage 5: 3-Tier Hybrid Thermo-Fluid Architecture (IAPWS-IF97 + Zero-`if` PINO + Full Boil-Off)

Activate this skill when building or calibrating `thermo_1d.py`, `thermo_3d_gpu.py`, `pino_models.py`, or `bonsai_copilot.py`.

## 1. Golden 3-Tier Division of Labor (`Gate 10 & Gate 17`)
1. **L1 Analytical Conservation Core (`2000 Hz` synchronous with MuJoCo)**:
   - Integrate water/steam mass $M$ and internal energy $U$ strictly via IAPWS-IF97 $(v, u) \mapsto (P, T_{\text{sat}}, x)$, plus open-top tender tank 2D shallow-water conservation (Domain 6).
   - **Never** let neural networks overwrite conserved mass/energy directly; neural operators predict fluxes/pressure drops fed into the conservative finite-volume integrator.
2. **L2 Multi-Operator BF16 PINO Layer (`250 Hz` on GPU)**:
   - Train lightweight FNO/PINO operators (`BurnerFlueSmokeboxPINO`, `SlideValveCylinderPINO`, `CabManifoldSightGlassPINO`, `TenderHydraulicPINO`) supervised by converged 3D FVM/PDE ground truth + continuity/momentum PDE residual losses.
   - **Zero-`if` Atmospheric Emergence**: Smokebox door breach, sight-glass purge blowdown, 5-check-ball bypass diversion, and Mariotte fuel-cap flood MUST emerge solely from imposing atmospheric Dirichlet boundary projection $p|_{\Gamma_{\text{atm}}} = p_{\text{atm}}$ — **strictly zero** `if not airtight: vac *= 0.05` or `if bypass_open: divert = 0.9` hardcoded multipliers!
   - Suffix every training tensor/label with explicit physical units (`_ml_s`, `_kg_s`, `_pa`, `_kpa`) to prevent `1000x` unit double-conversion bugs (Mistake 18.3).
3. **L3 Async Local `bonsai-2:27b` Copilot (`0.5 Hz` background daemon thread)**:
   - Non-blocking queue handoff (`< 0.5 ms` main-loop overhead).
   - **Cold-Start Physical Consistency Guard (`Gate 13`)**: Bind diagnostic prompts to `P_boiler_kgf` and `mdot_jet_gs` so `bonsai-2:27b` never false-alarms "smokebox leak / clogged flue" during cold electric `chimney_fan` warm-up at $P_{\text{boiler}} = 0$ (`-65 ~ -105 Pa` draft, `1x` real-time `25°C -> 2.0 kgf/cm²` calibrated to manual's `3 ~ 4.5 min`).

## 2. Full-Range Cylindrical Boil-Off Down to `0 ml` (`Gate 19.2`, Mistake 23)
- **Never** clamp evaporation at sight-glass bottom (`0% SG` $\approx 140\text{ ml}$) or bind fire-tube boiling strictly to firebox crown-sheet immersion `f_wet`!
- Invert horizontal cylindrical boiler net volume (subtracting 5 flues & inner firebox displacement) via exact circular chord-segment lookup `water_volume_to_free_z_mm(ml)` all the way from full capacity down to `0.0 ml` (`Z_boiler_bottom`).
- Decouple **lower flue / water-leg immersion** `f_wet_tubes` from **crown-sheet immersion** `f_wet_crown`, and retain high-conductivity C1100 copper shell residual-puddle conduction $\propto (V / 18\text{ ml})^{2/3}$ + Stefan-Boltzmann flame radiation floor so dry-firing boils continuously down to `0.0 ml` while overheating the bare copper crown sheet (`> 420°C` annealing damage).

## 3. CPU-Only GitHub Actions CI Compatibility (Mistake 25)
- Always install CPU PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) in `.github/workflows/*.yml`.
- Cache static STL voxel grids by content hash (`blake2b`), and in long boil-off gates check `torch.cuda.is_available()` (run full 3D PDE steps on GPU; run 1D core + initial/final 3D voxel sync check on CPU CI). Always pre-test locally with `CUDA_VISIBLE_DEVICES=""`.

## 4. Full-Vehicle 3D Fluid-LVS Verification ([fluid_topology_lvs.py](./scripts/fluid_topology_lvs.py))
- Solid-only topology gates (`boundary_edges == 0`, `connected_components == 1`) and port Euclidean proximity (`gap <= 0.50 mm`) **cannot** catch un-drilled multi-wall partitions, boolean re-sealing bugs, chassis/bracket intrusions pinching pipe lumens, or inter-compartment short-circuits!
- Always author a declarative full-vehicle fluid netlist `configs/<loco>_fluid_topology.json` (enumerating every fluid subnet, chamber node, pipe/valve edge with `spec_id_mm` / `spec_od_mm`, compartment isolation pairs, and dynamic valve `qpos` states) and run [fluid_topology_lvs.py](./scripts/fluid_topology_lvs.py):
  `python3 ~/.gemini/config/plugins/live-steam-twin/skills/live-steam-03-hybrid-pino-thermo/scripts/fluid_topology_lvs.py --xml models/<loco>_track.xml --topology configs/<loco>_fluid_topology.json --out configs/<loco>_fluid_lvs_report.json`
- Enforce all 4 LVS invariants across both straight and minimum-curve (`R2m`) postures:
  1. **`Zero-Open`**: 6-face FVM flood-fill inside wall-bounded inner sleeve (`r_sleeve < r_od` so ambient air outside the pipe can NEVER shortcut a blocked bore!) using per-mesh `torch.maximum` solid union reconstructed directly from MuJoCo `d.geom_xpos + verts_mj @ d.geom_xmat.T` (including visible structural primitives);
  2. **`Zero-Necking`**: C¹ circular-arc filleted centerline (`_fillet_polyline_3d`) axial-slab hydraulic diameter $d_{\text{eq}}(s) = 2\sqrt{A_{\text{open}}(s)/\pi} \ge d_{\min}$;
  3. **`Zero-Missing-Wall`**: Pipe-wall annular coverage ratio check preventing deleted/missing pipes in open air from false-passing;
  4. **`Zero-Short` & Dynamic Valve States**: Hydrostatic port-capped multi-compartment flood-fill (`short_voxels == 0`) + live joint `qpos` valve opening/closing topology verification, exporting `edge_conductances` directly into L1/L2 physics and Web X-Ray flow vectors with zero `if` cheats.
