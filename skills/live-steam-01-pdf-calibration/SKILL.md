---
name: live-steam-01-pdf-calibration
description: >-
  Use this skill at Stage 1 & Stage 2 when starting a new live-steam locomotive or mechanical assembly from scanned PDFs, extracting BOM/manual specs, calibrating millimeter dimensions/angles from drawings and photos, resolving multi-view part orientations, or running local Ollama bonsai-2:27b semantic cross-audits.
---

# Stage 1–2: Multi-Drawing PDF Calibration, Truth Matrix & Bonsai Semantic Audit

Activate this workflow whenever extracting dimensions, part orientations, valve axes, or piping topologies from scanned assembly PDFs (`イラスト 1..N`), detail sections (`図 x.y`), 1:1 templates, orthogonal photos, and operator manuals (`部品表 / 組立順序 / 試運転`).

## Mandatory 5-Step Calibration & Prior-Sourcing Protocol

0. **Manufacturer Historical Catalog & Gauge-1 Prior Knowledge Lookup (`asterhobby.com/railwayhistory.html` + `aster-manual.pdf`)**:
   - **Before** modeling any Aster Gauge-1 locomotive, consult the manufacturer's historical specification catalog (`https://www.asterhobby.com/railwayhistory.html` — 110 locomotives from 1975 to 2023) and download linked model-year spec PDFs into `data/raw_pdfs/` to disambiguate generational differences sharing the same road class (e.g., `1986/1992 D51` Watanabe `Mixed Type` boiler vs `2007/2010 JNR D51 Standard` van Riemsdijk `Type "C"` recessed-barrel boiler with separate `SUS304` ceramic-lined dry firebox `12-4`, zero wet copper waterlegs) and exact scale ratio (`1:30` for JNR `1067 mm` Cape-gauge prototypes on `45 mm` track vs `1:32` for `1435 mm` standard gauge).
   - Enforce the **Two-Tier Truth Hierarchy**:
     1. **Concrete part dimensions & capacities** MUST anchor to the assembly blueprints (`イラスト 1..N`, standard seal/ring codes such as `PR-15 = 15.0 mm`, `PF3-1.9`, `PN3.2-1`) and operator manual (`取扱説明書`), never overriding exact part fits with rounded brochure numbers;
     2. **System-level physics & implicit internal mechanisms** (e.g., `Type "C"` dry-liner reradiation + Hausen flue entry heat transfer, `0.50 mm` slide-valve buckle vertical lift overpressure/condensate relief, concealed downward-draining stopped brass whistle two-phase Wood-Kieffer aero-acoustics, Roscoe internal standpipe displacement lubrication, `0.20–0.33` adhesion factor) MUST follow *MANUAL OF ASTER MODEL LIVE STEAM ENGINES* (`aster-manual.pdf`).

1. **Headless Raster Deskew & BOM Extraction (Never `get_drawings()` Alone)**:
   - Scanned assembly PDFs are raster images, not vector CAD (`page.get_drawings()` will return empty/borders only). Render target pages via `pymupdf` (`get_pixmap(dpi=300)`), deskew via Hough line angle on border/track lines, and extract BOM tables & assembly text to structured `.json` first.
   - Explicitly lock:
     - **Driving handedness** (e.g., JNR Left-Hand Drive `+Y` vs UK/US Right-Hand Drive `-Y`),
     - **Every `ダミー` (Dummy / decorative) part** (`dummy: true`, forbidden from fluid loops and forbidden from fake joints, e.g., dummy check valves, dummy whistles, static manhole covers),
     - **Exact umbilical hose count, material code (`ST` silicone vs `BT` Viton), and shared-pump check-ball routing** from the `試運転` text.

2. **Shadow- & Camera-Pitch-Invariant Circle/Dial Calibration ([cv_rim_calibrator.py](./scripts/cv_rim_calibrator.py))**:
   - **NEVER** threshold bright pixels of a dial/flange and plug `cv2.boundingRect` `h / w` into `arccos(h / w)` (cab roof/bezel shadows clip top pixels and photographer tripod depression angle $\theta_{\text{cam}}$ masquerades as part upward tilt — see Mistake 27 in [mistakes_27_encyclopedia.md](./references/mistakes_27_encyclopedia.md)).
   - Always call `calibrate_dial_tilt_differential()` from [cv_rim_calibrator.py](./scripts/cv_rim_calibrator.py), which:
     1. Detects circular rims via `cv2.HoughCircles` + 4-quadrant radial first-derivative intensity peaks (`measure_circle_rim_aspect`),
     2. Divides by `D_vert / D_horiz` of a **known-vertical coplanar reference circle** on the same wall (canceling camera pitch), and
     3. Cross-checks mating pipe stem orthogonality (`HoughLinesP` $\approx 90^\circ$) on isometric/side assembly drawings.

3. **Build & Gate the Multi-Drawing Evidence Matrix (`configs/<loco>_part_evidence.json`)**:
   - Before writing any CAD code, register every moving part and every fluid/thermal-affecting component in `configs/<loco>_part_evidence.json` and validate it with:
     `python3 ~/.gemini/config/plugins/live-steam-twin/skills/live-steam-01-pdf-calibration/scripts/cv_rim_calibrator.py --verify-evidence configs/<loco>_part_evidence.json`
   - Every entry MUST cite `>= 2` complementary views (e.g., `["イラスト 13 (YZ backhead)", "イラスト 17 (XZ side)", "p43_left_photo"]`), an explicit `joint_type` (`hinge`/`slide`/`static`), `axis_xyz`, and `mating_parent`.

4. **Run Independent Local Ollama `bonsai-2:27b` Semantic Cross-Audit ([bonsai_semantic_auditor.py](./scripts/bonsai_semantic_auditor.py))**:
   - Cross-check your extracted manual text + `*_boiler_piping.json` + evidence matrix using the local zero-cost LLM auditor:
     `python3 ~/.gemini/config/plugins/live-steam-twin/skills/live-steam-01-pdf-calibration/scripts/bonsai_semantic_auditor.py --manual-txt logs/manual_text.txt --spec-json configs/<loco>_boiler_piping.json`
   - Resolve every flagged contradiction (e.g., 90-deg rotated blowdown needle, reversed smokebox exhaust/steam-tee order, missing linkage intermediate rockers, or fake hinged lids) before Stage 3 CAD modeling.
