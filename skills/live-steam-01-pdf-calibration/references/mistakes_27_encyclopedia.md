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

<!-- AUTO-SYNCED-INCREMENTAL-LESSONS-START -->
## E. Incremental Auto-Synced Lessons (Mistakes 28+)
- **M28 (图纸与 BOM 多源联立取证的四类隐蔽盲区（爆炸图长 S 引出线空白摆放误读、阀体形态/轴向混淆、原厂 BOM 印刷笔误与干式内火室 `12-4` 盲区）)**: 防再犯铁律**： 爆炸图中零件的**页面摆放坐标 ≠ 装配坐标**，必须追踪每条细引出线的真实安装孔落点； 锅炉后封头任何进水/取汽口必须叠加内部火室罩（`12-4`）与烟管剖面校核所连通的真实相区（水空间 / 汽空间 / 干式火室）； 对原厂 BOM 文字建立「图纸引出线 + 装配正文 + 零件尺寸兼容性」三源互审，自动拦截手册印刷笔误。
- **M29 (粗多边形实心圆棒 `.stl` 伪管路 vs 解析曲面 `ISO-10303-21 .step` 母本 + 真壁厚空心圆滑 `.stl` + 7 域三维内腔多相物理)**: 工程权衡与三大 OpenCascade 空心建模铁律**： `.step` 解析曲面母本 + 圆滑空心 `.stl` 双轨输出制**：MuJoCo 底层 `stl_decoder` 不直接读取 `.step` 且硬性要求单个 `.stl` 三角面数 `<= 200,000`。因此 `02_build_cad_meshes.py` 统一为全车 **98 个零件**同步导出无限精度解析 B-Rep 母本 `models/step/*.step`（`ISO-10303-21`）与圆滑水密空心 `models/meshes/*.stl`（高曲率管件 `ang_tol = 0.185 rad` 消除棱角，大型钣金外壳设 `stl_tol = 0.085, ang_tol = 0.28` 将单件控制在 `< 140k` 面
- **M30 (安全阀布尔迟滞 `if` 作弊或二次面积比蠕动泄漏 vs 1 比例蓄压环腔（Huddling Chamber）突跳安全阀与第 5 算子 `SafetyValvePopPINO`)**: - **问题剖析**：原厂手册仅给出 `28-6 ×2` 安全阀起跳表压 `4.0 kgf/cm²`、回座 `3.5~3.6 kgf/cm²` 及顶针 `上部の小さな突起`（未绘内部剖面）。若写 `if P > 4.0: open = True elif P < 3.55: open = False` 属于违背第一性原理的布尔作弊；而若仅建普通平阀座或用二次帘口面积比 $(A_1/A_2)^2$ 计算蓄压腔升力，由于 $\frac{d}{dy}[(A_1(y)/A_2)^2]\big|_{y=0^+} = 0 < k_s$，阀瓣在 `4.0 kgf/cm²` 只会微张几微米蠕动漏汽（Simmer），无法产生真蒸汽机车干脆的突跳泄压（Pop）与低压回座（Blowdown）。 - **物理机制与永久规范（`Gate
- **M31 (高功耗多核工作站重任务并发与后台本地 LLM 空载轮询叠加触发过热/OCP 断电关机，以及四层热安全与功耗护栏（`sim/hw_guard.py`）)**: 四层稳态防护强制规范**： 事件驱动 + 遥测签名去重唤醒 L3 大模型**：`sim/bonsai_copilot.py` 必须检查 `_payload_dirty` 与关键物理量量化签名（无客户端连接或稳态无显著变化时 **0.00 Hz 零推理唤醒**，最小间隔 `>= 12s`）； 零 `sudo` 微秒级硬件护栏（`sim/hw_guard.py`）**：所有批处理与门禁脚本入口自动调用 `apply_cpu_thread_and_nice_caps(max_threads=8, nice_delta=5)` 与 `stop_background_gpu_server_for_heavy_job()`（严禁后台常驻 GPU 推流与离线重计算双倍叠加！），并通过直读内核 `coretemp` 与 `li
