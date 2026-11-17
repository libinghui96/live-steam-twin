---
name: live-steam-04-webgl-fluid-ha-server
description: >-
  Use this skill at Stage 6 when building or updating the Three.js / MuJoCo WASM Web 3D frontend, custom GLSL volumetric steam/water/flame shaders, asset cache-busting manifests, GitHub Pages deployment, or the high-availability Python WebSocket GPU streaming server.
---

# Stage 6: WebGL PBR/GLSL Volumetric Fluid Rendering, SHA-256 Asset Sync & HA Live Streaming

Activate this skill whenever modifying `web/src/*`, `05_sync_web_assets.py`, or `gpu_3d_server.py`.

## 1. Four-Way Material, UV & Real Steam Jet Sync (Mistakes 18 & 23)
- **PBR Material Parity**: Every `<material name="...">` in MJCF (especially `mat_pyrex_glass`, `mat_gauge_dial`, `mat_copper`, `mat_brass`, `mat_machined_steel`, `mat_black_iron`) MUST be registered in Three.js `pbrMatByName` so transparent glass and white porcelain dials never fall back to dark cast iron.
- **Binary STL Planar UV Injection**: Binary `.stl` files contain no `uv` attribute. For textured dials (e.g. strictly vertical `cab_pg_dial` in the `YZ` plane with normal `[-1, 0, 0]`), project CAD vertex coordinates orthogonally (`u = 0.5 - py / D`, `v = 0.5 + pz / D`) and attach `new THREE.BufferAttribute(uvArr, 2)` before centering by `model.mesh_pos` / `model.mesh_quat`.
- **Volumetric Steam & Real Mass-Flow Plumes**: Stream physical `exhaust_g_s`, `blower_g_s`, and `safety_g_s` directly from the thermo solver (never multiply exhaust pulse by post-blowdown `cylPKgf ≈ 0`!). In X-Ray mode, render both the pressurized boiler headspace steam cloud (`z > z_water`) and the smokebox Venturi blast-jet cone (`blast_nozzle -> petticoat -> chimney`), and hide the 3D water surface / bubbles cleanly when `V_water -> 0 ml`.

## 2. SHA-256 Content-Hashed Static Asset Sync (Mistake 25)
- Vite hashes JS bundles but does **not** hash `public/models/*.stl` or `*.xml`.
- `05_sync_web_assets.py` MUST compute SHA-256 digests for every `.xml`, `.stl`, and `.json` file into `manifest.json` (`asset_hashes` + `build_hash`), clean up deleted `.stl` files in `web/public/models/meshes/`, and fetch `manifest.json?t=<timestamp>` with `{ cache: "no-cache" }` on startup so every mesh/XML URL uses `?v=<sha256>`.
- On pure static hosts (`*.github.io` without `?ws=` query param), skip same-origin `wss://.../ws` probing to avoid 404 reconnect noise, and ensure all visual joints (including screw-reverser wheels/nuts) are bound in WASM fallback mode.

## 3. High-Availability WebSocket GPU Server Five-Point Contract (Mistake 21)
1. **Closed-PTY `EIO` Immunity**: Ignore `signal.SIGHUP` and wrap `sys.stdout` / `sys.stderr` in a `_SafeTeeStream` that catches `OSError (Errno 5 EIO)` and logs to `logs/gpu_server.log` so closing the launching SSH/IDE terminal never crashes `print()` inside `ws_handler`.
2. **Graceful Old-Instance Takeover & Safe `pkill`**: Auto-terminate stale same-user server processes holding port `8765` on startup; if killing via shell, always write `pkill -f "[s]im.gpu_3d_server"` so regex never kills the invoking subshell.
3. **Same-Origin Proxy & LAN Binding**: Bind backend to `0.0.0.0:8765`, proxy `/ws` (`ws: true`) and `/healthz` in `vite.config.ts`, and connect via same-origin `ws(s)://<window.location.host>/ws` first so single-port SSH tunnels and HTTPS IDE proxies work seamlessly.
4. **Event-Loop Yielding & Backpressure**: `await asyncio.sleep(0)` every `<= 8` physics substeps (`< 4 ms`), drop frames gracefully if `transport.get_write_buffer_size() > 256 KB`, and set `heartbeat = 55s`.
5. **Strict Finite JSON (`allow_nan=False`)**: Sanitize non-finite floats before `json.dumps(..., allow_nan=False)`, auto-reset diverged `qpos`, and run a `5s` client-side stale-frame watchdog reconnect.
