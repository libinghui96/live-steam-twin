# Live-Steam Locomotive Digital Twin Kit (`live-steam-twin`)

从扫描栅格版组装图 PDF 与运转说明书从零构建 **1:1 精密真蒸汽机车三维机械 + 热流体多相数字孪生** 的全局可复用工程套件（沉淀自 Aster Hobby JNR D51 `Gates 1–20` 与 27 项实战踩坑经验）。

## 四层经验继承机制

1. **全局常驻红线 (`rules/AGENTS.md`)**：精炼跨车型最高准则（严禁循环直读高清 PDF/大图防 Token 雪崩、全可动/行为影响零件多图纸联立 1:1 还原、三层黄金物理分工）。
2. **生命周期硬拦截钩子 (`hooks.json` + `hooks/pre_tool_guard.py`)**：
   - 自动拦截直接对 `.pdf` 或超大图调用 `read_file` / `view_file` 的行为，强制先用本地 Python 脚本裁剪低分辨率局部 ROI；
   - 自动拦截在像素标定脚本中使用高阈值 `boundingRect` + `acos/arccos` 测圆盘倾角的失误模式（失误 27）。
3. **分阶段渐进式披露技能包 (`skills/`) + 可复用代码模板**：
   - `live-steam-01-pdf-calibration`：图纸纠偏、BOM 提取、抗阴影四象限边缘梯度差分标定 (`cv_rim_calibrator.py`)、多图纸证据矩阵校验与本地 Ollama `bonsai-2:27b` 语义互审 (`bonsai_semantic_auditor.py`)；
   - `live-steam-02-cad-mjcf-closed-loop`：`make_swept_tube` 圆弧导角管路原语 (`cad_swept_primitives.py`)、单连通水密/端口零断口/360° 净空通用门禁 (`generic_cad_mjcf_gates.py`)；
   - `live-steam-03-hybrid-pino-thermo`：IAPWS-IF97 硬守恒 + 圆筒弓形全量程干烧至 `0 ml` 蒸干 + CUDA STL 体素化与零 `if` PINO 涌现；
   - `live-steam-04-webgl-fluid-ha-server`：SHA-256 静态资产防缓存同步、PBR/UV 注入、体积蒸汽 GLSL 与防伪终端 `EIO` 秒断的高可用 WebSocket 服务。
4. **本地 Ollama `bonsai-2:27b` 异构双盲审计**：零 API 成本对《部品表 / 組立順序 / 試運転》与生成的 JSON/CAD 差异做语义一致性挑刺。
