#!/usr/bin/env python3
"""
PreToolUse lifecycle guardrail for live-steam-twin:
  1. Blocks direct read_file/view_file on raw PDFs (>1.5 MB or data/raw_pdfs/*.pdf) and huge uncropped scan PNGs (>1.5 MB in ortho_views/), instructing the agent to run a local Python crop/text script instead (prevents Vision Token Snowball).
  2. Blocks introducing the Mistake-27 photogrammetry anti-pattern (cv2.boundingRect + acos/arccos without coplanar rim-gradient normalization) in Python calibration scripts.
Always fails open ({"decision": "allow"}) on unrelated files or unexpected payloads.
"""
import json
import os
from pathlib import Path
import re
import sys


def allow() -> None:
    sys.stdout.write(json.dumps({"decision": "allow"}))
    sys.exit(0)


def deny(reason: str) -> None:
    sys.stdout.write(json.dumps({"decision": "deny", "reason": reason}, ensure_ascii=False))
    sys.exit(0)


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            allow()
        payload = json.loads(raw)
        tc = payload.get("toolCall") or {}
        name = str(tc.get("name") or "").lower()
        args = tc.get("args") or {}

        # 1. Guard against direct raw PDF / giant full-page ortho scan reads
        if "read" in name or "view" in name:
            target = str(args.get("Path") or args.get("AbsolutePath") or args.get("file_path") or "")
            if target:
                p = Path(target)
                low = target.lower()
                if low.endswith(".pdf"):
                    if "raw_pdfs" in low or (p.exists() and p.stat().st_size > 1_500_000):
                        deny(
                            "[live-steam-twin Guardrail] 禁止直接读取体积较大的原始扫描 PDF！"
                            "请先使用本地 Python 脚本 (pymupdf + cv2) 提取纯文本或裁剪 <=1280px 局部低分辨率对比条后再单次查看。"
                        )
                if low.endswith((".png", ".jpg", ".jpeg")) and "ortho_views" in low:
                    if p.exists() and p.stat().st_size > 1_500_000:
                        deny(
                            "[live-steam-twin Guardrail] 禁止直接读取 data/ortho_views/ 下的整页高分辨率大图 (>1.5MB)！"
                            "请先用 Python 脚本裁剪局部目标 ROI 并降采样至 <=1280px 后再单次查看。"
                        )

        # 2. Guard against Mistake 27 (threshold boundingRect + arccos tilt estimation)
        if any(k in name for k in ("write", "edit", "replace")):
            target = str(args.get("Path") or args.get("TargetFile") or "")
            code_blob = str(args.get("Content") or args.get("Replacement") or args.get("ReplacementContent") or "")
            if target.endswith(".py") and code_blob:
                has_bbox = "boundingRect" in code_blob or "connectedComponentsWithStats" in code_blob
                has_acos = bool(re.search(r"\b(math\.acos|np\.arccos)\b", code_blob))
                has_ref_norm = "measure_circle_rim_aspect" in code_blob or "coplanar_ref" in code_blob
                if has_bbox and has_acos and not has_ref_norm:
                    deny(
                        "[live-steam-twin Mistake-27 Guardrail] 检测到在 Python 脚本中同时使用阈值外接框 (boundingRect/connectedComponents) 与 acos/arccos 反推倾角！"
                        "表盘/法兰顶部阴影截断与相机俯拍外参会导致严重的虚假仰角（如垂直 PG-ø20 被误算为 24°）。"
                        "请激活 live-steam-01-pdf-calibration 技能并改用 cv_rim_calibrator.measure_circle_rim_aspect + 同平面垂直基准件差分消元法。"
                    )
    except Exception:
        pass
    allow()


if __name__ == "__main__":
    main()
