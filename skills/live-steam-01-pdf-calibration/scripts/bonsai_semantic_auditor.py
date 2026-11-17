#!/usr/bin/env python3
"""
Zero-cost Independent Semantic Cross-Auditor powered by local Ollama (default: bonsai-2:27b).
Cross-checks extracted locomotive BOM/assembly manual text, JSON piping/kinematics specs, and git diffs
against the 27 historical Live-Steam Digital Twin failure modes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

AUDIT_SYSTEM_PROMPT = """你是真蒸汽机车三维多体与热流数字孪生项目的独立总工审计员。
请严格对照以下 8 条历史高发致命失误清单，对用户提供的【原厂说明书/部品表摘录】与【当前工程 JSON 配置 / 代码变更摘要】进行逐项挑刺核查：
1. [假件误通水/误加关节 (失误13/17/26)]：带 ダミー (Dummy) 字样的装饰阀/假汽笛/静态假人孔盖 (如 15-3, 16-2, 30-2, 35-6) 是否被错误接入了流体回路或添加了可动关节？
2. [左/右驾与正交旋向颠倒 (失误13/16/26)]：日本国铁左驾 (+Y) vs 右驾 (-Y) 是否放反？床下排水针阀、水面计、逆止阀、汽笛阀的朝向 (+Y/-Y/-X/-Z) 是否与多张组装图一致？
3. [单图阴影/相机俯角误算仪表仰角 (失误27)]：压力表/圆法兰倾角是否经过了同平面垂直基准件差分校正（严禁直接用阈值白斑 h/w 算 arccos 把垂直表盘误算为 24°/38° 仰角）？
4. [烟室前后与高低管路层级颠倒 (失误22/26)]：排汽喷管、给汽横歧管、过热器水平直插段、通风 J 型弯管、注油毛细管侧偏绕行及集烟筒直径是否零碰撞且前后层级正确？
5. [气缸与平面滑阀同轴真值 (失误15/26)]：阀芯棒、滑阀框、平面 D 阀与前后护套是否 100% 同轴（严禁捏造横向狗腿折弯、假跨桥连杆或虚构活塞前伸动尻棒）？
6. [操纵机构有头无尾或穿锅筒 (失误14/22/26)]：逆转横轴是否按子剖面从锅底绕行而非刺穿紫铜水空间？长拉杆是扁钢拱跃还是圆棒？手摇泵/丝杠滑块/排液拉杆是否还原了真实四连杆或滑块联动？
7. [跨车软管根数、材质与双泵共用拓扑 (失误13)]：机煤间软管精确根数、硅胶(ST)/氟胶(BT)材质与手摇泵/轴动泵串联止回球共用逻辑是否与《試運転》正文一致？
8. [煤水车固定隔板 vs 可拆盖板边界 (失误17/26)]：固定焊死的挡煤板与中间可提离水箱盖宽度是否区分清楚？醇箱与水箱是否物理隔离？

输出格式要求：
- 若发现疑似矛盾或漏项，按 `[RISK-编号] 零件号: 具体矛盾点与建议复核的图纸页码` 逐条列出；
- 若全部核对无矛盾，输出 `[AUDIT-PASS] 未发现与原厂文档矛盾的历史失误模式`。保持简练客观。"""


def query_ollama(model: str, url: str, user_content: str, timeout_s: float = 120.0) -> str:
    chat_url = url.replace("/api/generate", "/api/chat") if url.endswith("/api/generate") else url
    final_user_msg = (
        f"{user_content}\n\n"
        "=== 立即执行上述 8 项真蒸汽机车历史失误清单逐条核查（禁止反问需求，直接给出逐项 PASS/RISK 审计结论） ==="
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": final_user_msg},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": 900},
    }
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        msg = body.get("message") or {}
        return str(msg.get("content") or body.get("response") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local Ollama bonsai-2:27b semantic cross-audit against 27 lessons learned")
    ap.add_argument("--model", default="bonsai-2:27b", help="Local Ollama model tag (default: bonsai-2:27b)")
    ap.add_argument("--url", default="http://127.0.0.1:11434/api/generate", help="Local Ollama generate endpoint")
    ap.add_argument("--manual-txt", type=Path, help="Extracted BOM / manual assembly text file")
    ap.add_argument("--spec-json", type=Path, help="Generated kinematics or boiler_piping JSON file")
    ap.add_argument("--git-diff", action="store_true", help="Include current git diff --stat and key Python/JSON hunks")
    args = ap.parse_args()

    sections = []
    if args.manual_txt and args.manual_txt.exists():
        sections.append(f"=== 原厂说明书 / BOM 提取文本 ({args.manual_txt.name}) ===\n" + args.manual_txt.read_text(encoding="utf-8")[:12000])
    if args.spec_json and args.spec_json.exists():
        raw_spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
        compact_spec = json.dumps(raw_spec, ensure_ascii=False, separators=(",", ":"))
        sections.append(f"=== 当前工程结构化真值配置 ({args.spec_json.name}) ===\n" + compact_spec[:24000])
    if args.git_diff:
        diff_txt = subprocess.run(["git", "diff", "--unified=2", "--", "*.py", "*.json"], capture_output=True, text=True).stdout
        if diff_txt:
            sections.append("=== 当前代码/配置变更 Diff ===\n" + diff_txt[:10000])

    if not sections:
        print("Usage: provide at least one of --manual-txt, --spec-json, or --git-diff", file=sys.stderr)
        raise SystemExit(2)

    try:
        report = query_ollama(args.model, args.url, "\n\n".join(sections))
        print(f"================ [{args.model} 独立语义交叉审计报告] ================\n{report}\n" + "=" * 68)
    except urllib.error.URLError as exc:
        print(f"[WARN] 本地 Ollama ({args.url}) 未启动或不可达: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
