"""通过 Claude Code CLI 做结构化 — 不花 API token,吃 Max 订阅额度。"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..schemas import EnrichedSegment, NoteSection, Notes
from .prompts import STRUCTURING_PROMPT, build_content_block

log = logging.getLogger("kb.structuring")


def _find_claude() -> str | None:
    return shutil.which("claude")


def is_available() -> bool:
    return _find_claude() is not None


def structure_notes(
    enriched: list[EnrichedSegment],
    video_id: str,
    cfg: dict[str, Any],
) -> Notes:
    """用 Claude Code CLI(非交互)生成结构化笔记。

    调用形式: `claude -p "<prompt>"` 或 `claude --print` 透传 stdin → stdout
    需要 user 已登录 `claude login`,额度从 Max 订阅扣除。
    """
    claude_bin = _find_claude()
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请安装 Claude Code(`npm install -g @anthropic-ai/claude-code`)并登录。"
        )

    content_block = build_content_block(enriched)
    max_chars = cfg.get("max_input_chars", 200000)
    if len(content_block) > max_chars:
        log.warning(
            f"Input too long ({len(content_block)} chars), truncating to {max_chars}"
        )
        content_block = content_block[:max_chars]

    prompt = STRUCTURING_PROMPT.replace("{content}", content_block)

    model = cfg.get("claude_model", "sonnet")
    cmd = [claude_bin, "-p", "--output-format", "text", "--model", model]
    log.info(
        f"Invoking Claude Code CLI ({model}, prompt ~{len(prompt) // 1000}k chars)..."
    )

    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}, stdout={proc.stdout[-500:]}"
        )

    raw = proc.stdout.strip()
    try:
        notes = _parse_notes_json(raw, video_id, enriched)
    except json.JSONDecodeError:
        # 保存原始输出供 debug,然后尝试一次修复后重解析
        import os, tempfile

        dump_dir = os.environ.get("KB_DEBUG_DUMP_DIR", tempfile.gettempdir())
        dump_path = Path(dump_dir) / f"notes_raw_{video_id}.txt"
        dump_path.write_text(raw, encoding="utf-8")
        log.error(f"原始 LLM 输出已保存到: {dump_path}")

        log.warning("尝试修复 JSON 并重解析...")
        fixed = _attempt_json_repair(raw)
        notes = _parse_notes_json(fixed, video_id, enriched)
    return notes


def _attempt_json_repair(raw: str) -> str:
    """最大努力修复常见 LLM JSON 输出错误:
    1. 剥离 ```json ... ``` 围栏
    2. 字符串值内未转义的 ASCII 双引号 → \\"
       启发式: 只处理"双引号在字符串值中间且前后有中文字符"的情况
    3. 单引号 → 双引号(谨慎,只处理键名)
    """
    s = raw.strip()

    # 1. 剥围栏
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()

    # 2. 修内部未转义双引号
    # 思路: 匹配 "key": "...<中文>"<词>"<中文>...",
    #       把 value 内部的 " 转义成 \"
    # 简单启发: 把 `非\的" 紧跟非 JSON 结构字符(如中文/字母/数字)` 且前面不是 `:` 或 `,` 或 `{` 的情况修复
    # 更稳的做法: 逐字符扫描,跟踪是否在字符串内部
    s = _escape_inner_quotes(s)
    return s


def _escape_inner_quotes(s: str) -> str:
    """逐字符扫描,把字符串值内部未转义的 ASCII 双引号转成 \\".

    状态机:
    - out_of_string: 找到 `"` → 进入字符串
    - in_string: 找到 `\\"` → 跳过;找到 `"` 且后面紧跟 JSON 结构符(,}]:\\s) → 退出字符串
                  否则这个 `"` 是字符串内未转义引号 → 替换成 \\"
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # in_string
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            # 看后面第一个非空白字符
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            next_ch = s[j] if j < n else ""
            if next_ch in ",}]:" or next_ch == "":
                # 合法的字符串结束
                out.append(ch)
                in_str = False
                i += 1
                continue
            # 否则是字符串内部未转义的引号,转义之
            out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_notes_json(
    raw: str, video_id: str, enriched: list[EnrichedSegment]
) -> Notes:
    """从 LLM 输出提取 JSON 并构建 Notes 对象。"""
    # 去掉可能的 ```json ... ``` 围栏
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"LLM 返回不是合法 JSON: {raw[:500]}")
        raise

    sections = [
        NoteSection(
            title=s["title"],
            start_sec=float(s.get("start_sec", 0)),
            end_sec=float(s.get("end_sec", 0)),
            summary=s.get("summary", ""),
            concepts=s.get("concepts", []),
            questions=s.get("questions", []),
        )
        for s in data.get("sections", [])
    ]

    notes = Notes(
        video_id=video_id,
        title=data.get("title", ""),
        one_liner=data.get("one_liner", ""),
        sections=sections,
    )
    notes.toc_markdown = _build_toc(notes)
    notes.full_markdown = _build_full_markdown(notes, enriched)
    return notes


def _build_toc(notes: Notes) -> str:
    from ..utils import format_timestamp_short

    lines = ["## 目录 (TOC)\n"]
    for i, sec in enumerate(notes.sections, 1):
        ts = format_timestamp_short(sec.start_sec)
        lines.append(f"{i}. **[{ts}]** {sec.title}")
    return "\n".join(lines)


def _build_full_markdown(notes: Notes, enriched: list[EnrichedSegment]) -> str:
    from ..utils import format_timestamp_short

    parts: list[str] = []
    parts.append(f"# {notes.title}\n")
    parts.append(f"> {notes.one_liner}\n")
    parts.append(notes.toc_markdown + "\n")

    for sec in notes.sections:
        ts = format_timestamp_short(sec.start_sec)
        ts_end = format_timestamp_short(sec.end_sec)
        parts.append(f"## {sec.title}  *([{ts} – {ts_end}])*\n")
        parts.append(sec.summary + "\n")

        if sec.concepts:
            parts.append(
                "**关键概念:** " + ", ".join(f"`{c}`" for c in sec.concepts) + "\n"
            )
        if sec.questions:
            parts.append("**复习提问:**")
            for q in sec.questions:
                parts.append(f"- {q}")
            parts.append("")

    return "\n".join(parts)
