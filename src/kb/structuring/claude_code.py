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
from ..utils import repair_llm_json
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
        fixed = repair_llm_json(raw)
        notes = _parse_notes_json(fixed, video_id, enriched)
    return notes


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
