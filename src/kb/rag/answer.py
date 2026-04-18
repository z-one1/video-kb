"""RAG 答题 — 检索 Top-K + Claude Code CLI 生成带引用答案。

设计原则:
- 复用 storage.chroma_client.query(不改动检索层)
- 复用 structuring 里的 Claude CLI 调用模式(subprocess + --print)
- prompt 里把 chunks 格式化成带 [ep.N @ mm:ss] 引用标记的块,强制 LLM 保留引用
- 关键帧路径以"参考关键帧"列表放在答案末尾,用户点击可看原图
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..storage import chroma_client
from ..utils import format_timestamp_short

log = logging.getLogger("kb.rag")


ANSWER_PROMPT = """你是一个基于视频知识库的答题助手。用户提了一个问题,我从向量库里检索出了 Top-<<K>> 相关片段。

请基于这些片段综合出一个准确、简洁、有引用的答案。严格遵守以下规则:

1. **只使用下方提供的片段内容**,不要编造。片段里没说的,明确说"未在知识库中找到"。
2. **每个事实都必须带引用**,格式: `[ep.N @ mm:ss]`,例如 `[ep.3 @ 07:10]`。引用要紧贴具体陈述的末尾,不要集中在答案末尾。
3. **答案用中文写**,但保留讲师原话里的英文术语(如 CRT / Turtle Soup / Model #1 / Kiss of Death Turtle Soup)。
4. 如果多集都讲同一个概念,**对比指出他们的差异或递进关系**,不要简单罗列。
5. 如果某些片段明显是噪声(纯价位数字、单独字母、列表编号),**忽略它**,不要硬凑。
6. 开头不要说"根据检索到的片段"这种废话,直接答。
7. 答案结尾列出"**📎 参考关键帧:**",把所有引用涉及的关键帧路径列出(每行一个),方便用户查原图。

用户问题:<<QUESTION>>

--- 检索到的 <<K>> 个片段 ---
<<CHUNKS>>
--- 片段结束 ---

请开始回答。"""


_EP_RE = re.compile(r"ep[_\.\s]*(\d+)", re.IGNORECASE)


def _extract_ep_tag(video_id: str, video_title: str) -> str:
    """从 video_id 或 video_title 抽取 'ep.N',失败则返回 title 前缀。"""
    for s in (video_id, video_title):
        if not s:
            continue
        m = _EP_RE.search(s)
        if m:
            return f"ep.{m.group(1)}"
    # 兜底
    return (video_title or video_id or "video")[:30]


def _frame_path_for(video_id: str, start_sec: float, videos_dir: Path) -> Path | None:
    """推算这个 chunk 对应的 keyframe JPG 路径(基于 format_timestamp_short 命名约定)。"""
    if not video_id:
        return None
    ts_label = format_timestamp_short(start_sec).replace(":", "-")
    candidate = videos_dir / video_id / "frames" / f"{ts_label}.jpg"
    return candidate if candidate.exists() else None


def build_chunk_block(
    hits: list[dict[str, Any]], videos_dir: Path
) -> tuple[str, list[Path]]:
    """渲染 Top-K chunks 为 prompt 块;同时收集对应的关键帧路径列表。

    Returns:
        (chunk_block_str, list_of_frame_paths_for_answer_footer)
    """
    lines: list[str] = []
    frame_paths: list[Path] = []
    for i, h in enumerate(hits, 1):
        md = h["metadata"]
        video_id = md.get("video_id", "")
        video_title = md.get("video_title", "")
        ep_tag = _extract_ep_tag(video_id, video_title)
        start = float(md.get("start_sec", 0))
        ts = format_timestamp_short(start)
        section = md.get("section_title", "") or "(无章节)"
        has_visual = md.get("has_visual", False)
        visual_icon = " 📷" if has_visual else ""
        distance = h.get("distance")
        dist_str = f"{distance:.3f}" if distance is not None else "-"

        header = (
            f"【Chunk #{i}】 [{ep_tag} @ {ts}]{visual_icon}  "
            f"section={section}  dist={dist_str}"
        )
        lines.append(header)
        lines.append(h["text"].strip())

        if has_visual:
            fp = _frame_path_for(video_id, start, videos_dir)
            if fp:
                lines.append(f"(🖼 对应关键帧: {fp})")
                frame_paths.append(fp)
        lines.append("")  # 空行分隔

    return "\n".join(lines), frame_paths


def answer(
    question: str,
    cfg: dict[str, Any],
    n_results: int = 8,
    video_id: str | None = None,
) -> dict[str, Any]:
    """检索 + Claude CLI 生成答案。

    Args:
        question: 用户自然语言提问
        cfg: 从 load_config() 拿到的配置
        n_results: 检索 Top-K(默认 8,实测 8 覆盖大多数场景)
        video_id: 仅检索指定视频(可选)

    Returns:
        {
            "answer": str,          # LLM 生成的带引用答案
            "hits": list,            # 检索到的原始 chunks(方便 CLI 二次展示)
            "frame_paths": list,     # 引用涉及的关键帧文件路径
            "prompt_chars": int,     # prompt 体积(便于监控)
        }
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请安装 Claude Code"
            "(`npm install -g @anthropic-ai/claude-code`)并登录。"
        )

    # 1. 检索
    where = {"video_id": video_id} if video_id else None
    hits = chroma_client.query(
        question,
        cfg["paths"]["chroma_dir"],
        cfg["embedding"],
        n_results=n_results,
        where=where,
    )

    if not hits:
        return {
            "answer": "知识库中没有相关内容。",
            "hits": [],
            "frame_paths": [],
            "prompt_chars": 0,
        }

    # 2. 构造 prompt
    videos_dir = Path(cfg["paths"]["videos_dir"])
    chunk_block, frame_paths = build_chunk_block(hits, videos_dir)
    # 用占位符替换,避免 str.format 被 question/chunks 里的 { } 搞崩
    prompt = (
        ANSWER_PROMPT.replace("<<K>>", str(len(hits)))
        .replace("<<QUESTION>>", question)
        .replace("<<CHUNKS>>", chunk_block)
    )

    # 3. Claude CLI 生成(复用 structuring 的调用模式)
    ask_cfg = cfg.get("ask", {})
    # 优先 ask.claude_model,回退到 structuring.claude_model,最后默认 sonnet
    model = ask_cfg.get("claude_model") or cfg.get("structuring", {}).get(
        "claude_model", "sonnet"
    )
    timeout = ask_cfg.get("timeout_sec", 180)
    log.info(
        f"Asking Claude ({model}, prompt ~{len(prompt) // 1000}k chars, "
        f"{len(hits)} chunks, {len(frame_paths)} frame refs)..."
    )

    proc = subprocess.run(
        [claude_bin, "-p", "--output-format", "text", "--model", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}"
        )

    return {
        "answer": proc.stdout.strip(),
        "hits": hits,
        "frame_paths": frame_paths,
        "prompt_chars": len(prompt),
    }
