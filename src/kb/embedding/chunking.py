"""把 Notes + enriched transcript 切成可嵌入的块。"""
from __future__ import annotations

import logging
from typing import Any

from ..schemas import Chunk, EnrichedSegment, Notes

log = logging.getLogger("kb.embedding")


def chunk_notes(
    notes: Notes,
    enriched: list[EnrichedSegment],
    video_id: str,
    cfg: dict[str, Any],
) -> list[Chunk]:
    """以章节为边界切块,章节内再按 chunk_size 字符切。

    每块带 metadata: section_title / start_sec / end_sec / has_visual
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise ImportError("langchain-text-splitters 未安装") from e

    chunk_size = cfg.get("chunk_size", 500)
    chunk_overlap = cfg.get("chunk_overlap", 50)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "!", "?", " ", ""],
    )

    chunks: list[Chunk] = []
    counter = 0

    # 先切 section summary — 这些是 LLM 结构化的高密度内容
    for sec in notes.sections:
        # summary + 关键概念合成一段
        sec_text = (
            f"【章节】{sec.title}\n"
            f"{sec.summary}\n"
            + (f"关键概念: {', '.join(sec.concepts)}\n" if sec.concepts else "")
        )
        for piece in splitter.split_text(sec_text):
            # 判断该章节覆盖的时间段内是否有视觉
            has_vis = any(
                seg.visual_descriptions
                and sec.start_sec <= seg.start_sec <= sec.end_sec
                for seg in enriched
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{video_id}_sec_{counter:04d}",
                    video_id=video_id,
                    text=piece,
                    start_sec=sec.start_sec,
                    end_sec=sec.end_sec,
                    section_title=sec.title,
                    has_visual=has_vis,
                )
            )
            counter += 1

    # 再切 enriched transcript 的原始字幕 — 提供细粒度检索能力
    # 按 ~90 秒滑窗聚合 segments,保留时间戳
    window_sec = cfg.get("transcript_window_sec", 90)
    current_texts: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0
    current_has_vis = False

    def flush():
        nonlocal counter, current_texts, current_start, current_has_vis
        if not current_texts or current_start is None:
            return
        block = " ".join(current_texts)
        for piece in splitter.split_text(block):
            # 找包含在当前窗口的章节标题
            sec_title = _find_section_title(notes, current_start, current_end)
            chunks.append(
                Chunk(
                    chunk_id=f"{video_id}_tx_{counter:04d}",
                    video_id=video_id,
                    text=piece,
                    start_sec=current_start,
                    end_sec=current_end,
                    section_title=sec_title,
                    has_visual=current_has_vis,
                )
            )
            counter += 1
        current_texts = []
        current_start = None
        current_has_vis = False

    for seg in enriched:
        if current_start is None:
            current_start = seg.start_sec
        current_end = seg.end_sec
        current_texts.append(seg.text)
        if seg.visual_descriptions:
            current_has_vis = True
            # 把视觉描述也并入文本流
            for vd in seg.visual_descriptions:
                current_texts.append(f"[画面] {vd.description}")
                if vd.extracted_text:
                    current_texts.append(f"[屏幕文字] {vd.extracted_text}")

        if (current_end - current_start) >= window_sec:
            flush()
    flush()

    log.info(f"Produced {len(chunks)} chunks for {video_id}")
    return chunks


def _find_section_title(
    notes: Notes, start_sec: float, end_sec: float
) -> str | None:
    mid = (start_sec + end_sec) / 2
    for sec in notes.sections:
        if sec.start_sec <= mid <= sec.end_sec:
            return sec.title
    return None
