"""把 Notes + enriched transcript 切成可嵌入的块。"""
from __future__ import annotations

import logging
from typing import Any

from ..schemas import Chunk, EnrichedSegment, Notes
from .splitter import recursive_char_split

log = logging.getLogger("kb.embedding")

_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "!", "?", " ", ""]


def chunk_notes(
    notes: Notes,
    enriched: list[EnrichedSegment],
    video_id: str,
    cfg: dict[str, Any],
) -> list[Chunk]:
    """以章节为边界切块,章节内再按 chunk_size 字符切。

    每块带 metadata: section_title / start_sec / end_sec / has_visual
    """
    chunk_size = cfg.get("chunk_size", 500)
    chunk_overlap = cfg.get("chunk_overlap", 50)

    def split(text: str) -> list[str]:
        return recursive_char_split(
            text, chunk_size, chunk_overlap, _DEFAULT_SEPARATORS
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
        for piece in split(sec_text):
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
    # 视觉描述裁剪:防止单窗口被几十条高密度帧描述灌爆 → chunk 数量爆炸
    max_visuals_per_window = cfg.get("max_visuals_per_window", 3)
    max_visual_chars = cfg.get("max_visual_chars", 200)

    current_texts: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0
    current_has_vis = False
    current_visual_count = 0  # 本窗口已并入的视觉描述条数

    def flush():
        nonlocal counter, current_texts, current_start, current_has_vis, current_visual_count
        if not current_texts or current_start is None:
            return
        block = " ".join(current_texts)
        for piece in split(block):
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
        current_visual_count = 0

    for seg in enriched:
        if current_start is None:
            current_start = seg.start_sec
        current_end = seg.end_sec
        current_texts.append(seg.text)
        if seg.visual_descriptions:
            current_has_vis = True
            # 把视觉描述也并入文本流 — 但每窗口最多 N 条,每条不超过 M 字符
            for vd in seg.visual_descriptions:
                if current_visual_count >= max_visuals_per_window:
                    break
                desc = (vd.description or "")[:max_visual_chars]
                if desc:
                    current_texts.append(f"[画面] {desc}")
                    current_visual_count += 1
                if vd.extracted_text:
                    extracted = vd.extracted_text[:max_visual_chars]
                    current_texts.append(f"[屏幕文字] {extracted}")

        if (current_end - current_start) >= window_sec:
            flush()
    flush()

    # 后处理:合并过短的碎片 chunk(splitter 在标点处切出的尾巴)
    min_chunk_chars = cfg.get("min_chunk_chars", 120)
    if min_chunk_chars > 0:
        before = len(chunks)
        chunks = _merge_short_chunks(chunks, min_chunk_chars)
        if before != len(chunks):
            log.info(
                f"Merged short chunks: {before} → {len(chunks)} "
                f"(min_chars={min_chunk_chars})"
            )

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


def _chunk_kind(chunk_id: str) -> str:
    """从 chunk_id 里取类型标记:'sec' 或 'tx'(例:abc_sec_0003 → 'sec')。"""
    parts = chunk_id.rsplit("_", 2)
    return parts[-2] if len(parts) >= 2 else ""


def _merge_short_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """把文本短于 min_chars 的 chunk 合并到同类型 + 同章节的前一块。

    - 只在同 kind(sec/tx)+ 同 section_title 内合并,避免跨边界污染
    - 找不到可合并的前块就保留原样(不强合,避免把孤立短块硬拼到陌生章节)
    - 时间戳取合并后的 start/end 外延,has_visual 取或
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    for cur in chunks:
        if len(cur.text) >= min_chars or not merged:
            merged.append(cur)
            continue

        prev = merged[-1]
        same_kind = _chunk_kind(prev.chunk_id) == _chunk_kind(cur.chunk_id)
        same_sec = (prev.section_title or "") == (cur.section_title or "")
        if not (same_kind and same_sec):
            # 边界不同,不合并
            merged.append(cur)
            continue

        # 合并到 prev
        prev.text = prev.text.rstrip() + " " + cur.text.lstrip()
        prev.end_sec = max(prev.end_sec or 0.0, cur.end_sec or 0.0)
        if cur.start_sec is not None:
            prev.start_sec = min(
                prev.start_sec if prev.start_sec is not None else cur.start_sec,
                cur.start_sec,
            )
        prev.has_visual = bool(prev.has_visual) or bool(cur.has_visual)

    return merged
