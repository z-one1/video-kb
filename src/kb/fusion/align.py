"""时间戳对齐 — 字幕段 ↔ 视觉描述融合。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..schemas import EnrichedSegment, Transcript, VisualDescription

log = logging.getLogger("kb.fusion")


def align(
    transcript: Transcript,
    visuals: list[VisualDescription],
    mode: str = "cover",
) -> list[EnrichedSegment]:
    """把视觉描述合并进字幕段落。

    mode:
      - "cover": 字幕段 [start, end] 内所有视觉帧都附加
      - "nearest": 每个字幕段只取最近的 1 张帧
    """
    enriched: list[EnrichedSegment] = []

    if mode == "nearest" and visuals:
        visuals_sorted = sorted(visuals, key=lambda v: v.t_sec)

    for seg in transcript.segments:
        attached: list[VisualDescription] = []

        if mode == "cover":
            for v in visuals:
                if seg.start_sec <= v.t_sec <= seg.end_sec:
                    attached.append(v)
        elif mode == "nearest" and visuals:
            mid = (seg.start_sec + seg.end_sec) / 2
            closest = min(visuals, key=lambda v: abs(v.t_sec - mid))
            # 只在"合理距离内"挂(避免过远的帧污染)
            if abs(closest.t_sec - mid) <= max(30.0, (seg.end_sec - seg.start_sec) * 2):
                attached.append(closest)

        enriched.append(
            EnrichedSegment(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                text=seg.text,
                visual_descriptions=attached,
            )
        )

    log.info(
        f"Aligned {len(transcript.segments)} transcript segments with "
        f"{len(visuals)} visual frames (mode={mode})"
    )
    return enriched


def write_enriched_markdown(
    enriched: list[EnrichedSegment],
    out_path: Path | str,
    title: str = "",
) -> Path:
    """生成 enriched_transcript.md —— 字幕 + 视觉描述交错呈现。"""
    from ..utils import format_timestamp_short

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title} — Enriched Transcript\n\n")
        for seg in enriched:
            ts = format_timestamp_short(seg.start_sec)
            f.write(f"**[{ts}]** {seg.text}\n")
            for vd in seg.visual_descriptions:
                f.write(f"  - 🖼 *[{vd.frame_id}]* {vd.description}")
                if vd.extracted_text:
                    f.write(f"  \n    *Text on screen:* {vd.extracted_text}")
                f.write("\n")
            f.write("\n")
    return out_path


def dump_enriched_json(
    enriched: list[EnrichedSegment], out_path: Path | str
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [s.model_dump() for s in enriched], f, ensure_ascii=False, indent=2
        )
    return out_path
