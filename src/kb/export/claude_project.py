"""把视频知识库打包成 Claude Project 可上传格式。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from ..schemas import Notes, VideoMeta
from ..utils import ensure_dir, format_timestamp_short

log = logging.getLogger("kb.export")


def export_for_claude_project(
    video_dir: Path | str,
    out_dir: Path | str,
    meta: VideoMeta,
    notes: Notes,
    include_frames: bool = True,
    max_frames: int = 30,
) -> Path:
    """为一个视频生成 Claude Project 上传包。

    输出:
      out_dir/<video_id>/
        ├── README.md               (一句话 + TOC,首个入口)
        ├── notes.md                (完整结构化笔记)
        ├── transcript.md           (带时间戳的字幕)
        ├── meta.yaml
        └── frames/                 (最多 max_frames 张,按时间戳分布)
    """
    video_dir = Path(video_dir)
    out_dir = ensure_dir(Path(out_dir) / meta.video_id)

    # 1. notes.md
    (out_dir / "notes.md").write_text(notes.full_markdown, encoding="utf-8")

    # 2. transcript.md (如果存在就拷)
    tx_src = video_dir / "transcript.md"
    if tx_src.exists():
        shutil.copy2(tx_src, out_dir / "transcript.md")

    # 3. meta.yaml
    meta_dict = meta.model_dump()
    with open(out_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta_dict, f, allow_unicode=True, sort_keys=False)

    # 4. frames —— 均匀下采样
    if include_frames:
        frames_src = video_dir / "frames"
        if frames_src.exists():
            frames_dst = ensure_dir(out_dir / "frames")
            all_frames = sorted(frames_src.glob("*.jpg"))
            if len(all_frames) > max_frames and max_frames > 0:
                step = len(all_frames) / max_frames
                selected = [all_frames[int(i * step)] for i in range(max_frames)]
            else:
                selected = all_frames
            for fp in selected:
                shutil.copy2(fp, frames_dst / fp.name)
            log.info(f"Copied {len(selected)} frames to export")

    # 5. README.md —— 入口文件
    readme = _build_readme(meta, notes)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    log.info(f"Exported Claude Project bundle: {out_dir}")
    return out_dir


def _build_readme(meta: VideoMeta, notes: Notes) -> str:
    lines = [
        f"# {notes.title or meta.title}",
        "",
        f"> {notes.one_liner}",
        "",
        "## 元数据",
        f"- **来源**: {meta.source}" + (f" ({meta.url})" if meta.url else ""),
        f"- **时长**: {meta.duration_sec:.0f}s",
        f"- **语言**: {meta.language or 'n/a'}",
        f"- **处理时间**: {meta.ingested_at}",
        "",
        notes.toc_markdown,
        "",
        "## 文件说明",
        "- `notes.md` — 结构化学习笔记 (章节 / 摘要 / 关键概念 / 复习问题)",
        "- `transcript.md` — 完整带时间戳字幕",
        "- `frames/` — 关键帧截图 (均匀采样)",
        "- `meta.yaml` — 视频元数据",
    ]
    return "\n".join(lines)
