"""通用工具:日志、路径、时间戳格式化。"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("kb")


def slug_video_id(source: str) -> str:
    """从 URL 或文件名生成稳定 video_id"""
    base = Path(source).stem if "/" in source or "\\" in source else source
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", base)[:60]
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def format_timestamp(seconds: float) -> str:
    """0.0 → '00:00:00', 3661.5 → '01:01:01'"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_short(seconds: float) -> str:
    """0.0 → '00:00', 125.5 → '02:05' (小于 1 小时时)"""
    seconds = int(seconds)
    if seconds >= 3600:
        return format_timestamp(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p: Path | str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_dir(kb_root: Path | str, video_id: str) -> Path:
    """返回 kb/videos/<video_id>/,并确保存在。"""
    d = Path(kb_root) / "videos" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug_doc_id(source_path: str, source_type: str) -> str:
    """从文档路径生成稳定 doc_id,带类型前缀 ('pdf_xxx' / 'img_xxx')。

    前缀用于和 video_id 隔离 — 同样的 basename 存为不同 source_type 不冲突。
    """
    p = Path(source_path)
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", p.stem)[:50]
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    prefix = {"pdf": "pdf", "image": "img"}.get(source_type, "doc")
    return f"{prefix}_{base}_{h}"


def doc_dir(kb_root: Path | str, doc_id: str) -> Path:
    """返回 kb/docs/<doc_id>/,并确保存在。"""
    d = Path(kb_root) / "docs" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d
