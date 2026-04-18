"""视频获取 — yt-dlp 包装 + 本地文件注册。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..schemas import VideoMeta
from ..utils import ensure_dir, now_iso, slug_video_id, video_dir

log = logging.getLogger("kb.ingest")


def register_local_video(
    source_path: Path | str,
    kb_root: Path | str,
    video_id: str | None = None,
    title: str | None = None,
) -> tuple[VideoMeta, Path]:
    """把本地视频登记到 kb/ 目录,返回 (meta, 工作目录内的视频副本路径)。

    不复制原视频(节省磁盘),而是保存绝对路径引用。
    """
    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")

    vid = video_id or slug_video_id(src.name)
    vdir = video_dir(kb_root, vid)

    # 软链接(或 Windows 下拷贝)到工作目录,方便后续管线读取固定路径
    symlink_path = vdir / f"video{src.suffix}"
    if not symlink_path.exists():
        try:
            symlink_path.symlink_to(src)
        except (OSError, NotImplementedError):
            # symlink 失败(Windows 或权限问题),退化为 copy
            shutil.copy2(src, symlink_path)

    meta = VideoMeta(
        video_id=vid,
        source="local",
        source_path=str(src),
        title=title or src.stem,
        ingested_at=now_iso(),
    )
    log.info(f"Registered local video: {vid} ({src.name})")
    return meta, symlink_path


def download_youtube(
    url: str,
    kb_root: Path | str,
    video_id: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[VideoMeta, Path]:
    """yt-dlp 下载 YouTube / B 站等站点,返回 (meta, 视频文件路径)。"""
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError("yt-dlp 未安装 — pip install yt-dlp") from e

    cfg = cfg or {}
    fmt = cfg.get("format", "bestvideo[height<=720]+bestaudio/best[height<=720]")
    sub_langs = cfg.get("subtitle_langs", ["en", "zh"])
    write_auto = cfg.get("write_auto_subs", True)

    # 先用 yt-dlp 获取元信息 → 确定 video_id
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    yt_id = info.get("id", "")
    title = info.get("title", "")
    duration = float(info.get("duration", 0) or 0)

    vid = video_id or slug_video_id(f"yt_{yt_id}")
    vdir = ensure_dir(Path(kb_root) / "videos" / vid)

    out_tpl = str(vdir / "video.%(ext)s")
    ydl_opts = {
        "format": fmt,
        "outtmpl": out_tpl,
        "writesubtitles": True,
        "writeautomaticsub": write_auto,
        "subtitleslangs": sub_langs,
        "subtitlesformat": "srt/vtt/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": True,
    }
    log.info(f"Downloading YouTube: {url} → {vdir}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 查找下载下来的主文件
    candidates = list(vdir.glob("video.*"))
    video_files = [p for p in candidates if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")]
    if not video_files:
        raise RuntimeError(f"yt-dlp 未产出视频文件,只找到: {candidates}")
    video_path = video_files[0]

    meta = VideoMeta(
        video_id=vid,
        source="youtube",
        url=url,
        source_path=None,
        title=title,
        duration_sec=duration,
        ingested_at=now_iso(),
    )
    log.info(f"Downloaded: {vid} ({title}, {duration:.0f}s)")
    return meta, video_path
