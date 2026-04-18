"""ffmpeg 音视频提取。"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("kb.ingest")


def probe_duration(video_path: Path | str) -> float:
    """用 ffprobe 获取视频时长(秒)"""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except Exception as e:
        log.warning(f"ffprobe failed: {e}")
        return 0.0


def extract_audio(
    video_path: Path | str,
    out_path: Path | str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """用 ffmpeg 抽取单声道 16kHz WAV — Whisper 最佳输入。"""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.info(f"Audio already extracted: {out_path}")
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    log.info(f"Extracting audio: {video_path.name} → {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    return out_path


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
