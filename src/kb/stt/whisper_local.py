"""faster-whisper 本地转写。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..schemas import Transcript, TranscriptSegment

log = logging.getLogger("kb.stt")


def transcribe(
    audio_path: Path | str,
    video_id: str,
    cfg: dict[str, Any],
    out_json_path: Path | str | None = None,
) -> Transcript:
    """用 faster-whisper 转写音频文件。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError("faster-whisper 未安装 — pip install faster-whisper") from e

    audio_path = Path(audio_path)
    model_size = cfg.get("model_size", "large-v3-turbo")
    device = cfg.get("device", "cpu")
    compute_type = cfg.get("compute_type", "int8")
    language = cfg.get("language")  # None = 自动检测
    vad = cfg.get("vad_filter", True)
    beam = cfg.get("beam_size", 5)
    word_ts = cfg.get("word_timestamps", False)

    log.info(f"Loading Whisper model: {model_size} on {device}/{compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    log.info(f"Transcribing: {audio_path.name} (vad={vad}, beam={beam})")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad,
        beam_size=beam,
        word_timestamps=word_ts,
    )

    detected_lang = info.language
    duration = info.duration
    log.info(f"Detected language: {detected_lang}, duration: {duration:.1f}s")

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        words = []
        if word_ts and seg.words:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in seg.words
            ]
        segments.append(
            TranscriptSegment(
                start_sec=seg.start,
                end_sec=seg.end,
                text=seg.text.strip(),
                words=words,
            )
        )
        # 滚动打印进度
        if len(segments) % 20 == 0:
            pct = (seg.end / duration * 100) if duration else 0
            log.info(f"  ...{len(segments)} segments ({pct:.0f}%)")

    transcript = Transcript(
        video_id=video_id,
        language=detected_lang,
        duration_sec=duration,
        segments=segments,
    )

    if out_json_path:
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)
        log.info(f"Saved transcript JSON: {out_json_path}")

    return transcript


def write_srt(transcript: Transcript, srt_path: Path | str) -> Path:
    """把 Transcript 写成 .srt 字幕文件"""

    def fmt(t: float) -> str:
        # SRT: 00:00:01,000
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript.segments, 1):
            f.write(f"{i}\n{fmt(seg.start_sec)} --> {fmt(seg.end_sec)}\n{seg.text}\n\n")
    return srt_path


def write_transcript_md(transcript: Transcript, md_path: Path | str) -> Path:
    """人类可读的 markdown 版本,带时间戳。"""
    from ..utils import format_timestamp_short

    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript (language: {transcript.language}, "
                f"duration: {transcript.duration_sec:.0f}s)\n\n")
        for seg in transcript.segments:
            ts = format_timestamp_short(seg.start_sec)
            f.write(f"**[{ts}]** {seg.text}\n\n")
    return md_path
