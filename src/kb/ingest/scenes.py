"""关键帧抽取 — PySceneDetect 场景切换 + 固定间隔兜底。"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..schemas import KeyFrame
from ..utils import format_timestamp_short

log = logging.getLogger("kb.ingest")


def detect_scenes(
    video_path: Path | str,
    cfg: dict[str, Any],
    out_dir: Path | str,
) -> list[KeyFrame]:
    """检测场景切换 + 在每段取中间帧 + 稀疏段落用固定间隔补齐。

    返回 KeyFrame 列表(按时间排序,frame_id 唯一)。
    同时把 JPG 写入 out_dir。
    """
    try:
        from scenedetect import detect, ContentDetector, open_video
    except ImportError as e:
        raise ImportError("scenedetect 未安装 — pip install scenedetect[opencv]") from e

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = cfg.get("threshold", 27.0)
    min_scene_len = cfg.get("min_scene_len", 15)
    interval = cfg.get("fallback_interval_sec", 30)

    log.info(f"Detecting scenes (threshold={threshold}) ...")
    scenes = detect(
        str(video_path),
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
        show_progress=False,
    )

    # 从 scene 列表获取 (start_sec, end_sec) 对
    scene_ranges: list[tuple[float, float]] = []
    for s in scenes:
        start = s[0].get_seconds()
        end = s[1].get_seconds()
        scene_ranges.append((start, end))

    log.info(f"Found {len(scene_ranges)} scenes")

    # 每段取中间点
    scene_frames: list[tuple[float, float, float, str]] = []  # (t, start, end, source)
    for start, end in scene_ranges:
        mid = (start + end) / 2
        scene_frames.append((mid, start, end, "scene_change"))

    # 固定间隔兜底 — 始终跟 scene_change 取并集,保证稀疏段落也有覆盖
    # 跟之前"只在无场景时才兜底"的 bug 对比:现在即使检测到少量场景切换,
    # 兜底采样也会把未覆盖的时间段补齐(例如讲师静态讲解 20 分钟只出 3 个切换的情况)
    from .extractor import probe_duration

    duration = probe_duration(video_path)
    existing_times = [f[0] for f in scene_frames]
    near_threshold = interval / 4  # 兜底点跟已有 scene 相距小于这个值就跳过,防重复
    added_fallback = 0
    for t in range(int(interval / 2), int(duration), int(interval)):
        t_f = float(t)
        if any(abs(t_f - et) < near_threshold for et in existing_times):
            continue
        scene_frames.append((t_f, t_f, t_f + interval, "fixed_interval"))
        added_fallback += 1
    log.info(
        f"Fallback: added {added_fallback} fixed-interval frames "
        f"(every {interval}s over {duration:.0f}s, union with {len(scene_ranges)} scenes)"
    )

    # 按时间排序
    scene_frames.sort(key=lambda x: x[0])

    # 按时长动态计算 max_frames (取代旧的静态 200)
    max_frames = _compute_max_frames(duration, cfg)
    log.info(
        f"max_frames = {max_frames} "
        f"(duration={duration:.0f}s, rate={cfg.get('frames_per_minute', 5)}/min, "
        f"floor={cfg.get('max_frames_floor', 20)}, ceiling={cfg.get('max_frames_ceiling', 160)})"
    )

    # 帧数超限 → 均匀下采样
    if len(scene_frames) > max_frames:
        log.warning(
            f"Too many keyframes ({len(scene_frames)}), downsampling to {max_frames}"
        )
        step = len(scene_frames) / max_frames
        sampled = [scene_frames[int(i * step)] for i in range(max_frames)]
        scene_frames = sampled

    # 用 ffmpeg 批量抽帧
    frames: list[KeyFrame] = []
    for t, start, end, src in scene_frames:
        ts_label = format_timestamp_short(t).replace(":", "-")
        img_name = f"{ts_label}.jpg"
        img_path = out_dir / img_name
        if not img_path.exists():
            _extract_frame(video_path, t, img_path)

        frames.append(
            KeyFrame(
                frame_id=ts_label,
                t_sec=t,
                image_path=str(img_path.relative_to(out_dir.parent)),
                scene_start_sec=start,
                scene_end_sec=end,
                source=src,  # type: ignore[arg-type]
            )
        )

    # 保存 manifest
    manifest_path = out_dir.parent / "frames_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([fr.model_dump() for fr in frames], f, ensure_ascii=False, indent=2)

    log.info(f"Extracted {len(frames)} keyframes → {out_dir}")
    return frames


def _compute_max_frames(duration_sec: float, cfg: dict[str, Any]) -> int:
    """按视频时长计算 max_frames,公式:
        clamp(round(duration_min * frames_per_minute), floor, ceiling)

    向后兼容:如果 cfg 里有老字段 `max_frames`,优先用那个(静态 cap)。
    """
    # 向后兼容:老 config 里显式给了 max_frames 就直接用
    legacy = cfg.get("max_frames")
    if legacy is not None:
        return int(legacy)

    fpm = cfg.get("frames_per_minute", 5)
    floor = cfg.get("max_frames_floor", 20)
    ceiling = cfg.get("max_frames_ceiling", 160)
    duration_min = max(duration_sec, 0.0) / 60.0
    target = round(duration_min * fpm)
    return max(floor, min(ceiling, target))


def _extract_frame(video_path: Path, t_sec: float, out_path: Path) -> None:
    """用 ffmpeg 在指定时间抽一帧 JPG。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(t_sec),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(f"ffmpeg frame extract failed at {t_sec}s: {result.stderr[-200:]}")


def load_frames_manifest(video_id_dir: Path | str) -> list[KeyFrame]:
    """从 frames_manifest.json 重新加载"""
    manifest = Path(video_id_dir) / "frames_manifest.json"
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as f:
        return [KeyFrame(**d) for d in json.load(f)]
