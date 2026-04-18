"""端到端 pipeline — 把各阶段串起来。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .config import load_config, get_env
from .embedding import bge, chunking
from .export import claude_project
from .fusion import align as fusion_align
from .ingest import downloader, extractor, scenes
from .schemas import Notes, Transcript, VideoMeta
from .storage import chroma_client
from .stt import whisper_local
from .structuring import claude_code, gemini_fallback
from .utils import ensure_dir, setup_logging, video_dir
from .vision import claude_code as vision_claude
from .vision import gemini as vision_gemini

log = logging.getLogger("kb.pipeline")


def _save_meta(meta: VideoMeta, vdir: Path) -> None:
    with open(vdir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta.model_dump(), f, allow_unicode=True, sort_keys=False)


def _load_meta(vdir: Path) -> VideoMeta | None:
    p = vdir / "meta.yaml"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return VideoMeta(**yaml.safe_load(f))


def ingest(
    source: str,
    cfg_path: Path | str | None = None,
    skip_vision: bool = False,
    skip_structure: bool = False,
    skip_embedding: bool = False,
    skip_export: bool = False,
    force: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """完整 pipeline:source 可以是 YouTube URL 或本地文件路径。"""
    setup_logging(log_level)
    cfg = load_config(cfg_path)
    kb_root = Path(cfg["paths"]["kb_root"])
    chroma_path = Path(cfg["paths"]["chroma_dir"])
    ensure_dir(kb_root)
    ensure_dir(chroma_path)

    # ============ 1. 获取 ============
    log.info("=" * 60)
    log.info(f"Stage 1: Ingest — {source}")
    log.info("=" * 60)

    is_url = source.startswith(("http://", "https://"))
    if is_url:
        meta, video_file = downloader.download_youtube(source, kb_root, cfg=cfg["ingest"])
    else:
        meta, video_file = downloader.register_local_video(source, kb_root)

    vdir = video_dir(kb_root, meta.video_id)

    # 填充 duration if missing
    if meta.duration_sec == 0:
        meta.duration_sec = extractor.probe_duration(video_file)

    _save_meta(meta, vdir)

    # ============ 2. 抽音频 + STT ============
    log.info("=" * 60)
    log.info("Stage 2: Audio extraction + transcription")
    log.info("=" * 60)

    audio_path = vdir / "audio.wav"
    extractor.extract_audio(video_file, audio_path)

    transcript_json = vdir / "transcript.json"
    if transcript_json.exists() and not force:
        log.info(f"Transcript exists, loading cached: {transcript_json}")
        with open(transcript_json, encoding="utf-8") as f:
            transcript = Transcript(**json.load(f))
    else:
        transcript = whisper_local.transcribe(
            audio_path, meta.video_id, cfg["stt"], out_json_path=transcript_json
        )

    whisper_local.write_srt(transcript, vdir / "transcript.srt")
    whisper_local.write_transcript_md(transcript, vdir / "transcript.md")
    meta.language = transcript.language
    meta.has_transcript = True
    _save_meta(meta, vdir)

    # ============ 3. 关键帧 + 视觉理解 ============
    visuals: list = []
    if not skip_vision:
        log.info("=" * 60)
        log.info("Stage 3: Keyframe extraction + vision")
        log.info("=" * 60)

        frames_dir = ensure_dir(vdir / "frames")
        frames_manifest = vdir / "frames_manifest.json"
        if frames_manifest.exists() and not force:
            log.info("Using cached frames manifest")
            frames = scenes.load_frames_manifest(vdir)
        else:
            frames = scenes.detect_scenes(video_file, cfg["scenes"], frames_dir)
        meta.has_frames = True

        # Vision — 根据 provider 分发
        vision_json = vdir / "visuals.json"
        provider = cfg["vision"].get("provider", "gemini")
        if vision_json.exists() and not force:
            log.info("Using cached visual descriptions")
            # 两家模块有相同的 load_visual_descriptions;随便挑一个
            visuals = vision_gemini.load_visual_descriptions(vision_json)
        elif provider == "claude_code":
            if not vision_claude.is_available():
                raise RuntimeError(
                    "vision.provider=claude_code 但 claude CLI 未安装或未登录。"
                    "改成 'gemini' 或先 `claude login`。"
                )
            log.info("Vision provider: Claude Code CLI")
            visuals = vision_claude.describe_frames(
                frames, frames_dir, cfg["vision"], None, out_json_path=vision_json
            )
        elif provider == "gemini":
            api_key = get_env("GEMINI_API_KEY", required=True)
            log.info("Vision provider: Gemini API")
            visuals = vision_gemini.describe_frames(
                frames, frames_dir, cfg["vision"], api_key, out_json_path=vision_json
            )
        elif provider == "none":
            log.info("Vision provider: none (skipping frame descriptions)")
            visuals = []
        else:
            raise ValueError(
                f"未知的 vision.provider={provider!r} (支持: claude_code/gemini/none)"
            )
        meta.has_vision = True
        _save_meta(meta, vdir)

    # ============ 4. 融合 + 结构化 ============
    enriched = fusion_align.align(
        transcript, visuals, mode=cfg["fusion"].get("visual_attach_mode", "cover")
    )
    fusion_align.write_enriched_markdown(
        enriched, vdir / "enriched_transcript.md", title=meta.title
    )
    fusion_align.dump_enriched_json(enriched, vdir / "enriched.json")

    notes: Notes | None = None
    if not skip_structure:
        log.info("=" * 60)
        log.info("Stage 4: LLM structuring")
        log.info("=" * 60)

        notes_json = vdir / "notes.json"
        if notes_json.exists() and not force:
            log.info("Using cached notes")
            with open(notes_json, encoding="utf-8") as f:
                notes = Notes(**json.load(f))
        else:
            provider = cfg["structuring"].get("provider", "claude_code")
            if provider == "claude_code" and claude_code.is_available():
                notes = claude_code.structure_notes(enriched, meta.video_id, cfg["structuring"])
            elif provider == "gemini" or not claude_code.is_available():
                if not claude_code.is_available() and provider == "claude_code":
                    log.warning(
                        "claude CLI 未找到,回退到 Gemini 结构化"
                    )
                api_key = get_env("GEMINI_API_KEY", required=True)
                notes = gemini_fallback.structure_notes(
                    enriched, meta.video_id, cfg["structuring"], api_key
                )
            else:
                raise RuntimeError(f"Unknown structuring provider: {provider}")

            with open(notes_json, "w", encoding="utf-8") as f:
                json.dump(notes.model_dump(), f, ensure_ascii=False, indent=2)
            (vdir / "notes.md").write_text(notes.full_markdown, encoding="utf-8")

        meta.has_notes = True
        _save_meta(meta, vdir)

    # ============ 5. 分块嵌入 + ChromaDB ============
    if not skip_embedding and notes is not None:
        log.info("=" * 60)
        log.info("Stage 5: Chunking + embedding + ChromaDB")
        log.info("=" * 60)

        chunks = chunking.chunk_notes(notes, enriched, meta.video_id, cfg["embedding"])
        if chunks:
            embeddings = bge.embed_texts([c.text for c in chunks], cfg["embedding"])
            chroma_client.upsert_chunks(
                chunks,
                embeddings,
                chroma_path,
                video_meta={"title": meta.title, "source": meta.source},
            )

            # 保存 chunks.jsonl 供后续参考
            with open(vdir / "chunks.jsonl", "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        meta.has_embeddings = True
        _save_meta(meta, vdir)

    # ============ 6. Claude Project 导出 ============
    if not skip_export and notes is not None:
        log.info("=" * 60)
        log.info("Stage 6: Claude Project export")
        log.info("=" * 60)
        export_dir = Path(cfg["_project_root"]) / cfg["export"].get(
            "claude_project_dir", "claude_upload"
        )
        claude_project.export_for_claude_project(
            vdir,
            export_dir,
            meta,
            notes,
            include_frames=cfg["export"].get("include_frames", True),
        )

    log.info("=" * 60)
    log.info(f"✅ Done: {meta.video_id}")
    log.info("=" * 60)

    return {
        "video_id": meta.video_id,
        "vdir": str(vdir),
        "transcript_segments": len(transcript.segments),
        "visuals": len(visuals),
        "sections": len(notes.sections) if notes else 0,
    }
