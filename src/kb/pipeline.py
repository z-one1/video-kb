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
from .ingest import docs as ingest_docs
from .ingest import downloader, extractor, scenes
from .schemas import Notes, Transcript, VideoMeta
from .storage import chroma_client
from .stt import whisper_local
from .structuring import claude_code, gemini_fallback
from .utils import doc_dir, ensure_dir, setup_logging, video_dir
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


def reindex(
    video_id: str | None = None,
    cfg_path: Path | str | None = None,
    log_level: str = "INFO",
) -> list[dict[str, Any]]:
    """复用已有的 enriched.json + notes.json 重新切块+嵌入+入库。

    适用场景:chunking 参数调整(window/visual cap/min_chunk_chars)后,
    不想重跑 STT/Vision/LLM,只想重新切块。

    Args:
        video_id: 指定 video_id 只重建单个;None 则重建所有 kb/videos/* 下可用的。

    每个视频的步骤:
      1. load notes.json + enriched.json(两者都必须存在)
      2. chunking.chunk_notes(...)  — 用当前 cfg 重算
      3. chroma_client.delete_by_video_id(video_id)  — 清旧
      4. bge.embed_texts(...) + chroma_client.upsert_chunks(...)
      5. 重写 chunks.jsonl
    """
    from .schemas import EnrichedSegment

    setup_logging(log_level)
    cfg = load_config(cfg_path)
    videos_root = Path(cfg["paths"]["videos_dir"])
    chroma_path = Path(cfg["paths"]["chroma_dir"])

    # 找要处理的 video 列表
    targets: list[Path] = []
    if video_id:
        vdir = videos_root / video_id
        if not vdir.exists():
            raise FileNotFoundError(f"video dir not found: {vdir}")
        targets.append(vdir)
    else:
        if not videos_root.exists():
            log.warning(f"videos_root not found: {videos_root}")
            return []
        for vdir in sorted(videos_root.iterdir()):
            if not vdir.is_dir():
                continue
            if (vdir / "notes.json").exists() and (vdir / "enriched.json").exists():
                targets.append(vdir)

    if not targets:
        log.warning("No videos to reindex (need notes.json + enriched.json)")
        return []

    log.info("=" * 60)
    log.info(f"Reindex: {len(targets)} video(s)")
    log.info("=" * 60)

    results: list[dict[str, Any]] = []
    for vdir in targets:
        vid = vdir.name
        log.info(f"--- reindex {vid} ---")

        meta = _load_meta(vdir)
        if meta is None:
            log.warning(f"{vid}: meta.yaml missing, skip")
            continue

        # 加载 enriched + notes
        with open(vdir / "enriched.json", encoding="utf-8") as f:
            enriched_raw = json.load(f)
        enriched = [EnrichedSegment(**d) for d in enriched_raw]
        with open(vdir / "notes.json", encoding="utf-8") as f:
            notes = Notes(**json.load(f))

        # 重新切块
        chunks = chunking.chunk_notes(notes, enriched, meta.video_id, cfg["embedding"])
        if not chunks:
            log.warning(f"{vid}: produced 0 chunks, skip")
            continue

        # 清旧 + 写新
        deleted = chroma_client.delete_by_video_id(meta.video_id, chroma_path)
        embeddings = bge.embed_texts([c.text for c in chunks], cfg["embedding"])
        chroma_client.upsert_chunks(
            chunks,
            embeddings,
            chroma_path,
            video_meta={"title": meta.title, "source": meta.source},
        )

        # 覆盖 chunks.jsonl
        with open(vdir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        results.append(
            {
                "video_id": vid,
                "chunks_new": len(chunks),
                "chunks_deleted": deleted,
            }
        )
        log.info(f"{vid}: {deleted} → {len(chunks)} chunks")

    log.info("=" * 60)
    log.info(f"✅ Reindex done: {len(results)} video(s)")
    log.info("=" * 60)
    return results


# ============================================================
# ===============  独立文档 (PDF/图片) ingestion  ==============
# ============================================================


def ingest_doc(
    source: str | Path,
    cfg_path: Path | str | None = None,
    force: bool = False,
    pdf_provider: str | None = None,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """单个 PDF 或图片入库。

    流程:
      1. 探测类型 (pdf/image),生成 doc_id,创建 kb/docs/<doc_id>/
      2. PDF: 抽每页文本 → pages.jsonl;Image: Claude CLI 描述 → description.json
      3. 分块 (PDF 按页、Image 单块)
      4. bge-m3 嵌入
      5. ChromaDB: delete_by_source_id(旧) + upsert(新)
      6. chunks.jsonl + meta.yaml

    Args:
        source: 绝对路径或项目相对路径(PDF 或 png/jpg)
        force: 有缓存也重跑(当前仅重新调 Claude 描述图片 / 重切块 PDF)

    Returns:
        {doc_id, source_type, chunks_new, chunks_deleted, ddir}
    """
    setup_logging(log_level)
    cfg = load_config(cfg_path)
    kb_root = Path(cfg["paths"]["kb_root"])
    chroma_path = Path(cfg["paths"]["chroma_dir"])
    ensure_dir(kb_root)
    ensure_dir(chroma_path)

    src = Path(source).expanduser()
    if not src.is_absolute():
        src = (Path(cfg.get("_project_root", ".")) / src).resolve()
    else:
        src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    doc_type = ingest_docs.detect_doc_type(src)
    if doc_type is None:
        raise ValueError(
            f"Unsupported file type: {src.suffix!r}. "
            f"Supported: .pdf / .png / .jpg / .jpeg / .webp / .bmp / .gif"
        )

    # 1. Meta + 目录
    meta = ingest_docs.build_doc_meta(src, doc_type)
    ddir = doc_dir(kb_root, meta.doc_id)
    # 若已存在且 force=False,以落盘的 ingested_at 为准(保留首次入库时间)
    prev = ingest_docs.load_doc_meta(ddir)
    if prev and not force:
        meta.ingested_at = prev.ingested_at

    log.info("=" * 60)
    log.info(f"Ingest doc ({doc_type}): {src.name} → {meta.doc_id}")
    log.info("=" * 60)

    ingest_doc_cfg = cfg.get("ingest_doc", {})
    # 从 vision 段借 claude_model / timeout(因为图片描述复用 vision 通道)
    vision_cfg = cfg.get("vision", {})

    chunks = []

    if doc_type == "pdf":
        # Provider 选择:CLI 参数 > 配置 > 默认 claude_code
        provider = (
            pdf_provider
            or ingest_doc_cfg.get("pdf_provider")
            or "claude_code"
        )
        log.info(f"PDF provider: {provider}")
        pages_path = ddir / "pages.jsonl"
        if pages_path.exists() and not force:
            log.info(f"Using cached pages.jsonl: {pages_path}")
            with open(pages_path, encoding="utf-8") as f:
                pages = [json.loads(line) for line in f if line.strip()]
        elif provider == "claude_code":
            # Claude CLI 抽取 — claude_model / timeout 从 ingest_doc 段 > vision 段 借
            pdf_cfg = {
                "claude_model": ingest_doc_cfg.get(
                    "pdf_claude_model",
                    ingest_doc_cfg.get(
                        "claude_model", vision_cfg.get("claude_model", "sonnet")
                    ),
                ),
            }
            timeout_sec = ingest_doc_cfg.get(
                "pdf_timeout_sec",
                ingest_doc_cfg.get("timeout_sec", 600),
            )
            pages = ingest_docs.extract_pdf_pages_via_claude(
                src, pdf_cfg, timeout_sec=timeout_sec
            )
            with open(pages_path, "w", encoding="utf-8") as f:
                for p in pages:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        elif provider == "pypdf":
            pages = ingest_docs.extract_pdf_pages_pypdf(src)
            with open(pages_path, "w", encoding="utf-8") as f:
                for p in pages:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        else:
            raise ValueError(
                f"Unknown pdf_provider={provider!r} (支持: claude_code / pypdf)"
            )
        meta.page_count = len(pages)
        meta.has_text = any(p.get("text") for p in pages)
        chunks = ingest_docs.chunk_pdf(
            pages, meta.doc_id, str(src), ingest_doc_cfg
        )

    elif doc_type == "image":
        desc_path = ddir / "description.json"
        if desc_path.exists() and not force:
            log.info(f"Using cached description.json: {desc_path}")
            with open(desc_path, encoding="utf-8") as f:
                description = json.load(f)
        else:
            # claude_model / timeout_sec 从 vision 借;允许 ingest_doc 段覆盖
            desc_cfg = {
                "claude_model": ingest_doc_cfg.get(
                    "claude_model", vision_cfg.get("claude_model", "sonnet")
                ),
                "image_prompt_lang": ingest_doc_cfg.get(
                    "image_prompt_lang", "zh"
                ),
            }
            timeout_sec = ingest_doc_cfg.get(
                "timeout_sec", vision_cfg.get("timeout_sec", 180)
            )
            description = ingest_docs.describe_image(
                src, desc_cfg, timeout_sec=timeout_sec
            )
            with open(desc_path, "w", encoding="utf-8") as f:
                json.dump(description, f, ensure_ascii=False, indent=2)
        meta.page_count = 1
        meta.has_text = bool(description.get("description"))
        chunks = ingest_docs.chunk_image(description, meta.doc_id, str(src))

    if not chunks:
        log.warning(f"{meta.doc_id}: produced 0 chunks, skip embedding")
        ingest_docs.save_doc_meta(meta, ddir)
        return {
            "doc_id": meta.doc_id,
            "source_type": doc_type,
            "chunks_new": 0,
            "chunks_deleted": 0,
            "ddir": str(ddir),
        }

    # 2. 嵌入 + 入库 (清旧再写新,幂等)
    log.info(f"Embedding {len(chunks)} chunks with {cfg['embedding'].get('model')}")
    embeddings = bge.embed_texts([c.text for c in chunks], cfg["embedding"])
    deleted = chroma_client.delete_by_source_id(meta.doc_id, chroma_path)
    chroma_client.upsert_chunks(
        chunks,
        embeddings,
        chroma_path,
        # title/source 用于检索结果显示
        video_meta={"title": meta.title, "source": doc_type},
    )

    # 3. chunks.jsonl + meta
    with open(ddir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    meta.has_embeddings = True
    ingest_docs.save_doc_meta(meta, ddir)

    log.info("=" * 60)
    log.info(
        f"✅ Doc done: {meta.doc_id}  "
        f"({deleted} deleted → {len(chunks)} new chunks)"
    )
    log.info("=" * 60)

    return {
        "doc_id": meta.doc_id,
        "source_type": doc_type,
        "chunks_new": len(chunks),
        "chunks_deleted": deleted,
        "ddir": str(ddir),
    }


def list_docs(cfg_path: Path | str | None = None) -> list[dict[str, Any]]:
    """扫 kb/docs/ 列出所有已入库文档元信息(返回 dict 列表,便于 CLI/JSON 使用)。"""
    cfg = load_config(cfg_path)
    docs_root = Path(cfg["paths"].get("docs_dir", "kb/docs"))
    if not docs_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for ddir in sorted(docs_root.iterdir()):
        if not ddir.is_dir():
            continue
        m = ingest_docs.load_doc_meta(ddir)
        if m is None:
            continue
        out.append(m.model_dump())
    return out
