"""主 CLI 入口 — 基于 typer。"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import pipeline
from .config import load_config
from .storage import chroma_client

app = typer.Typer(
    name="kb",
    help="自动视频知识库 — YouTube/本地视频 → 结构化笔记 + RAG 向量库",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    source: str = typer.Argument(..., help="YouTube URL 或本地视频文件路径"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    skip_vision: bool = typer.Option(False, "--skip-vision", help="跳过视觉理解阶段"),
    skip_structure: bool = typer.Option(False, "--skip-structure", help="跳过 LLM 结构化"),
    skip_embedding: bool = typer.Option(False, "--skip-embedding", help="跳过嵌入到 ChromaDB"),
    skip_export: bool = typer.Option(False, "--skip-export", help="跳过 Claude Project 导出"),
    force: bool = typer.Option(False, "--force", "-f", help="忽略缓存,重跑所有阶段"),
    log_level: str = typer.Option("INFO", "--log", help="日志级别"),
):
    """处理一个视频:下载→转写→视觉→结构化→嵌入→导出。"""
    result = pipeline.ingest(
        source,
        cfg_path=config,
        skip_vision=skip_vision,
        skip_structure=skip_structure,
        skip_embedding=skip_embedding,
        skip_export=skip_export,
        force=force,
        log_level=log_level,
    )
    console.print("\n[bold green]✅ Ingest complete[/bold green]")
    console.print(result)


@app.command()
def batch(
    folder: Path = typer.Argument(..., help="包含视频文件的文件夹路径"),
    pattern: str = typer.Option(
        "*.mp4,*.mkv,*.webm,*.mov,*.m4v",
        "--pattern",
        "-p",
        help="视频文件通配符,逗号分隔。默认覆盖常见格式",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="递归扫描子目录"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing",
        help="默认跳过已有 notes.json 的视频(防止重复跑)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只打印会处理哪些视频,不真跑"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    skip_vision: bool = typer.Option(False, "--skip-vision"),
    skip_structure: bool = typer.Option(False, "--skip-structure"),
    skip_embedding: bool = typer.Option(False, "--skip-embedding"),
    skip_export: bool = typer.Option(False, "--skip-export"),
    force: bool = typer.Option(False, "--force", "-f"),
    log_level: str = typer.Option("INFO", "--log"),
):
    """批量处理一个目录下的所有视频 — 单个失败不中断整体。

    示例:
        kb batch ~/Documents/RomoeCRT                    # 跑该目录下所有 mp4
        kb batch ~/Downloads --pattern "*.mp4" -r        # 递归,只要 mp4
        kb batch ~/Documents/RomoeCRT --dry-run          # 预览清单
        kb batch ~/Documents/RomoeCRT --no-skip-existing # 强制重跑已处理过的
    """
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[red]不是目录:[/red] {folder}")
        raise typer.Exit(1)

    # 收集视频文件
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    files: list[Path] = []
    for pat in patterns:
        if recursive:
            files.extend(folder.rglob(pat))
        else:
            files.extend(folder.glob(pat))
    files = sorted(set(files))

    if not files:
        console.print(
            f"[yellow]没找到匹配 {patterns} 的视频文件于 {folder}[/yellow]"
        )
        raise typer.Exit(0)

    # 判断哪些已经处理过
    cfg = load_config(config)
    videos_root = Path(cfg["paths"]["videos_dir"])
    from .utils import video_dir as _video_dir  # 重用 hash 算法

    def _already_done(video_path: Path) -> bool:
        """通过匹配 kb/videos/*/meta.yaml 里的源路径判断是否跑过。"""
        if not videos_root.exists():
            return False
        import yaml

        for vdir in videos_root.iterdir():
            mp = vdir / "meta.yaml"
            if not mp.exists():
                continue
            try:
                with open(mp, encoding="utf-8") as f:
                    m = yaml.safe_load(f) or {}
                # 匹配文件名(不依赖绝对路径,方便移动视频)
                if m.get("title") == video_path.name and (vdir / "notes.json").exists():
                    return True
            except Exception:
                continue
        return False

    # 预览
    console.print(f"\n[bold]扫描到 {len(files)} 个视频:[/bold] {folder}\n")
    to_process: list[Path] = []
    skipped: list[Path] = []
    for f in files:
        if skip_existing and _already_done(f):
            skipped.append(f)
            console.print(f"  [dim]SKIP[/dim]  {f.name}  [dim](已有 notes.json)[/dim]")
        else:
            to_process.append(f)
            console.print(f"  [green]QUEUE[/green] {f.name}")

    console.print(
        f"\n[bold]队列 {len(to_process)} 个[/bold],[dim]跳过 {len(skipped)} 个[/dim]\n"
    )

    if dry_run:
        console.print("[yellow]--dry-run 模式,不真跑,退出。[/yellow]")
        return

    if not to_process:
        console.print("[yellow]没东西跑。加 --no-skip-existing 强制重跑。[/yellow]")
        return

    # 逐个处理,单个失败不中断
    results: list[dict] = []
    batch_t0 = time.time()
    for idx, video in enumerate(to_process, 1):
        console.rule(
            f"[bold cyan]({idx}/{len(to_process)}) {video.name}[/bold cyan]"
        )
        t0 = time.time()
        rec = {"file": video.name, "status": "?", "elapsed": 0.0, "error": ""}
        try:
            result = pipeline.ingest(
                str(video),
                cfg_path=config,
                skip_vision=skip_vision,
                skip_structure=skip_structure,
                skip_embedding=skip_embedding,
                skip_export=skip_export,
                force=force,
                log_level=log_level,
            )
            rec["status"] = "✅ ok"
            rec["video_id"] = result.get("video_id", "")
            rec["segments"] = result.get("transcript_segments", 0)
            rec["visuals"] = result.get("visuals", 0)
            rec["sections"] = result.get("sections", 0)
        except KeyboardInterrupt:
            console.print("\n[red]用户中断,停止批处理。[/red]")
            rec["status"] = "⏹ interrupted"
            results.append(rec)
            break
        except Exception as e:
            rec["status"] = "❌ fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            console.print(f"\n[red]❌ 这个视频挂了,继续下一个:[/red] {rec['error']}")
            console.print(
                "[dim]" + traceback.format_exc(limit=3)[-800:] + "[/dim]"
            )

        rec["elapsed"] = time.time() - t0
        results.append(rec)

    # 汇总报表
    total_elapsed = time.time() - batch_t0
    console.rule("[bold]批处理汇总[/bold]")
    table = Table(show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("文件", max_width=40)
    table.add_column("状态")
    table.add_column("耗时", justify="right")
    table.add_column("片段", justify="right")
    table.add_column("视觉", justify="right")
    table.add_column("章节", justify="right")
    table.add_column("错误", max_width=50)

    ok = fail = 0
    for i, r in enumerate(results, 1):
        if r["status"].startswith("✅"):
            ok += 1
        elif r["status"].startswith("❌"):
            fail += 1
        table.add_row(
            str(i),
            r["file"],
            r["status"],
            f"{r['elapsed']:.0f}s",
            str(r.get("segments", "")),
            str(r.get("visuals", "")),
            str(r.get("sections", "")),
            r.get("error", "")[:80],
        )
    console.print(table)
    console.print(
        f"\n[bold]总计[/bold] {len(results)} 个, "
        f"[green]成功 {ok}[/green], "
        f"[red]失败 {fail}[/red], "
        f"[dim]总耗时 {total_elapsed / 60:.1f} min[/dim]"
    )


@app.command()
def query(
    text: str = typer.Argument(..., help="检索问题"),
    n: int = typer.Option(5, "--n", help="返回结果数"),
    video_id: Optional[str] = typer.Option(None, "--video-id", help="仅在指定视频中搜索"),
    no_aliases: bool = typer.Option(
        False, "--no-aliases", help="禁用同义词扩展(默认开启,词典见 configs/aliases.yaml)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """语义检索 ChromaDB 向量库。"""
    cfg = load_config(config)
    where = {"video_id": video_id} if video_id else None

    # 加载别名词典(若启用且配置/文件存在)
    aliases_lookup = None
    if not no_aliases:
        from .retrieval.aliases import load_aliases
        ap = cfg.get("retrieval", {}).get("aliases_path")
        if ap:
            apath = Path(ap)
            if not apath.is_absolute():
                apath = Path(cfg.get("_project_root", ".")) / apath
            aliases_lookup = load_aliases(apath) or None

    hits = chroma_client.query(
        text,
        cfg["paths"]["chroma_dir"],
        cfg["embedding"],
        n_results=n,
        where=where,
        aliases_lookup=aliases_lookup,
    )
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title=f"Query: {text}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Video / 时间", style="cyan", max_width=30)
    table.add_column("Section", style="magenta", max_width=25)
    table.add_column("Distance", justify="right", style="green", width=10)
    table.add_column("Text", max_width=70)

    for i, h in enumerate(hits, 1):
        md = h["metadata"]
        vid = md.get("video_title") or md.get("video_id", "")
        ts = f"{int(md.get('start_sec', 0))//60:02d}:{int(md.get('start_sec', 0))%60:02d}"
        vis = " 🖼" if md.get("has_visual") else ""
        table.add_row(
            str(i),
            f"{vid}\n@ {ts}{vis}",
            md.get("section_title", ""),
            f"{h['distance']:.3f}" if h["distance"] is not None else "-",
            h["text"][:400],
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="用自然语言提问"),
    n: int = typer.Option(8, "--n", help="检索 Top-K chunk 数(默认 8)"),
    video_id: Optional[str] = typer.Option(
        None, "--video-id", help="仅在指定视频中搜索"
    ),
    no_aliases: bool = typer.Option(
        False, "--no-aliases", help="禁用同义词扩展(默认开启)"
    ),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", help="同时显示检索到的原始 chunks"
    ),
    save: Optional[Path] = typer.Option(
        None, "--save", "-o", help="把答案保存到 markdown 文件"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """基于视频知识库综合回答问题 — 检索 Top-K + Claude CLI 生成带引用答案。"""
    from .rag.answer import answer as rag_answer

    cfg = load_config(config)

    with console.status(
        f"[cyan]检索并生成答案... (Top-{n} chunks → Claude CLI)[/cyan]",
        spinner="dots",
    ):
        result = rag_answer(
            question=question,
            cfg=cfg,
            n_results=n,
            video_id=video_id,
            use_aliases=not no_aliases,
        )

    console.rule(f"[bold]Q: {question}[/bold]")

    if show_chunks:
        console.print(
            f"\n[dim]检索到 {len(result['hits'])} 个 chunks "
            f"(prompt {result['prompt_chars']} chars):[/dim]"
        )
        for i, h in enumerate(result["hits"], 1):
            md = h["metadata"]
            ts = (
                f"{int(md.get('start_sec', 0)) // 60:02d}:"
                f"{int(md.get('start_sec', 0)) % 60:02d}"
            )
            vis = " 📷" if md.get("has_visual") else ""
            vt = (md.get("video_title") or md.get("video_id", ""))[:35]
            dist = h.get("distance")
            dist_str = f"{dist:.3f}" if dist is not None else "-"
            console.print(
                f"  [dim]#{i}[/dim] [{vt} @ {ts}]{vis}  d={dist_str}"
            )

    console.rule("[bold green]Answer[/bold green]")
    # markup=False 防止 Rich 把 [ep.N @ mm:ss] 当 style tag 吞掉引用标记
    console.print(result["answer"], markup=False)

    if save:
        save.write_text(
            f"# Q: {question}\n\n{result['answer']}\n",
            encoding="utf-8",
        )
        console.print(f"\n[green]✅ 答案已保存:[/green] {save}")


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """显示向量库统计(按 source_type 和 source_id 拆分)。"""
    cfg = load_config(config)
    info = chroma_client.stats(cfg["paths"]["chroma_dir"])
    console.print(f"[bold]Total chunks:[/bold] {info['total_chunks']}")

    per_type = info.get("per_type") or {}
    if per_type:
        ttable = Table(title="By source_type", show_lines=False)
        ttable.add_column("type", style="magenta")
        ttable.add_column("chunks", justify="right")
        for t, n in sorted(per_type.items(), key=lambda x: -x[1]):
            ttable.add_row(t, str(n))
        console.print(ttable)

    if info["videos"]:
        table = Table(title="By source_id (video_id / pdf_xxx / img_xxx)")
        table.add_column("source_id", style="cyan")
        table.add_column("chunks", justify="right")
        for vid, n in sorted(info["videos"].items(), key=lambda x: -x[1]):
            table.add_row(vid, str(n))
        console.print(table)


@app.command()
def list_videos(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """列出 kb/videos/ 下所有已处理视频。"""
    cfg = load_config(config)
    videos_root = Path(cfg["paths"]["videos_dir"])
    if not videos_root.exists():
        console.print("[yellow]No videos processed yet.[/yellow]")
        return

    import yaml

    table = Table(title="Processed videos")
    table.add_column("video_id", style="cyan")
    table.add_column("title", max_width=40)
    table.add_column("duration", justify="right")
    table.add_column("status", style="green")

    for vdir in sorted(videos_root.iterdir()):
        if not vdir.is_dir():
            continue
        meta_p = vdir / "meta.yaml"
        if not meta_p.exists():
            continue
        with open(meta_p, encoding="utf-8") as f:
            m = yaml.safe_load(f)
        flags = []
        if m.get("has_transcript"):
            flags.append("tx")
        if m.get("has_vision"):
            flags.append("vis")
        if m.get("has_notes"):
            flags.append("notes")
        if m.get("has_embeddings"):
            flags.append("emb")
        table.add_row(
            m.get("video_id", ""),
            m.get("title", ""),
            f"{m.get('duration_sec', 0):.0f}s",
            " ".join(flags),
        )
    console.print(table)


@app.command()
def export(
    video_id: str = typer.Argument(..., help="要导出的 video_id"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """重新导出一个视频的 Claude Project 包。"""
    import json

    import yaml
    from .export import claude_project
    from .schemas import Notes, VideoMeta

    cfg = load_config(config)
    vdir = Path(cfg["paths"]["videos_dir"]) / video_id
    if not vdir.exists():
        console.print(f"[red]Not found: {vdir}[/red]")
        raise typer.Exit(1)

    with open(vdir / "meta.yaml", encoding="utf-8") as f:
        meta = VideoMeta(**yaml.safe_load(f))
    with open(vdir / "notes.json", encoding="utf-8") as f:
        notes = Notes(**json.load(f))

    out = Path(cfg["_project_root"]) / cfg["export"].get("claude_project_dir", "claude_upload")
    claude_project.export_for_claude_project(vdir, out, meta, notes)
    console.print(f"[green]Exported:[/green] {out / video_id}")


@app.command()
def reindex(
    video_id: Optional[str] = typer.Argument(
        None, help="要重建的 video_id;不填则重建所有 kb/videos/* 下可用的"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log"),
):
    """只重切块+重嵌入,复用已有 enriched.json + notes.json。

    用法:
        kb reindex                   # 全量重建
        kb reindex <video_id>        # 只重建一个

    典型场景:调整了 embedding.min_chunk_chars / max_visuals_per_window
    等参数,想应用到现有 KB,但不想重跑 STT/Vision/LLM。
    """
    results = pipeline.reindex(
        video_id=video_id,
        cfg_path=config,
        log_level=log_level,
    )
    if not results:
        console.print("[yellow]没东西可重建。[/yellow]")
        return

    table = Table(title="Reindex 结果", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("video_id", style="cyan")
    table.add_column("删掉", justify="right")
    table.add_column("新增", justify="right", style="green")
    table.add_column("Δ", justify="right")
    for i, r in enumerate(results, 1):
        delta = r["chunks_new"] - r["chunks_deleted"]
        sign = "+" if delta >= 0 else ""
        table.add_row(
            str(i),
            r["video_id"],
            str(r["chunks_deleted"]),
            str(r["chunks_new"]),
            f"{sign}{delta}",
        )
    console.print(table)
    console.print(
        f"\n[bold green]✅ 共重建 {len(results)} 个视频[/bold green]"
    )


# ========= kb ingest-doc / kb docs =========


@app.command("ingest-doc")
def ingest_doc_cmd(
    source: Path = typer.Argument(..., help="PDF 或图片文件路径(也可以是文件夹,递归入库所有支持的文件)"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r",
        help="source 是文件夹时递归扫子目录(默认开)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="忽略缓存重跑(PDF 重抽文本 / 图片重新调 Claude 描述)"
    ),
    pdf_provider: Optional[str] = typer.Option(
        None, "--pdf-provider",
        help="PDF 抽取 provider:claude_code(默认,慢但能 OCR+图表+表格) / pypdf(快,仅文本)",
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log"),
):
    """把独立 PDF / 图片 入库到同一 ChromaDB(和视频并列,支持统一检索)。

    PDF 默认走 Claude CLI (`--pdf-provider claude_code`),能读扫描版、转录图表、
    保留表格结构。如果 PDF 是纯文本、量大、想快速索引,可以 `--pdf-provider pypdf`。

    示例:
        kb ingest-doc notes.pdf                             # 单 PDF(Claude CLI)
        kb ingest-doc notes.pdf --pdf-provider pypdf        # 单 PDF(纯文本快速)
        kb ingest-doc ~/Desktop/chart.png                   # 单张图
        kb ingest-doc ~/Documents/trading_refs              # 整个目录(递归)
        kb ingest-doc notes.pdf --force                     # 忽略缓存重跑
    """
    from .ingest.docs import detect_doc_type

    src = source.expanduser().resolve()
    if not src.exists():
        console.print(f"[red]Not found: {src}[/red]")
        raise typer.Exit(1)

    # 收集目标文件
    targets: list[Path] = []
    if src.is_file():
        if detect_doc_type(src) is None:
            console.print(f"[red]不支持的文件类型:[/red] {src.suffix}")
            raise typer.Exit(1)
        targets = [src]
    else:
        # 目录 → 扫所有支持类型
        patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif"]
        for pat in patterns:
            if recursive:
                targets.extend(src.rglob(pat))
            else:
                targets.extend(src.glob(pat))
        targets = sorted(set(targets))

    if not targets:
        console.print(f"[yellow]在 {src} 下没找到可入库的文件[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]准备入库 {len(targets)} 个文件:[/bold]")
    for f in targets:
        t = detect_doc_type(f)
        console.print(f"  [dim]{t:>5}[/dim]  {f.name}")
    console.print()

    results: list[dict] = []
    t0 = time.time()
    for idx, f in enumerate(targets, 1):
        console.rule(f"[cyan]({idx}/{len(targets)}) {f.name}[/cyan]")
        try:
            res = pipeline.ingest_doc(
                f,
                cfg_path=config,
                force=force,
                log_level=log_level,
            )
            res["status"] = "✅"
            results.append(res)
        except KeyboardInterrupt:
            console.print("\n[red]用户中断[/red]")
            break
        except Exception as e:
            console.print(f"[red]❌ {type(e).__name__}: {e}[/red]")
            console.print(
                "[dim]" + traceback.format_exc(limit=3)[-600:] + "[/dim]"
            )
            results.append(
                {
                    "status": "❌",
                    "doc_id": "",
                    "source_type": detect_doc_type(f) or "?",
                    "chunks_new": 0,
                    "chunks_deleted": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # 汇总
    console.rule("[bold]入库汇总[/bold]")
    table = Table(show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("status")
    table.add_column("type", style="magenta", width=6)
    table.add_column("doc_id", style="cyan", max_width=40)
    table.add_column("删", justify="right")
    table.add_column("新", justify="right", style="green")
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.get("status", "?"),
            r.get("source_type", ""),
            r.get("doc_id", "") or r.get("error", "")[:40],
            str(r.get("chunks_deleted", 0)),
            str(r.get("chunks_new", 0)),
        )
    console.print(table)
    ok = sum(1 for r in results if r.get("status", "").startswith("✅"))
    fail = sum(1 for r in results if r.get("status", "").startswith("❌"))
    console.print(
        f"\n[bold]总计[/bold] {len(results)}, "
        f"[green]ok {ok}[/green], [red]fail {fail}[/red], "
        f"[dim]{(time.time() - t0) / 60:.1f} min[/dim]"
    )


docs_app = typer.Typer(
    name="docs",
    help="管理独立文档(PDF / 图片)— 和 kb ingest-doc 配套",
    no_args_is_help=True,
)
app.add_typer(docs_app, name="docs")


@docs_app.command("list")
def docs_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """列出 kb/docs/ 下所有已入库文档。"""
    items = pipeline.list_docs(cfg_path=config)
    if not items:
        console.print("[yellow]还没有入库文档。试试 `kb ingest-doc <file>`[/yellow]")
        return
    table = Table(title="Docs in KB")
    table.add_column("doc_id", style="cyan", max_width=40)
    table.add_column("type", style="magenta", width=6)
    table.add_column("title", max_width=40)
    table.add_column("pages", justify="right")
    table.add_column("emb", style="green", width=4)
    table.add_column("ingested_at", style="dim")
    for m in items:
        flag = "✓" if m.get("has_embeddings") else "-"
        table.add_row(
            m.get("doc_id", ""),
            m.get("source_type", ""),
            m.get("title", ""),
            str(m.get("page_count", 0)),
            flag,
            (m.get("ingested_at") or "")[:19],
        )
    console.print(table)


@docs_app.command("remove")
def docs_remove(
    doc_id: str = typer.Argument(..., help="要删除的 doc_id"),
    keep_files: bool = typer.Option(
        False, "--keep-files", help="只清向量库,保留 kb/docs/<doc_id>/ 文件"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """从向量库和 kb/docs/ 中删除一个文档。"""
    import shutil as _shutil

    cfg = load_config(config)
    docs_root = Path(cfg["paths"].get("docs_dir", "kb/docs"))
    ddir = docs_root / doc_id

    deleted = chroma_client.delete_by_source_id(
        doc_id, cfg["paths"]["chroma_dir"]
    )
    console.print(f"向量库:删除 [bold]{deleted}[/bold] 个 chunks")
    if not keep_files and ddir.exists():
        _shutil.rmtree(ddir)
        console.print(f"文件:删除目录 {ddir}")
    elif not ddir.exists():
        console.print(f"[yellow]目录不存在(可能已清):{ddir}[/yellow]")


# ========= kb aliases 子命令组 =========
aliases_app = typer.Typer(
    name="aliases",
    help="管理同义词词典 — 查询时做概念扩展,提升召回",
    no_args_is_help=True,
)
app.add_typer(aliases_app, name="aliases")


def _resolve_aliases_path(cfg: dict) -> Path | None:
    ap = cfg.get("retrieval", {}).get("aliases_path")
    if not ap:
        return None
    apath = Path(ap)
    if not apath.is_absolute():
        apath = Path(cfg.get("_project_root", ".")) / apath
    return apath


@aliases_app.command("list")
def aliases_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """列出当前词典所有组。"""
    from .retrieval.aliases import load_aliases

    cfg = load_config(config)
    apath = _resolve_aliases_path(cfg)
    if not apath:
        console.print(
            "[yellow]retrieval.aliases_path 未配置。"
            "在 configs/default.yaml 里加 retrieval.aliases_path。[/yellow]"
        )
        return
    if not apath.exists():
        console.print(f"[yellow]词典不存在: {apath}[/yellow]")
        console.print(
            "新建一个模板文件往里加词,或跑 `kb aliases check \"<query>\"` 先预览。"
        )
        return

    lookup = load_aliases(apath)
    if not lookup:
        console.print(f"[yellow]词典为空: {apath}[/yellow]")
        return

    # lookup 是 term→group,每组多次出现,用 id 去重
    seen_groups = set()
    groups: list[list[str]] = []
    for group in lookup.values():
        gid = id(group)
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        groups.append(group)

    table = Table(title=f"Aliases @ {apath}", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Canonical", style="cyan")
    table.add_column("Aliases", style="green")
    for i, g in enumerate(groups, 1):
        table.add_row(str(i), g[0], ", ".join(g[1:]) if len(g) > 1 else "-")
    console.print(table)
    console.print(
        f"\n[dim]共 {len(groups)} 组 / {len(lookup)} 个词[/dim]"
    )


@aliases_app.command("check")
def aliases_check(
    query: str = typer.Argument(..., help="想预览扩展的查询文本"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """预览一个查询被词典扩展后长什么样(不真跑检索)。"""
    from .retrieval.aliases import preview_expansion

    cfg = load_config(config)
    apath = _resolve_aliases_path(cfg)
    result = preview_expansion(query, apath)

    console.print(f"[bold]原始:[/bold] {result['original']}")
    if not result["hits"]:
        console.print("[yellow]未命中词典中任何词,查询不会扩展。[/yellow]")
        return
    console.print(
        f"[bold]命中:[/bold] [cyan]{', '.join(result['hits'])}[/cyan]"
    )
    console.print(
        f"[bold]新增:[/bold] [green]{', '.join(result['added']) or '(无)'}[/green]"
    )
    console.print(f"[bold]扩展后:[/bold] {result['expanded']}")


@aliases_app.command("suggest")
def aliases_suggest(
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="输出文件,默认写到 configs/aliases.suggested.yaml",
    ),
    sample_size: int = typer.Option(
        80, "--sample-size", help="采样几个 chunk 喂给 Claude(默认 80)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="claude 模型(默认复用 structuring.claude_model,再兜底 sonnet)",
    ),
    timeout_sec: int = typer.Option(
        240, "--timeout", help="Claude CLI 超时秒数(默认 240)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """从向量库语料里自动抽专业术语,生成 aliases.yaml 建议稿。

    流程:ChromaDB 采样 → Claude CLI → 输出 YAML 建议到 --out 路径。
    ⚠️ 这是建议稿,**不是**正式词典。审阅后手工合并到 configs/aliases.yaml。

    示例:
        kb aliases suggest                          # 写到 configs/aliases.suggested.yaml
        kb aliases suggest --sample-size 120        # 采更多样本,Claude 上下文压力大一些
        kb aliases suggest --model opus             # 用 opus 更细致(慢 3-5x)
        kb aliases suggest -o /tmp/terms.yaml       # 自定义输出路径
    """
    from .retrieval.extract_terms import suggest_aliases

    cfg = load_config(config)
    # 默认模型:ask.claude_model → structuring.claude_model → sonnet
    if model is None:
        model = (
            cfg.get("ask", {}).get("claude_model")
            or cfg.get("structuring", {}).get("claude_model")
            or "sonnet"
        )
    # 默认输出路径:configs/aliases.suggested.yaml (和 aliases.yaml 同目录)
    if out is None:
        apath = _resolve_aliases_path(cfg)
        if apath:
            out = apath.parent / "aliases.suggested.yaml"
        else:
            out = Path("configs/aliases.suggested.yaml")

    with console.status(
        f"[cyan]采样 → Claude ({model}) 抽术语...[/cyan]", spinner="dots"
    ):
        try:
            result = suggest_aliases(
                chroma_path=cfg["paths"]["chroma_dir"],
                out_path=out,
                claude_model=model,
                sample_size=sample_size,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            console.print(f"[red]失败:[/red] {type(e).__name__}: {e}")
            raise typer.Exit(1)

    console.print(
        f"[green]✅ 已生成建议稿:[/green] {result['out_path']}\n"
        f"  采样 {result['sample_size']}/{result['total_chunks']} chunks, "
        f"prompt {result['prompt_chars'] // 1000}k chars, "
        f"输出 {result['raw_output_chars']} chars"
    )
    if not result["parsed_ok"]:
        console.print(
            "[yellow]⚠️  Claude 输出不是合法 YAML。"
            "打开文件手工修正,或改 --model opus 重跑。[/yellow]"
        )
    console.print(
        "\n[dim]下一步:[/dim]\n"
        f"  1. 打开 {result['out_path']} 看看有没有垃圾词\n"
        "  2. 挑想要的条目,**手工合并**到 configs/aliases.yaml\n"
        "  3. `kb aliases list` 确认最终词典生效\n"
        "  4. `kb aliases check \"<你常用的查询>\"` 验证扩展效果"
    )


if __name__ == "__main__":
    app()
