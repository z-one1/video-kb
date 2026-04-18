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
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """语义检索 ChromaDB 向量库。"""
    cfg = load_config(config)
    where = {"video_id": video_id} if video_id else None

    hits = chroma_client.query(
        text,
        cfg["paths"]["chroma_dir"],
        cfg["embedding"],
        n_results=n,
        where=where,
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
    """显示向量库统计。"""
    cfg = load_config(config)
    info = chroma_client.stats(cfg["paths"]["chroma_dir"])
    console.print(f"[bold]Total chunks:[/bold] {info['total_chunks']}")
    if info["videos"]:
        table = Table(title="Videos in KB")
        table.add_column("video_id", style="cyan")
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


if __name__ == "__main__":
    app()
