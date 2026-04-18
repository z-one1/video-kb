"""video-kb MCP Server — 把知识库暴露给 Claude Code / Claude Desktop / Cowork。

运行方式:
    kb-mcp                   # stdio transport (给 Claude Code 用)

配置 Claude Code:
    在项目根目录新建 .mcp.json:
    {
      "mcpServers": {
        "video-kb": {
          "command": "/path/to/.venv/bin/kb-mcp",
          "env": {}
        }
      }
    }
    或用: claude mcp add video-kb /path/to/.venv/bin/kb-mcp

启动后 Claude Code 会把下面这几个函数当成原生工具调用:
    - kb_ask:       用自然语言问问题,返回带 [ep.N @ mm:ss] 引用的综合答案
    - kb_query:     语义检索,返回 Top-K chunks(Claude 自己综合用)
    - kb_list_videos:  列出已入库的视频
    - kb_stats:     向量库总览

设计取舍:
    - 用 FastMCP 高级 API(自动从 type hints 生成 JSON schema)
    - 工具调用全部捕获异常,返回 error 字符串 — 避免 Claude Code 侧爆栈
    - 配置文件路径:环境变量 KB_PROJECT_ROOT > 上溯查找 configs/default.yaml > cwd
    - 所有工具都是 sync — Chroma 查询和 Claude CLI 调用本身是 blocking,async 包装没意义
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("kb.mcp")


# ---------- 项目根目录发现 ----------


def _find_project_root() -> Path:
    """按优先级查找项目根目录:
    1. 环境变量 KB_PROJECT_ROOT
    2. 从 cwd 上溯,找到含 configs/default.yaml 的目录
    3. 报错
    """
    env = os.environ.get("KB_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "configs" / "default.yaml").exists():
            return p
        raise RuntimeError(
            f"KB_PROJECT_ROOT={env} 存在但找不到 configs/default.yaml"
        )

    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "configs" / "default.yaml").exists() and (p / "src" / "kb").exists():
            return p
    raise RuntimeError(
        "找不到 video_kb 项目根目录。请设置环境变量 KB_PROJECT_ROOT=/path/to/video_kb,"
        "或在项目根目录下运行 kb-mcp。"
    )


# ---------- 懒加载配置和依赖 ----------


_cfg: dict[str, Any] | None = None
_project_root: Path | None = None


def _get_cfg() -> dict[str, Any]:
    """首次调用时加载配置并缓存。"""
    global _cfg, _project_root
    if _cfg is not None:
        return _cfg

    _project_root = _find_project_root()
    # 把 src 加到 sys.path,让 `from kb...` 可以 import
    src_dir = _project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from kb.config import load_config  # noqa: WPS433

    _cfg = load_config(None)  # 用默认路径 configs/default.yaml
    log.info(f"MCP server config loaded from {_project_root}")
    return _cfg


# ---------- 创建 FastMCP app ----------


def _create_app():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "未安装 mcp Python SDK。请运行: pip install 'mcp[cli]>=1.0'"
        ) from e

    mcp = FastMCP(
        name="video-kb",
        instructions=(
            "视频知识库检索。用户问跟已入库视频内容有关的问题时,"
            "优先调用 kb_ask 获取带引用的综合答案;"
            "只需要原始 chunk 供自己综合时用 kb_query;"
            "不知道库里有哪些视频时先 kb_list_videos。"
        ),
    )

    @mcp.tool()
    def kb_ask(
        question: str,
        n: int = 8,
        video_id: str | None = None,
    ) -> str:
        """对视频知识库用自然语言提问,返回带 [ep.N @ mm:ss] 引用的综合答案。

        这是首选工具:它先做语义检索,然后用 LLM 把 Top-K chunks 综合成一段
        带原文引用的完整回答,适合"CRT 模型是什么"/"liquidity 在哪几集被展开"
        这类需要跨片段综合的问题。

        Args:
            question: 用户问题,中英文均可
            n: 检索 Top-K chunk 数,默认 8,复杂综合题可提到 12-15
            video_id: 可选,仅限指定视频(从 kb_list_videos 获取 video_id)

        Returns:
            带引用的答案文本;失败时返回 "[ERROR] ..." 字符串
        """
        try:
            cfg = _get_cfg()
            from kb.rag.answer import answer as rag_answer  # noqa: WPS433

            result = rag_answer(
                question=question,
                cfg=cfg,
                n_results=n,
                video_id=video_id,
            )
            return result["answer"]
        except Exception as e:
            log.exception("kb_ask failed")
            return f"[ERROR] kb_ask 失败: {type(e).__name__}: {e}"

    @mcp.tool()
    def kb_query(
        query: str,
        n: int = 5,
        video_id: str | None = None,
    ) -> str:
        """语义检索视频知识库,返回 Top-K 原始 chunk(不含 LLM 综合)。

        什么时候用:
        - 需要看原始素材片段(例如核对讲师原话)
        - 自己决定如何综合,不希望 kb_ask 提前判断
        - 找特定时间点的内容

        Args:
            query: 检索文本
            n: Top-K,默认 5
            video_id: 可选,限定单个视频

        Returns:
            JSON 字符串,结构:
            [
              {"rank": 1, "video_title": "...", "ep": "ep.3", "timestamp": "07:10",
               "section": "...", "distance": 0.842, "has_visual": true,
               "text": "...", "frame_path": "/path/to/07-10.jpg 或 null"},
              ...
            ]
        """
        try:
            import re

            cfg = _get_cfg()
            from kb.storage import chroma_client  # noqa: WPS433
            from kb.utils import format_timestamp_short  # noqa: WPS433

            where = {"video_id": video_id} if video_id else None
            hits = chroma_client.query(
                query,
                cfg["paths"]["chroma_dir"],
                cfg["embedding"],
                n_results=n,
                where=where,
            )
            if not hits:
                return json.dumps([], ensure_ascii=False)

            ep_re = re.compile(r"ep[_\.\s]*(\d+)", re.IGNORECASE)
            videos_dir = Path(cfg["paths"]["videos_dir"])
            if not videos_dir.is_absolute():
                videos_dir = (_project_root or Path.cwd()) / videos_dir

            out = []
            for i, h in enumerate(hits, 1):
                md = h["metadata"]
                vid = md.get("video_id", "")
                title = md.get("video_title", "")
                ep_m = ep_re.search(vid) or ep_re.search(title)
                ep_tag = f"ep.{ep_m.group(1)}" if ep_m else title[:30]
                start = float(md.get("start_sec", 0))
                ts = format_timestamp_short(start)
                frame_path = None
                if md.get("has_visual"):
                    candidate = (
                        videos_dir / vid / "frames" / f"{ts.replace(':', '-')}.jpg"
                    )
                    if candidate.exists():
                        frame_path = str(candidate)

                out.append(
                    {
                        "rank": i,
                        "video_id": vid,
                        "video_title": title,
                        "ep": ep_tag,
                        "timestamp": ts,
                        "section": md.get("section_title", ""),
                        "distance": h.get("distance"),
                        "has_visual": md.get("has_visual", False),
                        "text": h["text"],
                        "frame_path": frame_path,
                    }
                )
            return json.dumps(out, ensure_ascii=False, indent=2)
        except Exception as e:
            log.exception("kb_query failed")
            return f"[ERROR] kb_query 失败: {type(e).__name__}: {e}"

    @mcp.tool()
    def kb_list_videos() -> str:
        """列出知识库里所有已 ingest 的视频。

        Claude 在不清楚库里有什么内容时应该先调用这个,了解范围。

        Returns:
            JSON 字符串,每项含 video_id / title / duration_sec / status flags
        """
        try:
            import yaml

            cfg = _get_cfg()
            videos_root = Path(cfg["paths"]["videos_dir"])
            if not videos_root.is_absolute():
                videos_root = (_project_root or Path.cwd()) / videos_root
            if not videos_root.exists():
                return json.dumps([], ensure_ascii=False)

            out = []
            for vdir in sorted(videos_root.iterdir()):
                if not vdir.is_dir():
                    continue
                meta_p = vdir / "meta.yaml"
                if not meta_p.exists():
                    continue
                with open(meta_p, encoding="utf-8") as f:
                    m = yaml.safe_load(f) or {}
                out.append(
                    {
                        "video_id": m.get("video_id", ""),
                        "title": m.get("title", ""),
                        "duration_sec": m.get("duration_sec", 0),
                        "has_transcript": bool(m.get("has_transcript")),
                        "has_vision": bool(m.get("has_vision")),
                        "has_notes": bool(m.get("has_notes")),
                        "has_embeddings": bool(m.get("has_embeddings")),
                    }
                )
            return json.dumps(out, ensure_ascii=False, indent=2)
        except Exception as e:
            log.exception("kb_list_videos failed")
            return f"[ERROR] kb_list_videos 失败: {type(e).__name__}: {e}"

    @mcp.tool()
    def kb_stats() -> str:
        """向量库总览:总 chunk 数 + 每个视频的 chunk 数。

        Returns:
            JSON 字符串: {"total_chunks": N, "videos": {video_id: chunk_count, ...}}
        """
        try:
            cfg = _get_cfg()
            from kb.storage import chroma_client  # noqa: WPS433

            info = chroma_client.stats(cfg["paths"]["chroma_dir"])
            return json.dumps(info, ensure_ascii=False, indent=2)
        except Exception as e:
            log.exception("kb_stats failed")
            return f"[ERROR] kb_stats 失败: {type(e).__name__}: {e}"

    return mcp


# ---------- 入口 ----------


def main() -> None:
    """stdio transport 启动 — 给 Claude Code 用。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # MCP stdio 协议要求 stdout 纯净,log 只能走 stderr
    )
    app = _create_app()
    app.run()  # 默认 stdio


if __name__ == "__main__":
    main()
