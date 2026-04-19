"""从 ChromaDB 语料抽取领域专业术语,生成 aliases 建议。

设计思路:
- 从向量库里按 source_id 均衡采样 (避免被单个最大视频主导)
- 编号成片段块喂给 Claude CLI
- 让 Claude 按『canonical + aliases』格式输出 YAML
- 写到 configs/aliases.suggested.yaml,用户人工 review 后合并到 aliases.yaml
  (不自动合并 — 自动合并容易污染主词典)
"""
from __future__ import annotations

import logging
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("kb.retrieval")


SUGGEST_PROMPT = """我有一个基于视频 + PDF 文档的专业知识库(领域:金融交易教育,特别是 ICT / Smart Money Concepts 方向)。下面是从语料里代表性抽样的 <<N>> 段内容。

请从这些内容里**抽取领域专业术语**,按『同义词组』的格式输出,供我们在 RAG 的查询扩展使用。

## 抽取规则

1. **只要专业术语**:交易机制 (CRT / Turtle Soup / Key Level / Displacement / FVG / Order Block / BOS / CHoCH / Kill Zone)、分析方法、讲师个人命名 (Model #1, Kiss of Death Turtle Soup) 等。
2. **忽略通用词**:像"然后"、"非常"、"一个"、"蜡烛"、"买入"这种不是专业术语的日常词 **不要** 抽。
3. **一组覆盖多种表达方式** — 这是关键,因为 aliases 的目的是让用户无论用哪个叫法都能命中:
   - 英文原文 + 中文译名(如 "Kill Zone" / "杀戮时段" / "交易时段")
   - 缩写 + 全称(如 "KL" / "Key Level" / "关键价位" / "POI")
   - 讲师惯用的特殊叫法(如果这组术语在语料里出现了特定别名)
   - 常见同义词(如 "invalidation" / "stop loss" / "SL" / "止损")
4. **每组 3-6 个词**,太多会稀释信号,太少起不到扩展作用。
5. **canonical 选最常见/最清晰的那个**作为规范词。中英任选,哪个更通用用哪个。
6. **只从语料里出现过的词抽**,不要自己脑补领域里应该有但语料里没出现的词。

## 输出格式

严格的 YAML 列表。**不要 markdown 围栏,不要任何解释或前后文**,直接给出 YAML 内容,就像这样:

- canonical: 入场信号
  aliases: [entry signal, entry, trigger, setup, entry model]

- canonical: Key Level
  aliases: [关键价位, KL, key level, POI, point of interest, liquidity level]

- canonical: 止损
  aliases: [stop loss, SL, invalidation, risk]

## 语料片段 (共 <<N>> 段)

<<SNIPPETS>>

## 开始输出

直接给 YAML,别的都不要:"""


def _sample_chunks(
    all_chunks: list[dict[str, Any]],
    sample_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """从所有 chunk 里分层采样:按 source_id 分桶,每桶均摊,保证覆盖多源。

    如果 total <= sample_size,全量返回(不采样)。
    """
    if len(all_chunks) <= sample_size:
        return list(all_chunks)

    # 按 source_id 分桶
    buckets: dict[str, list[dict[str, Any]]] = {}
    for c in all_chunks:
        sid = c.get("metadata", {}).get("video_id", "unknown")
        buckets.setdefault(sid, []).append(c)

    # 每桶分到的额度(向上取整,最后截断到 sample_size)
    n_buckets = len(buckets)
    per_bucket = max(1, sample_size // n_buckets)
    rng = random.Random(seed)

    picked: list[dict[str, Any]] = []
    for sid, items in buckets.items():
        rng.shuffle(items)
        picked.extend(items[:per_bucket])

    # 还没满就从整体里补足
    if len(picked) < sample_size:
        remaining = [c for c in all_chunks if c not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: sample_size - len(picked)])
    # 太多就截
    return picked[:sample_size]


def _format_snippets(chunks: list[dict[str, Any]], max_chars_per_chunk: int = 400) -> str:
    """把 chunk 编号拼成 Claude prompt 里喂的片段块。每段截断到 max_chars。"""
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        md = c.get("metadata", {})
        sid = md.get("video_id", "?")
        stype = md.get("source_type", "video")
        text = (c.get("text") or "")[:max_chars_per_chunk]
        lines.append(f"--- #{i} ({stype}/{sid}) ---\n{text.strip()}")
    return "\n\n".join(lines)


def suggest_aliases(
    chroma_path: Path | str,
    out_path: Path | str,
    claude_model: str = "sonnet",
    sample_size: int = 80,
    timeout_sec: int = 180,
    max_chars_per_chunk: int = 400,
) -> dict[str, Any]:
    """扫 ChromaDB 语料 → Claude CLI 抽术语 → 写 suggestions YAML。

    Returns:
        {
            "sample_size": 实际采样数,
            "total_chunks": 库中总 chunk 数,
            "out_path": 结果文件路径,
            "prompt_chars": prompt 字符数,
            "raw_output_chars": Claude 原始返回字符数,
        }
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请安装 Claude Code 并 `claude login`。"
        )

    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb 未安装") from e

    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_or_create_collection("videos")  # 集合名常量与 chroma_client 对齐
    total = col.count()
    if total == 0:
        raise RuntimeError(
            "ChromaDB 里没有 chunk,先跑 `kb batch` 或 `kb ingest-doc`。"
        )

    got = col.get(include=["documents", "metadatas"])
    all_chunks = [
        {"id": i, "text": d, "metadata": m}
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    sampled = _sample_chunks(all_chunks, sample_size)
    snippets = _format_snippets(sampled, max_chars_per_chunk=max_chars_per_chunk)

    prompt = (
        SUGGEST_PROMPT.replace("<<N>>", str(len(sampled))).replace(
            "<<SNIPPETS>>", snippets
        )
    )

    log.info(
        f"Suggest aliases: sampled {len(sampled)}/{total} chunks "
        f"→ prompt {len(prompt) // 1000}k chars → Claude ({claude_model})..."
    )

    proc = subprocess.run(
        [claude_bin, "-p", "--output-format", "text", "--model", claude_model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}"
        )

    raw = (proc.stdout or "").strip()
    # 清掉 Claude 偶尔加的 markdown 围栏
    if raw.startswith("```"):
        # 去首尾围栏
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
            if raw.lstrip().startswith("yaml"):
                raw = raw.lstrip()[4:].lstrip("\n")

    # 验证能被 YAML 解析(失败就原样写盘,让用户看)
    parsed_ok = False
    try:
        import yaml
        data = yaml.safe_load(raw)
        if isinstance(data, list):
            parsed_ok = True
    except Exception as e:
        log.warning(f"Claude 输出非合法 YAML,原样写盘让你手工修: {e}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# kb aliases suggest — 自动抽取的术语建议\n"
        "# ⚠️ 这是建议稿,不是正式词典。审阅后把想要的条目合并到 configs/aliases.yaml\n"
        f"# 基于语料采样 {len(sampled)}/{total} 个 chunks 生成 (model={claude_model})\n"
        + ("#" if parsed_ok else "# ⚠️ YAML 解析失败 —— 下面可能是纯文本,请手工修正")
        + "\n\n"
    )
    out_path.write_text(header + raw + "\n", encoding="utf-8")

    log.info(f"✅ Suggestions written: {out_path}")
    return {
        "sample_size": len(sampled),
        "total_chunks": total,
        "out_path": str(out_path),
        "prompt_chars": len(prompt),
        "raw_output_chars": len(raw),
        "parsed_ok": parsed_ok,
    }
