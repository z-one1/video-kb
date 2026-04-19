"""ChromaDB 本地向量库 — 写入与查询。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..schemas import Chunk

log = logging.getLogger("kb.storage")

COLLECTION_NAME = "videos"


def _client(db_path: Path | str):
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb 未安装 — pip install chromadb") from e
    return chromadb.PersistentClient(path=str(db_path))


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    db_path: Path | str,
    video_meta: dict[str, Any] | None = None,
) -> int:
    """把 chunk 批量写入 ChromaDB,去重(按 chunk_id)。

    向后兼容:video_meta 里的 title/source 原来只给视频用,现在也用于 doc 的显示。
    metadata 新增字段:
      - source_type: 'video' | 'pdf' | 'image'
      - page_num: PDF 专用
      - source_path: doc 专用(原始文件名)
    旧数据读取时:缺失字段由调用方(answer.py)按 source_type 默认 'video' 处理。
    """
    assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"
    if not chunks:
        return 0

    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)

    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = []
    for c in chunks:
        m: dict[str, Any] = {
            "video_id": c.video_id,  # 语义 = source_id
            "start_sec": c.start_sec if c.start_sec is not None else 0.0,
            "end_sec": c.end_sec if c.end_sec is not None else 0.0,
            "section_title": c.section_title or "",
            "has_visual": bool(c.has_visual),
            "source_type": c.source_type or "video",
        }
        # Chroma 不接受 None 值的 metadata,按需写入
        if c.page_num is not None:
            m["page_num"] = int(c.page_num)
        if c.source_path:
            m["source_path"] = c.source_path
        if video_meta:
            # 对 doc 也复用这两个字段做显示标题
            m["video_title"] = video_meta.get("title", "")
            m["source"] = video_meta.get("source", "")
        metas.append(m)

    # Chroma 自动 upsert 靠 ids 去重
    col.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    log.info(f"Upserted {len(chunks)} chunks to ChromaDB ({db_path})")
    return len(chunks)


def query(
    query_text: str,
    db_path: Path | str,
    embed_cfg: dict[str, Any],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
    aliases_lookup: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """基于查询文本检索最近似的 N 条 chunk。

    Args:
        aliases_lookup: 可选的同义词 lookup(由 retrieval.aliases.load_aliases 产出),
            提供则对 query_text 做一次扩展再嵌入。
    """
    from ..embedding.bge import embed_texts

    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)

    # 查询扩展(可选)— 把命中组内其他同义词拼到 query 末尾
    effective_query = query_text
    if aliases_lookup:
        from ..retrieval.aliases import expand_query
        effective_query = expand_query(query_text, aliases_lookup)
        if effective_query != query_text:
            log.info(
                f"Query expanded: {len(query_text)}ch → {len(effective_query)}ch"
            )

    q_emb = embed_texts([effective_query], embed_cfg)[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        where=where,
    )

    hits: list[dict[str, Any]] = []
    if not res["ids"]:
        return hits
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "id": res["ids"][0][i],
                "distance": res["distances"][0][i] if res.get("distances") else None,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
            }
        )
    return hits


def delete_by_video_id(video_id: str, db_path: Path | str) -> int:
    """按 video_id 删除该 source 的所有 chunk — 用于 reindex / re-ingest 前清库。

    命名为 video_id 是历史沿用,实际对 video / pdf / image 三种 source 都通用
    (它们共享 video_id metadata 字段,doc 侧用 pdf_xxx / img_xxx 前缀避免冲突)。
    集合不存在/为空时返回 0。
    """
    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)
    # Chroma 的 where 过滤:直接按 metadata.video_id 取
    existing = col.get(where={"video_id": video_id})
    ids = existing.get("ids", []) or []
    if not ids:
        log.info(f"delete_by_video_id({video_id}): nothing to delete")
        return 0
    col.delete(ids=ids)
    log.info(f"Deleted {len(ids)} chunks for video_id={video_id}")
    return len(ids)


# 新语义别名 — 后续代码优先用这个
delete_by_source_id = delete_by_video_id


def stats(db_path: Path | str) -> dict[str, Any]:
    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)
    count = col.count()

    # 汇总:每个 source 的 chunk 数 + 按 source_type 拆分
    per_source: dict[str, int] = {}
    per_type: dict[str, int] = {}
    if count > 0:
        all_meta = col.get(include=["metadatas"])
        for m in all_meta.get("metadatas", []) or []:
            sid = m.get("video_id", "unknown")
            per_source[sid] = per_source.get(sid, 0) + 1
            stype = m.get("source_type", "video")  # 旧数据默认 video
            per_type[stype] = per_type.get(stype, 0) + 1

    return {
        "total_chunks": count,
        "videos": per_source,  # 保留原键名,向后兼容 CLI
        "per_type": per_type,
    }
