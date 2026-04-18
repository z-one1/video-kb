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
    """把 chunk 批量写入 ChromaDB,去重(按 chunk_id)。"""
    assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"
    if not chunks:
        return 0

    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)

    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = []
    for c in chunks:
        m = {
            "video_id": c.video_id,
            "start_sec": c.start_sec if c.start_sec is not None else 0.0,
            "end_sec": c.end_sec if c.end_sec is not None else 0.0,
            "section_title": c.section_title or "",
            "has_visual": bool(c.has_visual),
        }
        if video_meta:
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
) -> list[dict[str, Any]]:
    """基于查询文本检索最近似的 N 条 chunk。"""
    from ..embedding.bge import embed_texts

    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)

    q_emb = embed_texts([query_text], embed_cfg)[0]
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


def stats(db_path: Path | str) -> dict[str, Any]:
    client = _client(db_path)
    col = client.get_or_create_collection(COLLECTION_NAME)
    count = col.count()

    # 汇总每个 video 的 chunk 数量
    per_video: dict[str, int] = {}
    if count > 0:
        all_meta = col.get(include=["metadatas"])
        for m in all_meta.get("metadatas", []) or []:
            vid = m.get("video_id", "unknown")
            per_video[vid] = per_video.get(vid, 0) + 1

    return {
        "total_chunks": count,
        "videos": per_video,
    }
