"""bge-m3 本地嵌入模型。"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kb.embedding")

_cached_model = None


def get_model(name: str = "BAAI/bge-m3"):
    """懒加载嵌入模型(首次会下载 ~2GB)。"""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers 未安装 — pip install sentence-transformers"
        ) from e

    log.info(f"Loading embedding model: {name} (first time ~2GB download)")
    _cached_model = SentenceTransformer(name)
    return _cached_model


def embed_texts(
    texts: list[str], cfg: dict[str, Any]
) -> list[list[float]]:
    model = get_model(cfg.get("model", "BAAI/bge-m3"))
    batch_size = cfg.get("batch_size", 32)
    normalize = cfg.get("normalize", True)
    log.info(f"Embedding {len(texts)} texts (batch={batch_size})")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return vecs.tolist()
