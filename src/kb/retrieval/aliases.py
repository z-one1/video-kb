"""概念别名词典 — 查询时做同义词扩展,提升召回。

设计目标:
- 词典是用户可编辑的纯文本 YAML,不依赖向量库
- 查询时子串匹配(兼容中文无空格),命中任一同义词就把整组加到查询末尾
- 一次嵌入,一次检索 — 不做 multi-query,保持 Claude 调用次数不变
- 文件不存在/为空时静默跳过,不影响默认流程

YAML 格式 (configs/aliases.yaml):
    - canonical: 入场信号
      aliases: [entry signal, entry, trigger, setup]
    - canonical: 关键价位
      aliases: [key level, POI, liquidity level]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("kb.retrieval")


def load_aliases(path: Path | str | None) -> dict[str, list[str]]:
    """读 aliases.yaml,返回 {term_lower → [组内所有词(原大小写)]}。

    任一词都指向整组,实现双向扩展(查 canonical 或任一 alias 都能触发)。
    文件不存在或为空返回 {}。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError as e:
        raise ImportError("pyyaml 未安装") from e

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or []

    if not isinstance(data, list):
        log.warning(f"aliases.yaml 格式异常 (应为列表): {p}")
        return {}

    lookup: dict[str, list[str]] = {}
    for group in data:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical")
        aliases = group.get("aliases") or []
        if not canonical:
            continue
        all_terms = [canonical] + [a for a in aliases if a]
        # 去重保持顺序
        seen = set()
        uniq = []
        for t in all_terms:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        for term in uniq:
            lookup[str(term).lower()] = uniq
    log.info(f"Loaded {len(lookup)} alias term(s) from {p}")
    return lookup


def _expand(query: str, lookup: dict[str, list[str]]) -> tuple[str, list[str]]:
    """内部函数:返回 (扩展后字符串, 新增别名列表[原大小写])。

    算法:
      1. 子串匹配找到命中的词典 term
      2. 每个命中组只扩展一次(用 id() 去重)
      3. 已出现在原 query 的词不重复加
    """
    if not lookup or not query:
        return query, []

    q_lower = query.lower()
    additions: list[str] = []
    added: set[str] = set()

    already_in_query = {t for t in lookup if t in q_lower}

    expanded_groups: set[int] = set()
    for term in already_in_query:
        group = lookup[term]
        gid = id(group)
        if gid in expanded_groups:
            continue
        expanded_groups.add(gid)
        for other in group:
            other_lower = other.lower()
            if other_lower in q_lower or other_lower in added:
                continue
            additions.append(other)
            added.add(other_lower)

    if not additions:
        return query, []
    return query + " " + " ".join(additions), additions


def expand_query(query: str, lookup: dict[str, list[str]]) -> str:
    """若 query 命中任一 alias term(子串,忽略大小写),把该组其他词拼到末尾。

    命中多组时各组扩展都加。已出现在 query 里的词不重复加。
    """
    expanded, added = _expand(query, lookup)
    if added:
        log.debug(f"Query expanded: {query!r} + {added} → {expanded!r}")
    return expanded


def preview_expansion(query: str, path: Path | str | None) -> dict[str, Any]:
    """给 CLI `kb aliases check` 用的一步式预览。

    Returns:
        {
            "original": str,
            "expanded": str,
            "added": list[str],    # 本次新增的同义词(原大小写,多词别名保持完整)
            "hits": list[str],     # 命中的词典 term
        }
    """
    lookup = load_aliases(path)
    if not lookup:
        return {"original": query, "expanded": query, "added": [], "hits": []}

    q_lower = query.lower()
    hits = [t for t in lookup if t in q_lower]
    expanded, added = _expand(query, lookup)
    return {
        "original": query,
        "expanded": expanded,
        "added": added,
        "hits": hits,
    }
