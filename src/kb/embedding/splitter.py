"""轻量级递归字符切分器 — 替换 langchain_text_splitters 的 RecursiveCharacterTextSplitter。

替换动机:
- 仅为 split_text() 一个方法引入 langchain → nltk → sklearn → pandas → numexpr
  → bottleneck 的传递依赖,拖出 NumPy 2.x 编译兼容警告,每次启动污染日志几百行。
- video-kb 对 splitter 的需求是"按结构分隔符递归切、保持 chunk_size 上限附近、支持
  overlap",~100 行纯 Python 能覆盖,无需第三方。

算法(对齐 langchain 的 RecursiveCharacterTextSplitter):
1. 按 separators 顺序挑出第一个在 text 里出现的分隔符(都没命中则字符级硬切)
2. 用该分隔符切 text
3. 每个片段:若长度 <= chunk_size 即"合格",否则用后续 separators 递归切
4. 合格片段列表做 greedy merge:累加到接近 chunk_size 就 flush;flush 时保留
   最后 ~chunk_overlap 个字符作为下一块的起始(滑动窗口)
5. 硬切分支直接按 step = chunk_size - chunk_overlap 下标切,片段天然带 overlap,
   merge 阶段遇到满尺寸片段会丢掉 tail 避免超长
"""
from __future__ import annotations

from typing import Sequence


def recursive_char_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> list[str]:
    """按分隔符递归切分 text,合并相邻小片段到接近 chunk_size 的块。

    Args:
        text: 要切的整段文字
        chunk_size: 单 chunk 目标字符数上限(greedy merge 尽量不超)
        chunk_overlap: 相邻 chunk 重叠字符数(硬切和 merge 两阶段都生效)
        separators: 候选分隔符列表,顺序即优先级(前者匹配不到才用后者)

    Returns:
        chunk 字符串列表,过滤掉 strip 后为空的项
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    pieces = _split_recursive(text, chunk_size, chunk_overlap, list(separators))
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def _hard_split(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, str]]:
    """字符级硬切,相邻切片带 chunk_overlap 字符重叠。"""
    step = max(1, chunk_size - chunk_overlap)
    return [(text[i : i + chunk_size], "") for i in range(0, len(text), step)]


def _split_recursive(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str]
) -> list[tuple[str, str]]:
    """递归切到所有片段 <= chunk_size 或 separators 用尽。

    返回 [(片段内容, 切出该片段的分隔符)],后者给 merge 阶段还原原 join 字符用。
    """
    if not separators:
        return _hard_split(text, chunk_size, chunk_overlap)

    # 挑出第一个实际出现的 separator;"" 代表"直接退字符级"
    sep: str | None = None
    remaining: list[str] = []
    for i, s in enumerate(separators):
        if s == "":
            return _hard_split(text, chunk_size, chunk_overlap)
        if s in text:
            sep = s
            remaining = separators[i + 1 :]
            break

    if sep is None:
        # 没命中任何 separator → 退字符级
        return _hard_split(text, chunk_size, chunk_overlap)

    parts = text.split(sep)
    out: list[tuple[str, str]] = []
    for p in parts:
        if not p:
            continue
        if len(p) <= chunk_size:
            out.append((p, sep))
        else:
            out.extend(_split_recursive(p, chunk_size, chunk_overlap, remaining))
    return out


def _merge_pieces(
    pieces: list[tuple[str, str]], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """把切碎的小片段贪心合并到接近 chunk_size 的块,带 overlap。"""
    if not pieces:
        return []

    chunks: list[str] = []
    buf: list[str] = []  # 当前正在累积的片段(含各片段原始 sep)
    buf_len = 0  # buf 里内容的字符总长(含中间 sep)

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        merged = "".join(buf).strip()
        if merged:
            chunks.append(merged)
        # 保留尾部 chunk_overlap 字符作为下一块的起始
        if chunk_overlap > 0 and merged:
            tail = merged[-chunk_overlap:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf = []
            buf_len = 0

    for piece, sep in pieces:
        piece_len = len(piece)
        sep_len = len(sep) if buf else 0
        if buf_len + piece_len + sep_len > chunk_size and buf:
            flush()
            # flush 完 buf 里只剩 overlap tail。若 tail + piece 仍然超 chunk_size
            # (典型:hard_split 出的满尺寸片段),丢掉 tail — piece 自己已经带
            # 了硬切阶段的内置 overlap,不需要再叠一层。
            if buf_len + piece_len > chunk_size:
                buf = []
                buf_len = 0
        if buf:
            buf.append(sep)
            buf_len += len(sep)
        buf.append(piece)
        buf_len += piece_len
    flush()

    return chunks
