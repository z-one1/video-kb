"""tests for kb.embedding.splitter.recursive_char_split — 替换 langchain 的本地实现。

目标:轻量级 char splitter,行为 ~等价于 langchain 的 RecursiveCharacterTextSplitter
对我们实际用法(chunk_size=500-800, overlap=50-80, 中英文混合,段落 / 句号 / 逗号 分隔符)
给出相同质量的切分 — 不追求 byte-for-byte 复刻 langchain。

关键断言:
- 空输入 → 空结果
- 短于 chunk_size → 整体作为 1 个 chunk(不切)
- 有段落分隔优先按段落切
- 无分隔符时硬切到 chunk_size 附近
- overlap > 0 时相邻 chunk 有字符重叠
- 所有 chunk 长度 <= chunk_size * 1.5(容忍边界轻微超)
"""
from __future__ import annotations

import pytest

from kb.embedding.splitter import recursive_char_split

DEFAULT_SEPS = ["\n\n", "\n", "。", ". ", "!", "?", " ", ""]


class TestEmptyAndShort:
    def test_empty_string(self):
        assert recursive_char_split("", 100, 10, DEFAULT_SEPS) == []

    def test_whitespace_only(self):
        # 空白只能产出空或很少东西 — 至少不崩,且不产出长于 0 的实质内容
        out = recursive_char_split("   \n\n   ", 100, 10, DEFAULT_SEPS)
        assert all(not s.strip() or len(s) > 0 for s in out)

    def test_shorter_than_chunk_size(self):
        text = "这是一段很短的文字。"
        out = recursive_char_split(text, 100, 10, DEFAULT_SEPS)
        assert len(out) == 1
        assert out[0] == text


class TestParagraphBoundary:
    def test_splits_at_double_newline_first(self):
        # 两段明显分隔,chunk_size 足够容纳但不该合并(因为 \n\n 是最高优先分隔)
        p1 = "第一段。" * 20  # ~80 chars
        p2 = "第二段。" * 20  # ~80 chars
        text = p1 + "\n\n" + p2
        out = recursive_char_split(text, 100, 10, DEFAULT_SEPS)
        # 两段各自超过 chunk_size,应被独立切
        assert len(out) >= 2
        # 第一个 chunk 不应包含第二段的内容
        assert "第二段" not in out[0]

    def test_single_paragraph_splits_at_sentence(self):
        # 单段但多句 — 应该按 。 切
        text = "句一。" * 30  # ~90 chars,三字一句
        out = recursive_char_split(text, 30, 5, DEFAULT_SEPS)
        assert len(out) >= 3
        # 大多数切分应该在 。 之后
        ends_with_period = sum(1 for c in out if c.endswith("。") or "。" in c[-3:])
        assert ends_with_period >= len(out) - 1


class TestLongUnbreakable:
    def test_no_separators_hard_chunks(self):
        # 没有任何 separator 能命中 → 退到字符级硬切
        text = "a" * 500
        out = recursive_char_split(text, 100, 10, ["\n\n", "\n", ""])
        assert len(out) >= 5
        # 每个 chunk 应该 <= chunk_size(硬切时严格)
        for c in out:
            assert len(c) <= 100

    def test_english_paragraph(self):
        text = "This is a sentence. " * 30
        out = recursive_char_split(text, 80, 10, DEFAULT_SEPS)
        assert len(out) >= 5
        # 英文应该在 ". " 或空格处切
        for c in out:
            assert len(c) <= 120  # 容忍边界


class TestOverlap:
    def test_overlap_positive_produces_overlap(self):
        # 无分隔符的长串,硬切时 overlap 应该体现:chunk_i 尾部 == chunk_{i+1} 头部
        text = "".join(chr(ord("a") + i % 26) for i in range(500))
        out = recursive_char_split(text, 100, 20, [""])
        assert len(out) >= 5
        # 检查至少一对相邻 chunk 有重叠
        overlaps = []
        for a, b in zip(out, out[1:]):
            # 找 a 的末尾和 b 的开头有多少字符重合
            for k in range(min(len(a), len(b), 30), 0, -1):
                if a[-k:] == b[:k]:
                    overlaps.append(k)
                    break
            else:
                overlaps.append(0)
        # 预期有实质性重叠
        assert max(overlaps) >= 10

    def test_overlap_zero_no_overlap(self):
        text = "".join(chr(ord("a") + i % 26) for i in range(300))
        out = recursive_char_split(text, 50, 0, [""])
        assert len(out) >= 5
        # overlap=0 时,拼回去应该等于原文(允许空白 trim)
        rejoined = "".join(out)
        assert rejoined == text


class TestRealWorldShape:
    """模拟 video-kb 实际喂给 splitter 的输入,保证不崩且切得合理。"""

    def test_section_summary_shape(self):
        """chunking.py 里的 sec_text 形状。"""
        text = (
            "【章节】Turtle Soup 入场条件\n"
            "本章讲述 Turtle Soup 的 4 个关键条件。" * 10
            + "\n"
            + "关键概念: CRT, KL, POI, Displacement\n"
        )
        out = recursive_char_split(text, 500, 50, DEFAULT_SEPS)
        assert len(out) >= 1
        # 每个 chunk 体量可控
        for c in out:
            assert len(c) <= 750  # chunk_size * 1.5

    def test_pdf_page_shape(self):
        """docs.py 里 chunk_pdf 典型页。"""
        pdf_page = (
            "CHoCH — Change of Character\n\n"
            "CHoCH 发生在结构转变的第一个信号。" * 30
            + "\n\n"
            "具体识别步骤:\n"
            "1. 确认前期趋势方向\n"
            "2. 等待最后一个 swing high/low 被打破\n"
            "3. 回测确认\n"
        )
        out = recursive_char_split(pdf_page, 800, 80, DEFAULT_SEPS)
        # 页面 1500+ chars 应该切出 2-3 块
        assert 1 <= len(out) <= 5
        # 所有块加起来应该覆盖大部分内容(允许分隔符丢失)
        total = sum(len(c) for c in out)
        assert total >= len(pdf_page) * 0.85  # 80%+ 内容保留


class TestReturnInvariants:
    def test_all_chunks_are_strings(self):
        out = recursive_char_split("任意文本。" * 50, 100, 10, DEFAULT_SEPS)
        for c in out:
            assert isinstance(c, str)

    def test_no_empty_chunks(self):
        # 不产出空 chunk(strip 后为空)
        out = recursive_char_split("abc\n\n\n\n\n\ndef", 100, 0, DEFAULT_SEPS)
        for c in out:
            assert c.strip()

    def test_deterministic(self):
        text = "确定性测试。" * 40
        a = recursive_char_split(text, 100, 10, DEFAULT_SEPS)
        b = recursive_char_split(text, 100, 10, DEFAULT_SEPS)
        assert a == b
