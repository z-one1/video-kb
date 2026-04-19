"""tests for kb.utils.parse_json_block — 通用 Claude CLI JSON 响应解析。

覆盖三种真实场景 + 边界:
- 纯 JSON
- ```json 围栏
- ``` 无语言标识围栏
- 前后有解释性文字("Here's the JSON: {...}")
- 嵌套对象
- 空输入 / 非 JSON / 破损 JSON → None
- 顶层是数组(不是 dict) → None(按契约)
"""
from __future__ import annotations

from kb.utils import parse_json_block


class TestCleanJson:
    def test_plain_object(self):
        raw = '{"description": "A diagram", "extracted_text": "CHoCH"}'
        out = parse_json_block(raw)
        assert out == {"description": "A diagram", "extracted_text": "CHoCH"}

    def test_with_whitespace(self):
        raw = '\n\n  {"a": 1}  \n'
        assert parse_json_block(raw) == {"a": 1}

    def test_nested_object(self):
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        out = parse_json_block(raw)
        assert out == {"outer": {"inner": [1, 2, 3]}, "flag": True}


class TestMarkdownFences:
    def test_json_fence(self):
        raw = '```json\n{"description": "Hello", "extracted_text": null}\n```'
        out = parse_json_block(raw)
        assert out == {"description": "Hello", "extracted_text": None}

    def test_bare_fence(self):
        raw = '```\n{"a": 1, "b": 2}\n```'
        assert parse_json_block(raw) == {"a": 1, "b": 2}

    def test_fence_with_prose_outside(self):
        raw = (
            "Sure! Here's the JSON:\n"
            "```json\n"
            '{"description": "diagram", "extracted_text": "KL"}\n'
            "```\n"
            "Hope this helps."
        )
        out = parse_json_block(raw)
        assert out == {"description": "diagram", "extracted_text": "KL"}


class TestEmbeddedInProse:
    def test_prose_prefix(self):
        raw = 'Here you go: {"description": "chart"}'
        assert parse_json_block(raw) == {"description": "chart"}

    def test_prose_suffix(self):
        raw = '{"description": "chart"} (end of response)'
        assert parse_json_block(raw) == {"description": "chart"}

    def test_prose_both_sides(self):
        raw = 'prefix {"x": 1, "y": 2} suffix'
        assert parse_json_block(raw) == {"x": 1, "y": 2}


class TestFailureModes:
    def test_empty(self):
        assert parse_json_block("") is None

    def test_whitespace_only(self):
        assert parse_json_block("   \n\n  ") is None

    def test_plain_text_no_json(self):
        assert parse_json_block("This is just prose, no braces at all") is None

    def test_malformed_json(self):
        # 有 {...} 但不合法
        assert parse_json_block('{"description": unclosed') is None

    def test_json_array_not_object(self):
        # 顶层数组按契约不接受(返回 None,让调用方换 parser)
        assert parse_json_block('[1, 2, 3]') is None

    def test_non_string_input_safe(self):
        # 防御性测试:None 不应崩
        assert parse_json_block(None) is None  # type: ignore[arg-type]


class TestRealWorldShapes:
    """从 video-kb 实际 Claude CLI 输出里见过的形状。"""

    def test_vision_response(self):
        # DEFAULT_PROMPT 指定的形式
        raw = (
            '{"description": "The frame shows a trading chart with CHoCH '
            'annotation.", "extracted_text": "Key Level 1.0850"}'
        )
        out = parse_json_block(raw)
        assert out is not None
        assert out["description"].startswith("The frame shows")
        assert out["extracted_text"] == "Key Level 1.0850"

    def test_claude_occasionally_wraps_in_fence(self):
        """Claude sometimes adds a fence despite being told not to."""
        raw = (
            '```json\n'
            '{\n'
            '  "description": "A whiteboard with ICT diagrams",\n'
            '  "extracted_text": "BOS / CHoCH / POI"\n'
            '}\n'
            '```'
        )
        out = parse_json_block(raw)
        assert out == {
            "description": "A whiteboard with ICT diagrams",
            "extracted_text": "BOS / CHoCH / POI",
        }
