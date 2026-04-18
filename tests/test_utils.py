"""核心工具函数单元测试 — 不需要外部依赖,快速验证环境。"""
from kb.utils import format_timestamp, format_timestamp_short, slug_video_id


def test_format_timestamp_basic():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(61) == "00:01:01"
    assert format_timestamp(3661) == "01:01:01"


def test_format_timestamp_short():
    assert format_timestamp_short(0) == "00:00"
    assert format_timestamp_short(59.9) == "00:59"
    assert format_timestamp_short(125) == "02:05"
    assert format_timestamp_short(3661) == "01:01:01"  # > 1h 时用长格式


def test_slug_video_id_stable():
    # 同样输入 → 同样 ID
    assert slug_video_id("lecture.mp4") == slug_video_id("lecture.mp4")
    # 不同输入 → 不同 ID
    assert slug_video_id("a.mp4") != slug_video_id("b.mp4")


def test_slug_video_id_handles_paths():
    s1 = slug_video_id("/tmp/lecture.mp4")
    s2 = slug_video_id("lecture.mp4")
    # 路径不同 → hash 不同,但都合法
    assert s1.startswith("lecture_")
    assert s2.startswith("lecture_")
    assert s1 != s2
