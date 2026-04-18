"""测试时间戳对齐逻辑 — 纯逻辑,无外部依赖。"""
from kb.fusion.align import align
from kb.schemas import Transcript, TranscriptSegment, VisualDescription


def _make_transcript():
    return Transcript(
        video_id="test",
        language="en",
        duration_sec=100.0,
        segments=[
            TranscriptSegment(start_sec=0, end_sec=10, text="Hello"),
            TranscriptSegment(start_sec=10, end_sec=20, text="World"),
            TranscriptSegment(start_sec=20, end_sec=30, text="Goodbye"),
        ],
    )


def _make_visuals():
    return [
        VisualDescription(
            frame_id="0000",
            t_sec=5.0,
            image_path="frames/0000.jpg",
            description="Intro slide",
        ),
        VisualDescription(
            frame_id="0015",
            t_sec=15.0,
            image_path="frames/0015.jpg",
            description="Diagram",
        ),
        VisualDescription(
            frame_id="0025",
            t_sec=25.0,
            image_path="frames/0025.jpg",
            description="Code sample",
        ),
    ]


def test_align_cover_mode():
    tx = _make_transcript()
    vis = _make_visuals()
    enriched = align(tx, vis, mode="cover")

    assert len(enriched) == 3
    assert len(enriched[0].visual_descriptions) == 1
    assert enriched[0].visual_descriptions[0].description == "Intro slide"
    assert enriched[1].visual_descriptions[0].description == "Diagram"
    assert enriched[2].visual_descriptions[0].description == "Code sample"


def test_align_nearest_mode():
    tx = _make_transcript()
    vis = _make_visuals()
    enriched = align(tx, vis, mode="nearest")

    # 每段字幕应该有最多 1 张近邻帧
    for seg in enriched:
        assert len(seg.visual_descriptions) <= 1


def test_align_no_visuals():
    tx = _make_transcript()
    enriched = align(tx, [], mode="cover")
    assert len(enriched) == 3
    assert all(len(s.visual_descriptions) == 0 for s in enriched)
