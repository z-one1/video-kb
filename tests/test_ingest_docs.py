"""kb ingest-doc 单元测试 — 覆盖纯函数 + CLI flag 传递,不调真 Claude CLI。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kb.ingest import docs as ingest_docs
from kb.ingest.docs import (
    _parse_claude_pdf_json,
    build_doc_meta,
    chunk_image,
    chunk_pdf,
    detect_doc_type,
    load_doc_meta,
    save_doc_meta,
)
from kb.schemas import DocMeta
from kb.utils import slug_doc_id


# ============ detect_doc_type ============


def test_detect_doc_type_pdf_lowercase():
    assert detect_doc_type(Path("notes.pdf")) == "pdf"


def test_detect_doc_type_pdf_uppercase():
    assert detect_doc_type(Path("FOO.PDF")) == "pdf"


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"])
def test_detect_doc_type_image_all_extensions(ext):
    assert detect_doc_type(Path(f"chart{ext}")) == "image"
    assert detect_doc_type(Path(f"CHART{ext.upper()}")) == "image"


@pytest.mark.parametrize("ext", [".docx", ".txt", ".mp4", ".md", ""])
def test_detect_doc_type_unsupported(ext):
    assert detect_doc_type(Path(f"file{ext}")) is None


# ============ slug_doc_id ============


def test_slug_doc_id_stable():
    a = slug_doc_id("/abs/path/notes.pdf", "pdf")
    b = slug_doc_id("/abs/path/notes.pdf", "pdf")
    assert a == b


def test_slug_doc_id_no_collision():
    a = slug_doc_id("/abs/path/notes.pdf", "pdf")
    b = slug_doc_id("/abs/other/notes.pdf", "pdf")
    assert a != b


def test_slug_doc_id_prefix():
    assert slug_doc_id("/x/foo.pdf", "pdf").startswith("pdf_")
    assert slug_doc_id("/x/bar.png", "image").startswith("img_")


def test_slug_doc_id_same_stem_different_type():
    # PDF 和同名 image 不应产生同 id(prefix 区分)
    a = slug_doc_id("/x/foo.pdf", "pdf")
    b = slug_doc_id("/x/foo.pdf", "image")
    assert a != b
    assert a.startswith("pdf_")
    assert b.startswith("img_")


# ============ _parse_claude_pdf_json ============


def test_parse_claude_pdf_json_clean():
    raw = '[{"page_num": 1, "text": "hello"}, {"page_num": 2, "text": "world"}]'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert pages == [
        {"page_num": 1, "text": "hello"},
        {"page_num": 2, "text": "world"},
    ]


def test_parse_claude_pdf_json_markdown_fence():
    raw = '```json\n[{"page_num": 1, "text": "a"}]\n```'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert pages == [{"page_num": 1, "text": "a"}]


def test_parse_claude_pdf_json_bare_fence():
    raw = '```\n[{"page_num": 1, "text": "a"}]\n```'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert pages == [{"page_num": 1, "text": "a"}]


def test_parse_claude_pdf_json_with_preamble():
    raw = '好的,这是 JSON:\n[{"page_num": 1, "text": "preamble ok"}]'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert pages == [{"page_num": 1, "text": "preamble ok"}]


def test_parse_claude_pdf_json_invalid_raises():
    with pytest.raises(ValueError):
        _parse_claude_pdf_json("not json at all", "x.pdf")


def test_parse_claude_pdf_json_empty_raises():
    with pytest.raises(ValueError):
        _parse_claude_pdf_json("", "x.pdf")


def test_parse_claude_pdf_json_not_array_raises():
    with pytest.raises(ValueError):
        _parse_claude_pdf_json('{"page_num": 1}', "x.pdf")


def test_parse_claude_pdf_json_strips_text():
    raw = '[{"page_num": 1, "text": "  hello  \\n"}]'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert pages[0]["text"] == "hello"


def test_parse_claude_pdf_json_fills_missing_page_num():
    # 某条缺 page_num → 用索引补
    raw = '[{"text": "first"}, {"text": "second"}]'
    pages = _parse_claude_pdf_json(raw, "x.pdf")
    assert [p["page_num"] for p in pages] == [1, 2]


# ============ chunk_pdf ============


def _cfg(**overrides: Any) -> dict[str, Any]:
    base = {"pdf_chunk_size": 800, "pdf_chunk_overlap": 80, "pdf_min_chunk_chars": 120}
    base.update(overrides)
    return base


def test_chunk_pdf_empty_pages():
    assert chunk_pdf([], "pdf_foo", "/x/foo.pdf", _cfg()) == []


def test_chunk_pdf_skips_blank_page_text():
    pages = [
        {"page_num": 1, "text": ""},
        {"page_num": 2, "text": ""},
    ]
    assert chunk_pdf(pages, "pdf_foo", "/x/foo.pdf", _cfg()) == []


def test_chunk_pdf_single_short_page_one_chunk():
    # 短于 chunk_size → 整页一个 chunk
    pages = [{"page_num": 1, "text": "Short page text."}]
    chunks = chunk_pdf(pages, "pdf_foo", "/x/foo.pdf", _cfg())
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "Short page text."
    assert c.page_num == 1
    assert c.section_title == "Page 1"
    assert c.source_type == "pdf"
    assert c.video_id == "pdf_foo"  # source_id 复用 video_id
    assert c.source_path == "/x/foo.pdf"
    assert c.has_visual is False


def test_chunk_pdf_multi_page_chunks_do_not_span_pages():
    # 两页都足够长,验证每 chunk 的 page_num 严格属于自己那页
    p1 = "A" * 200
    p2 = "B" * 200
    pages = [{"page_num": 1, "text": p1}, {"page_num": 2, "text": p2}]
    chunks = chunk_pdf(pages, "pdf_foo", "/x/foo.pdf", _cfg())
    assert len(chunks) >= 2
    for c in chunks:
        if "A" in c.text:
            assert c.page_num == 1
            assert "B" not in c.text
        if "B" in c.text:
            assert c.page_num == 2
            assert "A" not in c.text


def test_chunk_pdf_long_page_splits_but_keeps_page_num():
    # 远超 chunk_size 的一页 → 多 chunk,全部 page_num=1
    long_text = ("段落 。" * 500)[:3000]
    pages = [{"page_num": 1, "text": long_text}]
    chunks = chunk_pdf(pages, "pdf_foo", "/x/foo.pdf", _cfg(pdf_chunk_size=400))
    assert len(chunks) >= 2
    assert all(c.page_num == 1 for c in chunks)
    assert all(c.section_title == "Page 1" for c in chunks)


def test_chunk_pdf_chunk_ids_unique_and_increment():
    long_text = ("X" * 2000)
    pages = [{"page_num": 1, "text": long_text}]
    chunks = chunk_pdf(pages, "pdf_foo", "/x/foo.pdf", _cfg(pdf_chunk_size=400))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith("pdf_foo_p0001_") for c in chunks)


# ============ chunk_image ============


def test_chunk_image_normal():
    desc = {
        "description": "一张 EURUSD H4 图表",
        "extracted_text": "EURUSD 1.0850 Kill Zone",
    }
    chunks = chunk_image(desc, "img_foo", "/x/foo.png")
    assert len(chunks) == 1
    c = chunks[0]
    assert "[图片描述]" in c.text
    assert "[屏幕文字]" in c.text
    assert "EURUSD" in c.text
    assert c.source_type == "image"
    assert c.has_visual is True
    assert c.section_title == "Image"
    assert c.video_id == "img_foo"
    assert c.source_path == "/x/foo.png"
    assert c.chunk_id == "img_foo_img_0000"


def test_chunk_image_missing_extracted_text():
    desc = {"description": "只有描述", "extracted_text": None}
    chunks = chunk_image(desc, "img_foo", "/x/foo.png")
    assert len(chunks) == 1
    assert "[图片描述]" in chunks[0].text
    assert "[屏幕文字]" not in chunks[0].text


def test_chunk_image_missing_description_still_ok_if_text():
    desc = {"description": "", "extracted_text": "just OCR text"}
    chunks = chunk_image(desc, "img_foo", "/x/foo.png")
    assert len(chunks) == 1
    assert "[屏幕文字]" in chunks[0].text
    assert "[图片描述]" not in chunks[0].text


def test_chunk_image_empty_description_returns_empty():
    desc = {"description": "", "extracted_text": None}
    assert chunk_image(desc, "img_foo", "/x/foo.png") == []


def test_chunk_image_missing_keys_returns_empty():
    # 完全空 dict 也不应崩
    assert chunk_image({}, "img_foo", "/x/foo.png") == []


# ============ build / save / load DocMeta ============


def test_build_doc_meta_pdf(tmp_path: Path):
    src = tmp_path / "notes.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    meta = build_doc_meta(src, "pdf")
    assert isinstance(meta, DocMeta)
    assert meta.source_type == "pdf"
    assert meta.doc_id.startswith("pdf_")
    assert meta.title == "notes.pdf"
    assert meta.source_path == str(src.resolve())
    assert meta.ingested_at  # 非空 ISO 串


def test_build_doc_meta_image(tmp_path: Path):
    src = tmp_path / "chart.png"
    src.write_bytes(b"\x89PNG fake")
    meta = build_doc_meta(src, "image")
    assert meta.source_type == "image"
    assert meta.doc_id.startswith("img_")


def test_doc_meta_roundtrip(tmp_path: Path):
    src = tmp_path / "notes.pdf"
    src.write_bytes(b"fake")
    meta = build_doc_meta(src, "pdf")
    meta.page_count = 7
    meta.has_text = True
    save_doc_meta(meta, tmp_path)

    loaded = load_doc_meta(tmp_path)
    assert loaded is not None
    assert loaded.doc_id == meta.doc_id
    assert loaded.source_type == "pdf"
    assert loaded.page_count == 7
    assert loaded.has_text is True


def test_load_doc_meta_missing_returns_none(tmp_path: Path):
    assert load_doc_meta(tmp_path) is None


# ============ extract_pdf_pages_via_claude (subprocess mocked) ============


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_extract_pdf_pages_via_claude_happy(monkeypatch, tmp_path):
    pdf = tmp_path / "foo.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    captured: dict[str, Any] = {}

    def fake_run(cmd, input=None, **kw):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["kwargs"] = kw
        return _FakeProc(
            stdout='[{"page_num": 1, "text": "page 1 content"}, {"page_num": 2, "text": "page 2"}]'
        )

    monkeypatch.setattr(ingest_docs, "subprocess", type("M", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: "/usr/local/bin/claude")

    pages = ingest_docs.extract_pdf_pages_via_claude(
        pdf, {"claude_model": "sonnet"}, timeout_sec=123
    )
    assert len(pages) == 2
    assert pages[0] == {"page_num": 1, "text": "page 1 content"}
    assert "--model" in captured["cmd"]
    assert "sonnet" in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 123
    assert str(pdf.resolve()) in captured["input"]


def test_extract_pdf_pages_via_claude_no_cli(monkeypatch, tmp_path):
    pdf = tmp_path / "foo.pdf"
    pdf.write_bytes(b"fake")
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="claude CLI"):
        ingest_docs.extract_pdf_pages_via_claude(pdf, {}, timeout_sec=10)


def test_extract_pdf_pages_via_claude_nonzero_exit(monkeypatch, tmp_path):
    pdf = tmp_path / "foo.pdf"
    pdf.write_bytes(b"fake")
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: "/usr/local/bin/claude")
    monkeypatch.setattr(
        ingest_docs,
        "subprocess",
        type(
            "M",
            (),
            {
                "run": staticmethod(
                    lambda *a, **kw: _FakeProc(returncode=1, stderr="boom")
                )
            },
        ),
    )
    with pytest.raises(RuntimeError, match="claude CLI failed"):
        ingest_docs.extract_pdf_pages_via_claude(pdf, {}, timeout_sec=10)


# ============ describe_image (subprocess mocked) ============


def test_describe_image_happy(monkeypatch, tmp_path):
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG fake")

    fake_json = json.dumps(
        {"description": "A trading chart.", "extracted_text": "1.0850"}
    )

    captured: dict[str, Any] = {}

    def fake_run(cmd, input=None, **kw):
        captured["cmd"] = cmd
        captured["input"] = input
        return _FakeProc(stdout=fake_json)

    monkeypatch.setattr(ingest_docs, "subprocess", type("M", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: "/usr/local/bin/claude")

    out = ingest_docs.describe_image(img, {"claude_model": "haiku", "image_prompt_lang": "zh"})
    assert out["description"] == "A trading chart."
    assert out["extracted_text"] == "1.0850"
    assert "haiku" in captured["cmd"]
    assert str(img.resolve()) in captured["input"]
    # 中文 prompt 模板标志
    assert "分析这张图片" in captured["input"]


def test_describe_image_english_prompt(monkeypatch, tmp_path):
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG fake")
    captured: dict[str, Any] = {}

    def fake_run(cmd, input=None, **kw):
        captured["input"] = input
        return _FakeProc(stdout='{"description": "x", "extracted_text": null}')

    monkeypatch.setattr(ingest_docs, "subprocess", type("M", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: "/usr/local/bin/claude")

    ingest_docs.describe_image(img, {"image_prompt_lang": "en"})
    assert "Analyze this image" in captured["input"]


def test_describe_image_no_cli(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    monkeypatch.setattr(ingest_docs.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="claude CLI"):
        ingest_docs.describe_image(img, {})


# ============ extract_pdf_pages_pypdf (real pypdf on empty minimal PDF) ============


def test_extract_pdf_pages_pypdf_on_minimal_pdf(tmp_path):
    """pypdf 在最小合法 PDF 上不崩(空文本页也算成功)。"""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    out = tmp_path / "minimal.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    with open(out, "wb") as f:
        w.write(f)

    pages = ingest_docs.extract_pdf_pages_pypdf(out)
    assert len(pages) == 2
    assert pages[0]["page_num"] == 1
    assert pages[1]["page_num"] == 2
    # 空白页无文本
    assert pages[0]["text"] == ""


# ============ Regression: CLI 把 --pdf-provider 传给 pipeline ============


def test_cli_forwards_pdf_provider_to_pipeline(tmp_path: Path, monkeypatch):
    """回归 Phase 1 bug — 确保 `kb ingest-doc foo.pdf --pdf-provider pypdf`
    真的把 'pypdf' 传到 pipeline.ingest_doc,而不是被吞掉。"""
    from typer.testing import CliRunner

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    captured: dict[str, Any] = {}

    def fake_ingest_doc(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return {
            "doc_id": "pdf_fake_deadbeef",
            "source_type": "pdf",
            "chunks_new": 0,
            "chunks_deleted": 0,
            "ddir": str(tmp_path),
        }

    # patch pipeline 里的 ingest_doc,拦截实际入库
    from kb import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "ingest_doc", fake_ingest_doc)

    from kb.cli import app
    result = CliRunner().invoke(
        app, ["ingest-doc", str(pdf), "--pdf-provider", "pypdf"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("pdf_provider") == "pypdf", (
        f"--pdf-provider was not forwarded. captured kwargs: {captured}"
    )


def test_cli_pdf_provider_default_is_none(tmp_path: Path, monkeypatch):
    """不带 --pdf-provider → 传 None,让 pipeline 走 config 默认。"""
    from typer.testing import CliRunner

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    captured: dict[str, Any] = {}

    def fake_ingest_doc(source, **kwargs):
        captured.update(kwargs)
        return {
            "doc_id": "pdf_fake_deadbeef",
            "source_type": "pdf",
            "chunks_new": 0,
            "chunks_deleted": 0,
            "ddir": str(tmp_path),
        }

    from kb import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "ingest_doc", fake_ingest_doc)

    from kb.cli import app
    result = CliRunner().invoke(app, ["ingest-doc", str(pdf)])
    assert result.exit_code == 0, result.output
    assert captured.get("pdf_provider") is None
