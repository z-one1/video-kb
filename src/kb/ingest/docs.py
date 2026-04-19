"""独立文档 (PDF + 图片) 入库 — 复用 embedding / ChromaDB / vision 基础设施。

设计要点:
- PDF:pypdf 抽每页文本 → 按页打块(每页可再切) → chunk.source_type='pdf', page_num=N
- 图片:复用 vision.claude_code.describe_frames 的单帧调用模式 → 生成 description + extracted_text
       → 单图一 chunk(偶尔切成 2-3 小块,仅当描述超过 chunk_size)
- video_id 字段复用为 source_id(前缀 'pdf_' / 'img_'),避免和视频 id 碰撞
- has_visual:图片 chunk 强制 True(方便 UI 和 RAG 过滤)

产出目录:
    kb/docs/<doc_id>/
        meta.yaml        # DocMeta
        pages.jsonl      # PDF: {"page_num": N, "text": "..."} per line
        description.json # Image: Claude vision 原始输出
        chunks.jsonl     # 分块结果 (用于 reindex / debug)
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..schemas import Chunk, DocMeta
from ..utils import now_iso, slug_doc_id

log = logging.getLogger("kb.ingest.docs")


# ============ 通用 ============

_PDF_EXT = {".pdf"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def detect_doc_type(path: Path) -> str | None:
    """返回 'pdf' / 'image' / None (不支持)。"""
    ext = path.suffix.lower()
    if ext in _PDF_EXT:
        return "pdf"
    if ext in _IMAGE_EXT:
        return "image"
    return None


# ============ PDF ============


def extract_pdf_pages_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    """pypdf provider — 纯文本抽取,快且免费。

    缺点:扫描版 PDF 抽不到(没 OCR);图表内容被忽略;多栏版式可能错乱。
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf 未安装 — 跑 `pip install pypdf`"
        ) from e

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for i, pg in enumerate(reader.pages, 1):
        try:
            text = pg.extract_text() or ""
        except Exception as e:
            log.warning(f"Page {i} extract_text failed: {e}")
            text = ""
        pages.append({"page_num": i, "text": text.strip()})
    empty = sum(1 for p in pages if not p["text"])
    if empty == len(pages):
        log.warning(
            f"⚠️  {pdf_path.name}: pypdf 所有 {len(pages)} 页都抽不到文本 — "
            f"可能是扫描版。换 --provider claude_code 走 Claude CLI(能 OCR)。"
        )
    elif empty > 0:
        log.info(f"{pdf_path.name}: {empty}/{len(pages)} 页无文本")
    return pages


PDF_CLAUDE_PROMPT = """请从这个 PDF 里逐页提取内容: @{pdf_path}

## 输出格式(严格)

纯 JSON 数组,不要 markdown 围栏,不要前后文解释。每页一个对象:

[
  {{"page_num": 1, "text": "..."}},
  {{"page_num": 2, "text": "..."}}
]

## 提取规则

1. **逐字转录正文** — 不要概括、不要总结。保留专业术语原文(英文 / 中英混合)。
2. **图表 / 截图** — 用一段话描述(包括可见数字、标注、标签),在文本里用 `[图: ...]` 前缀,接原文续写。
3. **表格** — 用 markdown 表格语法 (`| col1 | col2 |`) 保留行列结构。
4. **页眉页脚 / 水印** — 跳过(除非包含关键内容如章节号)。
5. **多栏版式** — 按阅读顺序(通常从左到右 / 从上到下)串联,不要硬按坐标切。
6. **页码**从 1 开始,按 PDF 顺序。

## 其他

- 输出必须是能被 `json.loads` 解析的合法 JSON。
- 如果某页完全空白或只有图案装饰,给 `"text": ""`(仍要保留 page_num)。
- **不要**在 JSON 前后加任何文字(包括 "这是提取的结果:" 这种)。

直接开始输出 JSON:"""


def extract_pdf_pages_via_claude(
    pdf_path: Path, cfg: dict[str, Any], timeout_sec: int = 600
) -> list[dict[str, Any]]:
    """Claude CLI provider — 通过 `claude -p @pdf_path` 让模型读整个 PDF。

    优势:扫描版能 OCR;图表被描述;表格保结构;多栏版式正确。
    劣势:慢(几秒到几十秒);非确定性;超大 PDF 可能吐不全。

    上游必须已安装 claude CLI 并 `claude login`。
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请先 `npm install -g @anthropic-ai/claude-code` 并 `claude login`。"
        )
    model = cfg.get("claude_model", "sonnet")

    prompt = PDF_CLAUDE_PROMPT.replace(
        "{pdf_path}", str(pdf_path.resolve())
    )

    log.info(
        f"Extracting PDF via Claude CLI: {pdf_path.name} (model={model}, "
        f"timeout={timeout_sec}s)"
    )
    # --permission-mode bypassPermissions: 非交互 -p 会话必须显式跳权限检查,
    # 否则 Claude 会拿 @pdf_path 触发 Read tool 然后卡在"请授权"上,subprocess
    # 拿到的 stdout 是权限提示文本而不是我们要的 JSON。
    # 用户主动敲 `kb ingest-doc <file>` 即为同意读该文件,故可直接 bypass。
    proc = subprocess.run(
        [
            claude_bin,
            "-p",
            "--output-format", "text",
            "--model", model,
            "--permission-mode", "bypassPermissions",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}"
        )

    raw = (proc.stdout or "").strip()
    _raise_if_permission_refusal(raw, pdf_path.name)
    pages = _parse_claude_pdf_json(raw, pdf_path.name)
    log.info(f"Claude extracted {len(pages)} pages from {pdf_path.name}")
    return pages


# 识别 Claude CLI 吐回"请授权"这类拒读提示的关键词 — 做智能错误消息
_PERMISSION_REFUSAL_MARKERS = (
    "需要你授权", "授权读取", "权限提示", "无法访问", "没有权限",
    "permission", "authorize", "cannot access", "don't have access",
    "not allowed to",
)


def _raise_if_permission_refusal(raw: str, fname: str) -> None:
    """Claude CLI 在 -p 模式下被权限系统拦住时,stdout 是中文/英文"请授权"提示,
    不是我们要的 JSON。捕获这种情况并抛出可操作的错误消息。
    """
    head = raw[:300].lower()
    if any(m.lower() in head for m in _PERMISSION_REFUSAL_MARKERS):
        raise RuntimeError(
            f"Claude CLI 被权限系统拦住了,没读到 {fname}。返回片段:\n"
            f"  {raw[:200].strip()}\n\n"
            f"修复方式(任选其一):\n"
            f"  1. 确认 kb 已加了 --permission-mode bypassPermissions(本版本已加,"
            f"     若仍报错,请 `pip install -e .` 重装或检查 claude CLI 版本 >= 2.x)\n"
            f"  2. 在 shell 里先跑一次 `claude --allow-dangerously-skip-permissions` "
            f"一次性授权,之后 subprocess 就不会被拦\n"
            f"  3. 临时绕过:`kb ingest-doc {fname} --pdf-provider pypdf`(仅纯文本 PDF 可接受)"
        )


def _parse_claude_pdf_json(raw: str, fname: str) -> list[dict[str, Any]]:
    """从 Claude 返回里挖 JSON 数组。容忍围栏、前后杂字、偶尔的非法转义。"""
    import json
    import re

    if not raw:
        raise ValueError(f"Claude 对 {fname} 返回空内容")

    # 1. 去 markdown 围栏
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    candidate = m.group(1) if m else raw

    # 2. 找第一个 [...] 数组
    m2 = re.search(r"\[[\s\S]*\]", candidate)
    if not m2:
        raise ValueError(
            f"Claude 对 {fname} 的输出里找不到 JSON 数组。"
            f"前 300 字符:{raw[:300]!r}"
        )

    try:
        data = json.loads(m2.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude 对 {fname} 的 JSON 解析失败 ({e})。"
            f"前 300 字符:{m2.group(0)[:300]!r}"
        ) from e

    if not isinstance(data, list):
        raise ValueError(f"Claude 返回不是列表,而是 {type(data).__name__}")

    # 规范化
    pages: list[dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        page_num = item.get("page_num", i)
        text = item.get("text", "") or ""
        pages.append(
            {"page_num": int(page_num), "text": str(text).strip()}
        )
    return pages


# 向后兼容:pipeline 旧路径还在调这个名字
extract_pdf_pages = extract_pdf_pages_pypdf


def chunk_pdf(
    pages: list[dict[str, Any]],
    doc_id: str,
    source_path: str,
    cfg: dict[str, Any],
) -> list[Chunk]:
    """把 PDF 页文本切块。

    每页作为一个章节边界 (section_title='Page N') — 避免跨页合并稀释。
    页内按字符 splitter 切;如果一页文本太短(< min_chunk_chars)就整页当一个 chunk。
    """
    from ..embedding.splitter import recursive_char_split

    chunk_size = cfg.get("pdf_chunk_size", 800)
    chunk_overlap = cfg.get("pdf_chunk_overlap", 80)
    min_chars = cfg.get("pdf_min_chunk_chars", 120)
    separators = ["\n\n", "\n", "。", ". ", "!", "?", " ", ""]

    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        page_num = page["page_num"]
        text = page["text"]
        if not text:
            continue

        # 超短的页直接整页当一个 chunk,不切
        pieces = (
            [text]
            if len(text) < chunk_size
            else recursive_char_split(text, chunk_size, chunk_overlap, separators)
        )
        for piece in pieces:
            if len(piece) < min_chars and chunks and chunks[-1].page_num == page_num:
                # 同页的碎尾合并到前一块
                chunks[-1].text = chunks[-1].text.rstrip() + " " + piece.lstrip()
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page_num:04d}_{counter:04d}",
                    video_id=doc_id,  # source_id
                    text=piece,
                    section_title=f"Page {page_num}",
                    has_visual=False,
                    source_type="pdf",
                    page_num=page_num,
                    source_path=source_path,
                )
            )
            counter += 1
    log.info(f"PDF {doc_id}: {len(pages)} pages → {len(chunks)} chunks")
    return chunks


# ============ 图片 ============


IMAGE_PROMPT_ZH = """分析这张图片: @{image_path}

这是一张用户提供的参考资料(可能是交易图表、笔记截图、示意图、幻灯片等)。用 4-8 句中文详细描述:
1. 主要视觉内容 (是图表/截图/示意图/手写笔记/幻灯片?)
2. 画面上的关键元素 (数值、指标名、图形、标注、箭头等)
3. 如果是交易图表:包含的品种、时间框架、价位、画线、标注、指标
4. 所有可读文字逐字转录 (英文保留原文,最多 500 字符)

**只描述你看到的,不要推测含义或交易逻辑。** 目标是生成可被语义检索的视觉内容索引。

输出 JSON(不要 markdown 围栏,不要其他评论),键:
  "description" (string, 4-8 句中文描述)
  "extracted_text" (string 或 null, 画面文字逐字,最多 500 字符)
"""

IMAGE_PROMPT_EN = """Analyze this image: @{image_path}

This is user-provided reference material (chart, notes screenshot, diagram, slide, etc.).
Describe in 4-8 sentences:
1. Main visual content (chart / screenshot / diagram / handwritten notes / slide?)
2. Key elements on screen (numbers, indicator names, shapes, annotations, arrows)
3. If a trading chart: instrument, timeframe, price levels, drawn lines, annotations, indicators
4. All readable text transcribed verbatim (max 500 chars)

**Describe only what you see — no speculation on meaning or trading logic.**
Goal: create a searchable visual-content index.

Output JSON only (no markdown fences, no commentary), keys:
  "description" (string, 4-8 sentences)
  "extracted_text" (string or null, verbatim screen text, max 500 chars)
"""


def describe_image(
    img_path: Path,
    cfg: dict[str, Any],
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """调 Claude CLI 对单图生成描述。返回 {'description', 'extracted_text'}。

    依赖 claude CLI + `claude login`;不复用 vision/claude_code.describe_frames
    是因为那个函数签名是 list[KeyFrame],对独立图片太重了 —— 这里直接起 subprocess。
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请先 `npm install -g @anthropic-ai/claude-code` 并 `claude login`。"
        )

    model = cfg.get("claude_model", "sonnet")
    lang = cfg.get("image_prompt_lang", "zh")
    prompt_tpl = IMAGE_PROMPT_ZH if lang == "zh" else IMAGE_PROMPT_EN
    prompt = prompt_tpl.replace("{image_path}", str(img_path.resolve()))

    # --permission-mode bypassPermissions 理由同 extract_pdf_pages_via_claude
    proc = subprocess.run(
        [
            claude_bin,
            "-p",
            "--output-format", "text",
            "--model", model,
            "--permission-mode", "bypassPermissions",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}"
        )

    raw = (proc.stdout or "").strip()
    _raise_if_permission_refusal(raw, img_path.name)

    # 复用现有 JSON parser
    from ..vision.claude_code import _parse_json_response

    desc, ext = _parse_json_response(raw)
    return {"description": desc, "extracted_text": ext}


def chunk_image(
    description: dict[str, Any],
    doc_id: str,
    source_path: str,
) -> list[Chunk]:
    """图片 → 1 个 chunk(描述 + 屏幕文字拼成一段)。

    图片信息量有限,不分块;section_title 固定 'Image'。
    """
    desc = description.get("description", "").strip()
    ext_text = description.get("extracted_text")
    parts: list[str] = []
    if desc:
        parts.append(f"[图片描述] {desc}")
    if ext_text:
        parts.append(f"[屏幕文字] {ext_text}")

    if not parts:
        log.warning(f"Image {doc_id}: 描述为空,不入库")
        return []

    text = "\n".join(parts)
    chunk = Chunk(
        chunk_id=f"{doc_id}_img_0000",
        video_id=doc_id,
        text=text,
        section_title="Image",
        has_visual=True,
        source_type="image",
        source_path=source_path,
    )
    return [chunk]


# ============ Meta 落盘 ============


def save_doc_meta(meta: DocMeta, ddir: Path) -> None:
    import yaml
    with open(ddir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta.model_dump(), f, allow_unicode=True, sort_keys=False)


def load_doc_meta(ddir: Path) -> DocMeta | None:
    import yaml
    p = ddir / "meta.yaml"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return DocMeta(**yaml.safe_load(f))


def build_doc_meta(source_path: Path, doc_type: str) -> DocMeta:
    """根据源文件路径和类型构造 DocMeta(不写盘)。"""
    did = slug_doc_id(str(source_path), doc_type)
    return DocMeta(
        doc_id=did,
        source_type=doc_type,  # 'pdf' | 'image'
        source_path=str(source_path.resolve()),
        title=source_path.name,
        ingested_at=now_iso(),
    )
