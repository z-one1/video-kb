# Plan: video-kb 文档入库审计收尾

**Status:** Ready for `/executing-plans`
**Context:** 刚给 video-kb 加完 PDF + 独立图片入库(`kb ingest-doc`)。审计发现 1 个 BLOCKING bug + 若干缺口。此 plan 按 superpowers 方法论分阶段执行,每阶段有明确 verification。
**Branch:** `chore/doc-ingest-audit`(建议新开)

---

## 目标 (What "done" looks like)

审计完成后,`kb ingest-doc` 这条功能线应该达到:

1. ✅ `--pdf-provider` flag 真的起作用(当前被静默吞)
2. ✅ `tests/` 里有 doc ingestion 的独立单元测试,`pytest` 全绿
3. ✅ README 里明确告诉新用户怎么用 PDF / 图片入库
4. ✅ `kb reindex` 要么覆盖 docs,要么明确说明"只处理视频"
5. ✅ `kb docs remove` 不再能一键误删

非目标(明确不做):
- 不加 docx / epub / pptx 支持(后续迭代)
- 不重构 Chunk.video_id → source_id(破坏性,留给未来大版本)
- 不动 vision / structuring / fusion 这些成熟模块

---

## Phase 1 — 修 BLOCKING bug (预计 10 分钟)

### 1.1 事实陈述

`src/kb/cli.py:513-606`(`ingest_doc_cmd`)接受 `--pdf-provider` 参数,但调 `pipeline.ingest_doc(...)` 时**没有**把这个参数往下传:

```python
# cli.py:581-586 — 当前代码,有 bug
res = pipeline.ingest_doc(
    f,
    cfg_path=config,
    force=force,
    log_level=log_level,
)
```

而 `pipeline.py:423-427` 的 provider 选择链是:

```python
provider = (
    pdf_provider                                   # 函数参数 — 永远是 None,因为 CLI 没传
    or ingest_doc_cfg.get("pdf_provider")
    or "claude_code"
)
```

→ 所以 `kb ingest-doc foo.pdf --pdf-provider pypdf` 实际行为:**走 config 默认 `claude_code`**,用户参数被完全吞掉。

### 1.2 动作

改 `cli.py:581-586` 加一行:

```python
res = pipeline.ingest_doc(
    f,
    cfg_path=config,
    force=force,
    pdf_provider=pdf_provider,  # ← 加这一行
    log_level=log_level,
)
```

### 1.3 Verification(必做,不能跳)

**A. 编译通过:**
```bash
python3 -c "import ast; ast.parse(open('src/kb/cli.py').read())"
```

**B. Flag 真的走通了** — 拿同一个 PDF 跑两遍,对比 `pages.jsonl`:

```bash
# 清一下缓存
kb docs remove pdf_<hash> 2>/dev/null || true

# A. Claude CLI (默认)
kb ingest-doc docs/samples/test.pdf
head -5 kb/docs/pdf_*/pages.jsonl > /tmp/pages_claude.txt

kb docs remove pdf_<hash>

# B. pypdf(应该和 A 内容不同)
kb ingest-doc docs/samples/test.pdf --pdf-provider pypdf
head -5 kb/docs/pdf_*/pages.jsonl > /tmp/pages_pypdf.txt

diff /tmp/pages_claude.txt /tmp/pages_pypdf.txt
# 期望:diff 非空,claude 版本有 "[图:...]" 前缀,pypdf 没有
```

**C. 负向测试:** `kb ingest-doc foo.pdf --pdf-provider invalid_name` 应该报错(`pipeline.py:459+` 的 else 分支),不是静默默认。

**如果 B 的 diff 为空** → pass-through 没修好,回到 1.2 重查。

---

## Phase 2 — 补测试 (预计 45 分钟)

**先跑 `superpowers:test-driven-development`,按 red-green-refactor 来。**

### 2.1 新建 `tests/test_ingest_docs.py`

覆盖六组:

```
test_detect_doc_type_pdf_lowercase
test_detect_doc_type_pdf_uppercase       # FOO.PDF
test_detect_doc_type_image_all_extensions  # png/jpg/jpeg/webp/bmp/gif
test_detect_doc_type_unsupported          # .docx / .txt / .mp4 → None

test_slug_doc_id_stable                    # 同路径 2 次 → 同 id
test_slug_doc_id_no_collision              # 不同路径 → 不同 id
test_slug_doc_id_prefix                    # pdf_ / img_ 前缀正确

test_parse_claude_pdf_json_clean           # 正常 JSON 数组
test_parse_claude_pdf_json_markdown_fence  # ```json ... ``` 包裹
test_parse_claude_pdf_json_with_preamble   # "好的,这是 JSON: [...]"
test_parse_claude_pdf_json_invalid         # 非法 → 抛异常 or 返回空

test_chunk_pdf_empty_pages                 # pages=[] → chunks=[]
test_chunk_pdf_single_short_page           # 短于 min_chunk_chars
test_chunk_pdf_multi_page_boundary         # 验证 chunk 不跨页
test_chunk_pdf_short_tail_merge            # 尾部短 chunk 合并到前

test_chunk_image_normal                    # 完整描述 → 1 chunk
test_chunk_image_missing_fields            # 缺 [屏幕文字] → 降级 ok

test_build_doc_meta_roundtrip              # build → save → load 三步一致
```

### 2.2 Mock 策略

测试不能调真 Claude CLI。`_parse_claude_pdf_json` / `chunk_pdf` / `chunk_image` 都是纯函数,直接喂字符串和 dict。
只有 `describe_image` 和 `extract_pdf_pages_via_claude` 需要 monkeypatch subprocess。

```python
# conftest.py fragment
@pytest.fixture
def mock_claude_subprocess(monkeypatch):
    def fake_run(cmd, input, **kw):
        class R:
            returncode = 0
            stdout = '[{"page_num": 1, "text": "fake page"}]'
            stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
```

### 2.3 Verification

```bash
pytest tests/test_ingest_docs.py -v
# 期望:所有 test 绿,无 skip、无 warning
pytest --cov=src/kb/ingest/docs --cov-report=term-missing
# 期望:docs.py 覆盖率 ≥ 80%
```

---

## Phase 3 — 文档补齐 (预计 20 分钟)

### 3.1 README.md 新增一节

位置:现有 "🎥 Ingest videos" 段之后、"🤖 Ask questions" 段之前。

模板:

````markdown
## 📄 Ingest PDFs and images

视频以外的学习材料(讲师笔记 PDF、图表截图)也能进同一个知识库,`kb ask` 会跨源检索:

```bash
# 单个 PDF(默认 Claude CLI 抽取,能 OCR + 描述图表 + 保留表格)
kb ingest-doc trading_notes.pdf

# 大量纯文本 PDF,求快
kb ingest-doc thesis.pdf --pdf-provider pypdf

# 单张图(调 Claude vision 自动描述画面 + 提取屏幕文字)
kb ingest-doc chart_screenshot.png

# 整个目录递归
kb ingest-doc ~/Desktop/CRT_refs/
```

检索结果里的引用格式会告诉你来源:

- `[ep.3 @ 12:45]` — 视频第 3 集 12 分 45 秒
- `[trading_notes.pdf p.7]` — PDF 第 7 页
- `[img: chart_screenshot.png]` — 图片

文档管理:

```bash
kb docs list              # 列出所有已入库的独立文档
kb docs remove pdf_<id>   # 删除某个文档的所有 chunks
```

> ⚠️ PDF 走 Claude CLI 时,每 PDF 要占你 Max 订阅几百 tokens(视 PDF 页数)。
> 如果 PDF 是扫描版或含大量图表,这笔开销值得;纯文字 PDF 建议 `--pdf-provider pypdf`。
````

### 3.2 Verification

```bash
grep -c "kb ingest-doc" README.md
# 期望:≥ 3(至少 3 处示例)

grep "pdf_provider" README.md
# 期望:有一行(说明 provider 选择)
```

---

## Phase 4 — reindex 覆盖 docs (预计 40 分钟)

这是优化而非 blocking。如果时间紧可以拆成单独 PR。

### 4.1 现状

`pipeline.reindex()` (~line 285) 只扫 `kb/videos/*`。给每个视频的 `enriched.json` + `notes.json` 重新切块 → 重新嵌入 → 重新 upsert。

文档的对应落盘是:
- PDF: `kb/docs/pdf_<id>/pages.jsonl` + `meta.yaml`
- 图片: `kb/docs/img_<id>/description.txt` + `meta.yaml`

这些已经是 "提取阶段的产物",重新切块不用重跑 Claude CLI — 关键价值就是**不重烧 tokens**。

### 4.2 动作草案

在 `reindex()` 现有视频循环后加一段 docs 循环,调 `ingest_docs.chunk_pdf()` / `chunk_image()` 即可(已经是纯函数)。

```python
# pipeline.py 新加
docs_root = kb_root / "docs"
for ddir in sorted(docs_root.glob("*")):
    meta = ingest_docs.load_doc_meta(ddir)
    if not meta:
        continue
    if meta.source_type == "pdf":
        pages = [json.loads(l) for l in (ddir / "pages.jsonl").read_text().splitlines() if l.strip()]
        chunks = ingest_docs.chunk_pdf(pages, meta.doc_id, meta.source_path, ingest_doc_cfg)
    elif meta.source_type == "image":
        desc = (ddir / "description.txt").read_text(encoding="utf-8")
        chunks = ingest_docs.chunk_image(desc, meta.doc_id, meta.source_path)
    # embed + upsert 和视频流程一样
    ...
```

### 4.3 Verification

```bash
# 改 configs/default.yaml 的 pdf_chunk_size: 800 → 400
kb reindex
kb stats
# 期望:PDF 对应的 chunk 数大约翻倍(不严格,但量级对)
# 期望:ChromaDB 里 PDF chunks 全部被替换,meta 里的 source_type/page_num 仍正确
```

---

## Phase 5 — remove 加确认 (预计 5 分钟)

```python
# cli.py kb docs remove
@docs_app.command("remove")
def docs_remove(
    doc_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认,脚本用"),
):
    ...
    if not yes:
        typer.confirm(f"真的要删除 {doc_id} 及其所有 chunks?", abort=True)
    ...
```

### Verification

- 不带 `-y` 敲 `n` → 中止,不动文件
- 带 `-y` → 直接删,无 prompt

---

## Phase 6 — 提取 _parse_json_response (可选, 15 分钟)

低优先级。把 `src/kb/ingest/vision.py` 里的 `_parse_json_response` 移到 `src/kb/utils.py`(或新建 `src/kb/parsing.py`),vision 和 docs 都 import 共享版本。
**除非另外两处消费方已经确认需要共用,否则可以直接跳过此阶段。**

---

## Phase 7 — 最终验证(全流程冒烟)

所有改动做完后,跑一次真实端到端:

```bash
# 1. 干净状态
rm -rf kb/kb_db kb/docs

# 2. 进 3 种源各 1 个
kb ingest-doc samples/notes.pdf
kb ingest-doc samples/chart.png
kb batch samples/videos/*.mp4  # 假设已有 1 个视频样本

# 3. 跨源查询
kb ask "ICT 的 Key Level 定义是什么?在视频和笔记里怎么解释?" --n 10

# 期望:answer 里同时出现 [ep.N @ mm:ss] 和 [notes.pdf p.N] 的引用

# 4. 统计
kb stats
# 期望:By source_type 那张表里三种源都有

# 5. 测试全绿
pytest tests/ -v
```

全部通过 → 关 task #21–#25(#26 如果跳了保持 pending),合并 PR。

---

## Rollback plan

每个 phase 都是独立 commit。某个 phase 崩了:

```bash
git log --oneline -10
git reset --hard <上一个绿色 commit>
```

最坏情况 Phase 1 那行 `pdf_provider=pdf_provider` 撤掉,回到今天之前的 "flag 被吞但至少不崩" 的状态 — 不会 break 已入库的数据。

---

## 喂给 Claude Code 的方式

在 Mac 终端进项目根:

```bash
cd ~/Project/video_kb
claude
```

进了 Claude Code 后:

```
/executing-plans docs/PLAN_doc_ingest_audit.md
```

让 superpowers 按 phase 逐个跑,每 phase 做完会停下来等你 review 再继续 — 这是 superpowers 的标准流程,别手动跳过 checkpoint。
