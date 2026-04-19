"""RAG 答题 — 检索 Top-K + Claude Code CLI 生成带引用答案。

设计原则:
- 复用 storage.chroma_client.query(不改动检索层)
- 复用 structuring 里的 Claude CLI 调用模式(subprocess + --print)
- prompt 里把 chunks 格式化成带 [ep.N @ mm:ss] 引用标记的块,强制 LLM 保留引用
- 关键帧路径以"参考关键帧"列表放在答案末尾,用户点击可看原图
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..storage import chroma_client
from ..utils import format_timestamp_short

log = logging.getLogger("kb.rag")


ANSWER_PROMPT = """你是一个基于视频知识库的专业知识提取助手。用户提了一个专业问题,我从向量库里检索出了 Top-<<K>> 相关片段。

**核心目标:细节优先,而不是概括。** 用户做 RAG 不是要学泛泛的知识,而是要还原讲师在视频里真正讲的专业细节。宁可答得长、宁可大段引用原话,也不要用"总的来说""综合来看"这种空话把具体内容抹平。

严格遵守以下规则:

## 1. 只用检索到的片段,不编造
片段里没说的内容,不要从一般金融/交易常识里补。如果用户问的某个子点片段里没有,就在『🔍 覆盖情况』里标明,不要糊弄。

## 2. 每个事实必须带引用 — 按 chunk 类型选 citation 格式
引用紧贴具体陈述末尾(不是堆在段末)。三种格式(chunk header 里 `type=` 已标明来源):

- **视频 chunk** (`type=video`): `[ep.N @ mm:ss]`,例:
  > Key Level 的定义是 "a price point at which you expect price to give you a bounce" [ep.5 @ 07:54]
- **PDF chunk** (`type=pdf`): `[filename.pdf p.N]`,例:
  > ICT 定义 displacement 为 "aggressive single-directional price movement" [ict_lectures.pdf p.12]
- **图片 chunk** (`type=image`): `[img: filename.png]`,例:
  > 图中标注了 EURUSD H4 级别的 Kill Zone 区间 [img: eurusd_killzone.png]

chunk header 的 `[...]` 就是现成的 citation,照抄即可。

## 3. 关键内容原话引用(Verbatim),不要改写
对于**定义、判定条件、数值阈值、操作步骤、讲师明确的断言**,必须用英文原话 + 中文 gloss,格式:
> 讲师原话:"..." [ep.N @ mm:ss]
> (译:...)

改写原话会丢失专业细节。只有承接/解释性的描述才允许用你自己的中文。

## 4. 禁止抽象化填充词
不要写"综合来看""总的来说""从整体看""本质上"这类空话。不要用"讲师强调了 XX 的重要性"代替讲师到底怎么讲 XX。如果片段里只给了一个笼统说法,就原话引用那个笼统说法,别自己再加一层概括。

## 5. 按问题类型选结构
- **定义型**("什么是 X"): 先原话给定义,再列讲师给出的判定条件 / 关键属性,每条带引用。
- **操作机制型**("如何判断/如何画/怎么确认 X"): 按讲师讲的步骤或信号逐条列出,每步带原话或关键短语引用,包括否定条件(什么情况下不成立)。
- **分类对比型**("X 有几种 / A 和 B 的区别"): 按集数/分类列,每类底下再列属性,末尾一小段指出跨集的递进或差异。
- **应用场景型**("什么时候用 X / 怎么交易 X"): 按讲师给的 setup 条件 → 入场 → 止损 → 目标顺序组织,缺哪一环就标"未覆盖"。

## 6. 过滤噪声,不硬凑
纯价位数字、单独字母、列表编号("1.""2.")这种片段直接忽略,不要为了凑长度引用它们。

## 7. 开头直接答,不要元描述
不要写"根据检索到的片段""从知识库来看"这种废话。

## 8. 强制『🔍 覆盖情况』区块
答案正文之后必须有一节标题为 `### 🔍 覆盖情况` 的块,把问题拆成子点,每个子点标注以下三种状态之一:

- ✅ **已明确覆盖**: 讲师讲清楚了,正文有引用。
- ⚠️ **讲师列出但跳过**: 讲师在视频里提到这个点但明确说不展开(如"I won't be hitting all of these questions")。必须原话引用那句"跳过"的话 + 时间戳,让用户知道这个知识在视频里但未被讲解。
- ❌ **本次检索未覆盖**: 当前 Top-K 片段里没有相关内容(可能在别的集或别的片段)。

这一节是用户判断"答案够不够用/要不要再查"的关键,不能省略。

## 9. 结尾『📎 参考关键帧』
最后一行加 `### 📎 参考关键帧:`,把正文引用涉及的所有关键帧路径列出(每行一个)。如果没有关键帧引用就省略这一节。

---

用户问题:<<QUESTION>>

--- 检索到的 <<K>> 个片段 ---
<<CHUNKS>>
--- 片段结束 ---

请开始回答。记住:**细节优先,原话优先,诚实标注覆盖情况**。"""


_EP_RE = re.compile(r"ep[_\.\s]*(\d+)", re.IGNORECASE)


def _extract_ep_tag(video_id: str, video_title: str) -> str:
    """从 video_id 或 video_title 抽取 'ep.N',失败则返回 title 前缀。"""
    for s in (video_id, video_title):
        if not s:
            continue
        m = _EP_RE.search(s)
        if m:
            return f"ep.{m.group(1)}"
    # 兜底
    return (video_title or video_id or "video")[:30]


def _source_citation(md: dict[str, Any]) -> str:
    """按 source_type 生成 citation 标签。

    - video: [ep.N @ mm:ss]
    - pdf:   [notes.pdf p.3]
    - image: [img: chart.png]
    向后兼容:metadata 没有 source_type 时默认当作 video。
    """
    stype = md.get("source_type", "video")
    if stype == "pdf":
        title = md.get("video_title") or Path(
            md.get("source_path", "document")
        ).name
        page = md.get("page_num")
        page_s = f" p.{int(page)}" if page is not None else ""
        return f"[{title}{page_s}]"
    if stype == "image":
        title = md.get("video_title") or Path(
            md.get("source_path", "image")
        ).name
        return f"[img: {title}]"
    # video (默认)
    ep_tag = _extract_ep_tag(
        md.get("video_id", ""), md.get("video_title", "")
    )
    start = float(md.get("start_sec", 0))
    ts = format_timestamp_short(start)
    return f"[{ep_tag} @ {ts}]"


def _frame_path_for(video_id: str, start_sec: float, videos_dir: Path) -> Path | None:
    """推算 video chunk 对应的 keyframe JPG 路径(基于 format_timestamp_short 命名约定)。"""
    if not video_id:
        return None
    ts_label = format_timestamp_short(start_sec).replace(":", "-")
    candidate = videos_dir / video_id / "frames" / f"{ts_label}.jpg"
    return candidate if candidate.exists() else None


def build_chunk_block(
    hits: list[dict[str, Any]], videos_dir: Path
) -> tuple[str, list[Path]]:
    """渲染 Top-K chunks 为 prompt 块;同时收集对应的关键帧 / 图片文档路径列表。

    Returns:
        (chunk_block_str, list_of_ref_paths_for_answer_footer)
    """
    lines: list[str] = []
    ref_paths: list[Path] = []
    for i, h in enumerate(hits, 1):
        md = h["metadata"]
        stype = md.get("source_type", "video")
        cite = _source_citation(md)
        section = md.get("section_title", "") or "(无章节)"
        has_visual = md.get("has_visual", False)
        visual_icon = " 📷" if has_visual else ""
        distance = h.get("distance")
        dist_str = f"{distance:.3f}" if distance is not None else "-"

        header = (
            f"【Chunk #{i}】 {cite}{visual_icon}  "
            f"type={stype}  section={section}  dist={dist_str}"
        )
        lines.append(header)
        lines.append(h["text"].strip())

        # 视频 chunk 关键帧
        if stype == "video" and has_visual:
            start = float(md.get("start_sec", 0))
            fp = _frame_path_for(md.get("video_id", ""), start, videos_dir)
            if fp:
                lines.append(f"(🖼 对应关键帧: {fp})")
                ref_paths.append(fp)
        # 图片文档原图
        elif stype == "image":
            sp = md.get("source_path")
            if sp:
                p = Path(sp)
                if p.exists():
                    lines.append(f"(🖼 原图: {p})")
                    ref_paths.append(p)

        lines.append("")  # 空行分隔

    return "\n".join(lines), ref_paths


def answer(
    question: str,
    cfg: dict[str, Any],
    n_results: int = 8,
    video_id: str | None = None,
    use_aliases: bool = True,
) -> dict[str, Any]:
    """检索 + Claude CLI 生成答案。

    Args:
        question: 用户自然语言提问
        cfg: 从 load_config() 拿到的配置
        n_results: 检索 Top-K(默认 8,实测 8 覆盖大多数场景)
        video_id: 仅检索指定视频(可选)

    Returns:
        {
            "answer": str,          # LLM 生成的带引用答案
            "hits": list,            # 检索到的原始 chunks(方便 CLI 二次展示)
            "frame_paths": list,     # 引用涉及的关键帧文件路径
            "prompt_chars": int,     # prompt 体积(便于监控)
        }
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请安装 Claude Code"
            "(`npm install -g @anthropic-ai/claude-code`)并登录。"
        )

    # 1. 检索 — 可选别名扩展
    where = {"video_id": video_id} if video_id else None
    aliases_lookup: dict[str, list[str]] | None = None
    if use_aliases:
        from ..retrieval.aliases import load_aliases
        aliases_path = cfg.get("retrieval", {}).get("aliases_path")
        if aliases_path:
            # 相对路径解析到 _project_root
            ap = Path(aliases_path)
            if not ap.is_absolute():
                ap = Path(cfg.get("_project_root", ".")) / ap
            aliases_lookup = load_aliases(ap) or None

    hits = chroma_client.query(
        question,
        cfg["paths"]["chroma_dir"],
        cfg["embedding"],
        n_results=n_results,
        where=where,
        aliases_lookup=aliases_lookup,
    )

    if not hits:
        return {
            "answer": "知识库中没有相关内容。",
            "hits": [],
            "frame_paths": [],
            "ref_paths": [],
            "prompt_chars": 0,
        }

    # 2. 构造 prompt
    videos_dir = Path(cfg["paths"]["videos_dir"])
    chunk_block, ref_paths = build_chunk_block(hits, videos_dir)
    # 用占位符替换,避免 str.format 被 question/chunks 里的 { } 搞崩
    prompt = (
        ANSWER_PROMPT.replace("<<K>>", str(len(hits)))
        .replace("<<QUESTION>>", question)
        .replace("<<CHUNKS>>", chunk_block)
    )

    # 3. Claude CLI 生成(复用 structuring 的调用模式)
    ask_cfg = cfg.get("ask", {})
    # 优先 ask.claude_model,回退到 structuring.claude_model,最后默认 sonnet
    model = ask_cfg.get("claude_model") or cfg.get("structuring", {}).get(
        "claude_model", "sonnet"
    )
    timeout = ask_cfg.get("timeout_sec", 180)
    log.info(
        f"Asking Claude ({model}, prompt ~{len(prompt) // 1000}k chars, "
        f"{len(hits)} chunks, {len(ref_paths)} visual refs)..."
    )

    proc = subprocess.run(
        [claude_bin, "-p", "--output-format", "text", "--model", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"stderr={proc.stderr[-500:]}"
        )

    return {
        "answer": proc.stdout.strip(),
        "hits": hits,
        "frame_paths": ref_paths,  # 向后兼容别名,实际含视频关键帧 + 图片文档原图
        "ref_paths": ref_paths,
        "prompt_chars": len(prompt),
    }
