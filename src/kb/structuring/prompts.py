"""结构化笔记的 prompt 模板。"""
from __future__ import annotations


STRUCTURING_PROMPT = """你是一位视频内容整理专家。你会收到一段带时间戳的视频字幕,部分段落附有关键帧的视觉描述。

你的任务是产出一份**结构化的学习笔记**,便于后续检索和回顾。

请严格输出 JSON(不要额外的 markdown 代码块围栏),结构如下:

{
  "title": "一句话概括视频主题",
  "one_liner": "1 句话视频总结 (<= 40 字)",
  "sections": [
    {
      "title": "章节标题 (简练,<= 30 字)",
      "start_sec": 0,
      "end_sec": 180,
      "summary": "3-5 句核心摘要,保留关键细节和引用",
      "concepts": ["概念1", "概念2", "..."],
      "questions": ["潜在的学习问题1", "问题2"]
    }
  ]
}

规则:
1. 章节要覆盖整个视频时间轴,不留空白,不重叠
2. 每段时长建议 2-8 分钟,总章节数 4-15 段
3. `summary` 必须保留关键数字/术语/人名,不要过度抽象
4. `concepts` 是名词短语列表,适合做知识点标签
5. `questions` 是面向学习者的提问,帮助未来复习时提醒"我理解这里了吗"
6. 视觉描述里的代码/公式/图表要显式在 summary 中提及
7. 语言与输入语言一致(输入是中文 → 用中文,输入是英文 → 用英文)

**极其重要的 JSON 输出规则:**
A. 字符串值里**禁止使用 ASCII 双引号 `"`**,因为会破坏 JSON 结构
   - 中文里想表达"引用/强调"时,请用中文引号「」或『』 或 书名号《》
   - 例: 错 "summary": "讲师强调"时间优先于价格"的理念"
         对 "summary": "讲师强调「时间优先于价格」的理念"
B. 字符串内如确实需要 ASCII 引号(如引用英文术语),务必转义为 \"
   - 例: "summary": "讲师引用 \"Candle Range Theory\" 原文"
C. 换行用 `\n` 而不是真实换行符
D. 不要输出 markdown ```json ... ``` 围栏,只输出裸 JSON 对象
E. 确保所有字符串值、键名用 ASCII 双引号包围(这是 JSON 规范)

现在请处理以下输入:

---
{content}
---

只输出上述 JSON,不要任何前后缀或解释。"""


def build_content_block(enriched_segments: list) -> str:
    """把 EnrichedSegment 列表转成给 LLM 的纯文本块。"""
    from ..utils import format_timestamp_short

    lines: list[str] = []
    for seg in enriched_segments:
        ts = format_timestamp_short(seg.start_sec)
        ts_end = format_timestamp_short(seg.end_sec)
        lines.append(f"[{ts}-{ts_end}] {seg.text}")
        for vd in seg.visual_descriptions:
            desc = vd.description.replace("\n", " ")
            lines.append(f"  (visual @{vd.frame_id}) {desc}")
            if vd.extracted_text:
                lines.append(f"    (screen text) {vd.extracted_text}")
    return "\n".join(lines)
