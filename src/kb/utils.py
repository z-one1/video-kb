"""通用工具:日志、路径、时间戳格式化、Claude CLI 响应解析。"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("kb")


def slug_video_id(source: str) -> str:
    """从 URL 或文件名生成稳定 video_id"""
    base = Path(source).stem if "/" in source or "\\" in source else source
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", base)[:60]
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def format_timestamp(seconds: float) -> str:
    """0.0 → '00:00:00', 3661.5 → '01:01:01'"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_short(seconds: float) -> str:
    """0.0 → '00:00', 125.5 → '02:05' (小于 1 小时时)"""
    seconds = int(seconds)
    if seconds >= 3600:
        return format_timestamp(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p: Path | str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_dir(kb_root: Path | str, video_id: str) -> Path:
    """返回 kb/videos/<video_id>/,并确保存在。"""
    d = Path(kb_root) / "videos" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug_doc_id(source_path: str, source_type: str) -> str:
    """从文档路径生成稳定 doc_id,带类型前缀 ('pdf_xxx' / 'img_xxx')。

    前缀用于和 video_id 隔离 — 同样的 basename 存为不同 source_type 不冲突。
    """
    p = Path(source_path)
    base = re.sub(r"[^a-zA-Z0-9_\-]", "_", p.stem)[:50]
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    prefix = {"pdf": "pdf", "image": "img"}.get(source_type, "doc")
    return f"{prefix}_{base}_{h}"


def doc_dir(kb_root: Path | str, doc_id: str) -> Path:
    """返回 kb/docs/<doc_id>/,并确保存在。"""
    d = Path(kb_root) / "docs" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Claude CLI 响应解析 ----------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _escape_inner_quotes(s: str) -> str:
    """状态机逐字符扫描,把字符串值内部未转义的 ASCII `"` 转成 `\\"`。

    LLM 吐 JSON 常见失败模式:在 "text" 值里引用原话(如讲师说的 "Key Level"
    或题目里的 "..."),但忘了给内部 `"` 加反斜杠,导致 json.loads 崩在某个
    「Expecting ',' delimiter」错误上。

    状态机:
    - out_of_string: 遇到 `"` → 进入字符串
    - in_string: 遇到 `\\?` → 整对跳过(已转义);
                  遇到 `"` 且后面紧跟 `,}]:` 或空白+上述字符 → 退出字符串;
                  否则是内部未转义 `"`,替换成 `\\"` 修补。
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # in_string
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            next_ch = s[j] if j < n else ""
            if next_ch in ",}]:" or next_ch == "":
                out.append(ch)
                in_str = False
                i += 1
                continue
            out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def repair_llm_json(raw: str) -> str:
    """最大努力修复常见 LLM JSON 输出错误,返回修复后的字符串。

    覆盖模式:
    1. 剥离 ```json ... ``` 或 ``` ... ``` 围栏(如果有的话)
    2. 状态机修复字符串值内部未转义的双引号

    使用模式:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            fixed = repair_llm_json(raw)
            data = json.loads(fixed)  # 还不行就让它抛,调用方自己降级

    注意:这个函数**只修补**,不验证。修复后的字符串仍可能非法 JSON,
    调用方要么继续 try/except,要么接受失败。
    """
    s = (raw or "").strip()
    if not s:
        return s

    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    return _escape_inner_quotes(s)


def parse_json_block(raw: str) -> dict[str, Any] | None:
    """从 Claude CLI 的文本输出里抽第一个有效的 JSON 对象。

    容忍三种常见情况:
    1. 纯 JSON(直接 json.loads)
    2. Markdown 围栏包裹(```json ... ``` 或 ``` ... ```)
    3. JSON 混在前后解释性文字里(贪心匹配第一对 `{...}`)

    找不到合法 JSON 对象时返回 None — 由调用方决定 fallback(报错 / 降级 / 跳过)。
    只处理对象(dict),顶层是数组的场景请用专门 parser。

    Args:
        raw: Claude CLI stdout(已 strip 过更好,但本函数会自己 strip)

    Returns:
        解析出的 dict,或 None
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    # 1. 尝试整体当 JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. 剥围栏
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text

    # 3. 找第一个 {...} 对
    m2 = _OBJECT_RE.search(candidate)
    if not m2:
        return None
    try:
        data = json.loads(m2.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return None
