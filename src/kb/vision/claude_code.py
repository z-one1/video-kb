"""通过 Claude Code CLI 做视觉理解 — 不吃 Gemini 配额,走 Max 订阅。

调用形式:
    claude -p --output-format text --model sonnet \
           "Analyze this frame: @/abs/path/to/frame.jpg ..."

Claude CLI 会自动识别 prompt 里的 `@<file>` 引用,把图片作为多模态附件
发送给模型。要求 user 已 `claude login` 并有 Max/Pro 订阅或 API 额度。
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..schemas import KeyFrame, VisualDescription
from ..utils import parse_json_block

log = logging.getLogger("kb.vision")


DEFAULT_PROMPT = """Analyze this frame from an educational/informational video: @{image_path}

Describe in 2-4 sentences:
1. The main visual content (diagram, code, chart, person talking, slides, text on screen, UI)
2. Any visible text — transcribe it literally if it's readable (max 200 chars)
3. The apparent context (whiteboard lecture, slide deck, screencast, talking head, demo)

Be concise and factual. Only describe what you see, don't speculate.

Output ONLY valid JSON (no markdown fences, no commentary) with keys:
  "description" (string, 2-4 sentences)
  "extracted_text" (string or null — literal text visible in frame, max 200 chars)
"""


def _find_claude() -> str | None:
    return shutil.which("claude")


def is_available() -> bool:
    return _find_claude() is not None


def describe_frames(
    frames: list[KeyFrame],
    frames_dir: Path | str,
    cfg: dict[str, Any],
    api_key: str | None = None,  # 保留签名兼容 — claude CLI 不用 API key
    out_json_path: Path | str | None = None,
) -> list[VisualDescription]:
    """逐帧调用 Claude Code CLI 描述关键帧。

    与 vision/gemini.describe_frames 接口对偶,上层 pipeline 可无痛切换。

    Args:
        frames: 关键帧列表
        frames_dir: 帧图片所在目录 (绝对路径)
        cfg: vision 段配置 (claude_model, prompt, retry_max, retry_delay_sec, timeout_sec)
        api_key: 未使用,保留接口兼容
        out_json_path: 增量落盘路径
    """
    claude_bin = _find_claude()
    if not claude_bin:
        raise RuntimeError(
            "未找到 claude CLI。请先 `npm install -g @anthropic-ai/claude-code` 并 `claude login`。"
        )

    model = cfg.get("claude_model", "sonnet")
    retry_max = cfg.get("retry_max", 3)
    retry_delay = cfg.get("retry_delay_sec", 2)
    timeout_sec = cfg.get("timeout_sec", 120)
    prompt_tpl = cfg.get("prompt", DEFAULT_PROMPT)

    frames_dir = Path(frames_dir).resolve()
    results: list[VisualDescription] = []

    log.info(
        f"Describing {len(frames)} frames with Claude CLI (model={model}, "
        f"timeout={timeout_sec}s, retry_max={retry_max})"
    )

    for i, fr in enumerate(frames, 1):
        img_path = (frames_dir / Path(fr.image_path).name).resolve()
        if not img_path.exists():
            log.warning(f"Frame missing: {img_path}")
            continue

        description = ""
        extracted_text: str | None = None
        last_err: str | None = None

        for attempt in range(retry_max):
            try:
                prompt = prompt_tpl.replace("{image_path}", str(img_path))
                cmd = [
                    claude_bin,
                    "-p",
                    "--output-format",
                    "text",
                    "--model",
                    model,
                ]
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude exit={proc.returncode} "
                        f"stderr={proc.stderr[-300:]!r} stdout={proc.stdout[-300:]!r}"
                    )

                raw = (proc.stdout or "").strip()
                description, extracted_text = _parse_json_response(raw)
                break  # 成功

            except subprocess.TimeoutExpired as e:
                last_err = f"timeout after {timeout_sec}s"
                log.warning(
                    f"  frame {fr.frame_id} attempt {attempt + 1} timed out"
                )
            except Exception as e:
                last_err = str(e)
                log.warning(
                    f"  frame {fr.frame_id} attempt {attempt + 1} failed; "
                    f"err={type(e).__name__}: {str(e)[:180]}"
                )

            if attempt < retry_max - 1:
                wait = retry_delay * (attempt + 1)
                time.sleep(wait)

        results.append(
            VisualDescription(
                frame_id=fr.frame_id,
                t_sec=fr.t_sec,
                image_path=fr.image_path,
                description=description,
                extracted_text=extracted_text,
                model=f"claude-cli:{model}",
                error=last_err if not description else None,
            )
        )

        # 每 5 帧增量落盘 — Claude CLI 慢,掉链子也能续
        if out_json_path and (i % 5 == 0 or i == len(frames)):
            _save_partial(results, out_json_path)
            log.info(f"  ... {i}/{len(frames)} frames described (saved)")

    if out_json_path:
        _save_partial(results, out_json_path)
        log.info(f"Saved visual descriptions: {out_json_path}")

    return results


def _parse_json_response(raw: str) -> tuple[str, str | None]:
    """从 Claude CLI 文本输出里抽 {description, extracted_text}。

    通用 JSON 抽取走 `utils.parse_json_block`;本函数只负责:
    - 按视觉响应契约挑 description / extracted_text 两个键
    - fallback:解析失败时把整段 raw 当作 description 降级使用(不扔掉信息)
    """
    if not raw:
        return "", None

    data = parse_json_block(raw)
    if data is not None:
        desc = str(data.get("description", "")).strip()
        text = data.get("extracted_text")
        if isinstance(text, str):
            text = text.strip() or None
        elif text is not None:
            text = str(text)
        return desc, text

    # fallback: Claude 完全没给 JSON,把纯文本当描述(截到 2000 char)
    return raw.strip()[:2000], None


def _save_partial(
    results: list[VisualDescription], out_json_path: Path | str
) -> None:
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results], f, ensure_ascii=False, indent=2
        )


def load_visual_descriptions(json_path: Path | str) -> list[VisualDescription]:
    p = Path(json_path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [VisualDescription(**d) for d in json.load(f)]
