"""Gemini 视觉理解 — 批量描述关键帧。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..schemas import KeyFrame, VisualDescription

log = logging.getLogger("kb.vision")


DEFAULT_PROMPT = """Analyze this frame from an educational/informational video.

Describe in 2-4 sentences:
1. The main visual content (diagram, code, chart, person talking, slides, text on screen, UI)
2. Any visible text — transcribe it literally if it's readable (max 200 chars)
3. The apparent context (whiteboard lecture, slide deck, screencast, talking head, demo)

Be concise and factual. Only describe what you see, don't speculate.
Output JSON with keys: "description" (string), "extracted_text" (string or null)."""


def describe_frames(
    frames: list[KeyFrame],
    frames_dir: Path | str,
    cfg: dict[str, Any],
    api_key: str,
    out_json_path: Path | str | None = None,
) -> list[VisualDescription]:
    """并发调用 Gemini 2.0 Flash 描述每张关键帧。"""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as e:
        raise ImportError("google-genai 未安装 — pip install google-genai") from e

    from PIL import Image

    model_name = cfg.get("gemini_model", "gemini-2.0-flash-exp")
    retry_max = cfg.get("retry_max", 3)
    retry_delay = cfg.get("retry_delay_sec", 2)
    prompt = cfg.get("prompt", DEFAULT_PROMPT)

    client = genai.Client(api_key=api_key)
    frames_dir = Path(frames_dir)
    results: list[VisualDescription] = []

    log.info(f"Describing {len(frames)} frames with {model_name}")

    for i, fr in enumerate(frames, 1):
        img_path = frames_dir / Path(fr.image_path).name
        if not img_path.exists():
            log.warning(f"Frame missing: {img_path}")
            continue

        description = ""
        extracted_text = None
        last_err: str | None = None

        for attempt in range(retry_max):
            try:
                img = Image.open(img_path)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img],
                    config=gtypes.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                raw = resp.text or ""
                try:
                    parsed = json.loads(raw)
                    description = parsed.get("description", raw)
                    extracted_text = parsed.get("extracted_text")
                except json.JSONDecodeError:
                    description = raw.strip()
                break
            except Exception as e:
                last_err = str(e)
                # 从 Gemini error 里提取 retryDelay 建议
                wait = _parse_retry_delay(str(e), retry_delay * (attempt + 1))
                log.warning(
                    f"  frame {fr.frame_id} attempt {attempt + 1} failed; "
                    f"waiting {wait:.0f}s. err={type(e).__name__}: {str(e)[:120]}"
                )
                if attempt < retry_max - 1:
                    time.sleep(wait)

        results.append(
            VisualDescription(
                frame_id=fr.frame_id,
                t_sec=fr.t_sec,
                image_path=fr.image_path,
                description=description,
                extracted_text=extracted_text,
                model=model_name,
                error=last_err if not description else None,
            )
        )

        # 每 10 帧增量落盘 — 崩溃后不丢失
        if out_json_path and (i % 10 == 0 or i == len(frames)):
            _save_partial(results, out_json_path)
            log.info(f"  ... {i}/{len(frames)} frames described (saved)")

    if out_json_path:
        _save_partial(results, out_json_path)
        log.info(f"Saved visual descriptions: {out_json_path}")

    return results


def _save_partial(
    results: list[VisualDescription], out_json_path: Path | str
) -> None:
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results], f, ensure_ascii=False, indent=2
        )


def _parse_retry_delay(err_str: str, default: float) -> float:
    """从 Gemini 429 错误里解析 retryDelay,比如 'retryDelay': '2s' 或 '2.8s'。
    解析不到就返回 default。上限 60s 避免等太久。"""
    import re

    m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", err_str)
    if m:
        try:
            return min(60.0, float(m.group(1)) + 0.5)  # 多等 0.5s 保险
        except ValueError:
            pass
    return default


def load_visual_descriptions(json_path: Path | str) -> list[VisualDescription]:
    p = Path(json_path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [VisualDescription(**d) for d in json.load(f)]
