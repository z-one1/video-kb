"""Gemini fallback for structuring — 当 Claude CLI 不可用或超额时。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..schemas import EnrichedSegment, Notes
from .claude_code import _parse_notes_json  # 复用 parser
from .prompts import STRUCTURING_PROMPT, build_content_block

log = logging.getLogger("kb.structuring")


def structure_notes(
    enriched: list[EnrichedSegment],
    video_id: str,
    cfg: dict[str, Any],
    api_key: str,
) -> Notes:
    """用 Gemini 做结构化(免费层 1.5 Pro / 2.0 Flash)。"""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as e:
        raise ImportError("google-genai 未安装") from e

    content_block = build_content_block(enriched)
    max_chars = cfg.get("max_input_chars", 200000)
    if len(content_block) > max_chars:
        log.warning(f"Input too long ({len(content_block)} chars), truncating")
        content_block = content_block[:max_chars]

    prompt = STRUCTURING_PROMPT.replace("{content}", content_block)

    model_name = cfg.get("gemini_model", "gemini-2.0-flash-exp")
    client = genai.Client(api_key=api_key)

    log.info(f"Invoking Gemini for structuring: {model_name}")
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )

    raw = resp.text or ""
    return _parse_notes_json(raw, video_id, enriched)
