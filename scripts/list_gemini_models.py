"""List all Gemini models available to the current GEMINI_API_KEY.

Usage (from project root):
    source .venv/bin/activate
    python scripts/list_gemini_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("❌ GEMINI_API_KEY not set in .env", file=sys.stderr)
    sys.exit(1)

from google import genai

client = genai.Client(api_key=key)

print("Available Gemini models (supporting generateContent):\n")
for m in client.models.list():
    methods = getattr(m, "supported_actions", None) or getattr(
        m, "supported_generation_methods", []
    )
    name = m.name
    display = getattr(m, "display_name", "")
    # 只显示支持 generateContent 的
    if methods and "generateContent" not in methods:
        continue
    print(f"  {name:<50} {display}")
