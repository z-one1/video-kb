"""Smoke test: verify Claude CLI can describe an image.

Usage (from project root):
    source .venv/bin/activate
    python scripts/test_vision_claude.py                    # uses first frame of an existing video
    python scripts/test_vision_claude.py <image_path>       # any image

Run this before `kb ingest` so you don't crash at stage 3.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_test_image(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg).expanduser().resolve()
        if not p.exists():
            sys.exit(f"❌ 图片不存在: {p}")
        return p

    # 自动找已有视频的第一帧
    videos_dir = ROOT / "kb" / "videos"
    if not videos_dir.exists():
        sys.exit(
            f"❌ 找不到 {videos_dir},也没传参数。\n"
            "用法: python scripts/test_vision_claude.py <image_path>"
        )

    for vdir in sorted(videos_dir.iterdir()):
        frames = vdir / "frames"
        if frames.is_dir():
            jpgs = sorted(frames.glob("*.jpg"))
            if jpgs:
                print(f"ℹ️  自动选中: {jpgs[0]}")
                return jpgs[0]

    sys.exit("❌ 找不到任何已抽取的关键帧,请先跑过一次 kb ingest。")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    img_path = find_test_image(arg)

    claude_bin = shutil.which("claude")
    if not claude_bin:
        sys.exit(
            "❌ 没找到 claude CLI。先 `npm install -g @anthropic-ai/claude-code` + `claude login`。"
        )
    print(f"✅ claude CLI: {claude_bin}")
    print(f"✅ 测试图片:  {img_path}  ({img_path.stat().st_size / 1024:.0f} KB)")
    print()

    prompt = f"""Analyze this frame from an educational/informational video: @{img_path}

Describe in 2-4 sentences:
1. The main visual content (diagram, code, chart, person talking, slides, text)
2. Any visible text — transcribe it literally if readable (max 200 chars)
3. The apparent context (whiteboard, slides, screencast, talking head, demo)

Output ONLY valid JSON (no markdown fences) with keys:
  "description" (string)
  "extracted_text" (string or null)
"""

    cmd = [
        claude_bin,
        "-p",
        "--output-format", "text",
        "--model", "sonnet",
    ]
    print(f"▶  运行: {' '.join(cmd)}")
    print(f"   stdin prompt 长度: {len(prompt)} 字符")
    print(f"   (首次调用要启动 CLI + 建会话,大约 5-15 秒)\n")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        sys.exit("❌ 超时 120s — CLI 可能没登录或网络有问题。")
    elapsed = time.time() - t0

    print(f"⏱  耗时: {elapsed:.1f}s")
    print(f"   exit code: {proc.returncode}")
    print(f"   stdout 长度: {len(proc.stdout)}")
    print(f"   stderr 长度: {len(proc.stderr)}")
    print()

    if proc.returncode != 0:
        print("❌ CLI 报错")
        print("--- stderr ---")
        print(proc.stderr[-1500:])
        print("--- stdout ---")
        print(proc.stdout[-500:])
        sys.exit(1)

    raw = proc.stdout.strip()
    print("=" * 60)
    print("📝 Claude 原始输出:")
    print("=" * 60)
    print(raw)
    print()

    # 尝试解析 JSON
    import re

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            print("=" * 60)
            print("✅ JSON 解析成功")
            print("=" * 60)
            print(f"description: {data.get('description', '')}")
            print(f"extracted_text: {data.get('extracted_text')}")
            print()
            # 判断是否真的"看到"了图
            desc = str(data.get("description", "")).lower()
            hints = ["cannot see", "unable to", "no image", "don't see", "didn't provide"]
            if any(h in desc for h in hints):
                print("⚠️  警告: 描述里带 'cannot see' 之类字样,可能没真正读到图片")
                print("    检查 @file 引用是否被 CLI 识别,或改用绝对路径")
                sys.exit(2)
            print("🎉 Claude CLI 成功读取并描述了图片。")
            print("   现在可以放心跑: kb ingest ... (provider=claude_code)")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析失败: {e}")
            print("    CLI 能返回文字但格式不对 — 可能需要调 prompt。")
            sys.exit(3)
    else:
        print("⚠️  输出里没找到 JSON 对象。")
        sys.exit(4)


if __name__ == "__main__":
    main()
