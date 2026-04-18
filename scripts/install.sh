#!/usr/bin/env bash
# One-shot installer — run from project root.
# Usage: bash scripts/install.sh

set -e

cd "$(dirname "$0")/.."
echo "=== video_kb installer ==="
echo "Project root: $(pwd)"
echo

# 1. 检查 ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg 未安装"
    echo "请先跑: brew install ffmpeg"
    exit 1
fi
echo "✅ ffmpeg 已安装"

# 2. 选 Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &> /dev/null; then
    echo "❌ $PYTHON 未找到"
    exit 1
fi
PY_VERSION=$($PYTHON --version | awk '{print $2}')
echo "✅ Python: $PY_VERSION ($(which $PYTHON))"

# 3. 选择 uv 或 venv
if command -v uv &> /dev/null; then
    echo "✅ 使用 uv"
    if [ ! -d ".venv" ]; then
        uv venv -p "$PYTHON"
    fi
    source .venv/bin/activate
    uv pip install -e .
else
    echo "ℹ️  uv 未安装,使用标准 venv"
    if [ ! -d ".venv" ]; then
        $PYTHON -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .
fi

# 4. 初始化 .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "⚠️  请编辑 .env 并填入 GEMINI_API_KEY"
    echo "    获取地址: https://aistudio.google.com/apikey (免费)"
fi

echo
echo "=== ✅ Installation complete ==="
echo
echo "下一步:"
echo "  1. 编辑 .env 填 GEMINI_API_KEY"
echo "  2. source .venv/bin/activate"
echo "  3. bash scripts/doctor.sh    # 体检"
echo "  4. kb ingest <video_file>    # 试跑"
