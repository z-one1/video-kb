#!/usr/bin/env bash
# 环境体检脚本 — 检查外部依赖是否就绪
set -e

echo "=== video_kb environment doctor ==="
echo

check() {
    local label="$1"; shift
    if "$@" > /dev/null 2>&1; then
        echo "  ✅ $label"
    else
        echo "  ❌ $label  ($*)"
    fi
}

echo "System tools:"
check "ffmpeg"   ffmpeg -version
check "ffprobe"  ffprobe -version
check "yt-dlp"   yt-dlp --version
check "python"   python --version
check "claude"   claude --version

echo
echo "Python packages (from current venv):"
python - <<'PY'
import importlib
pkgs = [
    "yt_dlp", "faster_whisper", "scenedetect", "google.genai",
    "sentence_transformers", "chromadb", "langchain_text_splitters",
    "pydantic", "typer", "rich", "yaml", "PIL", "dotenv",
]
for p in pkgs:
    try:
        importlib.import_module(p)
        print(f"  ✅ {p}")
    except Exception as e:
        print(f"  ❌ {p}  ({type(e).__name__}: {e})")
PY

echo
echo "Environment variables:"
# 先尝试从 .env 加载
ENV_FILE="$(dirname "$0")/../.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
fi
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    echo "  ✅ GEMINI_API_KEY (length=${#GEMINI_API_KEY}, source=$([ -f "$ENV_FILE" ] && echo ".env" || echo "shell"))"
else
    echo "  ❌ GEMINI_API_KEY (not set; check $ENV_FILE)"
fi

echo
echo "Done."
