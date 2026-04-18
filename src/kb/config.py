"""加载 YAML 配置 + 环境变量。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """加载 YAML 配置 + .env"""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 路径规范化为绝对路径(基于项目根)
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((PROJECT_ROOT / rel).resolve())

    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    v = os.environ.get(key, default)
    if required and not v:
        raise RuntimeError(f"Missing env: {key}. Check .env file.")
    return v
