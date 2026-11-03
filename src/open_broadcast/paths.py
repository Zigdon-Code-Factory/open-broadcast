"""Filesystem locations (bundled resources, per-user config/cache/logs)."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

APP_NAME = "OpenBroadcast"

PACKAGE_DIR = Path(__file__).resolve().parent


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def bundled_models_dir() -> Path:
    """Models shipped alongside the code (populated by scripts/fetch_models.py)."""
    return PACKAGE_DIR / "models"


def assets_dir() -> Path:
    return PACKAGE_DIR / "assets"


def _base_dir(kind: str) -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Windows":
        root = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(root) / APP_NAME / kind
    if system == "Darwin":
        if kind == "config":
            return home / "Library" / "Application Support" / APP_NAME
        if kind == "logs":
            return home / "Library" / "Logs" / APP_NAME
        return home / "Library" / "Caches" / APP_NAME
    env = {"config": "XDG_CONFIG_HOME", "cache": "XDG_CACHE_HOME", "logs": "XDG_STATE_HOME"}[kind]
    default = {"config": ".config", "cache": ".cache", "logs": ".local/state"}[kind]
    return Path(os.environ.get(env) or home / default) / "open-broadcast"


def config_dir() -> Path:
    return _base_dir("config")


def cache_dir() -> Path:
    return _base_dir("cache")


def log_dir() -> Path:
    return _base_dir("logs")
