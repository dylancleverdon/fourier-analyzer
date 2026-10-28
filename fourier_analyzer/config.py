"""Filesystem locations and user settings."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "FourierAnalyzer"
GITHUB_REPO = "dylancleverdon/fourier-analyzer"


def app_dir() -> Path:
    """Root of the installed app (code, venv, data)."""
    env = os.environ.get("FA_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        p = Path.home() / ".local" / "share" / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def code_dir() -> Path:
    """Directory containing the fourier_analyzer package (the update target)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    p = app_dir() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = app_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def projects_dir() -> Path:
    p = data_dir() / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def exports_dir() -> Path:
    """User-visible export folder."""
    p = Path.home() / "Music" / "Fourier Analyzer" if sys.platform == "darwin" else data_dir() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def version() -> str:
    for cand in (code_dir() / "VERSION", app_dir() / "app" / "VERSION"):
        if cand.exists():
            return cand.read_text().strip()
    return "0.0.0"


_SETTINGS = None


def settings_path() -> Path:
    return app_dir() / "settings.json"


def settings() -> dict:
    global _SETTINGS
    if _SETTINGS is None:
        try:
            _SETTINGS = json.loads(settings_path().read_text())
        except (OSError, ValueError):
            _SETTINGS = {}
    return _SETTINGS


def save_settings(**kw) -> dict:
    s = settings()
    s.update(kw)
    settings_path().write_text(json.dumps(s, indent=1))
    return s


SERUM_TABLE_CANDIDATES = [
    "~/Documents/Xfer/Serum 2 Presets/Tables",
    "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Tables",
    "~/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Tables",
    "~/Music/Xfer/Serum 2 Presets/Tables",
]
SERUM_PRESET_CANDIDATES = [
    "~/Documents/Xfer/Serum 2 Presets/Presets",
    "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets",
    "~/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets",
]
SERUM_PLUGIN_CANDIDATES = [
    "/Library/Audio/Plug-Ins/Components/Serum2.component",
    "~/Library/Audio/Plug-Ins/Components/Serum2.component",
    "/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
    "~/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
]


def _first_existing(cands, key):
    s = settings().get(key)
    if s and Path(s).expanduser().exists():
        return Path(s).expanduser()
    for c in cands:
        p = Path(c).expanduser()
        if p.exists():
            return p
    return None


def serum_tables_dir() -> Path | None:
    return _first_existing(SERUM_TABLE_CANDIDATES, "serum_tables_dir")


def serum_presets_dir() -> Path | None:
    return _first_existing(SERUM_PRESET_CANDIDATES, "serum_presets_dir")


def serum_user_presets_dir() -> Path:
    """Where exported presets get copied so they show up in Serum's browser."""
    base = serum_presets_dir()
    if base is not None:
        p = base / "User" / "Fourier Analyzer"
    else:
        p = exports_dir() / "Serum Presets"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        p = exports_dir() / "Serum Presets"
        p.mkdir(parents=True, exist_ok=True)
    return p


def serum_plugin_path() -> Path | None:
    return _first_existing(SERUM_PLUGIN_CANDIDATES, "serum_plugin_path")
