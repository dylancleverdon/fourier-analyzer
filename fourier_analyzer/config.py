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
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_NAME
    else:
        p = Path.home() / ".local" / "share" / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_windows() -> bool:
    return sys.platform == "win32"


def venv_python(gui: bool = False) -> Path:
    """The app's private Python (pythonw.exe for windowless runs on Windows)."""
    v = app_dir() / "venv"
    if is_windows():
        return v / "Scripts" / ("pythonw.exe" if gui else "python.exe")
    return v / "bin" / "python"


def no_window_flags() -> int:
    """subprocess creationflags that keep helper processes from flashing a
    console window on Windows."""
    return 0x08000000 if is_windows() else 0          # CREATE_NO_WINDOW


def reveal(path: Path, select: bool = False) -> None:
    """Show a file or folder in Finder / Explorer."""
    import subprocess
    path = Path(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if select else ["open", str(path)])
    elif is_windows():
        if select:
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))                    # noqa: S606 (Windows only)
    else:
        subprocess.Popen(["xdg-open", str(path if not select else path.parent)])


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


def default_exports_dir() -> Path:
    if sys.platform in ("darwin", "win32"):
        return Path.home() / "Music" / "Fourier Analyzer"
    return data_dir() / "exports"


def exports_dir() -> Path:
    """User-visible export folder (Settings -> Export folder)."""
    s = settings().get("export_dir")
    p = Path(s).expanduser() if s else default_exports_dir()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        p = default_exports_dir()
        p.mkdir(parents=True, exist_ok=True)
    return p


def diagnostics_path() -> Path:
    return app_dir() / "serum_diagnostics.txt"


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


# Windows: Serum 2 keeps its content under Documents\Xfer (per user or Public)
_WIN_CONTENT = ["~/Documents/Xfer/Serum 2 Presets", "C:/Users/Public/Documents/Xfer/Serum 2 Presets",
                "C:/ProgramData/Xfer/Serum 2 Presets", "~/OneDrive/Documents/Xfer/Serum 2 Presets"]
SERUM_TABLE_CANDIDATES = [
    "~/Documents/Xfer/Serum 2 Presets/Tables",
    "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Tables",
    "~/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Tables",
    "~/Music/Xfer/Serum 2 Presets/Tables",
] + [c + "/Tables" for c in _WIN_CONTENT[1:]]
SERUM_PRESET_CANDIDATES = [
    "~/Documents/Xfer/Serum 2 Presets/Presets",
    "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets",
    "~/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets",
] + [c + "/Presets" for c in _WIN_CONTENT[1:]]
# VST3 first: it loads more reliably outside a DAW than the Audio Unit
SERUM_PLUGIN_CANDIDATES = [
    "/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
    "~/Library/Audio/Plug-Ins/VST3/Serum2.vst3",
    "/Library/Audio/Plug-Ins/Components/Serum2.component",
    "~/Library/Audio/Plug-Ins/Components/Serum2.component",
    "/Library/Audio/Plug-Ins/VST3/Serum 2.vst3",
    "/Library/Audio/Plug-Ins/Components/Serum 2.component",
    # Windows
    "C:/Program Files/Common Files/VST3/Serum2.vst3",
    "C:/Program Files/Common Files/VST3/Serum 2.vst3",
    "C:/Program Files/Common Files/VST3/Xfer Records/Serum2.vst3",
    "C:/Program Files/Common Files/VST3/Xfer/Serum2.vst3",
    "~/AppData/Local/Programs/Common/VST3/Serum2.vst3",
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
    """Where exported presets get copied so they show up in Serum's browser
    (Settings -> Install presets into)."""
    custom = settings().get("serum_preset_install_dir")
    if custom:
        p = Path(custom).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
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


def serum_plugin_paths() -> list[Path]:
    """Every installed Serum 2 plugin file, the user's setting first."""
    out = []
    s = settings().get("serum_plugin_path")
    if s and Path(s).expanduser().exists():
        out.append(Path(s).expanduser())
    for c in SERUM_PLUGIN_CANDIDATES:
        p = Path(c).expanduser()
        if p.exists() and p not in out:
            out.append(p)
    return out
