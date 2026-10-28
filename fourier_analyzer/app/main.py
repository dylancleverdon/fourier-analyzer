"""Desktop launcher: checks for updates, starts the local server and opens a
native window (pywebview) or, failing that, the default browser."""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
import webbrowser

from .. import config, updater


def _free_port(preferred: int = 47800) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def _auto_update():
    if not config.settings().get("auto_update", True) or "--no-update" in sys.argv:
        return
    info = updater.check(timeout=5)
    if info.get("available"):
        logging.info("updating %s -> %s", info["current"], info["latest"])
        try:
            if updater.apply(info).get("updated"):
                updater.relaunch()
        except Exception:
            logging.exception("auto update failed; starting current version")


def _ensure_path():
    """Make the venv's bin (ffmpeg symlink) and a bundled ffmpeg visible."""
    import os
    import shutil
    bindir = os.path.dirname(sys.executable)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        link_dir = config.cache_dir() / "bin"
        link_dir.mkdir(exist_ok=True)
        link = link_dir / "ffmpeg"
        if not link.exists():
            link.symlink_to(exe)
        os.environ["PATH"] = str(link_dir) + os.pathsep + os.environ["PATH"]
    except Exception:
        logging.warning("ffmpeg not found; mp3/m4a input may fail")


def run(open_window: bool = True, port: int | None = None):
    logfile = config.app_dir() / "app.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
    _ensure_path()
    _auto_update()
    import uvicorn
    from .server import app
    port = port or _free_port()
    url = f"http://127.0.0.1:{port}/"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    if not open_window:
        print(f"Fourier Analyzer running at {url}")
        th.join()
        return
    try:
        import webview
        webview.create_window(f"Fourier Analyzer {config.version()}", url, width=1360, height=900,
                              min_size=(900, 600))
        webview.start()
    except Exception:
        logging.exception("native window unavailable, opening browser")
        webbrowser.open(url)
        th.join()
