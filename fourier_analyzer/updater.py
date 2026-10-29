"""Self-update from GitHub.

Channel "release" (default): the latest GitHub Release of the repo, asset
``fourier-analyzer.tar.gz`` (built by .github/workflows/release.yml on every
push to the default branch).
Channel "branch:<name>": the tip of a branch, for testing unreleased work.

The install layout created by installer/install.sh is::

    ~/Library/Application Support/FourierAnalyzer/
        app/      <- code (this package, VERSION, requirements.txt)
        venv/     <- Python environment

Updating = download, unpack to app.new, install changed requirements into the
venv, swap app.new -> app (keeping app.old for rollback) and relaunch.
No signatures or checksums are verified (by request: speed over security).
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
API = "https://api.github.com/repos/{repo}"
ASSET = "fourier-analyzer.tar.gz"


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fourier-analyzer-updater",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return r.read()


def parse_version(v: str) -> tuple:
    v = v.strip().lstrip("v")
    parts = []
    for p in v.replace("-", ".").split("."):
        parts.append(int(p) if p.isdigit() else 0)
    return tuple(parts + [0] * (4 - len(parts)))


def channel() -> str:
    return config.settings().get("update_channel", "release")


def check(timeout: float = 10.0) -> dict:
    """Return {current, latest, available, url, notes} (never raises)."""
    cur = config.version()
    info = {"current": cur, "latest": None, "available": False, "url": None, "notes": "",
            "channel": channel()}
    repo = config.settings().get("github_repo", config.GITHUB_REPO)
    try:
        ch = channel()
        if ch.startswith("branch:"):
            br = ch.split(":", 1)[1]
            c = json.loads(_get(API.format(repo=repo) + f"/commits/{br}", timeout))
            sha = c["sha"]
            installed = (config.code_dir() / "COMMIT")
            cur_sha = installed.read_text().strip() if installed.exists() else ""
            info.update(latest=f"{cur.split('+')[0]}+{sha[:7]}", commit=sha,
                        url=f"https://codeload.github.com/{repo}/tar.gz/{sha}",
                        notes=f"branch {br}: {c['commit']['message'][:500]}")
            info["available"] = sha != cur_sha
            return info
        else:
            rel = json.loads(_get(API.format(repo=repo) + "/releases/latest", timeout))
            info["latest"] = rel.get("tag_name", "").lstrip("v")
            info["notes"] = (rel.get("body") or "")[:4000]
            for a in rel.get("assets", []):
                if a.get("name") == ASSET:
                    info["url"] = a["browser_download_url"]
            if not info["url"]:
                info["url"] = rel.get("tarball_url")
        info["available"] = bool(info["latest"]) and parse_version(info["latest"]) > parse_version(cur)
    except Exception as e:
        info["error"] = str(e)
    return info


def _install_root() -> Path:
    """The 'app' dir that holds the running code."""
    return config.code_dir()


def _venv_python() -> Path:
    vp = config.app_dir() / "venv" / "bin" / "python"
    return vp if vp.exists() else Path(sys.executable)


def _pip_install(req: Path, log_cb=None) -> None:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    py = str(_venv_python())
    if Path(uv).exists():
        cmd = [uv, "pip", "install", "--python", py, "-r", str(req)]
    else:
        # venvs made by uv have no pip unless seeded; bootstrap it if needed
        subprocess.run([py, "-m", "ensurepip", "--upgrade"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cmd = [py, "-m", "pip", "install", "-r", str(req)]
    if log_cb:
        log_cb("installing dependencies: " + " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def apply(info: dict | None = None, log_cb=None) -> dict:
    """Download and install the update. Returns {updated, version}."""
    info = info or check()
    if not info.get("available") or not info.get("url"):
        return {"updated": False, "reason": info.get("error") or "already up to date",
                "version": info.get("current")}
    root = _install_root()
    if not (root / "fourier_analyzer").exists():
        raise RuntimeError(f"unexpected install layout at {root}")
    if (root / ".git").exists():
        raise RuntimeError("running from a git checkout; use git pull instead")
    if log_cb:
        log_cb(f"downloading {info['url']}")
    data = _get(info["url"], timeout=120)
    new = root.with_name(root.name + ".new")
    old = root.with_name(root.name + ".old")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        # GitHub source tarballs wrap everything in one top-level folder
        tops = {m.name.split("/")[0] for m in members if m.name}
        strip = len(tops) == 1 and not any(m.name == "VERSION" for m in members)
        for m in members:
            name = m.name.split("/", 1)[1] if strip and "/" in m.name else (None if strip else m.name)
            if not name or name.startswith("/") or ".." in Path(name).parts:
                continue
            m.name = name
            try:
                tf.extract(m, new, filter="data")
            except TypeError:       # Python < 3.12
                tf.extract(m, new)
    if not (new / "fourier_analyzer").exists():
        raise RuntimeError("downloaded archive does not contain the app")
    if info.get("commit"):
        (new / "COMMIT").write_text(info["commit"] + "\n")
        (new / "VERSION").write_text(info["latest"] + "\n")
    elif info.get("latest"):
        (new / "VERSION").write_text(info["latest"] + "\n")
    req_new, req_old = new / "requirements.txt", root / "requirements.txt"
    if req_new.exists() and (not req_old.exists() or req_new.read_text() != req_old.read_text()):
        _pip_install(req_new, log_cb)
    shutil.rmtree(old, ignore_errors=True)
    os.replace(root, old)
    try:
        os.replace(new, root)
    except Exception:
        os.replace(old, root)
        raise
    shutil.rmtree(old, ignore_errors=True)
    # refresh the launcher/installer scripts that live outside the code dir
    try:
        launcher = config.app_dir() / "fourier-analyzer"
        src = root / "installer" / "launcher.sh"
        if src.exists():
            shutil.copy2(src, launcher)
            launcher.chmod(0o755)
    except OSError:
        pass
    if log_cb:
        log_cb(f"updated to {info.get('latest')}")
    return {"updated": True, "version": info.get("latest")}


def relaunch() -> None:
    """Replace the current process with a fresh app launch."""
    py = str(_venv_python())
    os.execv(py, [py, "-m", "fourier_analyzer", "app"] + [a for a in sys.argv[2:] if a != "--no-update"])
