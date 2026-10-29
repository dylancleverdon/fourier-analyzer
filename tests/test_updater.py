import io
import tarfile

from fourier_analyzer import updater


def test_parse_version():
    assert updater.parse_version("v0.1.12") > updater.parse_version("0.1.9")
    assert updater.parse_version("1.0") == (1, 0, 0, 0)


def _tarball(version):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("VERSION", version + "\n"), ("fourier_analyzer/__init__.py", ""),
                           ("requirements.txt", "numpy\n")):
            b = data.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


def test_apply_swaps_code_dir(tmp_path, monkeypatch):
    root = tmp_path / "app"
    (root / "fourier_analyzer").mkdir(parents=True)
    (root / "VERSION").write_text("0.1.1\n")
    (root / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(updater, "_install_root", lambda: root)
    monkeypatch.setattr(updater, "_get", lambda url, timeout=0: _tarball("0.1.5"))
    installs = []
    monkeypatch.setattr(updater, "_pip_install", lambda req, log_cb=None: installs.append(req))
    res = updater.apply({"available": True, "url": "x", "latest": "0.1.5", "current": "0.1.1"})
    assert res["updated"]
    assert (root / "VERSION").read_text().strip() == "0.1.5"
    assert not installs                    # requirements unchanged -> no reinstall
    assert not (tmp_path / "app.old").exists()


def test_apply_reinstalls_when_requirements_change(tmp_path, monkeypatch):
    root = tmp_path / "app"
    (root / "fourier_analyzer").mkdir(parents=True)
    (root / "requirements.txt").write_text("scipy\n")
    monkeypatch.setattr(updater, "_install_root", lambda: root)
    monkeypatch.setattr(updater, "_get", lambda url, timeout=0: _tarball("0.2.0"))
    installs = []
    monkeypatch.setattr(updater, "_pip_install", lambda req, log_cb=None: installs.append(req))
    updater.apply({"available": True, "url": "x", "latest": "0.2.0", "current": "0.1.1"})
    assert len(installs) == 1


def test_missing_dependency_triggers_install(monkeypatch):
    from fourier_analyzer.app import main
    calls = []
    monkeypatch.setattr(main, "REQUIRED_IMPORTS", ("json", "definitely_not_installed_xyz"))
    monkeypatch.setattr(updater, "_pip_install", lambda req, log_cb=None: calls.append(req))
    main._ensure_dependencies()
    assert len(calls) == 1 and calls[0].name == "requirements.txt"
    calls.clear()
    monkeypatch.setattr(main, "REQUIRED_IMPORTS", ("json",))
    main._ensure_dependencies()
    assert not calls
