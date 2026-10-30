"""Command line entry point.

    python -m fourier_analyzer app              # desktop app (default)
    python -m fourier_analyzer serve            # server only (open the printed URL)
    python -m fourier_analyzer separate song.wav
    python -m fourier_analyzer reconstruct bass.wav --start 12.5 --end 16.5 --target 90
    python -m fourier_analyzer update [--check]
    python -m fourier_analyzer models           # download ML models
    python -m fourier_analyzer calibrate        # measure Serum's cutoff curve (Mac + Serum 2)
    python -m fourier_analyzer selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _ensure_streams():
    """pythonw.exe (the windowless Windows launch) has no stdout/stderr; send
    them to a log file so prints and logging do not fail."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from . import config
    f = open(config.app_dir() / "launcher.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def main(argv=None):
    _ensure_streams()
    ap = argparse.ArgumentParser(prog="fourier-analyzer")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("app")
    a.add_argument("--no-update", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=47800)
    sp = sub.add_parser("separate")
    sp.add_argument("file")
    sp.add_argument("--out", default=None)
    rc = sub.add_parser("reconstruct")
    rc.add_argument("file")
    rc.add_argument("--start", type=float, default=None)
    rc.add_argument("--end", type=float, default=None)
    rc.add_argument("--bpm", type=float, default=None)
    rc.add_argument("--target", type=float, default=90)
    rc.add_argument("--time", type=float, default=240)
    rc.add_argument("--out", default=".")
    up = sub.add_parser("update")
    up.add_argument("--check", action="store_true")
    sub.add_parser("models")
    sub.add_parser("calibrate")
    sub.add_parser("selftest")
    sh = sub.add_parser("serum-host")          # internal: helper process for real Serum renders
    sh.add_argument("job")
    args = ap.parse_args(argv)
    cmd = args.cmd or "app"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if cmd == "app":
        from .app.main import run
        run(open_window=True)
    elif cmd == "serve":
        from .app.main import run
        run(open_window=False, port=args.port)
    elif cmd == "separate":
        from .separation.stems import separate_all
        out = Path(args.out or Path(args.file).with_suffix("")).resolve()
        stems = separate_all(Path(args.file), out, progress=print)
        print(json.dumps({k: str(v) for k, v in stems.items()}, indent=1))
    elif cmd == "reconstruct":
        _reconstruct_cli(args)
    elif cmd == "update":
        from . import updater
        info = updater.check()
        print(json.dumps(info, indent=1))
        if not args.check and info.get("available"):
            print(updater.apply(info, log_cb=print))
    elif cmd == "models":
        from .separation.stems import prefetch_models
        print(prefetch_models(progress=print))
    elif cmd == "calibrate":
        from .serum.host import calibrate
        print(json.dumps(calibrate(progress=print), indent=1, default=str))
    elif cmd == "serum-host":
        from .serum.host import host_main
        sys.exit(host_main(args.job))
    elif cmd == "selftest":
        sys.exit(_selftest())


def _reconstruct_cli(args):
    from . import audio, config
    from .analysis.features import analyze
    from .export import write_midi
    from .match.optimizer import OPT_SR, Options, Optimizer
    from .serum.mapping import build_preset_bytes
    from .synth.wavetables import TableBank
    y, sr = audio.load(args.file, sr=OPT_SR)
    a0 = int((args.start or 0) * sr)
    b0 = int((args.end or len(y) / sr) * sr)
    y = y[a0:b0]
    an = analyze(y, sr, bpm=args.bpm)
    bank = TableBank()
    opt = Optimizer(y, sr, an, Options(target_score=args.target, time_budget=args.time),
                    bank=bank, progress=lambda st, f, b, m, *_: m and print(f"[{st}] {m}"))
    res = opt.run()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.file).stem + " FA"
    (out / f"{name}.SerumPreset").write_bytes(build_preset_bytes(res.patch, bank, name=name))
    write_midi(out / f"{name}.mid", an.notes, an.bpm)
    print(f"waveform similarity {res.score:.1f}%  {res.details}")
    print("\n".join(res.patch.describe()))


def _selftest() -> int:
    """Render a known patch, reconstruct it, and check the preset round trip."""
    from .analysis.features import analyze
    from .match.optimizer import Options, Optimizer
    from .serum.mapping import build_preset_bytes
    from .serum.preset import read_preset_bytes
    from .synth.engine import Renderer
    from .synth.patch import Filter, Note, Osc, Patch
    r = Renderer()
    p = Patch(bpm=140)
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, cutoff=600, reso=0.2)
    notes = [Note(0, 0.4, 36), Note(0.43, 0.4, 39), Note(0.86, 0.8, 34)]
    y = r.render(p, notes, 1.8)
    an = analyze(y, r.sr, bpm=140)
    res = Optimizer(y, r.sr, an, Options(time_budget=40)).run()
    h, st = read_preset_bytes(build_preset_bytes(res.patch, name="selftest"))
    ok = res.score > 85 and h["presetName"] == "selftest"
    print(f"selftest accuracy {res.score:.1f} -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
