"""Exports: MIDI (notes + pitch bends at exact timing), rendered audio,
Serum preset and a JSON report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BEND_RANGE = 12  # semitones; the generated Serum preset uses the same range


def write_midi(path: str | Path, notes, bpm: float) -> Path:
    import mido
    tpb = 960
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    tr.append(mido.MetaMessage("track_name", name="Fourier Analyzer notes", time=0))
    # RPN 0 (pitch-bend range) = BEND_RANGE semitones
    for cc, v in ((101, 0), (100, 0), (6, BEND_RANGE), (38, 0), (101, 127), (100, 127)):
        tr.append(mido.Message("control_change", control=cc, value=v, time=0))
    sec_to_tick = tpb * bpm / 60.0
    events = []
    notes = sorted(notes, key=lambda n: n.start)
    for k, n in enumerate(notes):
        p = int(round(n.pitch))
        cents_off = (n.pitch - p)
        v = int(np.clip(round(n.velocity * 127), 1, 127))
        t0, t1 = n.start, n.start + n.duration
        # the bend stream belongs to the most recent note: stop where the next
        # note takes over (legato notes overlap their predecessor)
        t_next = notes[k + 1].start if k + 1 < len(notes) else None
        t_stop = min(t1, t_next) if t_next is not None else t1
        if n.bend or abs(cents_off) > 0.005:
            pts = n.bend or [[0.0, 0.0]]
            for t, semis in pts:
                if t0 + t >= t_stop and t > 0:
                    break
                val = int(np.clip((semis + cents_off) / BEND_RANGE * 8191, -8192, 8191))
                events.append((t0 + t, 1, mido.Message("pitchwheel", pitch=val)))
            if t_next is None or t_next >= t1:
                events.append((t1, 3, mido.Message("pitchwheel", pitch=0)))
        elif k > 0 and (notes[k - 1].bend or abs(notes[k - 1].pitch - round(notes[k - 1].pitch)) > 0.005):
            events.append((t0, 1, mido.Message("pitchwheel", pitch=0)))    # clear a previous note's bend
        events.append((t0, 2, mido.Message("note_on", note=p, velocity=v)))
        events.append((t1, 0, mido.Message("note_off", note=p, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t, _, msg in events:
        tick = int(round(max(t, 0) * sec_to_tick))
        msg.time = max(0, tick - last)
        last = max(last, tick)
        tr.append(msg)
    path = Path(path)
    mid.save(str(path))
    return path


def write_report(path: str | Path, data: dict) -> Path:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(data, indent=1, default=conv))
    return Path(path)
