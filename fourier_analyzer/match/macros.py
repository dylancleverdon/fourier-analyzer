"""Named macros for the knobs a producer reaches for first.

Each macro is wired so the patch sounds exactly the same at the macro's saved
position: sweeps (cutoff, warp amount, FM depth, LFO rate, drive) are bipolar
routes with the macro at 50 %, and LFO depth scales the LFO's routes through
the aux input with the macro at 100 %.
"""
from __future__ import annotations

from ..synth.patch import N_MACROS, Mod, Patch


def assign_macros(p: Patch) -> list[str]:
    """Map the patch's key parameters to named macros (in place). Returns
    the labels assigned, in macro order."""
    plans = []
    if p.filter.enabled:
        plans.append(("CUTOFF", [("filter.cutoff", 0.6)], 0.5, True))
    o0 = p.oscs[0]
    for slot, code in (("osc1.warp", o0.warp), ("osc1.warp2", o0.warp2)):
        if code and code != "none":
            label = "FM DEPTH" if code.upper().startswith(("FM", "PD")) else "WARP"
            if label not in [x[0] for x in plans]:
                plans.append((label, [(slot, 0.5)], 0.5, True))
    lfo_srcs = []
    for m in p.mods:
        if m.source.startswith("lfo") and m.amount and not m.bypass and m.source not in lfo_srcs:
            lfo_srcs.append(m.source)
    if lfo_srcs:
        k = int(lfo_srcs[0][3:]) - 1
        plans.append(("LFO RATE", [(f"@lfos.{k}.rate_hz", 0.4)], 0.5, True))
        plans.append(("LFO DEPTH", "aux:" + lfo_srcs[0], 1.0, False))
    dist = next((j for j, m in enumerate(p.fx_chain) if m.get("type") == "distortion"), None)
    if dist is not None:
        plans.append(("DRIVE", [(f"@fx_chain.{dist}.drive", 0.5)], 0.5, True))
    elif p.filter.enabled and p.filter.drive > 0:
        plans.append(("DRIVE", [("filter.drive", 0.5)], 0.5, True))
    used = {m.source for m in p.mods} | {m.aux for m in p.mods if m.aux}
    free = [i for i in range(N_MACROS) if f"macro{i + 1}" not in used and not p.macros[i]]
    labels = []
    names = list(getattr(p, "macro_names", None) or [""] * N_MACROS)
    for (label, routes, value, bipolar), i in zip(plans, free):
        src = f"macro{i + 1}"
        if isinstance(routes, str):                 # LFO depth: the macro scales the LFO's routes
            lfo = routes.split(":", 1)[1]
            for m in p.mods:
                if m.source == lfo and m.amount and not m.aux:
                    m.aux = src
        else:
            for dest, amt in routes:
                p.mods.append(Mod(src, dest, amt, bipolar=bipolar))
        p.macros[i] = value
        names[i] = label
        labels.append(label)
    p.macro_names = names
    return labels
