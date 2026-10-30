"""Synth patch description shared by the renderer, the optimizer and the
Serum preset writer.

Units are chosen to be musically meaningful (Hz, seconds, semitones) so the
optimizer can reason about them; conversion to Serum's internal parameter
scales happens in ``fourier_analyzer.serum.mapping``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

# Serum 2 has three primary oscillators (A, B, C) plus sub and noise.
MAX_OSCS = 3

N_ENVS = 4
N_LFOS = 10
N_MACROS = 8

LFO_SHAPES = ("sine", "triangle", "saw_down", "saw_up", "square", "custom")
LFO_TYPES = ("shape", "random", "lorenz", "rossler", "path")
# Filter types are Serum 2's own names (L24, MgL24, LadderAcid, Comb2, ...);
# the short legacy names lp12/lp24/hp12/hp24/bp12 are accepted too.
FILTER_TYPES = ("lp12", "lp24", "hp12", "hp24", "bp12")
# Distortion modes: Serum 2 names without the leading "k" (Overdrive,
# SoftSat, Diode1, ...); legacy soft/hard/tube/sinefold/linfold/diode accepted.
DIST_MODES = ("soft", "hard", "tube", "sinefold", "linfold", "diode")
# Warp modes: Serum 2 names without the leading "k" (Sync, PD_OSC, FM_SUB,
# BendPos, DistTube, FilterLPF, ...); legacy fm_b/fm_sub/am_b/rm_b/rm_sub ok.
WARP_MODES = ("none", "fm_b", "fm_sub", "am_b", "rm_b", "rm_sub")
MOD_SOURCES = (tuple(f"env{i}" for i in range(1, N_ENVS + 1)) + tuple(f"lfo{i}" for i in range(1, N_LFOS + 1))
               + ("velocity", "note", "modwheel") + tuple(f"macro{i}" for i in range(1, N_MACROS + 1))
               + ("fixed", "pitchbend", "aftertouch", "polyat", "rand1", "rand2", "rand_discrete", "alt1", "alt2",
                  "release_velocity", "active_voices", "voice_index", "voice_mod1", "voice_mod2",
                  "mpe_x", "mpe_y", "mpe_z", "osc1", "osc2", "osc3", "noise", "sub", "filter1", "filter2",
                  "lfobus"))
# amount 1.0 means: cutoff +10 octaves, pos +1 (whole table), level +1,
# pitch +48 semitones, drive +1, fine +100 cents, pan +1, warp +1, mix +1;
# "voiceamp" is Serum's Voice Amp knob (VELO -> Amp): knob 0.5 = unity gain, the
# modulation moves the knob (gain = 2 * knob); bipolar mods move it +-amount/2.
MOD_DESTS = ("osc1.pos", "osc2.pos", "osc3.pos", "osc1.level", "osc2.level",
             "osc3.level", "osc1.warp", "osc2.warp", "osc3.warp", "osc1.warp2", "osc2.warp2", "osc3.warp2",
             "osc1.fine", "osc2.fine", "osc3.fine", "osc1.pan", "osc2.pan", "osc3.pan",
             "filter.cutoff", "filter.reso", "filter.drive", "filter.var",
             "filter2.cutoff", "filter2.reso", "filter2.drive", "filter2.var",
             "pitch", "amp", "voiceamp", "dist.drive", "noise.level", "sub.level",
             "fx.<index>.mix", "fx.<index>.drive",
             # any other Serum parameter: "@<patch path>" (e.g. "@fx_chain.2.feedback",
             # "@oscs.0.detune"); amount 1.0 = the parameter's full knob range
             "@<path>")

# Effect modules (Patch.fx_chain / fx_buses entries are dicts with "type")
FX_TYPES = ("distortion", "flanger", "phaser", "chorus", "delay", "comp", "reverb", "eq", "filter",
            "hyperdim", "bode", "convolve", "utility", "split", "split3", "splitms")
FX_DEFAULTS = {
    # every module also takes "level" (linear output gain) and "enabled"
    "distortion": {"mode": "Overdrive", "drive": 0.3, "mix": 1.0, "filter": "off", "filter_freq": 0.5,
                   "filter_bw": 0.5, "filter_type": 0.0, "filter_keytrack": False, "stages": 2,
                   "curve_a": None, "curve_b": None},
    "flanger": {"rate_hz": 0.3, "sync": None, "depth": 0.5, "feedback": 0.5, "phase": 0.0, "mix": 0.5},
    "phaser": {"rate_hz": 0.3, "sync": None, "depth": 0.5, "depth2": 0.5, "freq": 800.0, "feedback": 0.4,
               "stages": 4, "phase": 0.0, "mix": 0.5},
    "chorus": {"rate_hz": 0.8, "sync": None, "depth": 0.3, "delay_ms": 7.0, "delay2_ms": 0.0, "feedback": 0.0,
               "filt_hz": 20000.0, "filt_mode": "lp", "mix": 0.3},
    "delay": {"mode": "normal", "time": 0.25, "time2": 0.25, "offset_l": 1.0, "offset_r": 1.0, "link": False,
              "feedback": 0.35, "freq": 2000.0, "bw": 3.0, "mix": 0.25},
    "comp": {"multiband": True, "depth": 0.5, "amount": 0.6, "threshold_db": -18.0, "ratio": 4.0,
             "attack_ms": 5.0, "release_ms": 80.0, "makeup_db": 0.0, "xover_lo": 200.0, "xover_hi": 3000.0,
             "band_cfg": None, "ratio_below": 0.0},
    "reverb": {"kind": "Hall", "size": 0.5, "decay": 1.5, "damping": 0.5, "predelay": 0.0, "mix": 0.2,
               "width": 1.0, "lo_cut": 0.0, "hi_cut": 0.0, "mode": 0.0, "spin": 0.0, "diffusion": 0.5},
    "eq": {"freq1": 120.0, "gain1": 0.0, "q1": 0.7, "type1": "lowshelf",
           "freq2": 6000.0, "gain2": 0.0, "q2": 0.7, "type2": "highshelf"},
    "filter": {"ftype": "L24", "cutoff": 2000.0, "reso": 0.2, "drive": 0.0, "var": 0.5, "mix": 1.0,
               "pan": 0.0, "keytrack": False, "clean": False, "pz": None},
    "hyperdim": {"rate": 0.4, "detune": 0.3, "voices": 4, "retrig": False, "mix": 0.4, "dim_size": 0.5,
                 "dim_mix": 0.0},
    "bode": {"shift_hz": 50.0, "range_hz": 100.0, "direction": 0.0, "width": 0.5, "delay_time": 0.0,
             "feedback": 0.0, "balance": 0.0, "blur": 0.0, "mono_input": False, "retrig": False, "mix": 0.5},
    "convolve": {"ir": "", "size": 0.5, "decay": 1.0, "tone": 0.0, "predelay": 0.0, "attack": 0.0,
                 "damp": 0.0, "min_phase": False, "ir_gain_db": 0.0, "mix": 0.2},
    "utility": {"gain_db": 0.0, "hpf": 0.0, "lpf": 0.0, "invert": False, "invert_r": False,
                "mono_bass": False, "mono_freq": 120.0, "width": 1.0, "balance": 0.0, "mix": 1.0},
    "split": {"freq": 300.0, "bands": [[], []]},
    "split3": {"freq": 200.0, "freq2": 2000.0, "bands": [[], [], []]},
    "splitms": {"bands": [[], []]},
}


def fx(fx_type: str, /, **kw) -> dict:
    """New effect-module dict with defaults."""
    d = {"type": fx_type, "enabled": True}
    d.update(copy.deepcopy(FX_DEFAULTS.get(fx_type, {})))
    d.update(kw)
    return d


@dataclass
class SampleCfg:
    """Sample source + playback settings shared by the Sample, Granular and
    Spectral oscillator modes (manual pp. 68-122)."""
    source: str = ""              # sample id in SampleBank ("embedded:<n>" | "serum:<rel path>" | "file:<path>")
    root: Optional[float] = None  # MIDI note of the original pitch (baseNote); None = from the file name
    detune: float = 0.0           # cents (baseDetune)
    start: float = 0.0            # 0..1 of the sample (kParamStart %)
    end: float = 1.0              # 0..1 (kParamEnd %)
    loop: str = "off"             # off (one shot) | forward | reverse | pingpong | tailed
    loop_start: float = 0.0       # 0..1
    loop_end: float = 1.0         # 0..1
    crossfade: float = 0.0        # 0..1 of the loop length
    link_loop: bool = False       # loop end follows loop start (kParamLinkLoopLength)
    relative_loop: bool = False   # loop moves with the start marker (kParamLoopStartLink)
    exit_on_release: bool = False  # kParamLoopEndsAtRelease
    scan: float = 1.0             # playback speed / scan rate, 1 = normal, negative = backwards
    scan_range: int = 0           # 0: +-200 %, 1: +-400 %, 2: +-800 %
    reverse: bool = False
    scan_keytrack: bool = False
    scan_tempo_lock: bool = False
    base_tempo: float = 0.0       # kParamBaseTempo (sample length to BPM)
    random_start: float = 0.0     # unison START (random start offset per voice) 0..1
    span: float = 0.0             # unison SPAN (fixed start offset per voice) 0..1
    slicing: str = "off"          # off | auto | manual
    slices: list = field(default_factory=list)  # slice positions 0..1
    slice_root: int = 36
    manual_position: bool = False  # granular/spectral manual X|Y mode
    position: float = 0.0         # manual playhead position 0..1
    fade_ms: float = 0.0          # edge fades
    gain: float = 1.0


@dataclass
class GranularCfg:
    density: float = 20.0         # grains per second (Free), division (BPM) or count (Grains)
    density_mode: str = "free"    # free | bpm | grains
    length: float = 0.08          # grain length in seconds (Free), beats (BPM) or 0..1 of period (Percent)
    length_mode: str = "free"     # free | bpm | percent
    window: str = "Hann"          # Hann Welch Gaussian BlackmanHarris Sinc Tukey Triangle Trapezoid ExpDec ExpDecRev
    window_amount: float = 1.0    # 0..1 influence of the window curve
    window_skew: float = 0.0      # -1..1
    rand_offset: float = 0.0      # 0..1 per-grain randomisations
    rand_dir: float = 0.0
    rand_pitch: float = 0.0       # semitones
    rand_length: float = 0.0
    rand_pan: float = 0.0
    rand_gain: float = 0.0
    rand_window: float = 0.0
    rand_skew: float = 0.0
    rand_warp: float = 0.0
    rand_warp2: float = 0.0
    reverse_grains: bool = False
    spawn: str = "together"       # together | even | exp | random (unison spawn pattern)
    jump_start: bool = True
    max_grains: int = 64
    loop_grains: bool = False


@dataclass
class SpectralCfg:
    freq_lo: float = 20.0         # Hz band limits of the resynthesis
    freq_hi: float = 20000.0
    smooth_edges: bool = False    # Butterworth edges (kParamLoHiIsSmooth)
    post_warp: bool = False
    phase_lock: bool = False
    transients: bool = False
    filter_shift: float = 0.0     # spectral filter CUT (kParamSpecFltShift)
    filter_mix: float = 0.0       # spectral filter wet/dry 0..1
    filter_curve: Optional[list] = None   # [[x, y, curve], ...] drawn spectral mask


@dataclass
class MultiCfg:
    sfz: str = ""                 # embedded SFZ text
    folder: str = ""              # base folder for the SFZ sample paths
    timbre: float = 0.0           # TIMBRE (zone shift, semitones-ish)
    env_override: bool = False
    env: Optional[dict] = None    # delay, attack, hold, decay, sustain, release (seconds / 0..1)
    vel_track: float = 1.0
    random: float = 0.0


UNISON_MODES = ("linear", "super", "exp", "inv", "random")


@dataclass
class Osc:
    enabled: bool = False
    type: str = "wavetable"       # wavetable | sample | granular | spectral | multisample
    table: str = "basic:saw"      # basic:<name> | serum:<relative path> | custom:<n>
    position: float = 0.0         # 0..1 across the table
    octave: int = 0
    semi: int = 0                 # SEM (kParamPitch)
    coarse: float = 0.0           # CRS continuous semitones (kParamCoarsePit)
    fine: float = 0.0             # cents
    level: float = 0.75           # 0..1 (linear amplitude)
    pan: float = 0.0              # -1..1
    pitch_mode: str = "semitones"  # semitones | harmonics | ratio
    ratio: float = 1.0            # harmonic number / ratio when pitch_mode != semitones
    ratio_source: int = -1        # oscillator the ratio refers to (-1 = played note)
    hz_offset: float = 0.0        # linear frequency offset in Hz (kParamHzOffset)
    pitch_track: bool = True      # follows the keyboard (off: fixed C-2 / C3)
    bend_track: bool = True
    unison: int = 1
    detune: float = 0.1           # 0..1, Serum-style detune amount
    blend: float = 0.75           # 0..1 level of the outer unison voices
    unison_mode: str = "linear"   # linear | super | exp | inv | random
    unison_width: float = 1.0     # stereo spread 0..1 (kParamUnisonStereo)
    unison_wtpos: float = 0.0     # WT POS spread across unison voices (-1..1)
    unison_warp: float = 0.0      # warp 1 spread (-1..1)
    unison_warp2: float = 0.0
    random_phase: float = 1.0     # 0..1 (0 = every note starts at `phase`)
    phase: float = 0.0            # 0..1 start phase of the cycle at note-on
    phase_memory: str = "all"     # all | contiguous | per_voice
    smooth_interp: bool = False   # WT POS smooth interpolation (kParamXfadeMode)
    # Warp modes use Serum's names without the "k" (Sync, PD_OSC, FM_SUB, BendPos,
    # DistTube, FilterLPF, Remap_1, SpectralComb, ...); legacy fm_b/fm_sub/am_b/rm_b/rm_sub ok.
    warp: str = "none"
    warp_amount: float = 0.0      # 0..1
    warp_var: float = 0.5         # WARP Var (sync softness, FM mode, ...) 0..1
    warp2: str = "none"           # second warp slot (applied after the first)
    warp2_amount: float = 0.0
    warp2_var: float = 0.5
    remap_curve: Optional[list] = None   # Remap warp graph [[x, y, curve], ...]
    unison_stack: str = "none"    # none | Octave1..3 | OctaveFifth1..3 | Center12 | Center24
    unison_range: float = 2.0     # semitones spread at detune 1.0 (Serum "range")
    key_range: tuple = (0, 127)   # OSC MAPPING note range (kParamKeyZoneMin/Max)
    vel_range: tuple = (0, 127)
    key_fold: bool = False
    key_warp: bool = False        # keeps warping others outside its range
    smp: SampleCfg = field(default_factory=SampleCfg)
    gran: GranularCfg = field(default_factory=GranularCfg)
    spec: SpectralCfg = field(default_factory=SpectralCfg)
    multi: MultiCfg = field(default_factory=MultiCfg)


@dataclass
class SubOsc:
    enabled: bool = False
    shape: str = "sine"           # sine | roundrect | triangle | saw | square | pulse
    octave: int = -1
    coarse: float = 0.0           # CRS semitones
    level: float = 0.5
    pan: float = 0.0
    phase: float = 0.0            # 0..1 start phase
    contiguous: bool = False      # new notes continue the previous phase
    pitch_track: bool = True


NOISE_COLORS = ("White", "Pink", "Brown", "Geiger")


@dataclass
class NoiseOsc:
    enabled: bool = False
    color: float = 0.5            # FILTER/colour knob: 0 = dark, 0.5 = flat, 1 = bright
    level: float = 0.2
    kind: str = "White"           # White | Pink | Brown | Geiger | sample
    sample: str = ""              # SampleBank id when kind == "sample"
    stereo: float = 0.0           # 0 = mono .. 1 = decorrelated L/R
    one_shot: bool = False
    start: float = 0.0            # 0..1 start phase (sample start)
    random_start: float = 1.0     # 0..1
    pitch: float = 0.0            # semitones from the file's own pitch
    fine: float = 0.0             # cents
    keytrack: bool = False
    pan: float = 0.0


@dataclass
class Filter:
    enabled: bool = False
    type: str = "lp24"
    cutoff: float = 2000.0        # Hz
    reso: float = 0.1             # 0..1
    drive: float = 0.0            # 0..1
    keytrack: float = 0.0         # 0..1
    var: float = 0.5              # 0..1 type-specific (morph, comb damping, vowel ...)
    mix: float = 1.0              # 0..1 wet
    level: float = 1.0            # output gain (linear)
    pan: float = 0.0              # -1..1 L/R cutoff offset (kParamStereo)
    clean: bool = False           # drive Clean Mode (-24 dB in / +24 dB out)
    pz: Optional[list] = None     # PZ SVF drawable filter sections
    x: float = 0.0                # PZ/morph X|Y controls
    y: float = 0.0


@dataclass
class Env:
    attack: float = 0.001
    hold: float = 0.0
    decay: float = 0.3
    sustain: float = 1.0
    release: float = 0.05
    attack_curve: float = 0.5     # 0..1 (0.5 = linear); Serum curve 0..100 / 100
    decay_curve: float = 0.666
    release_curve: float = 0.666
    start: float = 0.0            # level the attack starts from
    end: float = 0.0              # level the release ends at
    legato_inverted: bool = False  # always retrigger even with legato on


@dataclass
class LFO:
    shape: str = "sine"
    rate_hz: float = 2.0
    sync: Optional[str] = None    # e.g. "1/8", "1/4T", "1/16D"; overrides rate_hz
    phase: float = 0.0            # 0..1
    mode: str = "trig"            # trig | free | env (one shot)
    points: Optional[list] = None  # custom shape: [[x, y], ...] or [[x, y, curve], ...]
    type: str = "shape"           # shape | random (S&H) | lorenz | rossler | path
    rise: float = 0.0             # seconds to fade in after note-on
    delay: float = 0.0            # seconds before the LFO starts
    smooth: float = 0.0           # 0..1 slew
    direction: str = "forward"    # forward | reverse | pingpong
    loopback: Optional[float] = None  # envelope mode: x where the loop restarts
    mono: bool = False            # one LFO for all voices
    anchored: bool = False        # HOST: phase from the song position
    swing: bool = False
    path: Optional[list] = None   # Path LFO: [[x, y, curve], ...] 2-D path
    path_open: bool = False
    point_mods: list = field(default_factory=list)   # [{"point": i, "axis": "x"|"y"|"curve", "bus": b}]


@dataclass
class Mod:
    source: str = "lfo1"
    dest: str = "filter.cutoff"
    amount: float = 0.0           # -1..1, see MOD_DESTS comment
    bipolar: bool = False
    aux: Optional[str] = None     # optional second source that scales this one
    aux_invert: bool = False
    curve: float = 0.0            # -1..1 source response bend (CRV)
    curve_points: Optional[list] = None   # drawn remap curve for the source
    aux_curve: float = 0.0
    aux_curve_points: Optional[list] = None
    smooth_rise: float = 0.0      # slew limiter (seconds)
    smooth_fall: float = 0.0
    output: float = 1.0           # OUTPUT scaling 0..1
    bypass: bool = False


@dataclass
class Distortion:
    enabled: bool = False
    mode: str = "soft"
    drive: float = 0.3            # 0..1 (0..36 dB of gain)
    mix: float = 1.0


@dataclass
class OTT:
    enabled: bool = False
    depth: float = 0.5            # 0..1 wet
    amount: float = 0.6           # 0..1 compression strength (up + down)


@dataclass
class EQ:
    enabled: bool = False
    low_freq: float = 120.0
    low_gain: float = 0.0         # dB (low shelf)
    high_freq: float = 6000.0
    high_gain: float = 0.0        # dB (high shelf)


@dataclass
class Chorus:
    enabled: bool = False
    rate_hz: float = 0.8
    depth: float = 0.3
    mix: float = 0.3


@dataclass
class Clip:
    """A Serum CLIP slot (the reconstructed riff is stored here as the preview clip)."""
    notes: list = field(default_factory=list)   # [start_beats, length_beats, pitch, velocity]
    length_beats: float = 4.0
    name: str = ""


@dataclass
class Arp:
    enabled: bool = False
    shape: str = "Up"             # Up Down UpDown DownUp UpAndDown DownAndUp Played Chord Rand
                                  # RandNoDup RandDrift RandOnce Converge Diverge ConvAndDiv ThumbUp ThumbUD
                                  # PinkyUp PinkyUD Pattern
    rate: str = "1/16"            # tempo division
    gate: float = 0.75            # 0..1+ of the step
    shift: int = 12               # transpose per range step (semitones)
    range: int = 1                # octaves/repeats of the transpose
    transpose_shape: str = "Up"
    repeats: int = 1
    offset: int = 0
    chance: float = 1.0
    pattern: list = field(default_factory=list)  # custom steps [step, length, row, velocity]


@dataclass
class VoiceControl:
    """GLOBAL > Voice Control: per-voice random offsets and per-voice steps."""
    rand_pan: float = 0.0         # 0..1
    rand_detune: float = 0.0      # cents
    rand_cutoff: float = 0.0      # 0..1 (octaves * 4)
    rand_env: float = 0.0         # 0..1 time scaling
    env_scale: float = 1.0        # global envelope time scaling
    lfo_scale: float = 1.0        # global LFO time scaling
    steps: list = field(default_factory=list)   # per-voice dicts: detune (cents), pan, cutoff, env, mod1, mod2


@dataclass
class Patch:
    name: str = "Reconstructed"
    oscs: list = field(default_factory=lambda: [Osc(enabled=True), Osc(), Osc()])
    sub: SubOsc = field(default_factory=SubOsc)
    noise: NoiseOsc = field(default_factory=NoiseOsc)
    filter: Filter = field(default_factory=Filter)
    filter2: Filter = field(default_factory=Filter)
    # per source (osc1, osc2, osc3, noise, sub, filter1, filter2):
    #   dest filter|main|direct|none (filters: filter = the other filter),
    #   balance 0 = filter 1 only .. 1 = filter 2 only, send1/send2 0..1,
    #   env1: False makes the output free running (ENV 1 not applied)
    routing: dict = field(default_factory=dict)
    envs: list = field(default_factory=lambda: [Env(), Env(sustain=0.0), Env(sustain=0.0), Env(sustain=0.0)])
    lfos: list = field(default_factory=lambda: [LFO() for _ in range(N_LFOS)])
    macros: list = field(default_factory=lambda: [0.0] * N_MACROS)
    macro_names: list = field(default_factory=lambda: [""] * N_MACROS)   # labels shown in Serum
    mods: list = field(default_factory=list)
    lfo_buses: list = field(default_factory=list)   # LFO point-mod bus base values
    velo_curve: Optional[list] = None   # VELO graph [[x, y, curve], ...]
    note_curve: Optional[list] = None   # NOTE graph
    fx_chain: list = field(default_factory=list)     # main FX rack, in order
    fx_buses: list = field(default_factory=lambda: [[], []])
    bus_levels: list = field(default_factory=lambda: [1.0, 1.0])
    bus_dest: list = field(default_factory=lambda: ["main", "main"])   # main | direct | bus1 | bus2
    direct_level: float = 1.0
    distortion: Distortion = field(default_factory=Distortion)
    ott: OTT = field(default_factory=OTT)
    eq: EQ = field(default_factory=EQ)
    chorus: Chorus = field(default_factory=Chorus)
    mono: bool = True
    legato: bool = True
    poly: int = 16
    same_note_limit: bool = False
    glide: float = 0.0            # seconds
    glide_curve: float = 0.5      # 0..1 (0.5 = linear-in-pitch exponential approach)
    glide_always: bool = True
    glide_scaled: bool = False
    master: float = 0.7
    bpm: float = 140.0
    transpose: int = 0            # keyboard transpose (semitones)
    master_tune: float = 0.0      # Global > Main Tuning (semitones, all oscillators)
    tuning_a4: float = 440.0      # concert pitch
    tuning_table: Optional[list] = None   # 128 note frequencies (tuning file / MTS)
    scale_key: int = 0            # PitchQuantizer key (0 = C)
    scale_notes: Optional[list] = None    # 12 booleans of allowed pitch classes
    swing: float = 0.0            # 0..1 (0.5 = straight in Serum's display)
    bend_range: tuple = (12, -12)
    modwheel: float = 0.0         # 0..1 current mod wheel value
    voice_amp: float = 0.5        # Global Voice Amp knob: 0.5 = unity, 1 = +6 dB (VELO -> Amp target)
    voice: VoiceControl = field(default_factory=VoiceControl)
    arp: Arp = field(default_factory=Arp)
    clips: list = field(default_factory=list)
    preview_clip: int = 0
    # imported preset parts this model does not interpret (clip banks, arp
    # patterns, MIDI maps, unknown mod slots ...) are kept verbatim so a
    # preset survives import -> export unchanged
    serum_extra: dict = field(default_factory=dict)
    raw_mods: list = field(default_factory=list)
    # Custom wavetables referenced as custom:<n>; each is a list of frames
    # (stored separately as numpy arrays, not serialised here).

    def copy(self) -> "Patch":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        d = copy.deepcopy(d)
        p = cls()

        def build(klass, data):
            data = dict(data or {})
            names = {f.name for f in fields(klass)}
            sub = {"smp": SampleCfg, "gran": GranularCfg, "spec": SpectralCfg, "multi": MultiCfg}
            for k, kl in sub.items():
                if k in data and isinstance(data[k], dict):
                    data[k] = build(kl, data[k])
            return klass(**{k: v for k, v in data.items() if k in names})
        p.name = d.get("name", p.name)
        p.oscs = [build(Osc, o) for o in d.get("oscs", [])] or p.oscs
        while len(p.oscs) < MAX_OSCS:
            p.oscs.append(Osc())
        p.sub = build(SubOsc, d.get("sub"))
        p.noise = build(NoiseOsc, d.get("noise"))
        p.filter = build(Filter, d.get("filter"))
        p.filter2 = build(Filter, d.get("filter2"))
        p.routing = d.get("routing", {}) or {}
        p.envs = [build(Env, e) for e in d.get("envs", [])] or p.envs
        while len(p.envs) < N_ENVS:
            p.envs.append(Env(sustain=0.0))
        p.lfos = [build(LFO, l) for l in d.get("lfos", [])] or p.lfos
        while len(p.lfos) < N_LFOS:
            p.lfos.append(LFO())
        p.macros = (list(d.get("macros", [])) + [0.0] * N_MACROS)[:N_MACROS]
        p.mods = [build(Mod, m) for m in d.get("mods", [])]
        p.fx_chain = d.get("fx_chain", []) or []
        p.fx_buses = (d.get("fx_buses") or [[], []]) + [[]] * 0
        p.distortion = build(Distortion, d.get("distortion"))
        p.ott = build(OTT, d.get("ott"))
        p.eq = build(EQ, d.get("eq"))
        p.chorus = build(Chorus, d.get("chorus"))
        p.voice = build(VoiceControl, d.get("voice"))
        p.arp = build(Arp, d.get("arp"))
        p.clips = [build(Clip, c) for c in d.get("clips", [])]
        skip = {"name", "oscs", "sub", "noise", "filter", "filter2", "routing", "envs", "lfos", "macros", "mods",
                "fx_chain", "fx_buses", "distortion", "ott", "eq", "chorus", "voice", "arp", "clips"}
        for f in fields(cls):
            if f.name not in skip and f.name in d:
                setattr(p, f.name, d[f.name])
        return p

    def active_oscs(self):
        return [(i, o) for i, o in enumerate(self.oscs) if o.enabled]

    def mods_for(self, dest: str):
        return [m for m in self.mods if m.dest == dest and m.amount != 0.0]

    def route(self, source: str) -> dict:
        """Routing of one source; default: oscillators through the filters,
        filters to the main output."""
        r = {"dest": "main" if source.startswith("filter") else "filter", "balance": 0.0,
             "send1": 0.0, "send2": 0.0, "env1": True}
        r.update(self.routing.get(source, {}))
        return r

    def describe(self) -> list[str]:
        """Human-readable summary lines for the UI."""
        out = []
        for i, o in self.active_oscs():
            uni = f", unison {o.unison} x detune {o.detune:.2f}" if o.unison > 1 else ""
            warp = f", warp {o.warp} {o.warp_amount:.2f}" if o.warp != "none" and o.warp_amount else ""
            if o.warp2 != "none" and o.warp2_amount:
                warp += f" + {o.warp2} {o.warp2_amount:.2f}"
            if o.unison > 1 and o.unison_stack != "none":
                warp += f", stack {o.unison_stack}"
            out.append(f"Osc {'ABC'[i]}: {o.table} @ pos {o.position:.2f}, "
                       f"oct {o.octave:+d} semi {o.semi:+d} fine {o.fine:+.0f}c, "
                       f"level {o.level:.2f}{uni}{warp}")
        if self.sub.enabled:
            out.append(f"Sub: {self.sub.shape} oct {self.sub.octave:+d} level {self.sub.level:.2f}")
        if self.noise.enabled:
            out.append(f"Noise: color {self.noise.color:.2f} level {self.noise.level:.2f}")
        for label, f in (("Filter 1", self.filter), ("Filter 2", self.filter2)):
            if f.enabled:
                out.append(f"{label}: {f.type} cutoff {f.cutoff:.0f} Hz reso {f.reso:.2f} drive {f.drive:.2f}"
                           + (f" var {f.var:.2f}" if f.var != 0.5 else "") + (f" mix {f.mix:.2f}" if f.mix < 1 else ""))
        for src, r in sorted(self.routing.items()):
            out.append(f"Routing {src}: {r}")
        names = ["Env1 (amp)", "Env2", "Env3", "Env4"]
        for n, e in zip(names, self.envs):
            out.append(f"{n}: A {e.attack*1000:.1f}ms H {e.hold*1000:.0f}ms D {e.decay*1000:.0f}ms "
                       f"S {e.sustain:.2f} R {e.release*1000:.0f}ms")
        used = {m.source for m in self.mods}
        for i, l in enumerate(self.lfos):
            if f"lfo{i+1}" in used:
                rate = l.sync if l.sync else f"{l.rate_hz:.2f} Hz"
                kind = l.shape if l.type == "shape" else l.type
                extra = "".join(f", {k} {getattr(l, k):.2f}" for k in ("rise", "delay", "smooth") if getattr(l, k))
                out.append(f"LFO{i+1}: {kind} {rate} ({l.mode}){extra}")
        for m in self.mods:
            if m.amount:
                out.append(f"Mod: {m.source} -> {m.dest} amount {m.amount:+.3f}"
                           + (" (bipolar)" if m.bipolar else "") + (f" x {m.aux}" if m.aux else ""))
        for i, v in enumerate(self.macros):
            if v:
                out.append(f"Macro {i+1}: {v:.2f}")
        if self.distortion.enabled:
            d = self.distortion
            out.append(f"FX Distortion: {d.mode} drive {d.drive:.2f} mix {d.mix:.2f}")
        if self.ott.enabled:
            out.append(f"FX OTT/multiband: depth {self.ott.depth:.2f} amount {self.ott.amount:.2f}")
        if self.eq.enabled:
            out.append(f"FX EQ: low {self.eq.low_gain:+.1f} dB @ {self.eq.low_freq:.0f} Hz, "
                       f"high {self.eq.high_gain:+.1f} dB @ {self.eq.high_freq:.0f} Hz")
        if self.chorus.enabled:
            c = self.chorus
            out.append(f"FX Chorus: rate {c.rate_hz:.2f} Hz depth {c.depth:.2f} mix {c.mix:.2f}")
        def fx_lines(chain, prefix):
            for k, e in enumerate(chain):
                if not e.get("enabled", True):
                    continue
                params = ", ".join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}"
                                   for a, b in e.items() if a not in ("type", "enabled", "bands"))
                out.append(f"{prefix}{k + 1} {e['type']}: {params}")
                for bi, band in enumerate(e.get("bands", []) or []):
                    fx_lines(band, f"{prefix}{k + 1}.band{bi + 1}.")
        fx_lines(self.fx_chain, "FX ")
        for b, chain in enumerate(self.fx_buses):
            if chain:
                fx_lines(chain, f"Bus{b + 1} FX ")
        out.append(f"Voicing: {'mono' if self.mono else 'poly'}, glide {self.glide*1000:.0f} ms")
        return out


@dataclass
class Note:
    start: float                  # seconds
    duration: float               # seconds
    pitch: float                  # MIDI note number (float allows detuning)
    velocity: float = 1.0
    # optional pitch-bend curve relative to `pitch`: [[t_rel_seconds, semis], ...]
    bend: Optional[list] = None

    def to_dict(self):
        return asdict(self)


def sync_to_hz(sync: str, bpm: float) -> float:
    """Convert a tempo division such as '1/8', '1/4T', '1/16D', '2/1' to Hz."""
    s = sync.strip()
    mult = 1.0
    if s.endswith("T"):
        mult, s = 2.0 / 3.0, s[:-1]
    elif s.endswith("D"):
        mult, s = 1.5, s[:-1]
    num, den = s.split("/")
    beats = 4.0 * float(num) / float(den) * mult   # length in quarter notes
    return bpm / 60.0 / beats


SYNC_DIVISIONS = ["4/1", "2/1", "1/1", "1/2", "1/2D", "1/2T", "1/4", "1/4D", "1/4T",
                  "1/8", "1/8D", "1/8T", "1/16", "1/16D", "1/16T", "1/32", "1/32T", "1/64"]
