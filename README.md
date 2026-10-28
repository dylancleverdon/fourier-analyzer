# Fourier Analyzer

Drop in a section of a song and the app splits it into instruments. Pick one sound, for example the bass, and the app analyses it with Fourier analysis, then rebuilds it as a **Serum 2 preset**. It checks the rebuilt patch against the original and tells you how close it got.

## Install (macOS)

Open **Terminal** and paste:

```bash
curl -fsSL https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.sh | bash
```

The installer runs in your user account and needs no admin password. It:

1. installs a private Python 3.12 (via `uv`) plus all dependencies, including a bundled `ffmpeg`
2. installs the app to `~/Library/Application Support/FourierAnalyzer`
3. creates **/Applications/Fourier Analyzer.app**, clears the quarantine flag and ad-hoc signs it, so it opens like a normal app
4. downloads the separation models (about 1 GB)
5. finds Serum 2 (plugin, tables and presets folder) and indexes the stock wavetables
6. launches the app

You can also download `Install-Fourier-Analyzer.command` from the latest release and double-click it. Because the app is unsigned, macOS will block it the first time. Right-click it and choose **Open**, or go to **System Settings → Privacy & Security → Open Anyway**.

To remove the app, run `bash ~/Library/Application\ Support/FourierAnalyzer/app/installer/uninstall.sh`.

## Updates

Every push to the default branch runs the tests and publishes a new GitHub Release (`.github/workflows/release.yml`).

The app checks for a new release each time it launches. If there is one, it downloads it, reinstalls dependencies only when `requirements.txt` changed, swaps the code folder and restarts. The **Update** pill in the top bar does the same on demand.

**Settings → Update channel** can follow a branch (`branch:<name>`) instead of releases, for testing unreleased work.

There is deliberately no signing or checksum verification; update speed was prioritised over security.

## Workflow

1. **Drop a song section.** The app splits it in two stages:
   - Demucs v4 (`htdemucs_ft`) separates drums, bass, other and vocals.
   - MDX23C DrumSep separates the drums into kick, snare, toms, hi-hat, ride and crash.
   - Each stem is then labelled (sub bass vs growl bass, lead vs pad vs pluck, drum hit counts, and so on).
2. **Split layers.** For two stacked basses, select the bass stem and click **Split layers…**. NMF factorisation plus clustering separates layers that differ in register, timing or timbre. Layers that fully overlap in the same register can't be cleanly separated by any current method.
3. **Select a region and click Analyze.** You get:
   - the average spectrum and the harmonic profile
   - note pitches with exact start times, grid offset in ms and pitch-bend curves
   - the amplitude envelope
   - LFO detection: rate snapped to a tempo division such as 1/8, plus the shape and what it modulates
   - noise content and a unison/detune estimate
4. **Reconstruct.** The search runs in this order:
   1. **Initial guess** from the analysis.
   2. **Wavetable search**: basic shapes first, then your stock Serum tables, each quickly tuned.
   3. **Structure search**: it tries adding one component at a time and keeps it only if the score improves. Candidates are noise, unison, sub, a second or third oscillator, LFO routings, a filter envelope, a pitch envelope, distortion modes, OTT, EQ and glide.
   4. **Pruning**: it removes each component in turn. If the score doesn't drop, the component goes.
   5. **Fine-tuning** of every parameter with CMA-ES.
   6. **Custom wavetable**, only if still below target: built from the measured harmonic spectra. If an LFO was detected, the table is one LFO cycle swept by that LFO; otherwise it follows the note over time.
5. **Verification checks** ("check your work"):
   - The original is split into its tonal part and noise residual, and those are compared with the patch's oscillators alone and noise alone.
   - Each patch component is removed in turn to measure what it contributes.
   - It reports how much energy sits above the filter cutoff (evidence of distortion after the filter), timing error in ms and pitch error in cents.
   - If Serum 2 is installed, the preset is rendered in the **real Serum 2** plugin and scored too.
6. **Export.** You get:
   - the `.SerumPreset`, which **Install into Serum 2** copies to *User → Fourier Analyzer*
   - a MIDI file of the notes with exact timing and ±12 st pitch bends; the preset sets the same bend range
   - the render, the original clip and the report

   Everything is also copied to `~/Music/Fourier Analyzer/` so you can drag it into Logic.

### Accuracy score

The score runs from 0 to 100 and blends five comparisons against the original:

| Component | Weight | What it measures |
|---|---|---|
| Multi-resolution log-mel spectrogram | 40% | timbre over time |
| High-resolution averaged spectrum | 25% | waveform / harmonic structure |
| Loudness envelope | 15% | articulation and timing |
| Long-term spectrum | 10% | overall tone balance |
| Spectral-centroid movement | 10% | filter and LFO motion |

**Phase adjustment.** Unison, noise and random phase make every render of the *same* patch slightly different, and Serum behaves the same way. The app measures that ceiling by comparing the patch with itself across random phases, and reports `raw ÷ ceiling` as the accuracy.

**Target slider.** Lower targets stop earlier, with simpler patches built from basic and stock waves. Higher targets may need a custom wavetable. *Only Serum stock tables* forbids the generated basic shapes.

### What maps exactly and what is calibrated

Preset files use Serum 2's real format: a JSON header, then zstd-compressed CBOR. The encoder reproduces 626 factory presets byte-for-byte.

These come straight from Serum's own presets:
- parameter names
- the module layout
- modulation slot IDs
- LFO curve points
- FX entries
- embedded wavetables

These are **assumptions**:
- the filter cutoff curve
- coarse-pitch modulation depth
- the unison detune curve

**Settings → Calibrate with Serum 2** measures the cutoff curve in the real plugin and stores it. The other two can be overridden under `serum_scales` in `~/Library/Application Support/FourierAnalyzer/settings.json`. The in-app renderer approximates Serum; the real-Serum check shows how far apart they are.

## Command line

```bash
FA=~/Library/Application\ Support/FourierAnalyzer
"$FA/venv/bin/python" -m fourier_analyzer separate song.wav
"$FA/venv/bin/python" -m fourier_analyzer reconstruct bass.wav --start 12.5 --end 16.5 --target 90
"$FA/venv/bin/python" -m fourier_analyzer update --check
"$FA/venv/bin/python" -m fourier_analyzer calibrate
"$FA/venv/bin/python" -m fourier_analyzer selftest
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
FA_FORCE_DSP=1 python -m fourier_analyzer serve   # http://127.0.0.1:47800
```

### Layout

| Path | Contents |
|---|---|
| `fourier_analyzer/serum/` | preset codec, patch↔Serum mapping, real-Serum host |
| `fourier_analyzer/synth/` | patch model, wavetables, offline renderer |
| `fourier_analyzer/analysis/` | Fourier analysis (f0, harmonics, envelope, LFO, timing), instrument labelling |
| `fourier_analyzer/separation/` | stems, drum parts, NMF layer split |
| `fourier_analyzer/match/` | score, optimizer, verification checks |
| `fourier_analyzer/app/` | local web UI + native window |
| `installer/` | installer, launcher, uninstaller |
