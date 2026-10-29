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
5. checks that the separation stack (`onnxruntime`, `audioread`, `audio-separator`) imports, and reinstalls if not, then finds Serum 2 (plugin and presets folder)
6. launches the app

You can also download `Install-Fourier-Analyzer.command` from the latest release and double-click it. Because the app is unsigned, macOS will block it the first time. Right-click it and choose **Open**, or go to **System Settings → Privacy & Security → Open Anyway**.

To remove the app, run `bash ~/Library/Application\ Support/FourierAnalyzer/app/installer/uninstall.sh`.

## Updates

Every push to the default branch runs the tests and publishes a new GitHub Release (`.github/workflows/release.yml`).

The app checks for a new release each time it launches. If there is one, it downloads it, reinstalls dependencies only when `requirements.txt` changed, swaps the code folder and restarts. The **Update** pill in the top bar does the same on demand.

On every launch the app also checks that its key modules import (`onnxruntime`, `audioread`, `audio-separator`, `pedalboard` and so on). If any is missing it installs `requirements.txt` automatically, so older installs heal themselves when a dependency is added.

**Settings → Update channel** can follow a branch (`branch:<name>`) instead of releases, for testing unreleased work.

There is deliberately no signing or checksum verification; update speed was prioritised over security.

## Workflow

1. **Drop a song.** It opens on its waveform. Drag over the part you want (e.g. the drop) and click **Separate selection**, or **Separate whole file**. **Change clip…** picks a new region of the original later. Separation runs in two stages:
   - Demucs v4 (`htdemucs_ft`) separates drums, bass, other and vocals.
   - MDX23C DrumSep separates the drums into kick, snare, toms, hi-hat, ride and crash.
   - Each stem is then labelled (sub bass vs growl bass, lead vs pad vs pluck, drum hit counts, and so on).
2. **Split and recombine layers.**
   - **Split layers…** separates stacked parts (e.g. two basses) with NMF factorisation plus clustering. It works when layers differ in register, timing or timbre; layers that fully overlap in the same register can't be cleanly separated by any current method.
   - Tick any stems or layers and click **Combine selected** to sum them into a new stem before analysing or rebuilding it.
3. **Select a region and click Analyze.** You get:
   - the average spectrum and the harmonic profile
   - a MIDI piano roll and table: pitch to the cent, start times with grid offset in ms, velocity and pitch-bend curves, with kicks marked
   - the amplitude envelope, LFO detection (rate snapped to a tempo division such as 1/8, shape, what it modulates), noise content and a unison/detune estimate
   - **sidechain detection** (below)
4. **Reconstruct.** The patch always uses **custom wavetables** built from the measured harmonic magnitudes and phases of your sound. The search runs in this order:
   1. **Three wavetable designs** are built and briefly tuned: a single averaged cycle, frames evolving through the longest note (swept by Env2), and one LFO cycle folded into frames (swept by the LFO, which is how wub basses are built). The best is kept.
   2. **Structure search** tries adding one component at a time and keeps it only if the fit improves. Candidates include:
      - noise, unison, sub, and a second custom oscillator
      - **Serum warp modes on osc A: FM, AM and RM from osc B, and FM and RM from the sub**
      - LFO routings, filters, filter and pitch envelopes
      - distortion modes, OTT, EQ and glide
   3. **Pruning** removes each component in turn and drops it if the fit holds.
   4. **Fine-tuning** of every parameter with CMA-ES.
   5. **Waveform alignment**: start phase, fine tune, the amp envelope and each note's start time (±6 ms) are tuned to maximise literal waveform correlation. The refined note timing also goes into the exported MIDI.
5. **Verification checks** ("check your work"):
   - The original is split into its tonal part and noise residual, and those are compared with the patch's oscillators alone and noise alone.
   - Each patch component is removed in turn to measure what it contributes.
   - It reports energy above the filter cutoff, note-timing error in ms and pitch error in cents.
   - If Serum 2 is installed, the preset is rendered in the **real Serum 2** plugin and scored with the same metrics.
6. **Export.** You get:
   - the `.SerumPreset` with its wavetables embedded, which **Install into Serum 2** copies to *User → Fourier Analyzer*
   - the wavetables as `.wav` files too
   - a MIDI file with exact timing and ±12 st pitch bends; the preset sets the same bend range
   - `render.wav` (the patch alone) and `render_with_sidechain.wav` (as it sits in the track)
   - the report

   Everything is also copied to `~/Music/Fourier Analyzer/`. Old projects can be deleted from the sidebar (×).

### Sidechain compression

Kick times come from the separated kick stem, or the beat grid if there is no kick stem. The synth's loudness is averaged around each kick, using only kicks that land inside a sustained note, so note attacks don't count as ducking. The result is fitted with a compressor gain-reduction model (depth, attack, hold, release).

When ducking is found:
- it is **divided out before the analysis**, so it isn't mistaken for an amp envelope or LFO;
- it is **re-applied to every render before comparing**, so the patch is judged as it sounds in the track;
- the result shows matching **Logic Compressor sidechain settings**, because the ducking lives in the DAW, not in Serum.

### Similarity metrics (objective)

The headline number is **waveform similarity**: the Pearson correlation of the original and reconstructed waveforms at the best time alignment (±20 ms), × 100. It is the literal similarity of the two waveforms.

Reported alongside it, all standard definitions:

| Metric | Definition |
|---|---|
| Spectral similarity | Cosine similarity of the magnitude spectrograms (2048-point STFT); phase independent |
| SI-SDR | Scale-invariant signal-to-distortion ratio in dB (Le Roux et al. 2019) |
| Log-spectral distance | RMS dB difference of the power spectrograms |
| Envelope correlation | Pearson correlation of the RMS loudness envelopes |

Waveform similarity is phase sensitive. Sounds with random-phase unison, noise or reverb in the original can sound identical while correlating poorly sample by sample; spectral similarity shows how close the timbre is regardless. The **stop-early slider** applies to waveform similarity. The search itself optimises a blend of a perceptual spectro-temporal measure and spectral similarity, then waveform correlation in the final stage.

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
- FM depth at 100 % warp (`FM_INDEX` in the renderer)

**Settings → Calibrate with Serum 2** measures the cutoff curve in the real plugin and stores it. The other two can be overridden under `serum_scales` in `~/Library/Application Support/FourierAnalyzer/settings.json`. The in-app renderer approximates Serum; the real-Serum check shows how far apart they are.

## Command line

```bash
FA=~/Library/Application\ Support/FourierAnalyzer
"$FA/venv/bin/python" -m fourier_analyzer separate song.wav
"$FA/venv/bin/python" -m fourier_analyzer reconstruct bass.wav --start 12.5 --end 16.5 --target 90   # target = waveform similarity %
"$FA/venv/bin/python" -m fourier_analyzer update --check
"$FA/venv/bin/python" -m fourier_analyzer calibrate
"$FA/venv/bin/python" -m fourier_analyzer selftest
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
FA_NO_DEP_CHECK=1 FA_FORCE_DSP=1 python -m fourier_analyzer serve   # http://127.0.0.1:47800
```

### Layout

| Path | Contents |
|---|---|
| `fourier_analyzer/serum/` | preset codec, patch↔Serum mapping, real-Serum host |
| `fourier_analyzer/synth/` | patch model, wavetables, offline renderer |
| `fourier_analyzer/analysis/` | Fourier analysis (f0, harmonics, envelope, LFO, timing), sidechain detection, instrument labelling |
| `fourier_analyzer/separation/` | stems, drum parts, NMF layer split |
| `fourier_analyzer/match/` | objective metrics, search objective, optimizer, verification checks |
| `fourier_analyzer/app/` | local web UI + native window |
| `installer/` | installer, launcher, uninstaller |
