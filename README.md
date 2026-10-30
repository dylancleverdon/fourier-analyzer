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

## Install (Windows 10 / 11)

Download **[Install-Fourier-Analyzer-Windows.cmd](https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/Install-Fourier-Analyzer-Windows.cmd)** and double-click it. The app is unsigned, so Windows SmartScreen may say "Windows protected your PC": click **More info**, then **Run anyway**.

Or paste this into PowerShell:

```powershell
irm https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.ps1 | iex
```

The installer needs no admin rights. It:

1. installs a private Python 3.12 (via `uv`) plus all dependencies and `ffmpeg`
2. installs the app to `%LOCALAPPDATA%\FourierAnalyzer`
3. adds **Fourier Analyzer** to the Start menu and the desktop (plus **Uninstall Fourier Analyzer** in the Start menu)
4. downloads the separation models (about 1 GB; the first install downloads about 2 GB in total and takes 5–15 minutes)
5. finds Serum 2: the VST3 in `C:\Program Files\Common Files\VST3` and the presets under `Documents\Xfer\Serum 2 Presets`. If yours are elsewhere, set the paths in **Settings**
6. launches the app

The app opens in its own window (Microsoft Edge WebView2, built into Windows 11). Exports go to `Music\Fourier Analyzer`. Updates install automatically, as on the Mac.

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
      - phase and shape warps (Sync, Bend, PWM, Asym, Mirror, Quantize, Self-PD, Even/Odd, Flip, Dist), a second warp slot and unison stacks
      - LFO routings, filters, filter and pitch envelopes
      - Serum filter models (Ladder, Comb, Phase, Flange, Formant, Diffuser, Scream, morphing) and a parallel filter 2
      - FX-rack modules: distortion modes, chorus, flanger, phaser, hyperdimension, frequency shifter, FX filter, compressor, reverb, convolve and tempo-synced delay, plus a multiband split
      - OTT, EQ and glide
      - unison detune modes, warps on osc B and C, amp-envelope curve shapes and hold
      - **sample, granular and spectral oscillators** playing a note cut from the original (in place of osc A or as an extra layer); the audio is embedded in the preset. If the sound has an LFO wobble, the note's loudness and filter movement are flattened before it is embedded, the wobble goes into a tempo-synced LFO routing instead, the sampler loops less than half an LFO cycle, and granular / spectral scan with **tempo lock** (preferred over the plain sampler). Every such candidate is played at its root and a fifth either side; one whose wobble rate follows the pitch, or strays from the synced rate, is rejected
      With **confirm structure in real Serum 2** (on by default; used when Serum 2 is installed), each round's top candidates (3 by default, adjustable) are also rendered in the real plugin through one persistent Serum session, and a part is only kept if real Serum agrees it helps. Serum's renders are scored with the same measure as the search, so the two are compared like for like. If Serum isn't available, or its time runs out (its own budget in the reconstruct options), the app's engine decides alone.
   3. **Pruning** removes each component in turn and drops it if the fit holds (confirmed in real Serum 2 too, when available).
   4. **Fine-tuning** of every parameter with CMA-ES.
   5. **Exploration** (search times over 15 min): after the first full pass (about 15 min), the search keeps experimenting until the time runs out: fresh rounds of structure moves, pruning and CMA-ES restarts, keeping only what improves the fit.
   6. **Waveform alignment**: start phase, fine tune, the amp envelope and each note's start time (±6 ms) are tuned to maximise literal waveform correlation. The refined note timing also goes into the exported MIDI.

   7. **Fine-tune inside real Serum 2** (on by default, 2 min to 1 hour, skipped when Serum 2 isn't installed): the main search runs on the app's own copy of Serum's engine, which is fast but approximate. This stage keeps one copy of the real plugin open, renders candidate presets in it over and over, scores each against the original and nudges every continuous parameter with CMA-ES. It restarts whenever it converges, so it keeps working for the time you give it, and it never makes the result worse. The first **Stop** ends the main search and moves on to this stage; a second Stop ends it.

   **Synthesis first.** Warps, FM, filters, FX modules, FX buses and LFO / envelope / matrix routings cost nothing, however many a patch uses. Only sample, granular and spectral oscillators and multi-frame wavetables carry a cost, and candidates are ranked by similarity minus that cost (the **Prefer synthesis ↔ Prefer accuracy** slider sets how much). Sample-based oscillators are tried only once synthesis stops improving, or when it scores under 70, and must beat it by a margin; a kept sample is shortened to the shortest loop that keeps the score, and then one cycle of it is tried as a custom wavetable. When the best patch uses a sample, you get **two presets**: the best overall and the best synthesis-only one, with both scores.

   **Searching like a sound designer.** Besides single parts, the search tries 2–3 parts together (FM + distortion + OTT, distortion + EQ, filter + LFO on cutoff, warp + LFO on warp amount, unison + chorus …), re-orders the FX rack and moves time effects onto a parallel bus. Pruning removes a part, or a pair of parts, only when the score holds without it, so parts that work together stay together. The key knobs are mapped to named macros (CUTOFF, WARP or FM DEPTH, LFO RATE, LFO DEPTH, DRIVE) without changing the sound.

   **Checked against real Serum 2 all the way** (on by default when Serum 2 is installed): besides the structure finalists and pruning, the best patch is rendered in Serum after every stage (wavetable choice, each structure round, pruning, each part of the fine-tune) and every few minutes (setting); if Serum's score lags our engine's, it is tuned up inside Serum before the search continues. The best few fine-tune candidates are compared in Serum, not just the winner. Large patches are fine-tuned in groups (oscillators and warps, then filters and modulation, then FX) before all together, and long runs get more Serum time.

   **Our engine corrects itself during the run.** At every checkpoint the same patch exists twice (our engine's render and real Serum's), so the app fits quick corrections that make our engine sound like Serum for the rest of the run: overall level, brightness tilt, filter cutoff scale and amp-envelope timing. Each correction is kept only while it makes our engine match Serum better (it is dropped as soon as it doesn't), and corrections that held up are saved and start the next run (Engine accuracy shows them; Reset corrections clears them).

   **Real Serum decides.** When Serum 2 is installed, its score is the one that counts: the headline result is real Serum's waveform similarity (our engine's prediction is shown under it), the target is judged in Serum, structure and fine-tune choices are made by Serum, and at the end the best patch Serum heard during the run wins, even if our engine preferred a later one.

   **Works on every note.** The last note (or whatever lies past the first 6 s) is held out of the fit and the final patch is scored on it too, and the whole patch is played a fifth below and above its notes to make sure no wobble or LFO rate follows the pitch.

   **Tune this patch in Serum 2** (under every result) loads that result's preset back into the real plugin and keeps tuning it for the time you pick (2 min to 2 hours). The outcome is saved as a new result; the original stays as it is.

   **Serum only** (reconstruct option) skips tuning in the app's engine: starting from the analysis guess with a custom wavetable, parts are screened in the app's engine but only added or removed when real Serum 2 agrees (for the Serum time set in the options), then the parameters are tuned in Serum for the rest of the search time. It shows how fast real Serum is on your machine; every result shows the renders per second.

   **Search time** goes from 1.5 min to 8 hours, a custom number of minutes, or **Until I stop it**. Tick **keep improving after reaching the target** so a long run doesn't stop at the similarity target.

   **Pitch and velocity go into the patch** where Serum would put them:
   - vibrato or wobble becomes an LFO → Main Tuning (tempo synced);
   - a pitch drop or rise at each note start becomes an envelope → Main Tuning;
   - glide becomes legato portamento;
   - velocity drives Voice Amp through a velocity curve.

   Pitch movement explained by the patch is removed from the MIDI bends. One-off bends stay in the MIDI.
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
- the result shows matching **Logic Compressor sidechain settings**, because the ducking usually lives in the DAW;
- the preset gets a tempo-synced LFO → Voice Amp shaped like the ducking. Its matrix slot is **bypassed** unless you tick **put detected sidechain ducking into the preset**, in which case it ducks inside Serum.

### Learning from Serum (engine accuracy)

On by default (**learn from Serum** in the reconstruct options). After the preset is verified in real Serum 2, the app compares **its own render with Serum's render of the same preset**. That difference is the *engine gap*: what the app's copy of Serum gets wrong, kept separate from how well the patch fits the original.

For each run it:
- **documents exactly what differs** between the two renders: overall level, timing offset, every 1/3-octave band, spectral tilt and brightness, noisiness, and per note the attack, decay, sustain, release, onset, pitch (cents), harmonic balance and amplitude modulation. These become plain-language findings ("Serum is darker, −2.3 dB per octave", "Serum's release tail is 120 ms longer");
- **finds the cause**: every part of the patch (each FX module, filter, warp, unison, oscillator, noise, sub, matrix route, glide) is switched off in both engines in turn; the part whose removal closes the gap is the one emulated wrongly;
- **learns corrections**: the Serum tuning stage shows how far each parameter had to move to match the original in real Serum. Once at least 5 runs agree on the same offset for a parameter (for example "lp24 cutoff needs ×1.18"), that correction is applied to every preset the app writes;
- **logs everything** as numbers only (no audio) in `~/Library/Application Support/FourierAnalyzer/data/calibration/`, and writes `engine_gap.txt` next to each result.

Every structure decision checked in real Serum (our engine's score change vs Serum's, and whether they agreed) is logged too; the parts where Serum disagrees most are listed first on the Engine accuracy page.

The page also ranks the **FX modules and warps that differ most from real Serum** (gap explained in the culprit search plus how often and how far Serum disagreed), so those emulations get fixed first.

**Engine accuracy** (top bar) shows the totals: the score drop between our prediction and real Serum, the biggest causes across all runs, the features with the largest drop, the average band differences and the corrections in use. **Export report** saves it all as one JSON file to send in for engine fixes; **Reset corrections** forgets the learned corrections.

### Running jobs

- **Multiple windows.** **New window** (or ⌘N) opens another analyzer window. Each window can work on its own track, and up to 4 jobs run at once (setting `parallel_jobs`).
- **Cancel.** A running separation or layer split can be cancelled. **Stop (keep best)** ends a reconstruction early: it finishes a short alignment and exports the best patch found so far.
- **Stems as .wav.** **Save .wav** on a stem, or **Save all stems (.wav)**, copies the split instruments into your export folder (`<project>/stems/`) and shows them in Finder / File Explorer; **Download .zip** gets them all at once.
- **Clip first.** **Separate selection** separates only the part you selected, so long songs don't have to be separated in full.
- **Player.** Playback uses Web Audio, so it starts on the first sample and the playhead follows the audio clock.

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

### Serum 2 feature coverage

The patch model, renderer and preset mapping follow the Serum 2 user guide.

**Oscillators**
- 3 oscillators in every mode: wavetable, sample (loop modes, crossfade, scan), granular (window shapes, density, spawn patterns, randomisation), spectral (resynthesis, band limits, spectral filter and warps) and multisample (SFZ)
- pitch modes, ratio, Hz offset, pitch and bend tracking, key and velocity ranges
- both warp slots with every warp mode, plus remap curves
- unison: every mode, stacks, width, wavetable position and warp spread
- start phase, phase memory and random phase
- noise oscillator (colours, samples, one-shot, keytrack) and sub

**Filters and routing**
- both filters with the full Serum filter list (97 types in about 20 rendered families), with level, pan and the XY morph
- per-source routing: filter balance, filter → filter, direct out and FX bus sends

**Modulation**
- 4 envelopes with curves, start/end levels and legato
- 10 LFOs: shapes, custom paths, loopback, direction, swing, point modulation, S&H random, chaos, rise, delay and smooth
- 8 macros
- a 64-slot mod matrix with aux sources, curves, slew and bypass; any parameter is a destination
- velocity and note curves, Voice Amp

**Voicing and playback**
- mono, legato, poly limits, glide curve, always and scaled
- pitch quantiser and scales, tuning tables, Main Tuning
- the arpeggiator and clip player

**FX**
- the main FX rack and both FX buses with all module types and their full parameter sets, including nested 2-band, 3-band and mid/side splitters

Any Serum 2 preset can be imported and re-exported unchanged; settings that aren't interpreted are passed through. All 701 test presets load, render and round-trip.

**Limits**
- Factory samples, multisamples, noises and IRs are rendered only when Serum's content folder is installed (`/Library/Audio/Presets/Xfer Records/Serum 2 Presets`). User samples are embedded in the preset.
- The reconstruction starts from **custom wavetables** and also tries sample, granular and spectral oscillators built from a note of the original; multisample oscillators are used only when importing presets.
- The arpeggiator's Pattern mode is approximated.
- Some modulation source IDs (aftertouch, poly aftertouch, random, alternate, pitch bend) are inferred, not confirmed against Serum.

The FX and filter DSP reproduces how each module behaves, not Serum's exact algorithms. Calibrate against the real Serum check.

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
- coarse-pitch modulation depth (±48 st) and the Main Tuning range (±24 st)
- the units of synced LFO rates, module output levels and the compressor threshold
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
