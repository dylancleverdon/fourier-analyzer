// Fourier Analyzer UI (vanilla JS, talks to the local FastAPI server)
const $ = (s) => document.querySelector(s);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

const state = { pid: null, meta: null, stem: null, peaks: null, duration: 0, sel: null, result: null, polling: {} };
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (m) => { const r = Math.round(m); const c = Math.round((m - r) * 100); return `${NOTE_NAMES[((r % 12) + 12) % 12]}${Math.floor(r / 12) - 1}${c ? (c > 0 ? " +" : " ") + c + "c" : ""}`; };
const fmt = (x, d = 2) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(d));

// ------------------------------------------------------------------ updates
async function checkUpdate() {
  const b = $("#updateBtn");
  try {
    const u = await api("/api/update/check");
    if (u.available) { b.textContent = `Update to v${u.latest}`; b.classList.add("available"); b.onclick = applyUpdate; }
    else if (u.error) {
      b.textContent = /404/.test(u.error) ? "No releases published yet" : "Offline: can't check updates";
      b.title = u.error + " (click to retry)"; b.onclick = checkUpdate;
    }
    else { b.textContent = `Up to date (v${u.current})`; b.classList.remove("available"); b.onclick = checkUpdate; }
  } catch (e) { b.textContent = "Update check failed"; b.title = e.message; b.onclick = checkUpdate; }
}
async function applyUpdate() {
  const b = $("#updateBtn");
  b.textContent = "Updating…"; b.disabled = true;
  await post("/api/update/apply");
  const t = setInterval(async () => {
    try {
      const s = await api("/api/update/status");
      if (s.log.length) b.title = s.log.join("\n");
      if (!s.applying) {
        clearInterval(t);
        if (s.result && s.result.updated) { b.textContent = "Restarting…"; setTimeout(() => location.reload(), 6000); }
        else { b.disabled = false; b.textContent = "Update failed (hover)"; }
      }
    } catch (e) { /* server restarting */ clearInterval(t); setTimeout(() => location.reload(), 5000); }
  }, 1000);
}

// ------------------------------------------------------------------ projects
async function loadProjects() {
  const list = await api("/api/projects");
  const ul = $("#projects"); ul.innerHTML = "";
  list.forEach((p) => {
    const li = document.createElement("li");
    li.textContent = p.name; li.title = p.id;
    if (p.id === state.pid) li.classList.add("active");
    li.onclick = () => openProject(p.id);
    ul.appendChild(li);
  });
}

async function upload(file) {
  const fd = new FormData(); fd.append("file", file);
  showJob({ stage: "uploading", progress: 0.02, log: [] });
  const r = await fetch(`/api/projects?drums=${$("#optDrums").checked}`, { method: "POST", body: fd });
  if (!r.ok) { alert("Upload failed: " + (await r.text())); return; }
  const j = await r.json();
  await openProject(j.project.id);
  pollJob(j.job, () => openProject(j.project.id));
  loadProjects();
}

async function openProject(pid) {
  state.pid = pid;
  state.meta = await api(`/api/projects/${pid}`);
  $("#empty").hidden = true; $("#proj").hidden = false;
  $("#projName").textContent = state.meta.name;
  $("#projInfo").textContent = `${fmt(state.meta.duration, 1)} s · ${state.meta.status}`;
  $("#bpm").value = state.meta.bpm || "";
  renderInstruments(); renderStems(); renderPastResults();
  document.querySelectorAll("#projects li").forEach((li) => li.classList.toggle("active", li.title === pid));
}

function renderInstruments() {
  const box = $("#instruments"); box.innerHTML = "";
  const inst = state.meta.instruments || [];
  if (!inst.length) { box.innerHTML = `<span class="muted">Separating… instruments will appear here.</span>`; return; }
  inst.forEach((i) => {
    const c = document.createElement("span");
    c.className = "chip" + (i.present ? "" : " off");
    c.innerHTML = `${i.label}<span class="lvl">${i.hits ? i.hits + " hits · " : ""}${i.present ? fmt(i.level_db, 0) + " dB" : "not present"}</span>`;
    c.title = i.stem; c.onclick = () => selectStem(i.stem);
    box.appendChild(c);
  });
}

function stemLabel(name) {
  const i = (state.meta.instruments || []).find((x) => x.stem === name);
  return i ? i.label : name;
}

function renderStems() {
  const box = $("#stems"); box.innerHTML = "";
  Object.keys(state.meta.stems).forEach((name) => {
    const row = document.createElement("div");
    row.className = "stem" + (name === state.stem ? " active" : "");
    row.innerHTML = `<button class="iconbtn" title="Play">▶</button>
      <div><div class="name">${name}<small>${name === "mix" ? "full mix" : stemLabel(name)}</small></div><canvas></canvas></div>
      <div class="acts"><button class="ghost sel">Select</button><a class="btn ghost" href="/api/projects/${state.pid}/audio/${name}" download="${name}.wav">.wav</a></div>`;
    row.querySelector(".iconbtn").onclick = (e) => { e.stopPropagation(); playStem(name, row.querySelector(".iconbtn")); };
    row.querySelector(".sel").onclick = () => selectStem(name);
    box.appendChild(row);
    api(`/api/projects/${state.pid}/peaks/${name}?n=600`).then((p) => drawPeaks(row.querySelector("canvas"), p.peaks, null));
  });
}

function renderPastResults() {
  const r = state.meta.results || [];
  $("#resultsCard").hidden = !r.length;
  const ul = $("#pastResults"); ul.innerHTML = "";
  r.slice().reverse().forEach((x) => {
    const li = document.createElement("li");
    li.innerHTML = `<a href="#">${x.name}</a> · ${x.stem} · accuracy <b>${fmt(x.accuracy, 1)}</b>`;
    li.querySelector("a").onclick = async (e) => {
      e.preventDefault();
      const res = await api(`/api/projects/${state.pid}/results/${x.id}`);
      if (state.stem !== res.stem) await selectStem(res.stem);
      if (res.start != null) { state.sel = [res.start, res.end]; drawWave(); updateSelInfo(); }
      showResult(res);
      $("#result").scrollIntoView({ behavior: "smooth" });
    };
    ul.appendChild(li);
  });
}

// ------------------------------------------------------------------ audio
const player = $("#player");
let playingBtn = null;
function playStem(name, btn) {
  const url = `/api/projects/${state.pid}/audio/${name}`;
  if (playingBtn === btn && !player.paused) { player.pause(); btn.textContent = "▶"; return; }
  if (playingBtn) playingBtn.textContent = "▶";
  player.src = url; player.currentTime = 0; player.play(); btn.textContent = "❚❚"; playingBtn = btn;
  player.onended = () => (btn.textContent = "▶");
}

// ------------------------------------------------------------------ waveform editor
async function selectStem(name) {
  state.stem = name; state.sel = null;
  renderStems();
  $("#editor").hidden = false;
  $("#edTitle").textContent = `${name} — ${name === "mix" ? "full mix" : stemLabel(name)}`;
  const p = await api(`/api/projects/${state.pid}/peaks/${name}?n=2400`);
  state.peaks = p.peaks; state.duration = p.duration;
  drawWave(); updateSelInfo(); $("#spec").hidden = true;
  $("#editor").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawPeaks(cv, peaks, sel, dur) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight || cv.height;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  if (sel && dur) {
    g.fillStyle = "rgba(255,138,61,.22)";
    g.fillRect((sel[0] / dur) * w, 0, ((sel[1] - sel[0]) / dur) * w, h);
  }
  const max = Math.max(1e-6, ...peaks);
  g.fillStyle = "#b06cff";
  const step = peaks.length / w;
  for (let x = 0; x < w; x++) {
    let m = 0; for (let k = Math.floor(x * step); k < Math.floor((x + 1) * step); k++) m = Math.max(m, peaks[k] || 0);
    const v = (m / max) * (h / 2 - 1);
    g.fillRect(x, h / 2 - v, 1, Math.max(1, v * 2));
  }
}
function drawWave() { drawPeaks($("#wave"), state.peaks, state.sel, state.duration); }

(function setupSelection() {
  const cv = $("#wave"); let dragging = false, x0 = 0;
  const t = (e) => Math.max(0, Math.min(state.duration, ((e.clientX - cv.getBoundingClientRect().left) / cv.clientWidth) * state.duration));
  cv.addEventListener("mousedown", (e) => { dragging = true; x0 = t(e); state.sel = [x0, x0]; drawWave(); });
  window.addEventListener("mousemove", (e) => { if (!dragging) return; const x = t(e); state.sel = [Math.min(x0, x), Math.max(x0, x)]; drawWave(); updateSelInfo(); });
  window.addEventListener("mouseup", () => {
    if (!dragging) return; dragging = false;
    if (state.sel && state.sel[1] - state.sel[0] < 0.05) state.sel = null;
    drawWave(); updateSelInfo(); loadSpectrogram();
  });
})();

function selRange() { return state.sel ? { start: +state.sel[0].toFixed(4), end: +state.sel[1].toFixed(4) } : { start: null, end: null }; }
function updateSelInfo() {
  const s = state.sel;
  const bpm = parseFloat($("#bpm").value) || state.meta.bpm || 140;
  if (!s) { $("#selInfo").textContent = `whole stem · ${fmt(state.duration, 2)} s`; return; }
  const beats = ((s[1] - s[0]) * bpm) / 60;
  $("#selInfo").textContent = `${fmt(s[0], 3)}–${fmt(s[1], 3)} s · ${fmt(s[1] - s[0], 3)} s · ${fmt(beats, 2)} beats`;
}
function loadSpectrogram() {
  const r = selRange(); const img = $("#spec");
  const q = r.start != null ? `start=${r.start}&end=${r.end}` : "";
  img.src = `/api/projects/${state.pid}/spectrogram/${state.stem}?${q}&w=1400&h=260&_=${Date.now()}`;
  img.hidden = false;
}
$("#clearSel").onclick = () => { state.sel = null; drawWave(); updateSelInfo(); $("#spec").hidden = true; };
$("#playSel").onclick = () => {
  const r = selRange();
  player.src = `/api/projects/${state.pid}/audio/${state.stem}`;
  player.onloadedmetadata = () => {
    player.currentTime = r.start || 0; player.play();
    if (r.end) { const stop = () => { if (player.currentTime >= r.end) { player.pause(); player.removeEventListener("timeupdate", stop); } }; player.addEventListener("timeupdate", stop); }
  };
};
$("#splitBtn").onclick = async () => {
  const n = parseInt(prompt("Split this stem into how many layers? (2-4)\nWorks best on a selection where the layers differ in register or timing.", "2"), 10);
  if (!n) return;
  const r = await post(`/api/projects/${state.pid}/split`, { stem: state.stem, layers: n, ...selRange() });
  pollJob(r.job, () => openProject(state.pid));
};

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("#tab-fourier").hidden = b.dataset.tab !== "fourier";
  $("#tab-recon").hidden = b.dataset.tab !== "recon";
});

// ------------------------------------------------------------------ Fourier analysis
$("#runAnalysis").onclick = async () => {
  const b = $("#runAnalysis"); b.disabled = true; b.textContent = "Analyzing…";
  try {
    const a = await post(`/api/projects/${state.pid}/analyze`, { stem: state.stem, bpm: parseFloat($("#bpm").value) || null, ...selRange() });
    showAnalysis(a);
  } catch (e) { alert(e.message); }
  b.disabled = false; b.textContent = "Analyze selection";
};

function lineChart(cv, xs, ys, { logx = false, ymin = -80, ymax = 0, color = "#ff8a3d", bars = false, xlabel = "" } = {}) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const pad = 28;
  const xmin = logx ? Math.log10(xs[0]) : xs[0], xmax = logx ? Math.log10(xs[xs.length - 1]) : xs[xs.length - 1];
  const X = (x) => pad + ((logx ? Math.log10(x) : x) - xmin) / (xmax - xmin) * (w - pad - 6);
  const Y = (y) => 6 + (1 - (Math.max(ymin, Math.min(ymax, y)) - ymin) / (ymax - ymin)) * (h - pad - 6);
  g.strokeStyle = "#2a2f3a"; g.fillStyle = "#8a92a6"; g.font = "10px -apple-system";
  for (let d = ymin; d <= ymax; d += 20) { g.beginPath(); g.moveTo(pad, Y(d)); g.lineTo(w, Y(d)); g.stroke(); g.fillText(d, 2, Y(d) + 3); }
  if (logx) [50, 100, 200, 500, 1000, 2000, 5000, 10000].forEach((f) => { if (f >= xs[0] && f <= xs[xs.length - 1]) { g.fillText(f >= 1000 ? f / 1000 + "k" : f, X(f) - 8, h - 8); } });
  if (xlabel) g.fillText(xlabel, w - 60, h - 8);
  if (bars) {
    const bw = Math.max(1, (w - pad) / xs.length - 2);
    g.fillStyle = color;
    xs.forEach((x, i) => { const y = Y(ys[i]); g.fillRect(X(x) - bw / 2, y, bw, Y(ymin) - y); });
    return;
  }
  g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
  xs.forEach((x, i) => (i ? g.lineTo(X(x), Y(ys[i])) : g.moveTo(X(x), Y(ys[i])))); g.stroke();
}

function kv(el, pairs) { el.innerHTML = pairs.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join(""); }

function showAnalysis(a) {
  $("#analysis").hidden = false;
  lineChart($("#specChart"), a.spectrum.hz, a.spectrum.db, { logx: true });
  lineChart($("#harmChart"), a.harmonics_db.map((_, i) => i + 1), a.harmonics_db, { bars: true, color: "#b06cff", xlabel: "harmonic #" });
  const e = a.envelope;
  kv($("#envInfo"), [["attack", fmt(e.attack * 1000, 1) + " ms"], ["decay", fmt(e.decay * 1000, 0) + " ms"], ["sustain", fmt(e.sustain * 100, 0) + " %"], ["release", fmt(e.release * 1000, 0) + " ms"]]);
  const l = a.lfo;
  kv($("#lfoInfo"), l ? [["moves", l.feature], ["rate", l.sync ? `${l.sync} @ ${a.bpm} BPM` : fmt(l.rate_hz, 2) + " Hz"], ["period", fmt(l.period_s * 1000, 0) + " ms"], ["strength", fmt(l.strength, 2)], ["retrigger", l.mode]] : [["none detected", ""]]);
  kv($("#charInfo"), [["type", a.percussive ? "percussive / drum" : "pitched"], ["voiced", fmt(a.voiced_ratio * 100, 0) + " %"], ["noise", fmt(a.noise_ratio * 100, 0) + " %"], ["unison/detune", fmt(a.unison_hint, 2)], ["tempo", a.bpm + " BPM"]]);
  const tb = $("#notesBody"); tb.innerHTML = "";
  a.notes.forEach((n, i) => {
    const g = (a.timing.notes || [])[i] || {};
    const bend = n.bend ? `${fmt(Math.min(...n.bend.map((b) => b[1])), 1)} … ${fmt(Math.max(...n.bend.map((b) => b[1])), 1)} st` : "";
    tb.insertAdjacentHTML("beforeend", `<tr><td>${i + 1}</td><td class="mono">${fmt(n.start, 4)}</td><td class="mono">${fmt(g.grid_16th, 3)}</td><td class="mono">${fmt(g.offset_ms, 1)} ms</td><td>${fmt(g.length_16ths, 2)} × 1/16</td><td>${noteName(n.pitch)}</td><td>${bend}</td></tr>`);
  });
}

// ------------------------------------------------------------------ reconstruction
$("#target").oninput = () => ($("#targetVal").textContent = $("#target").value);
$("#runRecon").onclick = async () => {
  const body = {
    stem: state.stem, ...selRange(), bpm: parseFloat($("#bpm").value) || null,
    target: parseFloat($("#target").value), time_budget: parseFloat($("#budget").value),
    allow_custom: $("#allowCustom").checked, stock_only: $("#stockOnly").checked, verify_serum: $("#verifySerum").checked,
  };
  if (!body.start && state.duration > 12 && !confirm("No selection: the whole stem will be used (only the first 6 s drive the search). Continue?")) return;
  $("#runRecon").disabled = true;
  const r = await post(`/api/projects/${state.pid}/reconstruct`, body);
  pollJob(r.job, (job) => { $("#runRecon").disabled = false; if (job.result) { showResult(job.result); openProject(state.pid); } });
};

function showResult(r) {
  state.result = r;
  $("#result").hidden = false;
  document.querySelector('.tab[data-tab="recon"]').click();
  $("#accVal").textContent = fmt(r.accuracy, 1);
  const badge = $("#accBadge");
  badge.className = "badge " + (r.met_target ? "ok" : "warn");
  badge.textContent = r.met_target ? `target ${r.target}% reached${r.used_custom_wavetable ? " (custom wavetable)" : " with stock/basic waves"}` : `below target ${r.target}% — try a longer search or allow custom wavetables`;
  const d = r.details;
  $("#accBars").innerHTML = ["spectral", "harmonic", "envelope", "tone", "movement"].map((k) => `<span>${k}</span><div class="b"><div style="width:${d[k]}%"></div></div><span>${fmt(d[k], 0)}</span>`).join("") +
    `<span>raw</span><span class="muted small">${fmt(d.raw_score, 1)} raw ÷ ${fmt(d.self_similarity, 1)} phase-noise ceiling</span><span></span>`;
  $("#serumScore").textContent = r.serum && r.serum.ok ? `Real Serum 2 render scored ${fmt(r.serum.score, 1)} (raw)` : `Serum 2 check: ${r.serum ? r.serum.error : "not run"}`;
  const base = `/api/projects/${state.pid}/results/${r.id}/file/`;
  $("#aTarget").src = base + "target.wav";
  $("#aRender").src = base + "render.wav";
  $("#serumAB").hidden = !(r.serum && r.serum.ok);
  if (r.serum && r.serum.ok) $("#aSerum").src = base + "serum_render.wav";
  const preset = r.files.find((f) => f.endsWith(".SerumPreset"));
  $("#dlPreset").href = base + encodeURIComponent(preset); $("#dlPreset").download = preset;
  $("#dlMidi").href = base + "notes.mid"; $("#dlMidi").download = r.name + ".mid";
  $("#dlRender").href = base + "render.wav"; $("#dlRender").download = r.name + ".wav";
  $("#installBtn").onclick = async () => { const x = await post(`/api/projects/${state.pid}/results/${r.id}/install`); alert("Installed:\n" + x.installed + "\n\nIn Serum 2 open the preset browser → User → Fourier Analyzer."); };
  $("#revealBtn").onclick = () => post(`/api/projects/${state.pid}/results/${r.id}/reveal`);
  $("#patchDesc").innerHTML = r.description.map((l) => `<li>${l}</li>`).join("");
  $("#stages").innerHTML = r.stages.map((s) => `<li><b>${s.stage}</b>: ${fmt(s.score, 1)}${s.table ? " · " + s.table : ""}${s.steps && s.steps.length ? "<br><span class='muted small'>" + s.steps.map((x) => (x.added ? "+ " + x.added : "− " + x.removed) + " → " + fmt(x.score, 1)).join("<br>") + "</span>" : ""}</li>`).join("");
  $("#checks").innerHTML = r.checks.map((c) => `<li class="st-${c.status}"><b>${c.name}</b>: ${c.detail}${c.note ? ` <span class="muted">(${c.note})</span>` : ""}</li>`).join("");
}

// ------------------------------------------------------------------ jobs
function showJob(j) {
  $("#jobBox").hidden = false;
  $("#jobStage").textContent = `${j.kind || ""} · ${j.stage || j.status || ""}`;
  $("#jobBest").textContent = j.best != null ? `best ${fmt(j.best, 1)}` : "";
  $("#jobBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  const pre = $("#jobLog");
  pre.textContent = (j.log || []).map((l) => `${l.t}s  ${l.msg}`).join("\n");
  pre.scrollTop = pre.scrollHeight;
}
function pollJob(jid, done) {
  clearInterval(state.polling[jid]);
  state.polling[jid] = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jid}`);
      showJob(j);
      if (j.status === "done" || j.status === "error") {
        clearInterval(state.polling[jid]);
        if (j.status === "error") { $("#jobStage").textContent = "Error: " + j.error; }
        else setTimeout(() => ($("#jobBox").hidden = true), 2500);
        done && done(j);
      }
    } catch (e) { /* transient */ }
  }, 700);
}

// ------------------------------------------------------------------ settings
$("#settingsBtn").onclick = async () => {
  const s = await api("/api/system");
  kv($("#sysInfo"), [["version", s.version], ["ML separation", s.ml_separation ? "installed" : "missing (DSP fallback)"], ["Serum host (pedalboard)", s.pedalboard ? "available" : "missing"], ["Serum plugin", s.serum_plugin || "not found"], ["Serum tables", s.serum_tables || "not found"], ["exports", s.exports]]);
  const st = s.settings || {};
  const ch = $("#setChannel");
  if (st.update_channel && st.update_channel !== "release" && ![...ch.options].some((o) => o.value === st.update_channel)) ch.add(new Option(st.update_channel, st.update_channel));
  ch.value = st.update_channel || "release";
  $("#setAuto").checked = st.auto_update !== false;
  $("#setTables").value = st.serum_tables_dir || ""; $("#setPlugin").value = st.serum_plugin_path || "";
  $("#settings").showModal();
};
$("#saveSettings").onclick = async () => {
  await post("/api/settings", { values: { update_channel: $("#setChannel").value, auto_update: $("#setAuto").checked, serum_tables_dir: $("#setTables").value.trim(), serum_plugin_path: $("#setPlugin").value.trim() } });
  checkUpdate();
};
$("#calibrateBtn").onclick = async () => { const r = await post("/api/calibrate"); $("#settings").close(); pollJob(r.job, (j) => alert(j.status === "done" ? "Calibrated: " + JSON.stringify(j.result) : "Calibration failed: " + j.error)); };
$("#bpm").onchange = updateSelInfo;

// ------------------------------------------------------------------ drag & drop
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "drop" || e.target === drop) drop.classList.remove("over"); }));
document.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
$("#fileIn").onchange = (e) => { const f = e.target.files[0]; if (f) upload(f); e.target.value = ""; };
window.addEventListener("resize", () => { if (state.peaks) drawWave(); });

loadProjects();
checkUpdate();
setInterval(checkUpdate, 6 * 3600 * 1000);
