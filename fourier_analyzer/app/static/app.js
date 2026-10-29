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
    li.title = p.id;
    li.innerHTML = `<span class="pname"></span><button class="del" title="Delete project">×</button>`;
    li.querySelector(".pname").textContent = p.name;
    if (p.id === state.pid) li.classList.add("active");
    li.onclick = () => openProject(p.id);
    li.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete project "${p.name}" and all its stems and results?\n(Files already exported to ~/Music/Fourier Analyzer are kept.)`)) return;
      const r = await fetch(`/api/projects/${p.id}`, { method: "DELETE" });
      if (!r.ok) { alert(await r.text()); return; }
      if (state.pid === p.id) { state.pid = null; state.stem = null; $("#proj").hidden = true; $("#empty").hidden = false; }
      loadProjects();
    };
    ul.appendChild(li);
  });
}

async function upload(file) {
  const fd = new FormData(); fd.append("file", file);
  showJob({ stage: "uploading", progress: 0.02, log: [] });
  const r = await fetch(`/api/projects?drums=${$("#optDrums").checked}`, { method: "POST", body: fd });
  if (!r.ok) { alert("Upload failed: " + (await r.text())); return; }
  const j = await r.json();
  $("#jobBox").hidden = true;
  await openProject(j.project.id);
  if (j.job) pollJob(j.job, () => openProject(j.project.id));
  loadProjects();
}

async function openProject(pid) {
  state.pid = pid;
  state.meta = await api(`/api/projects/${pid}`);
  $("#empty").hidden = true; $("#proj").hidden = false;
  $("#projName").textContent = state.meta.name;
  $("#projInfo").textContent = `${fmt(state.meta.duration, 1)} s · ${state.meta.status}`;
  $("#bpm").value = state.meta.bpm || "";
  renderBpmAlts();
  if (!state.meta.bpm_candidates && state.meta.status === "ready") {
    post(`/api/projects/${pid}/tempo`).then((t) => { state.meta.bpm = t.bpm; state.meta.bpm_candidates = t.candidates; $("#bpm").value = t.bpm; renderBpmAlts(); });
  }
  api(`/api/jobs?project=${pid}`).then((jobs) => jobs.forEach((j) => pollJob(j.id, () => openProject(pid)))).catch(() => {});
  const st = state.meta.stems || {};
  // older projects have no "separation" field: the DSP fallback never produces toms/ride/crash
  const dsp = state.meta.separation === "dsp" || (!state.meta.separation && "drum_kick" in st && !("drum_toms" in st));
  $("#dspWarn").hidden = !dsp || state.meta.status !== "ready";
  $("#resepBtn").onclick = async () => {
    $("#dspWarn").hidden = true;
    const r = await post(`/api/projects/${pid}/separate`);
    pollJob(r.job, () => openProject(pid));
  };
  const fresh = state.meta.status === "new";
  $("#clipCard").hidden = !fresh;
  $("#instCard").hidden = fresh;
  renderInstruments(); renderStems(); renderPastResults();
  if (fresh && state.stem !== "mix") selectStem("mix");
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
    row.innerHTML = `<label class="tick" title="Tick to combine"><input type="checkbox" data-stem="${name}"></label><button class="iconbtn" title="Play">▶</button>
      <div><div class="name">${name}<small>${name === "mix" ? "full mix" : stemLabel(name)}</small></div><canvas></canvas></div>
      <div class="acts"><button class="ghost sel">Select</button><a class="btn ghost" href="/api/projects/${state.pid}/audio/${name}" download="${name}.wav">.wav</a></div>`;
    row.querySelector(".iconbtn").onclick = (e) => { e.stopPropagation(); playStem(name); };
    row.querySelector(".sel").onclick = () => selectStem(name);
    const cv = row.querySelector("canvas");
    cv.dataset.stem = name;
    cv.title = "Click to play from here";
    cv.onclick = (e) => { const r = cv.getBoundingClientRect(); playFrom(name, ((e.clientX - r.left) / r.width) * (cv._dur || 0)); };
    box.appendChild(row);
    api(`/api/projects/${state.pid}/peaks/${name}?n=600`).then((p) => { cv._peaks = p.peaks; cv._dur = p.duration; redrawRow(cv); });
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

// ------------------------------------------------------------------ audio + playhead
const player = $("#player");
state.playStem = null;   // stem loaded in the player
state.stopAt = null;     // end of selection playback
function rowButton(name) {
  const cv = document.querySelector(`#stems canvas[data-stem="${CSS.escape(name)}"]`);
  return cv ? cv.closest(".stem").querySelector(".iconbtn") : null;
}
function syncButtons() {
  document.querySelectorAll("#stems .iconbtn").forEach((b) => (b.textContent = "▶"));
  const b = state.playStem && rowButton(state.playStem);
  if (b && !player.paused) b.textContent = "❚❚";
  $("#playSel").textContent = !player.paused && state.playStem === state.stem ? "❚❚ Pause" : "▶ Play selection";
}
function playFrom(name, t, stopAt = null) {
  const go = () => { player.currentTime = Math.max(0, t || 0); state.stopAt = stopAt; player.play(); };
  if (state.playStem !== name) {
    state.playStem = name;
    player.src = `/api/projects/${state.pid}/audio/${name}`;
    player.addEventListener("loadedmetadata", go, { once: true });
  } else go();
}
function playStem(name) {
  if (state.playStem === name && !player.paused) { player.pause(); return; }
  const resume = state.playStem === name && player.currentTime > 0 && player.currentTime < (player.duration || 1e9) - 0.05;
  playFrom(name, resume ? player.currentTime : 0);
}
function headTime(name) { return state.playStem === name ? player.currentTime : null; }
function redrawRow(cv) {
  if (cv._peaks) drawPeaks(cv, cv._peaks, null, cv._dur, headTime(cv.dataset.stem));
}
function redrawHeads() {
  if (!state.playStem) return;
  const cv = document.querySelector(`#stems canvas[data-stem="${CSS.escape(state.playStem)}"]`);
  if (cv) redrawRow(cv);
  if (state.peaks && state.playStem === state.stem) drawWave();
  const pt = $("#playTime");
  if (pt) pt.textContent = state.playStem === state.stem ? `${fmt(player.currentTime, 2)} s` : "";
}
(function headLoop() {
  if (!player.paused) {
    if (state.stopAt != null && player.currentTime >= state.stopAt) { player.pause(); player.currentTime = state.stopAt; state.stopAt = null; }
    redrawHeads();
  }
  requestAnimationFrame(headLoop);
})();
["play", "pause", "ended", "seeked"].forEach((ev) => player.addEventListener(ev, () => { syncButtons(); redrawHeads(); }));
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || /INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) return;
  e.preventDefault();
  if (!player.paused) { player.pause(); return; }
  const name = state.stem || state.playStem;
  if (!name) return;
  if (state.playStem === name && player.currentTime > 0) player.play();
  else if (state.sel) playFrom(name, state.sel[0], state.sel[1]);
  else playFrom(name, 0);
});

// ------------------------------------------------------------------ waveform editor
async function selectStem(name) {
  state.stem = name; state.sel = null;
  renderStems();
  $("#editor").hidden = false;
  $("#edTitle").textContent = `${name} — ${name === "mix" ? "full mix" : stemLabel(name)}`;
  const p = await api(`/api/projects/${state.pid}/peaks/${name}?n=4000`);
  state.peaks = p.peaks; state.duration = p.duration;
  state.view = [0, p.duration]; state.viewData = null;
  drawWave(); updateSelInfo(); $("#spec").hidden = true;
  $("#editor").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawPeaks(cv, peaks, sel, dur, head = null) {
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
  if (head != null && dur) {
    const x = Math.round((head / dur) * w) + 0.5;
    g.strokeStyle = "#ffffff"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    g.fillStyle = "#ffffff";
    g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 7); g.closePath(); g.fill();
  }
}

// ---------------------------------------------------------------- zoomable editor
const MIN_VIEW = 0.005;           // seconds visible at maximum zoom
function view() { return state.view || [0, state.duration || 1]; }
function setView(a, b) {
  const D = state.duration || 1;
  let w = Math.max(MIN_VIEW, Math.min(D, b - a));
  a = Math.max(0, Math.min(D - w, a));
  state.view = [a, a + w];
  drawWave();
  fetchViewPeaks();
}
function zoomBy(f, center) {
  const [a, b] = view();
  const c = center == null ? (a + b) / 2 : center;
  setView(c - (c - a) * f, c + (b - c) * f);
}
let _vpTimer = null;
function fetchViewPeaks() {
  clearTimeout(_vpTimer);
  const [a, b] = view();
  if (a <= 0 && b >= state.duration - 1e-6) { state.viewData = null; return; }
  _vpTimer = setTimeout(async () => {
    const cv = $("#wave"); const stem = state.stem;
    const p = await api(`/api/projects/${state.pid}/peaks/${stem}?start=${a}&end=${b}&n=${Math.round(cv.clientWidth * 1.5)}`);
    const [a2, b2] = view();
    if (stem === state.stem && Math.abs(a2 - a) < 1e-9 && Math.abs(b2 - b) < 1e-9) { state.viewData = p; drawWave(); }
  }, 60);
}
function niceStep(span, px) {
  const target = span / Math.max(1, px / 90);
  const p = Math.pow(10, Math.floor(Math.log10(target)));
  for (const m of [1, 2, 5, 10]) if (m * p >= target) return m * p;
  return 10 * p;
}
function fmtT(t, step) {
  const d = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
  return t >= 60 ? `${Math.floor(t / 60)}:${(t % 60).toFixed(d).padStart(d ? 3 + d : 2, "0")}` : t.toFixed(d) + "s";
}
function drawWave() {
  const cv = $("#wave"); if (!cv || !state.peaks) return;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const [a, b] = view(), span = b - a, D = state.duration;
  const X = (t) => ((t - a) / span) * w;
  const top = 16, mid = top + (h - top) / 2, amp = (h - top) / 2 - 2;
  // beat / bar grid
  const bpm = parseFloat($("#bpm").value) || state.meta.bpm;
  if (bpm) {
    const beat = 60 / bpm;
    if (span / beat < w / 6) {
      for (let k = Math.floor(a / beat); k * beat <= b; k++) {
        g.fillStyle = k % 4 === 0 ? "#2b303c" : "#1a1e26";
        g.fillRect(Math.round(X(k * beat)), top, 1, h - top);
        if (k % 4 === 0 && span / beat < w / 25) { g.fillStyle = "#5b6475"; g.font = "9px -apple-system"; g.fillText(`bar ${k / 4 + 1}`, X(k * beat) + 3, h - 4); }
      }
    }
  }
  // time ruler
  const step = niceStep(span, w);
  g.fillStyle = "#8a92a6"; g.font = "10px -apple-system";
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) {
    const x = Math.round(X(t)); g.fillRect(x, 0, 1, 5); g.fillText(fmtT(t, step), x + 3, 10);
  }
  // selection
  if (state.sel) { g.fillStyle = "rgba(255,138,61,.22)"; g.fillRect(X(state.sel[0]), top, X(state.sel[1]) - X(state.sel[0]), h - top); }
  // waveform: zoomed data (signed min/max) when available, else the overview peaks
  const max = Math.max(1e-6, ...state.peaks);
  g.fillStyle = "#b06cff";
  const vd = state.viewData;
  if (vd && Math.abs(vd.start - a) < 1e-3 && Math.abs(vd.end - b) < 1e-3 && vd.mins) {
    const n = vd.mins.length;
    if (n < w / 2) {   // very deep zoom: draw the actual signal as a line
      g.strokeStyle = "#b06cff"; g.lineWidth = 1.5; g.beginPath();
      vd.maxs.forEach((v, i) => { const x = (i + 0.5) / n * w, y = mid - (v / max) * amp; i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
    } else {
      for (let x = 0; x < w; x++) {
        const i0 = Math.floor(x / w * n), i1 = Math.max(i0 + 1, Math.floor((x + 1) / w * n));
        let lo = 0, hi = 0;
        for (let i = i0; i < i1; i++) { lo = Math.min(lo, vd.mins[i]); hi = Math.max(hi, vd.maxs[i]); }
        const y0 = mid - (hi / max) * amp, y1 = mid - (lo / max) * amp;
        g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
      }
    }
  } else {
    const P = state.peaks, n = P.length;
    for (let x = 0; x < w; x++) {
      const t0 = a + (x / w) * span, t1 = a + ((x + 1) / w) * span;
      const i0 = Math.floor((t0 / D) * n), i1 = Math.max(i0 + 1, Math.ceil((t1 / D) * n));
      let m = 0; for (let i = i0; i < i1 && i < n; i++) m = Math.max(m, P[i]);
      const v = (m / max) * amp; g.fillRect(x, mid - v, 1, Math.max(1, v * 2));
    }
  }
  // playhead (+ follow it while playing)
  const head = headTime(state.stem);
  if (head != null) {
    if (!player.paused && (head > b || head < a) && span < D) { setView(head - span * 0.05, head + span * 0.95); return; }
    const x = Math.round(X(head)) + 0.5;
    g.strokeStyle = "#ffffff"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, top); g.lineTo(x, h); g.stroke();
    g.fillStyle = "#ffffff"; g.beginPath(); g.moveTo(x - 5, top); g.lineTo(x + 5, top); g.lineTo(x, top + 7); g.closePath(); g.fill();
  }
  drawOverview();
  $("#zoomSel").disabled = !state.sel;
}
function drawOverview() {
  const cv = $("#waveOverview"); if (!cv || !state.peaks) return;
  const [a, b] = view(), D = state.duration || 1;
  drawPeaks(cv, state.peaks, state.sel, D, headTime(state.stem));
  const g = cv.getContext("2d"), w = cv.clientWidth, h = cv.clientHeight;
  if (b - a < D - 1e-6) {
    g.fillStyle = "rgba(0,0,0,.55)";
    g.fillRect(0, 0, (a / D) * w, h); g.fillRect((b / D) * w, 0, w - (b / D) * w, h);
    g.strokeStyle = "#ffffff"; g.lineWidth = 1; g.strokeRect((a / D) * w + 0.5, 0.5, ((b - a) / D) * w - 1, h - 1);
  }
}

(function setupSelection() {
  const cv = $("#wave"); let dragging = false, x0 = 0, px0 = 0;
  const t = (e) => { const [a, b] = view(); return Math.max(0, Math.min(state.duration, a + ((e.clientX - cv.getBoundingClientRect().left) / cv.clientWidth) * (b - a))); };
  let prevSel = null;
  cv.addEventListener("mousedown", (e) => { dragging = true; x0 = t(e); px0 = e.clientX; prevSel = state.sel; state.sel = [x0, x0]; drawWave(); });
  window.addEventListener("mousemove", (e) => { if (!dragging) return; const x = t(e); state.sel = [Math.min(x0, x), Math.max(x0, x)]; drawWave(); updateSelInfo(); });
  window.addEventListener("mouseup", (e) => {
    if (!dragging) return; dragging = false;
    if (Math.abs(e.clientX - px0) < 4) {
      // a click (not a drag) moves the playhead and plays from there
      state.sel = prevSel;
      drawWave(); updateSelInfo();
      playFrom(state.stem, x0);
      return;
    }
    drawWave(); updateSelInfo(); loadSpectrogram();
  });
  // zoom: ctrl/cmd + wheel (trackpad pinch sends ctrl+wheel); pan: horizontal / shift + wheel
  cv.addEventListener("wheel", (e) => {
    const [a, b] = view(), span = b - a;
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      zoomBy(Math.exp(e.deltaY * 0.01), t(e));
    } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault();
      const d = (e.shiftKey ? e.deltaY : e.deltaX) / cv.clientWidth * span;
      setView(a + d, b + d);
    }
  }, { passive: false });
  // overview: drag the box to pan, click elsewhere to centre there
  const ov = $("#waveOverview"); let ovDrag = false, ovOff = 0;
  const ovT = (e) => ((e.clientX - ov.getBoundingClientRect().left) / ov.clientWidth) * state.duration;
  ov.addEventListener("mousedown", (e) => {
    const [a, b] = view(), tt = ovT(e);
    ovDrag = true; ovOff = tt >= a && tt <= b ? tt - a : (b - a) / 2;
    setView(tt - ovOff, tt - ovOff + (b - a));
  });
  window.addEventListener("mousemove", (e) => { if (!ovDrag) return; const [a, b] = view(); const tt = ovT(e); setView(tt - ovOff, tt - ovOff + (b - a)); });
  window.addEventListener("mouseup", () => { ovDrag = false; });
})();
$("#zoomIn").onclick = () => zoomBy(0.5, headTime(state.stem));
$("#zoomOut").onclick = () => zoomBy(2);
$("#zoomFit").onclick = () => setView(0, state.duration);
$("#zoomSel").onclick = () => { if (state.sel) { const pad = (state.sel[1] - state.sel[0]) * 0.08; setView(state.sel[0] - pad, state.sel[1] + pad); } };
document.addEventListener("keydown", (e) => {
  if (!state.peaks || $("#editor").hidden || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) || e.metaKey || e.ctrlKey) return;
  if (e.key === "+" || e.key === "=") zoomBy(0.5);
  else if (e.key === "-" || e.key === "_") zoomBy(2);
  else if (e.key === "0") setView(0, state.duration);
  else if (e.key === "z" || e.key === "Z") $("#zoomSel").click();
});

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
  if (!player.paused && state.playStem === state.stem) { player.pause(); return; }
  const r = selRange();
  playFrom(state.stem, r.start || 0, r.end);
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
  const sc = a.sidechain || {};
  $("#scAnalysis").innerHTML = sc.detected
    ? `<b>Detected</b> from the ${sc.source}: −${sc.depth_db} dB duck, attack ${sc.attack_ms} ms, hold ${sc.hold_ms} ms, release ${sc.release_ms} ms (${sc.kicks_used} kicks, fit error ${sc.fit_rms_db} dB). The analysis above is of the <i>un-ducked</i> synth.<br><span class="muted">${sc.logic || ""}</span>`
    : `<span class="muted">Not detected: ${sc.reason || "—"}</span>`;
  drawRoll(a, sc);
  const tb = $("#notesBody"); tb.innerHTML = "";
  a.notes.forEach((n, i) => {
    const g = (a.timing.notes || [])[i] || {};
    const bend = n.bend ? `${fmt(Math.min(...n.bend.map((b) => b[1])), 1)} … ${fmt(Math.max(...n.bend.map((b) => b[1])), 1)} st` : "";
    tb.insertAdjacentHTML("beforeend", `<tr><td>${i + 1}</td><td class="mono">${fmt(n.start, 4)}</td><td class="mono">${fmt(g.grid_16th, 3)}</td><td class="mono">${fmt(g.offset_ms, 1)} ms</td><td>${fmt(g.length_16ths, 2)} × 1/16</td><td>${noteName(n.pitch)}</td><td>${bend}</td></tr>`);
  });
}

function drawRoll(a, sc) {
  const cv = $("#roll"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const notes = a.notes || [];
  if (!notes.length) return;
  let lo = Math.min(...notes.map((n) => n.pitch + Math.min(0, ...(n.bend || [[0, 0]]).map((b) => b[1])))) - 2;
  let hi = Math.max(...notes.map((n) => n.pitch + Math.max(0, ...(n.bend || [[0, 0]]).map((b) => b[1])))) + 2;
  if (hi - lo < 12) { const m = (hi + lo) / 2; lo = m - 6; hi = m + 6; }
  const dur = a.duration, pad = 34;
  const X = (t) => pad + (t / dur) * (w - pad - 4), Y = (p) => 6 + (1 - (p - lo) / (hi - lo)) * (h - 12);
  g.font = "10px -apple-system"; g.fillStyle = "#8a92a6"; g.strokeStyle = "#20242d";
  for (let p = Math.ceil(lo); p <= hi; p++) {
    if (p % 12 === 0 || hi - lo < 25) { g.beginPath(); g.moveTo(pad, Y(p)); g.lineTo(w, Y(p)); g.stroke(); }
    if (p % 12 === 0) g.fillText(noteName(p), 2, Y(p) + 3);
  }
  const beat = 60 / a.bpm;
  for (let t = 0, k = 0; t < dur; t += beat, k++) { g.strokeStyle = k % 4 === 0 ? "#2f3542" : "#1b1f27"; g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), h); g.stroke(); }
  (sc && sc.kicks || []).forEach((k) => { g.fillStyle = "#ff8a3d"; g.fillRect(X(k) - 1, h - 8, 2, 8); });
  notes.forEach((n) => {
    g.fillStyle = `rgba(176,108,255,${0.35 + 0.65 * (n.velocity || 1)})`;
    g.fillRect(X(n.start), Y(n.pitch) - 4, Math.max(2, X(n.start + n.duration) - X(n.start)), 8);
    if (n.bend) {
      g.strokeStyle = "#ffffff"; g.lineWidth = 1.2; g.beginPath();
      n.bend.forEach((b, i) => { const x = X(n.start + b[0]), y = Y(n.pitch + b[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke(); g.lineWidth = 1;
    }
  });
}

// ------------------------------------------------------------------ reconstruction
$("#target").oninput = () => ($("#targetVal").textContent = $("#target").value);
$("#runRecon").onclick = async () => {
  const body = {
    stem: state.stem, ...selRange(), bpm: parseFloat($("#bpm").value) || null,
    target: parseFloat($("#target").value), time_budget: parseFloat($("#budget").value),
    verify_serum: $("#verifySerum").checked,
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
  badge.textContent = r.met_target ? `target ${r.target}% reached` : `below target ${r.target}%: try a longer search or a cleaner selection`;
  const d = r.details;
  const info = r.metric_info || {};
  if ("waveform_similarity" in d) {
    const rows = [
      ["waveform similarity", d.waveform_similarity, "%", d.waveform_similarity],
      ["spectral similarity", d.spectral_similarity, "%", d.spectral_similarity],
      ["envelope correlation", d.envelope_correlation, "%", d.envelope_correlation],
      ["SI-SDR", d.si_sdr_db, " dB", Math.max(0, Math.min(100, d.si_sdr_db * 5))],
      ["log-spectral distance", d.lsd_db, " dB", Math.max(0, 100 - d.lsd_db * 10)],
    ];
    const keys = ["waveform_similarity", "spectral_similarity", "envelope_correlation", "si_sdr_db", "lsd_db"];
    $("#accBars").innerHTML = rows.map((x, i) => `<span title="${info[keys[i]] || ""}">${x[0]}</span><div class="b"><div style="width:${x[3]}%"></div></div><span>${fmt(x[1], 1)}${x[2]}</span>`).join("");
  } else {   // results made by older versions
    $("#accBars").innerHTML = ["spectral", "harmonic", "envelope", "tone", "movement"].map((k) => `<span>${k}</span><div class="b"><div style="width:${d[k]}%"></div></div><span>${fmt(d[k], 0)}</span>`).join("");
  }
  const sc = r.sidechain;
  $("#scInfo").textContent = !sc ? "" : sc.detected
    ? `Sidechain detected from the ${sc.source}: -${sc.depth_db} dB duck, attack ${sc.attack_ms} ms, hold ${sc.hold_ms} ms, release ${sc.release_ms} ms (${sc.kicks_used} kicks). It was removed before fitting and is re-applied when comparing. ${r.sidechain_logic || ""}`
    : `No sidechain ducking found (${sc.reason || ""}).`;
  $("#serumScore").textContent = r.serum && r.serum.ok ? `Real Serum 2 render: waveform similarity ${fmt(r.serum.score, 1)}%, spectral ${fmt(r.serum.details.spectral_similarity, 1)}%` : `Serum 2 check: ${r.serum ? r.serum.error : "not run"}`;
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
const clock = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
function showJob(j) {
  $("#jobBox").hidden = false;
  const d = j.detail || {};
  const kind = { reconstruct: "Reconstructing patch", separate: "Separating stems", split: "Splitting layers", calibrate: "Calibrating" }[j.kind] || j.kind || "";
  $("#jobStage").textContent = `${kind}${j.status === "done" ? " · done" : ""}`;
  let timing = `running ${clock(j.elapsed)}`;
  if (j.status === "done") timing = `finished in ${clock(j.elapsed)}`;
  else if (d.budget && d.elapsed != null) timing += ` · search ${clock(d.elapsed)} of ${clock(d.budget)} (~${clock(d.budget - d.elapsed)} left)`;
  $("#jobTime").textContent = timing;
  $("#jobBest").textContent = j.best != null ? `best match ${fmt(j.best, 1)}` : "";
  let now = d.stage_text || j.stage || "";
  if (d.activity) now += ` — ${d.activity}`;
  const bits = [];
  if (d.stage_elapsed != null) bits.push(`this step ${clock(d.stage_elapsed)}`);
  if (d.stage_evals) bits.push(`${d.stage_evals.toLocaleString()} patches tested${d.rate ? ` (${fmt(d.rate, 0)}/s)` : ""}`);
  if (d.evals) bits.push(`${d.evals.toLocaleString()} total`);
  $("#jobNow").textContent = j.status === "error" ? "" : (now ? "Now: " + now + (bits.length ? "  ·  " + bits.join(" · ") : "") : "");
  $("#jobBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  const pre = $("#jobLog");
  const text = (j.log || []).map((l) => `${l.t}s  ${l.msg}`).join("\n");
  if (pre.textContent !== text) {
    // follow new lines only if the user is already at the bottom; never yank
    // them away from something they scrolled up to read
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    const keep = pre.scrollTop;
    pre.textContent = text;
    pre.scrollTop = atBottom ? pre.scrollHeight : keep;
  }
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
function renderBpmAlts() {
  const box = $("#bpmAlts"); box.innerHTML = "";
  const c = (state.meta.bpm_candidates || []).filter((x) => Math.abs(x - parseFloat($("#bpm").value)) > 0.01).slice(0, 4);
  if (!c.length) return;
  box.insertAdjacentHTML("beforeend", `<span class="muted small">or</span>`);
  c.forEach((x) => {
    const b = document.createElement("button"); b.className = "pill"; b.textContent = x;
    b.title = "Use this tempo (half/double-time and triplet feels are easy to confuse)";
    b.onclick = () => { $("#bpm").value = x; $("#bpm").onchange(); };
    box.appendChild(b);
  });
}
// ------------------------------------------------------------------ clip + combine
async function separate(clip) {
  const r = selRange();
  // clip=false still sends clip:true with no range: that restores the full original
  const body = { drums: $("#optDrums").checked, clip: true, start: clip ? r.start : null, end: clip ? r.end : null };
  if (clip && r.start == null) { alert("Drag over the part you want in the waveform first (or use 'Separate whole file')."); return; }
  $("#clipCard").hidden = true;
  state.sel = null;
  const j = await post(`/api/projects/${state.pid}/separate`, body);
  state.meta.status = "separating";
  pollJob(j.job, () => { state.stem = null; openProject(state.pid); });
}
$("#sepSelBtn").onclick = () => separate(true);
$("#sepAllBtn").onclick = () => {
  const m = state.meta.clip ? "This re-separates the WHOLE original file. Continue?" : null;
  if (m && !confirm(m)) return;
  separate(false);
};
$("#reclipBtn").onclick = async () => {
  if (!confirm("Pick a new region of the ORIGINAL file and separate it again? Current stems are replaced (finished reconstructions are kept).")) return;
  await post(`/api/projects/${state.pid}/clip`, {});
  state.stem = null;
  await openProject(state.pid);
};
$("#combineBtn").onclick = async () => {
  const picked = [...document.querySelectorAll('#stems input[type=checkbox]:checked')].map((c) => c.dataset.stem);
  if (picked.length < 2) { alert("Tick two or more stems/layers to combine (e.g. bass_layer1 + bass_layer3)."); return; }
  const name = prompt("Name for the combined stem:", picked.join("+"));
  if (name === null) return;
  try {
    const r = await post(`/api/projects/${state.pid}/combine`, { stems: picked, name });
    await openProject(state.pid);
    selectStem(r.stem);
  } catch (e) { alert(e.message); }
};
$("#bpm").onchange = () => {
  const v = parseFloat($("#bpm").value);
  if (v > 0 && state.pid) { post(`/api/projects/${state.pid}/bpm`, { bpm: v }); state.meta.bpm = v; }
  renderBpmAlts(); updateSelInfo();
};

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
