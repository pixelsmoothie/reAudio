"use strict";

/* AudioMind frontend: browse/search/recommend UI, filters, modals,
 * upload with analysis spinner, Spotify-style player bar with a live
 * Web Audio API canvas visualizer. */

const $ = (id) => document.getElementById(id);
const audioEl = $("audioElement");

const state = {
  presets: {},
  browseTracks: [],
  currentTrackId: null,
};

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

/* ------------------------------------------------------------------ */
/* filters                                                             */
/* ------------------------------------------------------------------ */

const filterDirty = new Set();
let fractalMode = "max"; // "max" => at most (smooth), "min" => at least (chaotic)

const SVG_PLAY = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
const SVG_PAUSE = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;

function updateFilterLabels() {
  $("durationValue").textContent = filterDirty.has("duration")
    ? `≤ ${$("durationSlider").value}s` : "any";
  const lo = $("bpmMin").value;
  const hi = $("bpmMax").value;
  $("bpmValue").textContent = filterDirty.has("bpm")
    ? `${Math.min(lo, hi)}–${Math.max(lo, hi)} BPM` : "any";
  const fv = parseFloat($("fractalSlider").value).toFixed(2);
  $("fractalValue").textContent = filterDirty.has("fractal")
    ? (fractalMode === "max" ? `≤ ${fv}` : `≥ ${fv}`) : "any";

  const badge = $("activeFiltersBadge");
  if (badge) {
    badge.textContent = filterDirty.size;
    badge.classList.toggle("hidden", filterDirty.size === 0);
  }
}

function currentFilters() {
  const filters = {};
  if (filterDirty.has("duration")) {
    filters.duration_max = parseFloat($("durationSlider").value);
  }
  if (filterDirty.has("bpm")) {
    const lo = parseInt($("bpmMin").value, 10);
    const hi = parseInt($("bpmMax").value, 10);
    filters.bpm_min = Math.min(lo, hi);
    filters.bpm_max = Math.max(lo, hi);
  }
  if (filterDirty.has("fractal")) {
    const value = parseFloat($("fractalSlider").value);
    if (fractalMode === "max") filters.max_fractal_dim = value;
    else filters.min_fractal_dim = value;
  }
  return filters;
}

function resetFilters() {
  filterDirty.clear();
  $("durationSlider").value = 300;
  $("bpmMin").value = 40;
  $("bpmMax").value = 200;
  $("fractalSlider").value = 1.5;
  updateFilterLabels();
}

/* ------------------------------------------------------------------ */
/* track cards                                                         */
/* ------------------------------------------------------------------ */

function hfdMeta(hfd) {
  if (hfd >= 1.6) return { cls: "chaotic", label: "Rough/Distorted" };
  if (hfd >= 1.35) return { cls: "textured", label: "Textured/Rhythmic" };
  return { cls: "smooth", label: "Smooth/Harmonic" };
}

function cardEl(track, showScore) {
  const card = document.createElement("article");
  card.className = "track-card glass";
  card.dataset.trackId = track.track_id;

  const hfd = hfdMeta(track.higuchi_fractal_dimension);
  const scoreHtml = (showScore && typeof track.match_score === "number")
    ? `<div class="match-row">
         <div class="match-bar"><div class="match-fill" style="width:${Math.max(0, Math.min(100, track.match_score))}%"></div></div>
         <span class="match-pct">${track.match_score.toFixed(1)}% Match</span>
       </div>`
    : "";

  card.innerHTML = `
    <div class="card-top">
      <button class="btn play-btn round" data-action="play" title="Play / Pause">${SVG_PLAY}</button>
      <div class="card-title-wrap">
        <h3 class="card-title">${esc(track.title)}</h3>
        <p class="card-artist">${esc(track.artist)}</p>
      </div>
    </div>
    <div class="card-tags">
      <span class="chip genre">${esc(track.primary_genre)}</span>
      <span class="chip bpm">${Math.round(track.bpm)} BPM</span>
      <span class="badge-hfd ${hfd.cls}">HFD: ${track.higuchi_fractal_dimension.toFixed(2)} (${hfd.label})</span>
    </div>
    ${scoreHtml}
    <div class="card-actions">
      <button class="btn btn-ghost small" data-action="why">Why did this match?</button>
      <button class="btn btn-ghost small" data-action="similar">Similar Tracks</button>
    </div>`;

  card.querySelector('[data-action="play"]').addEventListener("click", () => togglePlay(track));
  const whyBtn = card.querySelector('[data-action="why"]');
  if (track.explanation) {
    whyBtn.addEventListener("click", () => openExplainModal(track));
  } else {
    whyBtn.style.display = "none";
  }
  card.querySelector('[data-action="similar"]').addEventListener("click", () => loadSimilar(track));
  return card;
}

function renderResults(list, title, showScore) {
  $("resultsTitle").textContent = title;
  $("resultsCount").textContent = `${list.length} track${list.length === 1 ? "" : "s"}`;
  const grid = $("trackGrid");
  grid.innerHTML = "";
  $("emptyState").classList.toggle("hidden", list.length > 0);
  list.forEach((track) => grid.appendChild(cardEl(track, showScore)));
  syncPlayIcons();
}

async function loadBrowse() {
  state.browseTracks = await api("/api/tracks");
  renderResults(state.browseTracks, "All Tracks", false);
}

async function doSearch() {
  const query = $("searchInput").value.trim();
  if (!query) {
    toast("Type an intent first, or click a Game Dev preset.", true);
    return;
  }
  const body = { query, top_k: 8, ...currentFilters() };
  try {
    const results = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderResults(results, `Results for “${query}”`, true);
  } catch (err) {
    toast(err.message, true);
  }
}

async function loadSimilar(track) {
  try {
    const similar = await api(`/api/recommend/${encodeURIComponent(track.track_id)}`);
    renderResults(similar, `Similar to “${track.title}”`, false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    toast(err.message, true);
  }
}

/* ------------------------------------------------------------------ */
/* presets                                                             */
/* ------------------------------------------------------------------ */

async function loadPresets() {
  state.presets = await api("/api/presets");
  const wrap = $("presetPills");
  Object.entries(state.presets).forEach(([name, preset]) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "chip preset-pill";
    pill.title = preset.description;
    pill.textContent = name;
    pill.addEventListener("click", () => {
      $("searchInput").value = preset.query;
      doSearch();
    });
    wrap.appendChild(pill);
  });
}

/* ------------------------------------------------------------------ */
/* playback + visualizer                                               */
/* ------------------------------------------------------------------ */

let audioCtx = null;
let analyser = null;

function ensureAudioGraph() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") audioCtx.resume();
    return;
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  audioCtx = new Ctx();
  const source = audioCtx.createMediaElementSource(audioEl);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.82;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  startVisualizer();
}

function togglePlay(track) {
  if (state.currentTrackId === track.track_id) {
    if (audioEl.paused) audioEl.play();
    else audioEl.pause();
    return;
  }
  state.currentTrackId = track.track_id;
  audioEl.src = `/api/audio/${encodeURIComponent(track.track_id)}`;
  $("npTitle").textContent = track.title;
  $("npArtist").textContent = `${track.artist} · ${track.primary_genre}`;
  audioEl.play().catch((err) => toast(`Playback failed: ${err.message}`, true));
}

function syncPlayIcons() {
  const playing = !audioEl.paused && !!state.currentTrackId;
  const playerBar = $("playerBar");
  if (playerBar) playerBar.classList.toggle("is-playing", playing);

  document.querySelectorAll(".track-card").forEach((card) => {
    const btn = card.querySelector('[data-action="play"]');
    if (!btn) return;
    const isCurrent = card.dataset.trackId === state.currentTrackId;
    card.classList.toggle("active-track", isCurrent);
    btn.innerHTML = (playing && isCurrent) ? SVG_PAUSE : SVG_PLAY;
  });
  const mainBtn = $("playPause");
  if (mainBtn) mainBtn.innerHTML = playing ? SVG_PAUSE : SVG_PLAY;
}

function startVisualizer() {
  const canvas = $("visualizer");
  const ctx2d = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, rect.width * dpr);
    canvas.height = Math.max(1, rect.height * dpr);
  };
  window.addEventListener("resize", resize);
  resize();

  const timeData = new Uint8Array(analyser.fftSize);

  const draw = () => {
    requestAnimationFrame(draw);
    const w = canvas.width;
    const h = canvas.height;
    ctx2d.clearRect(0, 0, w, h);

    analyser.getByteTimeDomainData(timeData);

    ctx2d.save();
    ctx2d.beginPath();
    ctx2d.strokeStyle = "#38bdf8";
    ctx2d.shadowColor = "rgba(56, 189, 248, 0.55)";
    ctx2d.shadowBlur = 6 * dpr;
    ctx2d.lineWidth = 1.75 * dpr;
    for (let i = 0; i < timeData.length; i++) {
      const x = (i / (timeData.length - 1)) * w;
      const y = h / 2 + ((timeData[i] - 128) / 128) * h * 0.42;
      if (i === 0) ctx2d.moveTo(x, y);
      else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
    ctx2d.restore();
  };
  draw();
}

/* ------------------------------------------------------------------ */
/* modals                                                              */
/* ------------------------------------------------------------------ */

function openModal(modal) { modal.classList.remove("hidden"); }
function closeModal(modal) { modal.classList.add("hidden"); }
function closeAllModals() {
  document.querySelectorAll(".modal").forEach(closeModal);
}

function openExplainModal(track) {
  const ex = track.explanation || {};
  $("explainTrack").textContent =
    `${track.title} — ${track.artist} · ${track.primary_genre}`;
  $("explainScore").textContent =
    (typeof ex.score_pct === "number") ? `${ex.score_pct.toFixed(1)}% Match` : "";
  $("explainSummary").textContent = ex.summary ||
    "No AI explanation available for this view — run an intent search to generate one.";
  const list = $("explainInsights");
  list.innerHTML = "";
  (ex.insights || []).forEach((insight) => {
    const li = document.createElement("li");
    li.textContent = insight;
    list.appendChild(li);
  });
  openModal($("explainModal"));
}

/* ------------------------------------------------------------------ */
/* upload                                                              */
/* ------------------------------------------------------------------ */

function setupUpload() {
  $("uploadBtn").addEventListener("click", () => {
    $("uploadForm").reset();
    $("uploadStatus").textContent = "";
    openModal($("uploadModal"));
  });

  $("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file && !$("titleInput").value) {
      $("titleInput").value = file.name.replace(/\.[^.]+$/, "");
    }
  });

  $("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = $("fileInput");
    const submitBtn = $("uploadSubmit");
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("title", $("titleInput").value.trim() || fileInput.files[0].name);
    formData.append("artist", $("artistInput").value.trim() || "Unknown");

    submitBtn.disabled = true;
    $("uploadSpinner").classList.remove("hidden");
    $("uploadStatus").textContent = "Extracting DSP & fractal features, computing embeddings…";

    try {
      const track = await api("/api/upload", { method: "POST", body: formData });
      $("uploadStatus").textContent = "";
      closeAllModals();
      toast(`Indexed "${track.title}" — HFD ${track.higuchi_fractal_dimension.toFixed(2)}, ${Math.round(track.bpm)} BPM`);
      await loadBrowse();
    } catch (err) {
      $("uploadStatus").textContent = `Upload failed: ${err.message}`;
      toast(err.message, true);
    } finally {
      submitBtn.disabled = false;
      $("uploadSpinner").classList.add("hidden");
    }
  });
}

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  $("searchBtn").addEventListener("click", doSearch);
  $("searchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  $("filtersToggle").addEventListener("click", () => {
    $("filtersPanel").classList.toggle("hidden");
  });
  for (const id of ["durationSlider", "bpmMin", "bpmMax", "fractalSlider"]) {
    $(id).addEventListener("input", () => {
      filterDirty.add(
        id === "durationSlider" ? "duration"
        : id === "fractalSlider" ? "fractal" : "bpm"
      );
      updateFilterLabels();
    });
  }
  $("fractalMode").addEventListener("click", (e) => {
    fractalMode = fractalMode === "max" ? "min" : "max";
    e.target.textContent = fractalMode === "max" ? "≤ max" : "≥ min";
    updateFilterLabels();
  });
  $("resetFilters").addEventListener("click", resetFilters);

  $("playPause").addEventListener("click", () => {
    if (!state.currentTrackId) {
      const first = state.browseTracks[0];
      if (first) togglePlay(first);
      return;
    }
    if (audioEl.paused) audioEl.play();
    else audioEl.pause();
  });

  audioEl.addEventListener("play", () => { ensureAudioGraph(); syncPlayIcons(); });
  audioEl.addEventListener("pause", syncPlayIcons);
  audioEl.addEventListener("ended", syncPlayIcons);
  audioEl.addEventListener("loadedmetadata", () => {
    $("durTime").textContent = fmtTime(audioEl.duration);
  });
  audioEl.addEventListener("timeupdate", () => {
    $("curTime").textContent = fmtTime(audioEl.currentTime);
    if (audioEl.duration) {
      $("seekBar").value = (audioEl.currentTime / audioEl.duration) * 100;
    }
  });
  $("seekBar").addEventListener("input", (e) => {
    if (audioEl.duration) {
      audioEl.currentTime = (parseFloat(e.target.value) / 100) * audioEl.duration;
    }
  });

  document.querySelectorAll(".modal").forEach((modal) => {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal(modal);
    });
    modal.querySelector(".close-modal").addEventListener("click", () => closeModal(modal));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAllModals();
    if (e.key === "/" && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
      e.preventDefault();
      $("searchInput").focus();
    }
  });

  setupUpload();
  updateFilterLabels();
  loadPresets().catch((err) => toast(err.message, true));
  loadBrowse().catch((err) => toast(err.message, true));
});
