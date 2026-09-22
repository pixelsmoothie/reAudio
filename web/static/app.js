"use strict";

/* reAudio sound browser and player controller.
 * Direct track list rendering, real-time audio playback,
 * search filters, and audio visualizer.
 * Strict constraint: ZERO emojis. */

const $ = (id) => document.getElementById(id);
const audioEl = $("audioElement");

const state = {
  presets: {},
  browseTracks: [],
  currentTrackId: null,
};

const SVG_PLAY = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
const SVG_PAUSE = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
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
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3600);
}

/* ------------------------------------------------------------------ */
/* Filters                                                            */
/* ------------------------------------------------------------------ */

const filterDirty = new Set();
let fractalMode = "max"; // "max" => at most, "min" => at least

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
  const toggleBtn = $("filtersToggle");
  if (toggleBtn) {
    toggleBtn.classList.toggle("is-active", filterDirty.size > 0);
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
  const mc = $("mainContent");
  if (mc && !mc.classList.contains("is-landing")) {
    doSearch();
  }
}

/* ------------------------------------------------------------------ */
/* Track Rows                                                         */
/* ------------------------------------------------------------------ */

function hfdMeta(hfd) {
  if (hfd >= 1.6) return { cls: "chaotic", label: "Rough" };
  if (hfd >= 1.35) return { cls: "textured", label: "Textured" };
  return { cls: "smooth", label: "Smooth" };
}

function rowEl(track, showScore) {
  const row = document.createElement("div");
  row.className = "track-row";
  row.dataset.trackId = track.track_id;

  const hfd = hfdMeta(track.higuchi_fractal_dimension);
  const matchHtml = (showScore && typeof track.match_score === "number")
    ? `<div class="match-bar-mini"><div class="match-bar-fill" style="width:${Math.max(0, Math.min(100, track.match_score))}%"></div></div>
       <span>${track.match_score.toFixed(0)}%</span>`
    : `<span style="color:var(--text-dim)">&mdash;</span>`;

  row.innerHTML = `
    <div class="col col-play">
      <button class="row-play-btn" data-action="play" title="Play">${SVG_PLAY}</button>
    </div>
    <div class="col col-title" title="${esc(track.title)}">${esc(track.title)}</div>
    <div class="col col-artist" title="${esc(track.artist)}">${esc(track.artist)}</div>
    <div class="col col-genre"><span class="genre-tag">${esc(track.primary_genre)}</span></div>
    <div class="col col-bpm mono">${Math.round(track.bpm)}</div>
    <div class="col col-texture"><span class="texture-${hfd.cls}">${hfd.label}</span></div>
    <div class="col col-duration mono">${fmtTime(track.duration)}</div>
    <div class="col col-match mono">${matchHtml}</div>
    <div class="col col-actions">
      <button class="action-chip" data-action="why" title="Why did this match?">Why</button>
      <button class="action-chip" data-action="similar" title="Find similar tracks">Similar</button>
    </div>`;

  row.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    togglePlay(track);
  });

  row.querySelector('[data-action="play"]').addEventListener("click", (e) => {
    e.stopPropagation();
    togglePlay(track);
  });

  const whyBtn = row.querySelector('[data-action="why"]');
  if (track.explanation) {
    whyBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openExplainModal(track);
    });
  } else {
    whyBtn.style.display = "none";
  }

  row.querySelector('[data-action="similar"]').addEventListener("click", (e) => {
    e.stopPropagation();
    loadSimilar(track);
  });

  return row;
}

function showLanding() {
  const mc = $("mainContent");
  if (mc) mc.classList.add("is-landing");
  const ts = $("tracksSection");
  if (ts) ts.classList.add("hidden");
  $("searchInput").value = "";
  $("searchInput").focus();
}

function showResultsView() {
  const mc = $("mainContent");
  if (mc) mc.classList.remove("is-landing");
  const ts = $("tracksSection");
  if (ts) ts.classList.remove("hidden");
}

function renderResults(list, title, showScore) {
  showResultsView();
  $("resultsTitle").textContent = title;
  $("resultsCount").textContent = `${list.length} track${list.length === 1 ? "" : "s"}`;
  const body = $("trackGrid");
  body.innerHTML = "";
  $("emptyState").classList.toggle("hidden", list.length > 0);
  list.forEach((track) => body.appendChild(rowEl(track, showScore)));
  syncPlayIcons();
}

async function loadBrowse() {
  state.browseTracks = await api("/api/tracks");
  renderResults(state.browseTracks, "All Sounds", false);
}

async function doSearch() {
  const query = $("searchInput").value.trim();
  const filters = currentFilters();
  const hasFilters = filterDirty.size > 0;

  if (!query && !hasFilters) {
    showLanding();
    return;
  }

  const body = { query: query || "", top_k: 16, ...filters };
  try {
    const results = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const title = query ? `Results for "${query}"` : "Filtered Sounds";
    renderResults(results, title, !!query);
  } catch (err) {
    toast(err.message, true);
  }
}

async function loadSimilar(track) {
  try {
    const similar = await api(`/api/recommend/${encodeURIComponent(track.track_id)}`);
    renderResults(similar, `Similar to "${track.title}"`, false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    toast(err.message, true);
  }
}

/* ------------------------------------------------------------------ */
/* Presets                                                            */
/* ------------------------------------------------------------------ */

function wirePresetPills() {
  document.querySelectorAll(".preset-pill").forEach((pill) => {
    pill.onclick = () => {
      const q = pill.dataset.query || state.presets[pill.textContent]?.query;
      if (q) {
        $("searchInput").value = q;
        doSearch();
      }
    };
  });
}

async function loadPresets() {
  try {
    state.presets = await api("/api/presets");
    const wrap = $("presetPills");
    if (wrap) {
      wrap.innerHTML = "";
      Object.entries(state.presets).forEach(([name, preset]) => {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "preset-pill";
        pill.title = preset.description;
        pill.textContent = name;
        pill.dataset.query = preset.query;
        wrap.appendChild(pill);
      });
    }
  } catch (_) { /* keep fallback pills */ }
  wirePresetPills();
}

/* ------------------------------------------------------------------ */
/* Playback & Visualizer                                              */
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
  analyser.smoothingTimeConstant = 0.85;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  startVisualizer();
}

function togglePlay(track) {
  const playerBar = $("playerBar");
  if (playerBar) playerBar.classList.remove("hidden");

  if (state.currentTrackId === track.track_id) {
    if (audioEl.paused) audioEl.play();
    else audioEl.pause();
    return;
  }
  state.currentTrackId = track.track_id;
  audioEl.src = `/api/audio/${encodeURIComponent(track.track_id)}`;
  $("npTitle").textContent = track.title;
  $("npArtist").textContent = `${track.artist} · ${track.primary_genre}`;
  audioEl.play().catch((err) => toast(`Playback error: ${err.message}`, true));
}

function syncPlayIcons() {
  const playing = !audioEl.paused && !!state.currentTrackId;

  document.querySelectorAll(".track-row").forEach((row) => {
    const btn = row.querySelector('[data-action="play"]');
    if (!btn) return;
    const isCurrent = row.dataset.trackId === state.currentTrackId;
    row.classList.toggle("is-current", isCurrent);
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

    ctx2d.beginPath();
    ctx2d.strokeStyle = "#c8f24e";
    ctx2d.lineWidth = 1.2 * dpr;
    for (let i = 0; i < timeData.length; i++) {
      const x = (i / (timeData.length - 1)) * w;
      const y = h / 2 + ((timeData[i] - 128) / 128) * h * 0.44;
      if (i === 0) ctx2d.moveTo(x, y);
      else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
  };
  draw();
}

/* ------------------------------------------------------------------ */
/* Modals                                                             */
/* ------------------------------------------------------------------ */

function openModal(modal) { modal.classList.remove("hidden"); }
function closeModal(modal) { modal.classList.add("hidden"); }
function closeAllModals() {
  document.querySelectorAll(".modal-overlay, .modal").forEach(closeModal);
}

function openExplainModal(track) {
  const ex = track.explanation || {};
  $("explainTrack").textContent = `${track.title} — ${track.artist}`;
  $("explainScore").textContent = (typeof ex.score_pct === "number")
    ? `${ex.score_pct.toFixed(0)}% Match` : "";
  $("explainSummary").textContent = ex.summary || "No match diagnostics available.";
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
/* Upload Handling                                                    */
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
    formData.append("artist", $("artistInput").value.trim() || "Unknown Artist");

    submitBtn.disabled = true;
    $("uploadSpinner").classList.remove("hidden");
    $("uploadStatus").textContent = "Analyzing audio features...";

    try {
      const track = await api("/api/upload", { method: "POST", body: formData });
      $("uploadStatus").textContent = "";
      closeAllModals();
      toast(`Added "${track.title}" to library`);
      await loadBrowse();
    } catch (err) {
      $("uploadStatus").textContent = `Failed: ${err.message}`;
      toast(err.message, true);
    } finally {
      submitBtn.disabled = false;
      $("uploadSpinner").classList.add("hidden");
    }
  });
}

/* ------------------------------------------------------------------ */
/* Initialization                                                     */
/* ------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  $("searchBtn").addEventListener("click", doSearch);
  $("searchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  $("filtersToggle").addEventListener("click", () => {
    $("filtersPanel").classList.toggle("hidden");
  });

  let filterDebounce = null;
  for (const id of ["durationSlider", "bpmMin", "bpmMax", "fractalSlider"]) {
    $(id).addEventListener("input", () => {
      filterDirty.add(
        id === "durationSlider" ? "duration"
        : id === "fractalSlider" ? "fractal" : "bpm"
      );
      updateFilterLabels();
      const mc = $("mainContent");
      if (mc && !mc.classList.contains("is-landing")) {
        clearTimeout(filterDebounce);
        filterDebounce = setTimeout(doSearch, 220);
      }
    });
  }

  $("fractalMode").addEventListener("click", (e) => {
    fractalMode = fractalMode === "max" ? "min" : "max";
    e.target.textContent = fractalMode === "max" ? "≤ max" : "≥ min";
    updateFilterLabels();
    const mc = $("mainContent");
    if (mc && !mc.classList.contains("is-landing")) {
      doSearch();
    }
  });

  $("resetFilters").addEventListener("click", resetFilters);

  const applyBtn = $("applyFiltersBtn");
  if (applyBtn) {
    applyBtn.addEventListener("click", () => {
      doSearch();
    });
  }

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

  document.querySelectorAll(".modal-overlay, .modal").forEach((modal) => {
    modal.addEventListener("click", (e) => {
      if (e.target === modal || e.target.classList.contains("close-modal") || e.target.classList.contains("modal-backdrop")) {
        closeModal(modal);
      }
    });
    const closeBtn = modal.querySelector(".close-modal");
    if (closeBtn) closeBtn.addEventListener("click", () => closeModal(modal));
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAllModals();
    if (e.key === "/" && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
      e.preventDefault();
      $("searchInput").focus();
    }
  });

  const emptyClear = $("emptyClearBtn");
  if (emptyClear) emptyClear.addEventListener("click", showLanding);

  const brandLink = $("brandLink");
  if (brandLink) {
    brandLink.addEventListener("click", (e) => {
      e.preventDefault();
      showLanding();
    });
  }

  setupUpload();
  updateFilterLabels();
  loadPresets().catch((err) => toast(err.message, true));
  api("/api/tracks").then((tracks) => { state.browseTracks = tracks; }).catch(() => {});
});
