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
  activeView: "search",
  dbData: null,
  currentDbTable: "tracks",
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

  // DSP Metric Breakdown
  if ($("explainCosine")) {
    $("explainCosine").textContent = (typeof track.similarity === "number")
      ? track.similarity.toFixed(4)
      : (typeof ex.score_pct === "number" ? (ex.score_pct / 100).toFixed(4) : "—");
  }
  if ($("explainAlignment")) {
    $("explainAlignment").textContent = (typeof track.acoustic_alignment === "number")
      ? `${Math.round(track.acoustic_alignment * 100)}%`
      : "100%";
  }
  if ($("explainHFD")) {
    $("explainHFD").textContent = (typeof track.higuchi_fractal_dimension === "number")
      ? track.higuchi_fractal_dimension.toFixed(3)
      : "—";
  }
  if ($("explainBPM")) {
    $("explainBPM").textContent = (typeof track.bpm === "number" && track.bpm > 0)
      ? `${Math.round(track.bpm)} BPM`
      : "None";
  }

  const list = $("explainInsights");
  list.innerHTML = "";
  (ex.insights || []).forEach((insight) => {
    const li = document.createElement("li");
    li.textContent = insight;
    list.appendChild(li);
  });
function openVectorModal(trackId, colKey, blobMeta) {
  $("vectorModalTitle").textContent = `${colKey} (${trackId})`;
  $("vectorModalSub").textContent = `${blobMeta.dims}-D Float32 Array • ${blobMeta.bytes} Bytes in SQLite BLOB`;

  const metaBar = $("vectorMetaBar");
  metaBar.innerHTML = `
    <span>Storage Type: <strong>SQLAlchemy LargeBinary</strong></span>
    <span>Dtype: <strong>${blobMeta.dtype}</strong></span>
    <span>Total Dimensions: <strong>${blobMeta.dims}</strong></span>
    <span>Binary Footprint: <strong>${blobMeta.bytes} bytes</strong></span>
  `;

  const dimContainer = $("vectorDimensions");
  dimContainer.innerHTML = "";

  (blobMeta.preview || []).forEach((num, idx) => {
    const item = document.createElement("div");
    item.className = "vec-dim-item";
    item.innerHTML = `<span class="vec-dim-idx">[${idx}]</span><span class="vec-dim-val">${num.toFixed(4)}</span>`;
    dimContainer.appendChild(item);
  });

  const remainingNotice = document.createElement("div");
  remainingNotice.style.gridColumn = "1 / -1";
  remainingNotice.style.color = "var(--text-dim)";
  remainingNotice.style.padding = "8px 0 0";
  remainingNotice.textContent = `... and ${blobMeta.dims - (blobMeta.preview || []).length} more float32 dimensions stored in binary BLOB.`;
  dimContainer.appendChild(remainingNotice);

  openModal($("vectorModal"));
}

/* ------------------------------------------------------------------ */
/* View Switching & Database Inspector                                */
/* ------------------------------------------------------------------ */

function switchView(viewName) {
  state.activeView = viewName;
  const isDb = viewName === "database";

  const btnSearch = $("viewSearchBtn");
  const btnDb = $("viewDatabaseBtn");
  if (btnSearch) btnSearch.classList.toggle("is-active", !isDb);
  if (btnDb) btnDb.classList.toggle("is-active", isDb);

  const searchArea = $("searchArea");
  const tracksSec = $("tracksSection");
  const dbSec = $("databaseSection");
  const mainContent = $("mainContent");

  if (isDb) {
    if (searchArea) searchArea.classList.add("hidden");
    if (tracksSec) tracksSec.classList.add("hidden");
    if (dbSec) dbSec.classList.remove("hidden");
    if (mainContent) mainContent.classList.remove("is-landing");
    loadDatabaseView();
  } else {
    if (dbSec) dbSec.classList.add("hidden");
    if (searchArea) searchArea.classList.remove("hidden");
    showLanding();
  }
}

async function loadDatabaseView() {
  try {
    const data = await api("/api/database/inspect");
    state.dbData = data;

    const total = data.summary?.total_tracks || 0;
    if ($("dbTotalTracks")) $("dbTotalTracks").textContent = total;
    if ($("countTracks")) $("countTracks").textContent = data.tables?.tracks?.length || 0;
    if ($("countFeatures")) $("countFeatures").textContent = data.tables?.audio_features?.length || 0;
    if ($("countSemantic")) $("countSemantic").textContent = data.tables?.semantic_metadata?.length || 0;
    if ($("countJoined")) $("countJoined").textContent = total;

    renderDatabaseTable();
  } catch (err) {
    toast(`Failed to load database: ${err.message}`, true);
  }
}

const DB_TABLE_DESCRIPTIONS = {
  tracks: "Core relational audio tracks (duration, sample rate, file storage path, timestamps)",
  audio_features: "Extracted DSP acoustic features (BPM, Higuchi Fractal Dimension HFD, Spectral Centroid, ZCR)",
  semantic_metadata: "Semantic classification, tags, and 512-D L2-normalized float32 vector BLOBs",
  joined: "Consolidated relational view: audio metadata, DSP features, and vector embeddings combined",
};

function renderDatabaseTable() {
  if (!state.dbData) return;

  const currentTab = state.currentDbTable || "tracks";
  const descEl = $("dbTableDesc");
  if (descEl) descEl.textContent = DB_TABLE_DESCRIPTIONS[currentTab] || "";

  const thead = $("dbTableHead");
  const tbody = $("dbTableBody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const filterText = ($("dbFilterInput")?.value || "").toLowerCase().trim();

  let cols = [];
  let rows = [];

  if (currentTab === "tracks") {
    cols = ["track_id", "title", "artist", "duration (s)", "sample_rate", "bitrate", "channels", "file_path", "created_at"];
    rows = (state.dbData.tables?.tracks || []).map((t) => ({
      track_id: t.track_id,
      title: t.title,
      artist: t.artist,
      "duration (s)": t.duration,
      sample_rate: `${t.sample_rate} Hz`,
      bitrate: t.bitrate,
      channels: t.channels === 1 ? "Mono (1)" : "Stereo (2)",
      file_path: t.file_path,
      created_at: t.created_at ? t.created_at.split("T")[0] : "--",
    }));
  } else if (currentTab === "audio_features") {
    cols = ["track_id", "bpm", "higuchi_fractal_dimension", "spectral_centroid", "rms_energy", "zero_crossing_rate", "spectral_rolloff", "dynamic_range_db", "katz_fractal_dimension", "spectral_fractal_beta"];
    rows = (state.dbData.tables?.audio_features || []).map((f) => ({
      track_id: f.track_id,
      bpm: `${f.bpm} BPM`,
      higuchi_fractal_dimension: f.higuchi_fractal_dimension,
      spectral_centroid: `${f.spectral_centroid} Hz`,
      rms_energy: f.rms_energy,
      zero_crossing_rate: f.zero_crossing_rate,
      spectral_rolloff: `${f.spectral_rolloff} Hz`,
      dynamic_range_db: `${f.dynamic_range_db} dB`,
      katz_fractal_dimension: f.katz_fractal_dimension,
      spectral_fractal_beta: f.spectral_fractal_beta,
    }));
  } else if (currentTab === "semantic_metadata") {
    cols = ["track_id", "primary_genre", "mood", "tags", "audio_embedding", "text_embedding", "generated_description"];
    rows = (state.dbData.tables?.semantic_metadata || []).map((s) => ({
      track_id: s.track_id,
      primary_genre: s.primary_genre,
      mood: s.mood,
      tags: (s.tags || []).join(", "),
      audio_embedding: s.audio_embedding,
      text_embedding: s.text_embedding,
      generated_description: s.generated_description,
    }));
  } else if (currentTab === "joined") {
    cols = ["track_id", "title", "artist", "duration", "bpm", "higuchi (HFD)", "genre", "mood", "audio_embedding"];
    const tMap = Object.fromEntries((state.dbData.tables?.tracks || []).map((t) => [t.track_id, t]));
    const fMap = Object.fromEntries((state.dbData.tables?.audio_features || []).map((f) => [f.track_id, f]));
    const sMap = Object.fromEntries((state.dbData.tables?.semantic_metadata || []).map((s) => [s.track_id, s]));

    rows = Object.keys(tMap).map((id) => {
      const t = tMap[id] || {};
      const f = fMap[id] || {};
      const s = sMap[id] || {};
      return {
        track_id: id,
        title: t.title || "--",
        artist: t.artist || "--",
        duration: `${t.duration || 0}s`,
        bpm: `${f.bpm || 0} BPM`,
        "higuchi (HFD)": f.higuchi_fractal_dimension || "--",
        genre: s.primary_genre || "--",
        mood: s.mood || "--",
        audio_embedding: s.audio_embedding,
      };
    });
  }

  // Filter rows
  if (filterText) {
    rows = rows.filter((r) =>
      Object.values(r).some((v) => {
        if (typeof v === "object" && v !== null) return false;
        return String(v || "").toLowerCase().includes(filterText);
      })
    );
  }

  // Render Head
  const headerTr = document.createElement("tr");
  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    headerTr.appendChild(th);
  });
  thead.appendChild(headerTr);

  // Render Body
  if (rows.length === 0) {
    const emptyTr = document.createElement("tr");
    const emptyTd = document.createElement("td");
    emptyTd.colSpan = cols.length;
    emptyTd.style.textAlign = "center";
    emptyTd.style.padding = "24px";
    emptyTd.textContent = "No database records match the filter query.";
    emptyTr.appendChild(emptyTd);
    tbody.appendChild(emptyTr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    cols.forEach((colKey) => {
      const td = document.createElement("td");
      const val = row[colKey];

      if (colKey === "track_id") {
        td.className = "db-id-cell mono";
        td.textContent = val;
      } else if (colKey.includes("higuchi") || colKey === "bpm") {
        td.className = "db-highlight mono";
        td.textContent = val;
      } else if (typeof val === "object" && val !== null && val.dims) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "blob-btn mono";
        btn.textContent = `BLOB [${val.bytes} B] ${val.dims}-D`;
        btn.title = "Click to inspect raw float32 vector embedding";
        btn.onclick = () => openVectorModal(row.track_id, colKey, val);
        td.appendChild(btn);
      } else {
        td.textContent = val ?? "--";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
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

  // Universal modal close handler
  document.addEventListener("click", (e) => {
    const closeEl = e.target.closest && e.target.closest(".close-modal, .modal-close-btn, [data-action='close-modal']");
    if (closeEl) {
      e.preventDefault();
      e.stopPropagation();
      const modal = closeEl.closest(".modal-overlay, .modal");
      if (modal) closeModal(modal);
      else closeAllModals();
      return;
    }
    if (e.target.classList && (e.target.classList.contains("modal-overlay") || e.target.classList.contains("modal-backdrop"))) {
      e.preventDefault();
      e.stopPropagation();
      const modal = e.target.closest(".modal-overlay, .modal") || e.target;
      closeModal(modal);
    }
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
      switchView("search");
    });
  }

  // View Switcher (Sound Library vs Database Inspector)
  const viewSearchBtn = $("viewSearchBtn");
  const viewDatabaseBtn = $("viewDatabaseBtn");
  if (viewSearchBtn) viewSearchBtn.addEventListener("click", () => switchView("search"));
  if (viewDatabaseBtn) viewDatabaseBtn.addEventListener("click", () => switchView("database"));

  // DB Table Tabs
  document.querySelectorAll(".db-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".db-tab-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.currentDbTable = btn.dataset.table;
      renderDatabaseTable();
    });
  });

  // DB Filter Input
  const dbFilter = $("dbFilterInput");
  if (dbFilter) {
    dbFilter.addEventListener("input", () => {
      renderDatabaseTable();
    });
  }

  setupUpload();
  updateFilterLabels();
  loadPresets().catch((err) => toast(err.message, true));
  api("/api/tracks").then((tracks) => { state.browseTracks = tracks; }).catch(() => {});
});
