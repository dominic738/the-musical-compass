// Backend API URL - change this when deploying
// const API_URL = "http://localhost:8000";
const API_URL = "https://musical-compass-backend-v2-390523133390.us-central1.run.app"; // Use this for production

// Plot square in SVG user units; matches the viewBox in index.html
const GRAPH_SIZE = 700;
const SVG_NS = "http://www.w3.org/2000/svg";

let currentAxisX = null;
let currentAxisY = null;
let currentOffsetX = 0;
let currentOffsetY = 0;

function dot(vec1, vec2) {
  return vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
}

// Position of a song embedding along the current axes, centred so 0 sits
// midway between the two words of each axis.
function projectEmbedding(embedding) {
  // Older cached embeddings aren't unit length; normalise so all songs share a scale
  const norm = Math.sqrt(dot(embedding, embedding)) || 1;
  return {
    x: dot(embedding, currentAxisX) / norm + currentOffsetX,
    y: dot(embedding, currentAxisY) / norm + currentOffsetY,
  };
}

// Pull FastAPI's error detail out of a failed response, if there is one
async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (e) { /* not JSON */ }
  return `HTTP error! status: ${response.status}`;
}

function sigmoidScaled(x, scale = 30) {
  return 2 * ((1 / (1 + Math.exp(-x * scale))) - 0.5);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// "very happy" -> "Very Happy"
function titleCase(s) {
  return s.replace(/(^|\s)(\p{L})/gu, (m, sp, ch) => sp + ch.toUpperCase());
}

// Political-compass score: -10 to +10, real minus sign
function formatScore(v, digits = 2) {
  const n = Math.abs(v).toFixed(digits);
  return v < 0 ? `−${n}` : `+${n}`;
}

const CLOSE_ICON = '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3.5l9 9M12.5 3.5l-9 9" /></svg>';

// --- UI helpers ---------------------------------------------------------------

function toast(message, type = "info", duration = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 200);
  }, duration);
}

// Builds a dialog using the shared .modal styles. Returns { root, close }.
function openModal(innerHtml, { dismissable = true, onClose } = {}) {
  const root = document.createElement("div");
  root.className = "modal-root active";
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-modal", "true");
  root.innerHTML = `<div class="modal-backdrop"></div><div class="modal">${innerHtml}</div>`;

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKey);
    root.classList.add("closing");
    setTimeout(() => root.remove(), 150);
    if (onClose) onClose();
  };
  const onKey = (e) => { if (e.key === "Escape" && dismissable) close(); };

  if (dismissable) {
    root.querySelector(".modal-backdrop").addEventListener("click", close);
    document.addEventListener("keydown", onKey);
  }
  document.body.appendChild(root);
  return { root, close };
}

function setButtonLoading(btn, loading, loadingText) {
  if (loading) {
    btn.dataset.label = btn.textContent;
    btn.textContent = loadingText;
    btn.classList.add("is-loading");
    btn.disabled = true;
  } else {
    btn.textContent = btn.dataset.label || btn.textContent;
    btn.classList.remove("is-loading");
    btn.disabled = false;
  }
}

// --- Building the compass -----------------------------------------------------

// Progress is drawn on the chart itself, so the compass visibly builds in place
function createBuildProgress(xPos, xNeg, yPos, yNeg) {
  const row = (id, name, status) => `
    <div class="progress-row" data-id="${id}" data-state="idle">
      <span class="progress-icon"></span>
      <span class="progress-name">${escapeHtml(name)}</span>
      <span class="progress-status">${status}</span>
      <div class="progress-track"><div class="progress-bar"></div></div>
    </div>`;

  const root = document.createElement("div");
  root.className = "build-progress";
  root.innerHTML = `<div class="build-card">
    <h2 class="build-title">Generating your compass</h2>
    <p class="build-note">Writing phrases for each word and finding the line between each pair. New words take up to a minute; words someone has used before are quick.</p>
    <div class="progress-list">
      ${row('word_1', titleCase(xPos), 'Phrases')}
      ${row('word_2', titleCase(xNeg), 'Phrases')}
      ${row('word_3', titleCase(yPos), 'Phrases')}
      ${row('word_4', titleCase(yNeg), 'Phrases')}
      <hr class="progress-divider">
      ${row('axis_x', `${titleCase(xNeg)} / ${titleCase(xPos)}`, 'Axis')}
      ${row('axis_y', `${titleCase(yNeg)} / ${titleCase(yPos)}`, 'Axis')}
    </div>
  </div>`;

  const emptyState = document.getElementById("emptyState");
  emptyState.hidden = true;
  document.querySelector(".compass-wrap").appendChild(root);

  const close = () => {
    root.remove();
    if (!builtWords) emptyState.hidden = false;
  };
  return { root, close };
}

function updateItemStatus(modal, id, status) {
  const row = modal.root.querySelector(`[data-id="${id}"]`);
  if (row) row.dataset.state = status;
}

async function generateAxes() {
  if (document.getElementById("generateBtn").disabled) return;  // already building

  const xPos = document.getElementById("xPos").value.trim();
  const xNeg = document.getElementById("xNeg").value.trim();
  const yPos = document.getElementById("yPos").value.trim();
  const yNeg = document.getElementById("yNeg").value.trim();

  if (!xPos || !xNeg || !yPos || !yNeg) {
    toast("Fill in all four directions first.", "error");
    return;
  }

  if (xPos.toLowerCase() === xNeg.toLowerCase() || yPos.toLowerCase() === yNeg.toLowerCase()) {
    toast("Opposite directions need different words.", "error");
    return;
  }

  const generateBtn = document.getElementById("generateBtn");
  const addBtn = document.getElementById("addSongBtn");
  setButtonLoading(generateBtn, true, "Generating…");
  addBtn.disabled = true;

  const modal = createBuildProgress(xPos, xNeg, yPos, yNeg);

  try {
    // PHASE 1: Words
    updateItemStatus(modal, 'word_1', 'loading');
    setTimeout(() => updateItemStatus(modal, 'word_2', 'loading'), 200);
    setTimeout(() => updateItemStatus(modal, 'word_3', 'loading'), 400);
    setTimeout(() => updateItemStatus(modal, 'word_4', 'loading'), 600);

    // Call API
    const response = await fetch(`${API_URL}/generate-axes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x_pos: xPos, x_neg: xNeg, y_pos: yPos, y_neg: yNeg })
    });

    if (!response.ok) throw new Error(await errorDetail(response));
    const data = await response.json();

    // PHASE 2: Axes
    ['word_1', 'word_2', 'word_3', 'word_4'].forEach(id => updateItemStatus(modal, id, 'complete'));

    updateItemStatus(modal, 'axis_x', 'loading');
    updateItemStatus(modal, 'axis_y', 'loading');

    await new Promise(r => setTimeout(r, 600));
    updateItemStatus(modal, 'axis_x', 'complete');
    updateItemStatus(modal, 'axis_y', 'complete');

    await new Promise(resolve => setTimeout(resolve, 500));
    modal.close();

    // Draw Labels & Update Global Data
    drawAxisLabels(xNeg, xPos, yPos, yNeg);
    currentAxisX = data.axis_x;
    currentAxisY = data.axis_y;
    currentOffsetX = data.offset_x ?? 0;
    currentOffsetY = data.offset_y ?? 0;
    document.getElementById("songTitle").focus();

  } catch (error) {
    console.error("Error generating axes:", error);
    ['word_1', 'word_2', 'word_3', 'word_4', 'axis_x', 'axis_y'].forEach(id => {
      updateItemStatus(modal, id, 'error');
    });
    setTimeout(() => {
      modal.close();
      toast(`Couldn't generate the compass: ${error.message}`, "error", 6000);
    }, 1500);
  } finally {
    setButtonLoading(generateBtn, false);
    addBtn.disabled = false;
  }
}

// --- Adding songs ---------------------------------------------------------------

async function addSong() {
  const title = document.getElementById("songTitle").value.trim();
  const artist = document.getElementById("artistName").value.trim();
  const color = selectedColor;

  if (!title || !artist) {
    toast("Enter both a song title and an artist.", "error");
    return;
  }

  if (!currentAxisX || !currentAxisY) {
    toast("Generate a compass first (step 1).", "error");
    return;
  }

  const addBtn = document.getElementById("addSongBtn");
  setButtonLoading(addBtn, true, "Finding lyrics…");

  try {
    const checkRes = await fetch(`${API_URL}/check-song`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, artist })
    });

    if (!checkRes.ok) throw new Error(await errorDetail(checkRes));
    const checkData = await checkRes.json();

    if (checkData.found) {
      const { x, y } = projectEmbedding(checkData.embedding);
      plotSong(title, artist, x, y, color);
      clearSongInputs();
      return;
    }

    // Not cached yet: try LRCLIB before asking the user to paste lyrics
    const lyrics = await fetchLyricsFromLrclib(title, artist);
    if (lyrics) {
      addBtn.textContent = "Reading lyrics…";
      if (await embedAndPlotWithLyrics({ title, artist, lyrics, color })) clearSongInputs();
      return;
    }

    openLyricsModal({ title, artist, color });

  } catch (err) {
    console.error(err);
    toast(`Couldn't add that song: ${err.message}`, "error");
  } finally {
    setButtonLoading(addBtn, false);
  }
}

function clearSongInputs() {
  document.getElementById("songTitle").value = "";
  document.getElementById("artistName").value = "";
  document.getElementById("songTitle").focus();
}

// --- LRCLIB lyrics lookup ---------------------------------------------------
// Runs in the browser (LRCLIB allows CORS) so the backend never fetches lyrics.

const LRCLIB_URL = "https://lrclib.net/api/search";
const LRCLIB_TIMEOUT_MS = 8000;
// Alternate recordings we shouldn't silently substitute for the original
const ALT_VERSION_RE = /\b(live|acoustic|remix|demo|instrumental|karaoke|cover|unplugged|mix)\b/i;

// Lowercase, strip accents and punctuation; keeps letters from any script
function normalizeName(s) {
  return s.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

// Title without "(Remastered 2011)", "[Bonus]" or " - Single Version" style suffixes
function baseTitle(s) {
  return normalizeName(s.replace(/\s*[\(\[].*?[\)\]]/g, "").replace(/\s+-\s+.*$/, ""));
}

function pickLrclibMatch(results, title, artist) {
  const wantArtist = normalizeName(artist);
  const candidates = results.filter(r => {
    if (r.instrumental || !r.plainLyrics) return false;
    const gotArtist = normalizeName(r.artistName || "");
    // Allow "Artist feat. Someone" style credits
    return gotArtist === wantArtist || gotArtist.startsWith(wantArtist + " ");
  });

  const exact = candidates.find(r => normalizeName(r.trackName || "") === normalizeName(title));
  if (exact) return exact;

  // Looser match only if it isn't a live/acoustic/etc. version the user didn't ask for
  const userWantsAltVersion = ALT_VERSION_RE.test(title);
  return candidates.find(r =>
    baseTitle(r.trackName || "") === baseTitle(title) &&
    (userWantsAltVersion || !ALT_VERSION_RE.test(r.trackName || ""))
  ) || null;
}

// Returns plain lyrics text, or null if there's no confident match or LRCLIB is unreachable
async function fetchLyricsFromLrclib(title, artist) {
  try {
    const params = new URLSearchParams({ track_name: title, artist_name: artist });
    const res = await fetch(`${LRCLIB_URL}?${params}`, {
      // LRCLIB asks clients to identify themselves; browsers can't set User-Agent
      headers: { "Lrclib-Client": "MusicalCompass (https://mymusicalcompass.com)" },
      signal: AbortSignal.timeout(LRCLIB_TIMEOUT_MS),
    });
    if (!res.ok) return null;

    const match = pickLrclibMatch(await res.json(), title, artist);
    if (!match) return null;

    console.log(`Lyrics from LRCLIB: ${match.trackName} — ${match.artistName} (id ${match.id})`);
    return match.plainLyrics;
  } catch (e) {
    console.warn("LRCLIB lookup failed:", e);
    return null;
  }
}

function openLyricsModal({ title, artist, color }) {
  const modal = openModal(`
    <button class="modal-close" aria-label="Close">${CLOSE_ICON}</button>
    <h2 class="modal-title">Paste the lyrics</h2>
    <p>No lyrics found for <strong>${escapeHtml(title)}</strong> by <strong>${escapeHtml(artist)}</strong>. Paste them and the song goes on the compass. Only the result is saved, never the lyrics.</p>
    <textarea id="lyricsModalInput" placeholder="Paste the full lyrics here…"></textarea>
    <div class="modal-actions">
      <button class="btn btn-quiet" data-action="cancel">Cancel</button>
      <button class="btn" data-action="submit">Place song</button>
    </div>
  `);

  const textarea = modal.root.querySelector("#lyricsModalInput");
  textarea.focus();

  modal.root.querySelector(".modal-close").onclick = modal.close;
  modal.root.querySelector('[data-action="cancel"]').onclick = modal.close;

  const submitBtn = modal.root.querySelector('[data-action="submit"]');
  submitBtn.onclick = async () => {
    const lyrics = textarea.value.trim();
    if (!lyrics) {
      textarea.focus();
      toast("Paste the lyrics first.", "error");
      return;
    }

    setButtonLoading(submitBtn, true, "Reading lyrics…");
    if (await embedAndPlotWithLyrics({ title, artist, lyrics, color })) {
      modal.close();
      clearSongInputs();
    } else {
      setButtonLoading(submitBtn, false);
    }
  };
}

async function embedAndPlotWithLyrics({ title, artist, lyrics, color }) {
  try {
    const embedRes = await fetch(`${API_URL}/embed-lyrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, artist, lyrics })
    });

    if (!embedRes.ok) {
      toast(`Couldn't read those lyrics: ${await errorDetail(embedRes)}`, "error", 6000);
      return false;
    }
    const embedData = await embedRes.json();

    const { x, y } = projectEmbedding(embedData.embedding);

    plotSong(title, artist, x, y, color);
    return true;

  } catch (e) {
    console.error(e);
    toast("Something went wrong reading the lyrics. Please try again.", "error");
    return false;
  }
}

// --- The compass ------------------------------------------------------------------

// Songs on the map: { id, title, artist, color, score: {x, y}, circle, label }
let songs = [];
let nextSongId = 1;
let selectedId = null;

// Words the current axes were built from; pole labels look "pending" until they match
let builtWords = null;

const QUADRANT_COLORS = { tl: "#FF7575", tr: "#42AAFF", bl: "#9AED97", br: "#C09AEC" };

// "very happy" -> "VeryHappy", for meme-style quadrant flairs
function flairWord(s) {
  return titleCase(s).replace(/\s+/g, "");
}

function quadrantOf(score) {
  return (score.y >= 0 ? "t" : "b") + (score.x >= 0 ? "r" : "l");
}

function flairName(q) {
  if (!builtWords) return "";
  const vertical = q[0] === "t" ? builtWords.yPos : builtWords.yNeg;
  const horizontal = q[1] === "r" ? builtWords.xPos : builtWords.xNeg;
  return flairWord(vertical) + flairWord(horizontal);
}

// Dot "drops" from the centre to its place: exponential ease-out
function animatePlacement(circle, label, from, to, labelDx, labelDy = 0, onDone = null, duration = 700) {
  let start = null;
  const ease = t => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) duration = 1;

  function step(now) {
    // Time from the first frame: rAF timestamps can be earlier than performance.now()
    if (start === null) start = now;
    const t = Math.min((now - start) / duration, 1);
    const k = ease(t);
    const x = from.x + (to.x - from.x) * k;
    const y = from.y + (to.y - from.y) * k;
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    label.setAttribute("x", x + labelDx);
    label.setAttribute("y", y + labelDy * k);
    if (t < 1) requestAnimationFrame(step);
    else if (onDone) onDone();
  }
  requestAnimationFrame(step);
}

function plotSong(title, artist, rawX, rawY, color) {
  const layer = document.getElementById("songLayer");
  const nx = sigmoidScaled(rawX);
  const ny = sigmoidScaled(rawY);

  const cx = GRAPH_SIZE * (nx + 1) / 2;
  const cy = GRAPH_SIZE * (1 - (ny + 1) / 2);
  const id = nextSongId++;

  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("r", 7);
  circle.setAttribute("fill", color);
  circle.classList.add("song-dot");
  circle.dataset.songId = id;
  circle.addEventListener("mouseenter", () => previewSong(id));
  circle.addEventListener("mouseleave", () => previewSong(null));
  circle.addEventListener("click", () => selectSong(id));

  const tooltip = document.createElementNS(SVG_NS, "title");
  tooltip.textContent = `${title} — ${artist}`;
  circle.appendChild(tooltip);

  const label = document.createElementNS(SVG_NS, "text");
  label.textContent = title;
  label.classList.add("song-label");
  label.setAttribute("dominant-baseline", "middle");

  layer.appendChild(label);
  layer.appendChild(circle);

  // Keep labels inside the plot: flip to the left of the dot only when the
  // label would otherwise run past the right edge
  const gap = 12 * currentScale;
  const width = label.getComputedTextLength();
  const flip = cx + gap + width > GRAPH_SIZE - 6;
  label.setAttribute("text-anchor", flip ? "end" : "start");

  // Nudge the label up or down until it clears the labels already placed
  const lineH = 15 * currentScale;
  const left = flip ? cx - gap - width : cx + gap;
  const box = dy => ({ x1: left, x2: left + width, y1: cy + dy - lineH / 2, y2: cy + dy + lineH / 2 });
  const overlaps = (b, o) => b.x1 < o.x2 && b.x2 > o.x1 && b.y1 < o.y2 && b.y2 > o.y1;
  // Clear other labels and other dots (including this one)
  const dotR = 9 * currentScale;
  const dotBox = (x, y) => ({ x1: x - dotR, x2: x + dotR, y1: y - dotR, y2: y + dotR });
  const hits = b => overlaps(b, dotBox(cx, cy)) || songs.some(s =>
    overlaps(b, s.labelBox) || overlaps(b, dotBox(s.dot.x, s.dot.y)));
  let labelDy = 0;
  for (const step of [0, 1, -1, 2, -2, 3, -3]) {
    const dy = step * lineH;
    const b = box(dy);
    if (b.y1 < 0 || b.y2 > GRAPH_SIZE) continue;
    if (!hits(b)) { labelDy = dy; break; }
  }
  const labelBox = box(labelDy);

  // On small screens the form sits below the chart: bring the chart into view for the drop
  if (window.matchMedia("(max-width: 900px)").matches) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.querySelector(".compass-wrap").scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  }

  // A label nudged off its dot gets a hairline leader back to it
  let leader = null;
  if (labelDy !== 0) {
    leader = document.createElementNS(SVG_NS, "line");
    leader.classList.add("song-leader");
    leader.setAttribute("x1", cx);
    leader.setAttribute("y1", cy);
    leader.setAttribute("x2", flip ? labelBox.x2 + 2 * currentScale : labelBox.x1 - 2 * currentScale);
    leader.setAttribute("y2", cy + labelDy);
    leader.style.opacity = "0";
    layer.insertBefore(leader, layer.firstChild);
  }

  const centre = { x: GRAPH_SIZE / 2, y: GRAPH_SIZE / 2 };
  animatePlacement(circle, label, centre, { x: cx, y: cy }, flip ? -gap : gap, labelDy,
    () => { if (leader) leader.style.opacity = ""; });

  songs.push({ id, title, artist, color, score: { x: nx * 10, y: ny * 10 }, circle, label, labelBox, leader, dot: { x: cx, y: cy } });
  selectSong(id);
}

function selectSong(id) {
  selectedId = id;
  for (const song of songs) {
    const on = song.id === id;
    song.circle.classList.toggle("is-selected", on);
    song.label.classList.toggle("is-selected", on);
    if (on) {
      // Bring the selected song above its neighbours
      song.label.parentNode.appendChild(song.label);
      song.circle.parentNode.appendChild(song.circle);
    }
  }
  renderSongList();
  renderResult(songs.find(s => s.id === id) || null);
}

// Hovering a dot or row shows its result without changing the selection
function previewSong(id) {
  const song = songs.find(s => s.id === id) || songs.find(s => s.id === selectedId) || null;
  renderResult(song);
}

function removeSong(id) {
  const song = songs.find(s => s.id === id);
  if (!song) return;
  song.circle.remove();
  song.label.remove();
  if (song.leader) song.leader.remove();
  songs = songs.filter(s => s.id !== id);
  if (selectedId === id) {
    const last = songs[songs.length - 1];
    selectSong(last ? last.id : null);
  } else {
    renderSongList();
  }
}

function clearSongs() {
  for (const song of songs) {
    song.circle.remove();
    song.label.remove();
    if (song.leader) song.leader.remove();
  }
  songs = [];
  selectedId = null;
  renderSongList();
  renderResult(null);
}

function renderSongList() {
  const list = document.getElementById("songList");
  list.innerHTML = "";

  for (const song of songs) {
    const q = quadrantOf(song.score);
    const li = document.createElement("li");
    li.className = "song-item" + (song.id === selectedId ? " is-selected" : "");
    li.title = flairName(q);
    li.innerHTML = `
      <span class="flair-chip" style="--c:${QUADRANT_COLORS[q]}"></span>
      <span class="song-name"><b>${escapeHtml(song.title)}</b> <span>${escapeHtml(song.artist)}</span></span>
      <span class="song-score">${formatScore(song.score.x, 1)}, ${formatScore(song.score.y, 1)}</span>
      <button class="song-remove" aria-label="Remove ${escapeHtml(song.title)}">${CLOSE_ICON}</button>
    `;
    li.addEventListener("click", () => selectSong(song.id));
    li.addEventListener("mouseenter", () => previewSong(song.id));
    li.addEventListener("mouseleave", () => previewSong(null));
    li.querySelector(".song-remove").addEventListener("click", (e) => {
      e.stopPropagation();
      removeSong(song.id);
    });
    list.appendChild(li);
  }

  document.getElementById("songCount").textContent = songs.length;
  document.getElementById("songListEmpty").hidden = songs.length > 0;
  document.getElementById("clearBtn").disabled = songs.length === 0;
}

// The signature: each song's result, read like the political compass test
function renderResult(song) {
  const body = document.getElementById("resultBody");

  if (!song) {
    const hint = builtWords
      ? "Add a song to see its result."
      : "Your result will appear here.";
    body.innerHTML = `<p class="result-hint">${hint}</p>`;
    return;
  }

  const w = builtWords;
  const q = quadrantOf(song.score);
  body.innerHTML = `
    <p class="result-title">${escapeHtml(song.title)} <span>by ${escapeHtml(song.artist)}</span></p>
    <dl class="result-scores">
      <div><dt>${escapeHtml(titleCase(w.xNeg))} ↔ ${escapeHtml(titleCase(w.xPos))}</dt><dd>${formatScore(song.score.x)}</dd></div>
      <div><dt>${escapeHtml(titleCase(w.yNeg))} ↔ ${escapeHtml(titleCase(w.yPos))}</dt><dd>${formatScore(song.score.y)}</dd></div>
    </dl>
    <p class="result-flair"><span class="flair-chip" style="--c:${QUADRANT_COLORS[q]}"></span>${escapeHtml(flairName(q))}</p>
  `;
}

// --- Axis labels ----------------------------------------------------------------

const AXIS_INPUTS = { xPos: "labelXPos", xNeg: "labelXNeg", yPos: "labelYPos", yNeg: "labelYNeg" };

function syncAxisLabels() {
  for (const [inputId, labelId] of Object.entries(AXIS_INPUTS)) {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    const value = input.value.trim();
    label.textContent = titleCase(value || input.placeholder);
    label.classList.toggle("pending", !builtWords || builtWords[inputId] !== value);
  }
}

function drawFlairs() {
  for (const q of ["tl", "tr", "bl", "br"]) {
    document.getElementById("flair" + q.toUpperCase()).textContent = flairName(q);
  }
}

function drawAxisLabels(xLeft, xRight, yTop, yBottom) {
  builtWords = { xNeg: xLeft, xPos: xRight, yPos: yTop, yNeg: yBottom };
  syncAxisLabels();
  drawFlairs();
  document.getElementById("emptyState").hidden = true;
  document.getElementById("saveBtn").disabled = false;
  clearSongs();
}

// One faint line per unit on the -10..+10 scale, like the political compass chart
function drawGrid() {
  const g = document.getElementById("gridLines");
  const step = GRAPH_SIZE / 20;
  for (let i = 1; i < 20; i++) {
    if (i === 10) continue;  // the axes are drawn separately
    const p = i * step;
    for (const [x1, y1, x2, y2] of [[p, 0, p, GRAPH_SIZE], [0, p, GRAPH_SIZE, p]]) {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1);
      line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      g.appendChild(line);
    }
  }
}

// Dots and labels are sized in SVG units, so scale them up when the chart is drawn small
let currentScale = 1;
function updateScale() {
  const svg = document.getElementById("graph");
  const box = svg.viewBox.baseVal;
  const rendered = svg.getBoundingClientRect().width || box.width;
  currentScale = box.width / rendered;
  svg.style.setProperty("--s", currentScale.toFixed(3));
}

// --- Save image ---------------------------------------------------------------------

const EXPORT_WIDTH = 1600;
const STYLE_PROPS = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-linejoin",
  "paint-order", "font-family", "font-size", "font-weight", "opacity"];

async function embeddedFontCss() {
  try {
    const buf = await (await fetch("assets/fonts/arimo-latin.woff2")).arrayBuffer();
    let bin = "";
    new Uint8Array(buf).forEach(b => { bin += String.fromCharCode(b); });
    return `@font-face{font-family:"Arimo";font-weight:400 700;src:url(data:font/woff2;base64,${btoa(bin)}) format("woff2");}`;
  } catch (e) {
    return "";  // file:// or offline: the export falls back to Arial, same metrics
  }
}

async function saveImage() {
  const svg = document.getElementById("graph");

  // Render at desktop proportions regardless of the current screen size
  const prevScale = svg.style.getPropertyValue("--s");
  svg.style.setProperty("--s", (svg.viewBox.baseVal.width / 800).toFixed(3));

  const clone = svg.cloneNode(true);
  const src = [svg, ...svg.querySelectorAll("*")];
  const dst = [clone, ...clone.querySelectorAll("*")];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el);
    for (const p of STYLE_PROPS) dst[i].style.setProperty(p, cs.getPropertyValue(p));
    if (el.tagName === "circle") dst[i].setAttribute("r", parseFloat(cs.getPropertyValue("r")) || 7);
  });
  svg.style.setProperty("--s", prevScale);

  // Hairlines are screen-pixel strokes on the page; give them user-unit widths here
  clone.querySelectorAll(".grid-lines line").forEach(l => { l.style.strokeWidth = "0.9"; l.style.vectorEffect = "none"; });
  clone.querySelectorAll(".axes line, .frame").forEach(l => { l.style.strokeWidth = "2.5"; l.style.vectorEffect = "none"; });

  const style = document.createElementNS(SVG_NS, "style");
  style.textContent = await embeddedFontCss();
  clone.insertBefore(style, clone.firstChild);

  const box = svg.viewBox.baseVal;
  const height = Math.round(EXPORT_WIDTH * box.height / box.width);
  clone.setAttribute("width", EXPORT_WIDTH);
  clone.setAttribute("height", height);

  const markup = new XMLSerializer().serializeToString(clone);
  const img = new Image();
  img.onload = () => {
    const pad = 56, foot = 70, rowH = 46, headH = 56;
    // The legend carries each song's result, so the image makes sense on its own
    const legendH = songs.length ? headH + songs.length * rowH + 40 : 0;
    const canvas = document.createElement("canvas");
    canvas.width = EXPORT_WIDTH + pad * 2;
    canvas.height = height + pad * 2 + legendH + foot;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, pad, pad, EXPORT_WIDTH, height);

    const FONT = "Arimo, Arial, sans-serif";
    // Truncate text to a pixel width with an ellipsis
    const fit = (text, max) => {
      if (ctx.measureText(text).width <= max) return text;
      while (text.length > 1 && ctx.measureText(text + "…").width > max) text = text.slice(0, -1);
      return text + "…";
    };

    if (songs.length) {
      const w = builtWords;
      const x0 = pad;
      const col = { title: x0 + 40, artist: x0 + 640, sx: x0 + 1170, sy: x0 + 1370, flair: x0 + 1410 };
      let y = pad + height + 40;

      ctx.fillStyle = "#000000";
      ctx.fillRect(x0, y, EXPORT_WIDTH, 2);
      y += 38;
      ctx.font = `700 22px ${FONT}`;
      ctx.fillStyle = "#4D4D4D";
      ctx.textAlign = "left";
      ctx.fillText("Song", col.title, y);
      ctx.fillText("Artist", col.artist, y);
      ctx.fillText("Quadrant", col.flair, y);
      ctx.textAlign = "right";
      ctx.fillText(fit(`${titleCase(w.xNeg)} ↔ ${titleCase(w.xPos)}`, 190), col.sx, y);
      ctx.fillText(fit(`${titleCase(w.yNeg)} ↔ ${titleCase(w.yPos)}`, 190), col.sy, y);
      y += 18;

      for (const song of songs) {
        const q = quadrantOf(song.score);
        const bold = song.id === selectedId ? 700 : 400;
        y += rowH;
        ctx.fillStyle = "#CFCFCF";
        ctx.fillRect(x0, y - rowH + 12, EXPORT_WIDTH, 1);

        ctx.fillStyle = QUADRANT_COLORS[q];
        ctx.fillRect(x0 + 2, y - 22, 22, 22);
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 + 2, y - 22, 22, 22);

        ctx.fillStyle = "#000000";
        ctx.textAlign = "left";
        ctx.font = `700 26px ${FONT}`;
        ctx.fillText(fit(song.title, 580), col.title, y);
        ctx.font = `${bold} 26px ${FONT}`;
        ctx.fillStyle = "#4D4D4D";
        ctx.fillText(fit(song.artist, 480), col.artist, y);
        ctx.fillStyle = "#000000";
        ctx.font = `${bold} 26px ${FONT}`;
        ctx.fillText(fit(flairName(q), EXPORT_WIDTH + x0 - col.flair), col.flair, y);
        ctx.textAlign = "right";
        ctx.font = `700 26px ${FONT}`;
        ctx.fillText(formatScore(song.score.x), col.sx, y);
        ctx.fillText(formatScore(song.score.y), col.sy, y);
      }
    }

    ctx.textAlign = "left";
    ctx.fillStyle = "#000000";
    ctx.font = `700 30px ${FONT}`;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("The Musical Compass", pad, canvas.height - pad);
    ctx.font = `400 26px ${FONT}`;
    ctx.textAlign = "right";
    ctx.fillStyle = "#4D4D4D";
    ctx.fillText("mymusicalcompass.com", canvas.width - pad, canvas.height - pad);

    canvas.toBlob(blob => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "musical-compass.png";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  };
  img.onerror = () => toast("Couldn't create the image. Try a screenshot instead.", "error");
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(markup);
}

// --- Colour picker --------------------------------------------------------------

let selectedColor = "#E10600";

function selectColor(color, fromCustom = false) {
  selectedColor = color;
  document.querySelectorAll(".swatch[data-color]").forEach(btn => {
    btn.setAttribute("aria-pressed", String(!fromCustom && btn.dataset.color.toLowerCase() === color.toLowerCase()));
  });
  const custom = document.querySelector(".swatch-custom");
  custom.classList.toggle("is-set", fromCustom);
  custom.style.setProperty("--c", color);
  custom.setAttribute("aria-pressed", String(fromCustom));
}

// --- Init -------------------------------------------------------------------------

window.onload = () => {
  document.getElementById("generateBtn").addEventListener("click", generateAxes);
  document.getElementById("addSongBtn").addEventListener("click", addSong);
  document.getElementById("clearBtn").addEventListener("click", clearSongs);
  document.getElementById("saveBtn").addEventListener("click", saveImage);

  // Enter submits the relevant step
  for (const id of Object.keys(AXIS_INPUTS)) {
    const input = document.getElementById(id);
    input.addEventListener("input", syncAxisLabels);
    input.addEventListener("keydown", e => { if (e.key === "Enter") generateAxes(); });
  }
  for (const id of ["songTitle", "artistName"]) {
    document.getElementById(id).addEventListener("keydown", e => {
      if (e.key === "Enter" && !document.getElementById("addSongBtn").disabled) addSong();
    });
  }

  document.querySelectorAll(".swatch[data-color]").forEach(btn => {
    btn.addEventListener("click", () => selectColor(btn.dataset.color));
  });
  document.getElementById("customColor").addEventListener("input", e => selectColor(e.target.value, true));
  selectColor(selectedColor);

  const overlay = document.getElementById("tutorialOverlay");
  const closeBtn = document.getElementById("closeTutorialBtn");
  const closeOverlay = () => overlay.classList.remove("active");

  document.getElementById("tutorialBtn").addEventListener("click", () => {
    overlay.classList.add("active");
    closeBtn.focus();
  });
  closeBtn.addEventListener("click", closeOverlay);
  overlay.addEventListener("click", (e) => {
    if (e.target.classList.contains("overlay-backdrop")) closeOverlay();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && overlay.classList.contains("active")) closeOverlay();
  });

  drawGrid();
  syncAxisLabels();
  renderSongList();
  renderResult(null);
  updateScale();
  new ResizeObserver(updateScale).observe(document.getElementById("graph"));
};
