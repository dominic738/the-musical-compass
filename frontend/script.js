// Backend API URL - change this when deploying
// const API_URL = "http://localhost:8000";
const API_URL = "https://musical-compass-api-603407497726.us-central1.run.app"; // Use this for production

let currentAxisX = null;
let currentAxisY = null;

function dot(vec1, vec2) {
  return vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
}

function sigmoidScaled(x, scale = 30) {
  return 2 * ((1 / (1 + Math.exp(-x * scale))) - 0.5);
}

function createLoadingModal(xPos, xNeg, yPos, yNeg) {
  const modal = document.createElement('div');
  
  modal.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background-color: rgba(0, 0, 0, 0.6); 
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    display: flex; align-items: center; justify-content: center;
    z-index: 999999; opacity: 0; transition: opacity 0.3s ease;
  `;

  // Helper for rows
  const createRow = (id, label, subLabel = '') => `
    <div class="loading-item" data-id="${id}" style="margin-bottom: 12px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <div style="display: flex; align-items: center;">
          <div class="spinner" style="width: 16px; height: 16px; margin-right: 8px; font-size: 14px; display: flex; align-items: center;">⏳</div>
          <span class="label" style="font-weight: 600; font-size: 14px; color: #333; text-transform: capitalize;">${label}</span>
        </div>
        <span style="font-size: 12px; color: #888;">${subLabel}</span>
      </div>
      <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px; overflow: hidden;">
        <div class="bar" style="height: 100%; width: 0%; background: #667eea; transition: width 0.5s ease;"></div>
      </div>
    </div>
  `;

  modal.innerHTML = `
    <div style="
      background: white; padding: 30px; border-radius: 16px; 
      width: 90%; max-width: 420px; 
      box-shadow: 0 20px 60px rgba(0,0,0,0.3); font-family: sans-serif;
    ">
      <h2 style="margin: 0 0 5px 0; color: #333; font-size: 20px;">🎵 Building Compass</h2>
      <p style="color: #666; margin-bottom: 20px; font-size: 14px;">Fetching phrases & calculating vectors...</p>
      
      <div style="margin-bottom: 20px;">
        ${createRow('word_1', xPos, 'Fetching phrases...')}
        ${createRow('word_2', xNeg, 'Fetching phrases...')}
        ${createRow('word_3', yPos, 'Fetching phrases...')}
        ${createRow('word_4', yNeg, 'Fetching phrases...')}
      </div>

      <hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">

      <div>
        ${createRow('axis_x', 'X-Axis', 'Computing SGD...')}
        ${createRow('axis_y', 'Y-Axis', 'Computing SGD...')}
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  requestAnimationFrame(() => modal.style.opacity = '1');
  return modal;
}

function updateItemStatus(modal, id, status) {
  const row = modal.querySelector(`[data-id="${id}"]`);
  if (!row) return;

  const spinner = row.querySelector('.spinner');
  const bar = row.querySelector('.bar');
  const label = row.querySelector('.label');

  if (status === 'loading') {
    spinner.innerHTML = '⏳';
    bar.style.width = '90%'; 
    bar.style.backgroundColor = '#667eea'; // Original Purple
    bar.style.transition = 'width 2s ease-in-out'; 
  } 
  else if (status === 'complete') {
    spinner.innerHTML = '✅';
    label.style.color = '#4ade80'; // Original Mint Green
    
    bar.style.transition = 'width 0.3s ease'; 
    bar.style.width = '100%';
    bar.style.backgroundColor = '#4ade80'; // Original Mint Green
  } 
  else if (status === 'error') {
    spinner.innerHTML = '❌';
    label.style.color = '#ef4444';
    bar.style.width = '100%';
    bar.style.backgroundColor = '#ef4444'; 
  }
}

function updateOverallProgress(modal, completed, total) {
  const bar = modal.querySelector('.loading-overall-bar');
  const percentage = (completed / total) * 100;
  bar.style.width = `${percentage}%`;
}

function closeLoadingModal(modal) {
  modal.classList.remove('active');
  setTimeout(() => modal.remove(), 300);
}

async function generateAxes() {
  const xPos = document.getElementById("xPos").value.trim();
  const xNeg = document.getElementById("xNeg").value.trim();
  const yPos = document.getElementById("yPos").value.trim();
  const yNeg = document.getElementById("yNeg").value.trim();

  if (!xPos || !xNeg || !yPos || !yNeg) {
    alert("Please fill in all axis labels!");
    return;
  }

  const modal = createLoadingModal(xPos, xNeg, yPos, yNeg);

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

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();

    // PHASE 2: Axes
    ['word_1', 'word_2', 'word_3', 'word_4'].forEach(id => updateItemStatus(modal, id, 'complete'));

    updateItemStatus(modal, 'axis_x', 'loading');
    updateItemStatus(modal, 'axis_y', 'loading');
    
    await new Promise(r => setTimeout(r, 600)); 
    updateItemStatus(modal, 'axis_x', 'complete');
    updateItemStatus(modal, 'axis_y', 'complete');

    await new Promise(resolve => setTimeout(resolve, 800));
    
    modal.style.opacity = '0';
    setTimeout(() => modal.remove(), 300);

    // Safe Confetti (Run if available, ignore if not)
    try {
      confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
    } catch (e) { console.warn("Confetti not loaded"); }

    // Draw Labels & Update Global Data
    drawAxisLabels(xNeg, xPos, yPos, yNeg);
    currentAxisX = data.axis_x;
    currentAxisY = data.axis_y;

  } catch (error) {
    console.error("Error generating axes:", error);
    ['word_1', 'word_2', 'word_3', 'word_4', 'axis_x', 'axis_y'].forEach(id => {
      updateItemStatus(modal, id, 'error');
    });
    setTimeout(() => {
      modal.remove();
      alert("Something went wrong. Please try again.");
    }, 2000);
  }
}

async function addSong() {
  const title = document.getElementById("songTitle").value.trim();
  const artist = document.getElementById("artistName").value.trim();
  const color = document.getElementById("songColor").value;

  if (!title || !artist) {
    alert("Please enter both song title and artist.");
    return;
  }

  if (!currentAxisX || !currentAxisY) {
    alert("Please generate axes first.");
    return;
  }

  const spinner = document.getElementById("songSpinner");
  spinner.style.display = "block";

  try {
    const checkRes = await fetch(`${API_URL}/check-song`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, artist })
    });

    const checkData = await checkRes.json();


    if (checkData.found) {
      const x = dot(checkData.embedding, currentAxisX);
      const y = dot(checkData.embedding, currentAxisY);
      plotSong(`${title} — ${artist}`, x, y, color);
      return;
    }


    openLyricsModal({ title, artist, color });

  } catch (err) {
    console.error(err);
    alert("Failed to add song.");
  } finally {
    spinner.style.display = "none";
  }
}

function openLyricsModal({ title, artist, color }) {
  const modal = document.createElement("div");

  modal.style.cssText = `
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(6px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999999;
  `;

  modal.innerHTML = `
    <div style="
      background: white;
      padding: 28px;
      border-radius: 16px;
      width: 90%;
      max-width: 520px;
      font-family: sans-serif;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    ">
      <h2 style="margin: 0 0 8px;">🎵 Lyrics Needed</h2>
      <p style="color: #555; font-size: 14px; margin-bottom: 14px;">
        I don’t have <strong>${title} — ${artist}</strong> in my database yet.<br>
        Would you mind pasting the lyrics below?
      </p>

      <textarea id="lyricsModalInput" placeholder="Paste lyrics here..."
        style="
          width: 100%;
          height: 180px;
          padding: 12px;
          font-size: 13px;
          border-radius: 8px;
          border: 1px solid #ddd;
          resize: vertical;
        "></textarea>

      <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 14px;">
        <button id="cancelLyricsBtn">Cancel</button>
        <button id="submitLyricsBtn"
          style="background:#667eea;color:white;border:none;padding:8px 14px;border-radius:8px;">
          Submit
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  modal.querySelector("#cancelLyricsBtn").onclick = () => modal.remove();

  modal.querySelector("#submitLyricsBtn").onclick = async () => {
    const lyrics = modal.querySelector("#lyricsModalInput").value.trim();
    if (!lyrics) {
      alert("Please paste lyrics.");
      return;
    }

    modal.remove();
    await embedAndPlotWithLyrics({ title, artist, lyrics, color });
  };
}



function animateText(label, startX, startY, endX, endY, duration = 600) {
  const startTime = performance.now();

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const t = Math.min(elapsed / duration, 1);

    const x = startX + (endX - startX) * t;
    const y = startY + (endY - startY) * t;

    label.setAttribute("x", x);
    label.setAttribute("y", y);

    if (t < 1) {
      requestAnimationFrame(step);
    }
  }

  requestAnimationFrame(step);
}

async function addPlaylist() {
  const playlistUrl = document.getElementById('playlistUrl').value.trim();
  const color = document.getElementById('playlistColor').value;

  if (!currentAxisX || !currentAxisY) {
    alert("Please generate axes first.");
    return;
  }

  if (!playlistUrl) {
    alert("Please paste a Spotify playlist URL.");
    return;
  }

  const spinner = document.getElementById("playlistSpinner");
  spinner.style.display = "block";

  try {
    const response = await fetch(`${API_URL}/get-playlist-tracks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playlist_url: playlistUrl })
    });

    const data = await response.json();

    if (!data.tracks || data.tracks.length === 0) {
      alert("No tracks found in this playlist.");
      spinner.style.display = "none";
      return;
    }

    for (const track of data.tracks) {
      await embedAndPlotSong(track.title, track.artist, color);
    }
  } catch (err) {
    console.error("Error adding playlist:", err);
    alert("Something went wrong while adding the playlist. Please try again.");
  } finally {
    spinner.style.display = "none";
  }
}

function animateCircle(circle, startX, startY, endX, endY, duration = 600) {
  const startTime = performance.now();

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const t = Math.min(elapsed / duration, 1);

    const x = startX + (endX - startX) * t;
    const y = startY + (endY - startY) * t;

    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);

    if (t < 1) {
      requestAnimationFrame(step);
    }
  }

  requestAnimationFrame(step);
}

function plotSong(title, rawX, rawY, color, startX = 350, startY = 350) {
  const svg = document.getElementById("graph");
  const width = svg.clientWidth;
  const height = svg.clientHeight;

  const x = sigmoidScaled(rawX);
  const y = sigmoidScaled(rawY);

  const cx = width * (x + 1) / 2;
  const cy = height * (1 - (y + 1) / 2);

  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("r", 6);
  circle.setAttribute("fill", color);
  circle.classList.add('song-dot');

  circle.setAttribute("cx", startX);
  circle.setAttribute("cy", startY);
  animateCircle(circle, startX, startY, cx, cy, 600);

  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.textContent = title;
  label.setAttribute("font-size", "10px");
  label.classList.add('song-label');

  label.setAttribute("x", startX + 8);
  label.setAttribute("y", startY + 4);
  animateText(label, startX + 8, startY + 4, cx + 8, cy + 4, 600);

  svg.appendChild(circle);
  svg.appendChild(label);
}

async function clearSongs() {
  const svg = document.getElementById("graph");
  const dots = svg.querySelectorAll(".song-dot");
  const labels = svg.querySelectorAll(".song-label");

  dots.forEach(dot => dot.remove());
  labels.forEach(label => label.remove());
}

async function embedAndPlotWithLyrics({ title, artist, lyrics, color }) {
  try {
    const embedRes = await fetch(`${API_URL}/embed-lyrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, artist, lyrics })
    });

    const embedData = await embedRes.json();
    if (!embedData.embedding) {
      alert("Failed to embed lyrics.");
      return;
    }

    const x = dot(embedData.embedding, currentAxisX);
    const y = dot(embedData.embedding, currentAxisY);

    plotSong(`${title} — ${artist}`, x, y, color);

  } catch (e) {
    console.error(e);
    alert("Embedding failed.");
  }
}


function drawAxisLabels(xLeft, xRight, yTop, yBottom) {
  const svg = document.getElementById("graph");

  const oldLabels = svg.querySelectorAll(".axis-label");
  oldLabels.forEach(label => label.remove());

  const createLabel = (text, x, y, anchor, baseline) => {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x);
    label.setAttribute("y", y);
    label.setAttribute("font-size", "28"); 
    label.setAttribute("font-weight", "600");
    label.setAttribute("text-anchor", anchor);
    label.setAttribute("dominant-baseline", baseline);
    label.setAttribute("fill", "#667eea"); 
    label.classList.add("axis-label");
    label.textContent = text;
    svg.appendChild(label);
  };

  createLabel(xLeft, -25, 350, "end", "middle");       
  createLabel(xRight, 725, 350, "start", "middle");    
  createLabel(yTop, 350, -15, "middle", "baseline");   
  createLabel(yBottom, 350, 710, "middle", "hanging"); 

  clearSongs();
}

window.onload = () => {
  console.log("script.js loaded!");
  document.getElementById("generateBtn").addEventListener("click", generateAxes);
  document.getElementById("addSongBtn").addEventListener("click", addSong);
  //document.getElementById('addPlaylistBtn').addEventListener('click', addPlaylist);
  document.getElementById('clearBtn').addEventListener('click', clearSongs);

  const overlay = document.getElementById("tutorialOverlay");
  const openBtn = document.getElementById("tutorialBtn");
  const closeBtn = document.getElementById("closeTutorialBtn");

  openBtn.addEventListener("click", () => {
    overlay.classList.add("active");
  });

  closeBtn.addEventListener("click", () => {
    overlay.classList.remove("active");
  });

  overlay.addEventListener("click", (e) => {
    if (e.target.classList.contains("overlay-backdrop")) {
      overlay.classList.remove("active");
    }
  });
};

// Mouse repel effect for labels
const svg = document.getElementById("graph");

svg.addEventListener("mousemove", (e) => {
  const cursor = { x: e.clientX, y: e.clientY };

  document.querySelectorAll(".song-label").forEach(label => {
    const bbox = label.getBoundingClientRect();
    const labelCenter = {
      x: bbox.left + bbox.width / 2,
      y: bbox.top + bbox.height / 2
    };

    const dx = labelCenter.x - cursor.x;
    const dy = labelCenter.y - cursor.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    const repelRadius = 80;

    if (distance < repelRadius) {
      const repelStrength = (repelRadius - distance) / repelRadius;
      const offsetX = (dx / distance) * repelStrength * 10;
      const offsetY = (dy / distance) * repelStrength * 10;
      label.setAttribute("transform", `translate(${offsetX}, ${offsetY})`);
    } else {
      label.setAttribute("transform", "translate(0, 0)");
    }
  });
});