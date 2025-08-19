
let currentAxisX = null;
let currentAxisY = null;

function dot(vec1, vec2) {
  return vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
}

function sigmoidScaled(x, scale = 30) {
  return 2 * ((1 / (1 + Math.exp(-x * scale))) - 0.5);
}


async function generateAxes() {
  const xPos = document.getElementById("xPos").value;
  const xNeg = document.getElementById("xNeg").value;
  const yPos = document.getElementById("yPos").value;
  const yNeg = document.getElementById("yNeg").value;

  console.log("Button clicked — axis inputs:");
  console.log("X+:", xPos);
  console.log("X−:", xNeg);
  console.log("Y+:", yPos);
  console.log("Y−:", yNeg);

  const loaderContainer = document.getElementById("axisLoadingBar");
  const loader = loaderContainer.querySelector(".loading-bar");

  // Show loading bar
  loaderContainer.style.display = "block";
  loader.style.width = "0%";

  // Animate bar
  setTimeout(() => {
    loader.style.width = "100%";
  }, 100);

  try {
    const response = await fetch("http://localhost:8000/generate-axes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x_pos: xPos,
        x_neg: xNeg,
        y_pos: yPos,
        y_neg: yNeg
      })
    });

    const data = await response.json();


    setTimeout(() => {
      loaderContainer.style.display = "none";
      loader.style.width = "0%";

      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
      });


      drawAxisLabels(xNeg, xPos, yPos, yNeg);
      currentAxisX = data.axis_x;
      currentAxisY = data.axis_y;

    }, 700);

  } catch (error) {
    console.error("Error generating axes:", error);
    loaderContainer.style.display = "none";
    loader.style.width = "0%";
    alert("Something went wrong while generating axes.");
  }
}

async function addSong() {
  const title = document.getElementById("songTitle").value;
  const artist = document.getElementById("artistName").value;
  const color = document.getElementById("songColor").value;

  console.log(color)

  if (!currentAxisX || !currentAxisY) {
    alert("Please generate axes first.");
    return;
  }

  const spinner = document.getElementById("songSpinner");
  spinner.style.display = "block";

  const response = await fetch("http://localhost:8000/embed-song", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      artist
    })
  });

  const data = await response.json();

  if (!data.embedding) {
    alert("No lyrics found or embedding failed.");
    return;
  }

  console.log("Embedding length:", data.embedding.length);
  console.log("Axis X length:", currentAxisX.length);
  console.log("Axis Y length:", currentAxisY.length);

  const x = dot(data.embedding, currentAxisX);
  const y = dot(data.embedding, currentAxisY);

  console.log(sigmoidScaled(x))
  console.log(sigmoidScaled(y))

  spinner.style.display = "none"; 

  plotSong(`${title} — ${artist}`, x, y, color);

  }


function animateText(label, startX, startY, endX, endY, duration = 600) {
  const startTime = performance.now();

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const t = Math.min(elapsed / duration, 1);  // normalize to 0–1

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

  const playlistUrl = document.getElementById('playlistUrl').value;
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
    const response = await fetch("http://localhost:8000/get-playlist-tracks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playlist_url : playlistUrl
        })
      });

    const data = await response.json();
    console.log("Playlist response:", data);

    if (!data.tracks || data.tracks.length === 0) {
      alert("No tracks found in this playlist.");
      spinner.style.display = "none";
      return;
    }

    for (const track of data.tracks) {
      await embedAndPlotSong(track.title, track.artist, color)
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

  // Apply sigmoid scaling
  const x = sigmoidScaled(rawX);
  const y = sigmoidScaled(rawY);

  // Convert to SVG coordinates (normalized -1 → 1 → SVG space)
  const cx = width * (x + 1) / 2;
  const cy = height * (1 - (y + 1) / 2); // flip y axis (top = 0)

  // Draw circle
  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("r", 6);
  circle.setAttribute("fill", color);
  circle.classList.add('song-dot')

  circle.setAttribute("cx", startX);
  circle.setAttribute("cy", startY);
  animateCircle(circle, startX, startY, cx, cy, 600);

  // Label
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.textContent = title;
  label.setAttribute("font-size", "10px");
  label.classList.add('song-label')

  label.setAttribute("x", startX + 8);
  label.setAttribute("y", startY + 4);
  animateText(label, startX + 8, startY + 4, cx + 8, cy + 4, 600);

  // Add to graph
  svg.appendChild(circle);
  svg.appendChild(label);
}

async function clearSongs(){
  const svg = document.getElementById("graph");

  const dots = svg.querySelectorAll(".song-dot");
  const labels = svg.querySelectorAll(".song-label");

  dots.forEach(dot => dot.remove());
  labels.forEach(label => label.remove());

}

async function embedAndPlotSong(title, artist, color) {
  const response = await fetch("http://localhost:8000/embed-song", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, artist })
  });

  const data = await response.json();

  if (!data.embedding) {
    console.warn(`No embedding found for ${title} — ${artist}`);
    return;
  }

  const x = dot(data.embedding, currentAxisX);
  const y = dot(data.embedding, currentAxisY);

  plotSong(`${title} — ${artist}`, x, y, color);
}


function drawAxisLabels(xLeft, xRight, yTop, yBottom) {
  const svg = document.getElementById("graph");

  
  const oldLabels = svg.querySelectorAll(".axis-label");
  oldLabels.forEach(label => label.remove());
  
  

  
  const leftLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  leftLabel.setAttribute("x", -10);               // slightly off left edge
  leftLabel.setAttribute("y", 350);     
  leftLabel.setAttribute("font-size", "28");          // vertical center
  leftLabel.setAttribute("text-anchor", "end");   // align right edge
  leftLabel.setAttribute("dominant-baseline", "middle");
  leftLabel.classList.add("axis-label");
  leftLabel.textContent = xLeft;
  svg.appendChild(leftLabel);

  const rightLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  rightLabel.setAttribute("x", 710);
  rightLabel.setAttribute("y", 350);
  rightLabel.setAttribute("font-size", "28");
  rightLabel.setAttribute("text-anchor", "start");
  rightLabel.setAttribute('dominant-baseline', 'middle')
  rightLabel.classList.add("axis-label");
  rightLabel.textContent = xRight;
  svg.appendChild(rightLabel);


  const topLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  topLabel.setAttribute("x", 350);
  topLabel.setAttribute("y", -10);
  topLabel.setAttribute("font-size", "28");
  topLabel.setAttribute("text-anchor", "middle");
  topLabel.setAttribute("dominant-baseline", "baseline");
  topLabel.classList.add("axis-label");
  topLabel.textContent = yTop;
  svg.appendChild(topLabel);

  const bottomLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  bottomLabel.setAttribute("x", 350);
  bottomLabel.setAttribute("y", 710);
  bottomLabel.setAttribute("font-size", "28");
  bottomLabel.setAttribute("text-anchor", "middle");
  bottomLabel.setAttribute('dominant-baseline', 'hanging');
  bottomLabel.classList.add("axis-label");
  bottomLabel.textContent = yBottom;
  svg.appendChild(bottomLabel);

  clearSongs()
}


window.onload = () => {
  console.log("script.js loaded!");
  document.getElementById("generateBtn").addEventListener("click", generateAxes);
  document.getElementById("addSongBtn").addEventListener("click", addSong);
  document.getElementById('addPlaylistBtn').addEventListener('click', addPlaylist);
  document.getElementById('clearBtn').addEventListener('click', clearSongs)

  const overlay = document.getElementById("tutorialOverlay");
  const openBtn = document.getElementById("tutorialBtn");
  const closeBtn = document.getElementById("closeTutorialBtn");

  openBtn.addEventListener("click", () => {
    overlay.classList.add("active");
  });

  closeBtn.addEventListener("click", () => {
    overlay.classList.remove("active");
  });

  // Optional: click outside to close
  overlay.addEventListener("click", (e) => {
    if (e.target.classList.contains("overlay-backdrop")) {
      overlay.classList.remove("active");
    }
  });
};

// Place near the bottom of script.js
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