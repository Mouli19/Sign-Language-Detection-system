const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const predictionLabel = document.querySelector("#predictionLabel");
const confidenceText = document.querySelector("#confidenceText");
const confidenceBar = document.querySelector("#confidenceBar");
const connectionStatus = document.querySelector("#connectionStatus");
const emptyState = document.querySelector("#emptyState");
const fingerGrid = document.querySelector("#fingerGrid");
const signList = document.querySelector("#signList");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let stream = null;
let running = false;
let busy = false;
let timerId = null;

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

async function loadSigns() {
  const response = await fetch("/api/signs");
  const data = await response.json();
  signList.innerHTML = "";
  data.signs.forEach((sign) => {
    const item = document.createElement("span");
    item.textContent = sign;
    signList.appendChild(item);
  });
}

function setStatus(text, live = false) {
  connectionStatus.textContent = text;
  connectionStatus.classList.toggle("live", live);
}

function updatePrediction(result) {
  const confidence = Math.round((result.confidence || 0) * 100);
  predictionLabel.textContent = result.label || "Unknown";
  confidenceText.textContent = `Confidence ${confidence}%`;
  confidenceBar.style.width = `${confidence}%`;
  drawLandmarks(result.landmarks || []);
  updateFingerState(result.fingerState || {});
}

function updateFingerState(state) {
  const labels = ["thumb", "index", "middle", "ring", "pinky"];
  fingerGrid.innerHTML = "";
  labels.forEach((name) => {
    const label = document.createElement("span");
    const value = document.createElement("strong");
    const isOn = Boolean(state[name]);
    label.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    value.textContent = isOn ? "Open" : "Closed";
    value.classList.toggle("on", isOn);
    fingerGrid.append(label, value);
  });
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));
}

function drawLandmarks(landmarks) {
  resizeOverlay();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!landmarks.length) return;

  ctx.save();
  ctx.translate(overlay.width, 0);
  ctx.scale(-1, 1);

  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(69, 212, 131, 0.88)";
  ctx.fillStyle = "#ffffff";

  connections.forEach(([from, to]) => {
    const first = landmarks[from];
    const second = landmarks[to];
    if (!first || !second) return;
    ctx.beginPath();
    ctx.moveTo(first.x * overlay.width, first.y * overlay.height);
    ctx.lineTo(second.x * overlay.width, second.y * overlay.height);
    ctx.stroke();
  });

  landmarks.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * overlay.width, point.y * overlay.height, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

async function predictFrame() {
  if (!running || busy || video.readyState < 2) return;
  busy = true;

  const width = 480;
  const ratio = video.videoHeight / video.videoWidth || 0.75;
  captureCanvas.width = width;
  captureCanvas.height = Math.round(width * ratio);
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: captureCanvas.toDataURL("image/jpeg", 0.72),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");
    updatePrediction(result);
    setStatus("Live", true);
  } catch (error) {
    setStatus(error.message);
  } finally {
    busy = false;
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
  running = true;
  emptyState.classList.add("hidden");
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Live", true);
  timerId = window.setInterval(predictFrame, 220);
}

function stopCamera() {
  running = false;
  busy = false;
  window.clearInterval(timerId);
  timerId = null;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  emptyState.classList.remove("hidden");
  startButton.disabled = false;
  stopButton.disabled = true;
  predictionLabel.textContent = "Waiting";
  confidenceText.textContent = "Confidence 0%";
  confidenceBar.style.width = "0%";
  setStatus("Idle");
}

startButton.addEventListener("click", () => {
  startCamera().catch((error) => setStatus(error.message));
});

stopButton.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeOverlay);

loadSigns().catch(() => {
  signList.textContent = "Unable to load signs.";
});
