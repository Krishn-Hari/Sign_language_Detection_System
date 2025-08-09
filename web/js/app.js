const $ = (sel) => document.querySelector(sel);

// Tabs
const tabs = [
  { btn: '#tab-webcam', panel: '#panel-webcam' },
  { btn: '#tab-upload', panel: '#panel-upload' },
  { btn: '#tab-about', panel: '#panel-about' },
];

function switchTab(activeBtnId) {
  tabs.forEach(({ btn, panel }) => {
    const btnEl = $(btn);
    const panelEl = $(panel);
    const isActive = btn === activeBtnId;
    btnEl.classList.toggle('tab-active', isActive);
    panelEl.classList.toggle('hidden', !isActive);
  });
}

$('#tab-webcam').addEventListener('click', () => switchTab('#tab-webcam'));
$('#tab-upload').addEventListener('click', () => switchTab('#tab-upload'));
$('#tab-about').addEventListener('click', () => switchTab('#tab-about'));

// Speech helpers
function speakText(text) {
  if (!window.speechSynthesis) return;
  const uttr = new SpeechSynthesisUtterance(text);
  uttr.rate = 1.0;
  uttr.pitch = 1.0;
  uttr.lang = 'en-US';
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(uttr);
}

// Webcam & detection (existing logic retained)
const video = $('#video');
const canvas = $('#canvas');
const ctx = canvas.getContext('2d');
const aiBadge = $('#ai-badge');
const grid = $('#grid-overlay');
let stream = null;
let aiTimer = null;
let lastSpoken = '';
let lastLabel = '';
let stableStart = 0;
let lastActivityTs = Date.now();

function fitCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      fitCanvas();
    };
  } catch (e) {
    alert('Failed to access webcam: ' + e);
  }
}

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function captureFrame() {
  fitCanvas();
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85);
  });
}

async function predictBlob(blob, updateTargets) {
  const form = new FormData();
  form.append('image', blob, 'frame.jpg');
  const res = await fetch('/predict', { method: 'POST', body: form });
  const data = await res.json();
  updateTargets(data);
}

function getConfThreshold() {
  return parseFloat($('#conf-slider').value) || 0.8;
}

function updatePrediction({ label, confidence, message }) {
  $('#prediction').textContent = label ?? '–';
  $('#confidence').textContent = 'Confidence: ' + (confidence?.toFixed?.(2) ?? '–');
  $('#message').textContent = message ?? '';

  const confTh = getConfThreshold();
  const now = Date.now();

  if (label && confidence >= confTh) {
    if (label !== lastLabel) {
      lastLabel = label;
      stableStart = now;
    }
  } else {
    lastLabel = '';
    stableStart = 0;
  }

  const speakOn = $('#speak-toggle').checked;
  if (speakOn && label && confidence >= confTh && label !== lastSpoken) {
    lastSpoken = label;
    speakText(label);
  }

  const autoAdd = $('#auto-add-toggle').checked;
  if (autoAdd && lastLabel && stableStart && now - stableStart >= 800) {
    addToSentence(lastLabel);
    stableStart = 0;
  }

  lastActivityTs = now;
}

$('#conf-slider').addEventListener('input', () => {
  $('#conf-value').textContent = parseFloat($('#conf-slider').value).toFixed(2);
});

$('#btn-start').addEventListener('click', startWebcam);
$('#btn-stop').addEventListener('click', () => {
  stopAI();
  stopWebcam();
});
$('#btn-capture').addEventListener('click', async () => {
  if (!stream) await startWebcam();
  const blob = await captureFrame();
  await predictBlob(blob, updatePrediction);
});

async function aiLoop() {
  if (!stream) await startWebcam();
  const blob = await captureFrame();
  await predictBlob(blob, updatePrediction);
}

function startAI() {
  if (aiTimer) return;
  aiBadge.classList.remove('hidden');
  $('#btn-ai').textContent = 'Stop AI';
  aiTimer = setInterval(async () => {
    await aiLoop();
    maybeAutoSpeakSentence();
  }, 500);
}

function stopAI() {
  if (!aiTimer) return;
  clearInterval(aiTimer);
  aiTimer = null;
  aiBadge.classList.add('hidden');
  $('#btn-ai').textContent = 'Start AI';
}

$('#btn-ai').addEventListener('click', () => {
  if (aiTimer) stopAI(); else startAI();
});

// Upload flow
const fileInput = $('#file-input');
const preview = $('#preview');

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove('hidden');
});

$('#btn-upload-predict').addEventListener('click', async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    alert('Please choose an image first.');
    return;
  }
  await predictBlob(file, ({ label, confidence, message }) => {
    $('#prediction-upload').textContent = label ?? '–';
    $('#confidence-upload').textContent = 'Confidence: ' + (confidence?.toFixed?.(2) ?? '–');
    $('#message-upload').textContent = message ?? '';
  });
});

// Sentence builder
function currentSentence() { return $('#sentence').textContent || ''; }
function setSentence(text) { $('#sentence').textContent = text; }
function addToSentence(token) { setSentence((currentSentence() + (currentSentence().endsWith(' ') || currentSentence() === '' ? '' : ' ') + token).trimStart()); }

$('#btn-add').addEventListener('click', () => {
  const label = $('#prediction').textContent.trim();
  if (label && label !== '–') addToSentence(label);
});
$('#btn-space').addEventListener('click', () => setSentence(currentSentence() + ' '));
$('#btn-backspace').addEventListener('click', () => setSentence(currentSentence().slice(0, -1)));
$('#btn-clear-sentence').addEventListener('click', () => setSentence(''));
$('#btn-speak-sentence').addEventListener('click', () => speakText(currentSentence()));

function maybeAutoSpeakSentence() {
  const autoSpeakSentence = $('#auto-speak-sentence-toggle').checked;
  if (!autoSpeakSentence) return;
  const now = Date.now();
  if (now - lastActivityTs >= 3000) {
    const text = currentSentence().trim();
    if (text) speakText(text);
    lastActivityTs = now + 1e9;
  }
}

// New utility buttons
$('#btn-grid').addEventListener('click', () => grid.classList.toggle('hidden'));
$('#btn-reset').addEventListener('click', () => {
  stopAI();
  stopWebcam();
  setSentence('');
  $('#prediction').textContent = '–';
  $('#confidence').textContent = 'Confidence: –';
  $('#message').textContent = '';
});

const helpModal = $('#help-modal');
$('#btn-help').addEventListener('click', () => helpModal.classList.remove('hidden'));
$('#help-close').addEventListener('click', () => helpModal.classList.add('hidden'));

$('#btn-copy-sentence').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(currentSentence());
    alert('Sentence copied to clipboard');
  } catch { alert('Copy failed'); }
});

$('#btn-download-sentence').addEventListener('click', () => {
  const blob = new Blob([currentSentence()], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sentence.txt';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// Default to webcam tab
switchTab('#tab-webcam');
