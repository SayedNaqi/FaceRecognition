/* ============================================================
   Aurora UI — app.js
   Smart Face Recognition web client (RTL, Persian)
   ============================================================ */

"use strict";

const $ = (id) => document.getElementById(id);
const FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const fa = (n) => String(n).replace(/[0-9]/g, (d) => FA_DIGITS[+d]);

/* ---------------- inline SVG icons ---------------- */
const ICONS = {
  home: '<path d="M3 11 12 3l9 8"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
  add_user: '<circle cx="10" cy="8" r="4"/><path d="M3 20c0-4 3-6 7-6s7 2 7 6"/><path d="M18 8v6M15 11h6"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2 20c0-3.5 3-5.5 7-5.5s7 2 7 5.5"/><circle cx="17" cy="9" r="2.5"/><path d="M15.5 14.6c3 .3 6.5 2 6.5 5.4"/>',
  report: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/>',
  settings: '<path d="M4 8h16M4 16h16"/><circle cx="14" cy="8" r="2.5"/><circle cx="10" cy="16" r="2.5"/>',
  camera: '<rect x="3" y="7" width="18" height="13" rx="3"/><circle cx="12" cy="13.5" r="3.5"/><path d="M8 7l1.5-3h5L16 7"/>',
  play: '<path d="M7 4.5v15l12-7.5z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l5 5"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 20h16"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14"/><path d="M10 11v6M14 11v6"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
  pulse: '<path d="M3 12h5l2-6 4 12 2-6h5"/>',
  check: '<path d="M4 12.5l5 5L20 6.5"/>',
  sparkles: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  faceid: '<path d="M8 4v6M4 8h6" opacity="0"/><path d="M9 5v4M5 9h4" opacity="0"/><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>',
};

function drawFaceIdGlyph(svg) {
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.innerHTML = `
    <path d="M8 6v4M6 8h4"/><path d="M58 6v4M56 8h4"/>
    <path d="M8 58v-4M6 56h4"/><path d="M58 58v-4M56 56h4"/>
    <circle cx="32" cy="32" r="11"/>
    <path d="M27 29v3M37 29v3"/>
    <path d="M32 35v4"/>
    <path d="M26 41c2 2.4 4 3.4 6 3.4s4-1 6-3.4"/>`;
}

document.querySelectorAll("[data-icon]").forEach((el) => {
  const name = el.getAttribute("data-icon");
  if (name === "faceid") { drawFaceIdGlyph(el); return; }
  const path = ICONS[name] || ICONS.home;
  el.outerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
});
drawFaceIdGlyph($("hero-glyph"));

/* ---------------- helpers ---------------- */
async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.status }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function toast(msg, ok = true) {
  const el = document.createElement("div");
  el.className = "toast" + (ok ? "" : " err");
  el.textContent = msg;
  $("toast-host").appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 420);
  }, 2200);
}

function status(text, ok = true) {
  const el = $("st-status");
  el.textContent = "● " + text;
  el.className = ok ? "st-ok" : "";
  el.style.color = ok ? "" : "var(--bad)";
}

function initials(name) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return (parts[0][0] + (parts[1] ? parts[1][0] : "")).toUpperCase();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- navigation ---------------- */
let currentPage = "dashboard";
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
});

function showPage(name) {
  if (currentPage === "dashboard" && name !== "dashboard") stopCamera();
  if (currentPage === "enroll" && name !== "enroll") cancelScan(true);
  currentPage = name;
  document.querySelectorAll(".page").forEach((p) => p.classList.add("hidden"));
  $("page-" + name).classList.remove("hidden");
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.page === name));
  if (name === "people") loadPeople();
  if (name === "report") loadReport();
  if (name === "settings") loadSettings();
  if (name === "dashboard") loadStats();
}

/* ---------------- clock ---------------- */
setInterval(() => {
  $("clock").textContent = new Date().toLocaleString("en-GB",
    { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit",
      minute: "2-digit", second: "2-digit", hour12: false });
}, 1000);

/* ---------------- shared camera stream ---------------- */
let video = null;
let stream = null;
let cameraOn = false;
let rafId = null;
let recogTimer = null;
let fpsHistory = [];
let deviceId = localStorage.getItem("camera_device") || "default";

async function openCamera() {
  if (stream) return true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId === "default" ? undefined : { exact: deviceId },
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    toast("دوربین باز نشد: " + e.message, false);
    status("دوربین در دسترس نیست", false);
    return false;
  }
  if (!video) {
    video = document.createElement("video");
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
  }
  video.srcObject = stream;
  await video.play();
  cameraOn = true;
  $("cam-idle").style.display = "none";
  $("btn-cam").innerHTML = '<span data-icon="stop" style="width:17px;height:17px"></span>توقف دوربین';
  const ic = $("btn-cam").querySelector("svg");
  if (ic) { ic.outerHTML = stopSvg(); }
  status("دوربین روشن است");
  startLoops();
  return true;
}

function stopCamera() {
  cameraOn = false;
  if (rafId) cancelAnimationFrame(rafId), rafId = null;
  if (recogTimer) clearInterval(recogTimer), recogTimer = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  if (video) video.srcObject = null;
  $("cam-idle").style.display = "flex";
  $("btn-cam").textContent = "شروع دوربین";
  const span = document.createElement("span");
  span.innerHTML = stopSvg();
  span.firstChild.outerHTML = playSvg();
  $("btn-cam").prepend(span.firstChild);
  $("faces-count").textContent = "۰";
  $("fps").textContent = "— FPS";
  status("آماده");
}

function stopSvg() {
  return '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
}
function playSvg() {
  return '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4.5v15l12-7.5z"/></svg>';
}

$("btn-cam").addEventListener("click", async () => {
  if (cameraOn) stopCamera();
  else {
    await openCamera();
    if (cameraOn) startRecognition();
  }
});

/* ---------------- frame loops + HUD ---------------- */
function startLoops() {
  const draw = () => {
    if (!cameraOn) return;
    const canvas = currentPage === "enroll" ? $("scan-canvas") : $("dash-canvas");
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
    rafId = requestAnimationFrame(draw);
  };
  rafId = requestAnimationFrame(draw);
}

function drawFrame(ctx, canvas) {
  const vw = video.videoWidth || 960;
  const vh = video.videoHeight || 600;
  const cw = canvas.width, ch = canvas.height;
  // cover-fit
  const scale = Math.max(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  ctx.drawImage(video, (cw - dw) / 2, (ch - dh) / 2, dw, dh);

  // HUD: radar rings
  const t = performance.now() / 1000;
  const cx = cw / 2, cy = ch / 2;
  const pulse = 10 * (1 + Math.sin(t * 2.2)) / 2;
  ctx.strokeStyle = "rgba(96,106,160,0.5)";
  ctx.lineWidth = 1.5;
  for (const rr of [105 + pulse, 165 + pulse]) {
    for (let a = 0; a < 360; a += 45) {
      ctx.beginPath();
      ctx.ellipse(cx, cy, rr, rr * 0.72, 0, (a + 8) * Math.PI / 180,
                  (a + 34) * Math.PI / 180);
      ctx.stroke();
    }
  }
  // corner brackets
  ctx.strokeStyle = "rgba(34,211,238,0.9)";
  ctx.lineWidth = 3;
  const bm = 14, bl = 40;
  for (const [sx, sy] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const px = sx > 0 ? bm : cw - bm, py = sy > 0 ? bm : ch - bm;
    ctx.beginPath();
    ctx.moveTo(px, py); ctx.lineTo(px - sx * bl, py);
    ctx.moveTo(px, py); ctx.lineTo(px, py - sy * bl);
    ctx.stroke();
  }
  // LIVE + time
  ctx.font = "600 20px Vazirmatn, sans-serif";
  ctx.fillStyle = "#22d3ee";
  ctx.fillText("● LIVE", 24, ch - 20);
  ctx.font = "500 17px Vazirmatn, sans-serif";
  ctx.fillStyle = "rgba(180,200,220,0.9)";
  ctx.textAlign = "left";
  ctx.fillText(new Date().toLocaleTimeString("en-GB"), cw - 92, ch - 20);
  ctx.textAlign = "right";
}

function drawMatches(ctx, canvas, matches) {
  const vw = video.videoWidth || 960;
  const vh = video.videoHeight || 600;
  const cw = canvas.width, ch = canvas.height;
  const scale = Math.max(cw / vw, ch / vh);
  const ox = (cw - vw * scale) / 2, oy = (ch - vh * scale) / 2;
  for (const m of matches) {
    const x = ox + m.x * scale, y = oy + m.y * scale;
    const w = m.w * scale, h = m.h * scale;
    const col = m.known ? "#34d399" : "#fb7185";
    ctx.strokeStyle = col;
    ctx.lineWidth = 4;
    const arm = Math.max(14, Math.min(w, h) / 4);
    for (const [sx, sy] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      const px = sx > 0 ? x : x + w, py = sy > 0 ? y : y + h;
      ctx.beginPath();
      ctx.moveTo(px, py); ctx.lineTo(px - sx * arm, py);
      ctx.moveTo(px, py); ctx.lineTo(px, py - sy * arm);
      ctx.stroke();
    }
    // confidence rail
    ctx.fillStyle = "rgba(45,49,89,0.9)";
    ctx.fillRect(x, y + h + 6, w, 6);
    ctx.fillStyle = col;
    ctx.fillRect(x, y + h + 6, w * Math.max(0, Math.min(1, m.sim)), 6);
    // Persian label — browsers shape Arabic/Persian correctly on canvas
    const label = m.known ? `${m.name}  ${fa(Math.round(m.sim * 100))}٪`
                          : `ناشناس  ${fa(Math.round(m.sim * 100))}٪`;
    ctx.font = "700 21px Vazirmatn, sans-serif";
    const tw = ctx.measureText(label).width;
    const ly = Math.max(y - 14, 24);
    ctx.fillStyle = col;
    ctx.fillRect(x, ly - 24, tw + 18, 32);
    ctx.fillStyle = "#0b1030";
    ctx.fillText(label, x + 9, ly);
  }
}

/* ---------------- recognition ---------------- */
let lastMatches = [];

async function startRecognition() {
  if (recogTimer) clearInterval(recogTimer);
  recogTimer = setInterval(async () => {
    if (!cameraOn || !video || currentPage !== "dashboard") return;
    const canvas = $("dash-canvas");
    // grab a frame from the live canvas (already drawn by rAF)
    try {
      const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.8));
      if (!blob) return;
      const form = new FormData();
      form.append("image", blob, "frame.jpg");
      const res = await fetch("/api/recognize", { method: "POST", body: form });
      if (!res.ok) return;
      const data = await res.json();
      lastMatches = data.matches;
      const ctx = canvas.getContext("2d");
      drawFrame(ctx, canvas);
      drawMatches(ctx, canvas, lastMatches);
      $("faces-count").textContent = fa(lastMatches.length);
      if (data.last_known) showLastKnown(data.last_known);
      const now = performance.now();
      fpsHistory.push(now);
      fpsHistory = fpsHistory.filter((t) => now - t < 1000);
      $("fps").textContent = fa(fpsHistory.length) + " FPS";
    } catch (e) { /* transient */ }
  }, 420);
}

function showLastKnown(person) {
  $("last-avatar").textContent = initials(person.full_name);
  $("last-avatar").style.background =
    "linear-gradient(135deg,#22d3ee,#e879f9)";
  $("last-name").textContent = person.full_name;
  $("last-meta").textContent =
    [person.student_id, person.department].filter(Boolean).join("  ·  ") || "—";
  const pct = Math.round(person.sim * 100);
  $("gauge").style.setProperty("--deg", (person.sim * 360).toFixed(1) + "deg");
  $("gauge-v").textContent = fa(pct) + "٪";
  const card = $("last-card");
  card.style.borderColor = "var(--cyan)";
  setTimeout(() => (card.style.borderColor = ""), 1400);
}

/* ---------------- stats ---------------- */
async function loadStats() {
  try {
    const h = await api("/api/health");
    $("stat-people").textContent = fa(h.people);
    $("stat-today").textContent = fa(h.attendance_today);
    $("stat-model").textContent = h.models ? "آماده" : "خطا";
    $("stat-model").style.color = h.models ? "var(--magenta)" : "var(--bad)";
  } catch { /* ignore */ }
}

/* ---------------- enroll ---------------- */
const ZONES = ["center", "slight_left", "slight_right", "far_left",
               "far_right", "up", "down"];
const ZONE_LABELS = {
  center: "● مستقیم", slight_left: "← کمی چپ", slight_right: "→ کمی راست",
  far_left: "⟲ کامل چپ", far_right: "⟳ کامل راست", up: "↑ سر بالا",
  down: "↓ سر پایین",
};
const ZONE_TARGETS = { center: 4, slight_left: 4, slight_right: 4,
                       far_left: 3, far_right: 3, up: 3, down: 3 };
const PROMPTS = {
  center: "لطفاً مستقیم رو به دوربین نگاه کنید",
  slight_left: "سرتان را کمی به چپ بچرخانید ←",
  slight_right: "سرتان را کمی به راست بچرخانید →",
  far_left: "صورت خود را کامل‌تر به چپ بچرخانید",
  far_right: "صورت خود را کامل‌تر به راست بچرخانید",
  up: "سرتان را کمی بالا کنید ↑",
  down: "سرتان را کمی پایین کنید ↓",
};

let scanState = null;
let analyzeTimer = null;

(function buildZoneChips() {
  $("zones").innerHTML = ZONES.map((z) =>
    `<span class="zone-chip" id="chip-${z}">⬜ ${ZONE_LABELS[z]}</span>`).join("");
})();

function missingZones() {
  return ZONES.filter((z) => scanState.counts[z] < ZONE_TARGETS[z]);
}

$("btn-scan").addEventListener("click", async () => {
  const name = $("f-name").value.trim();
  if (name.length < 2) {
    $("form-error").textContent = "⚠ نام کامل الزامی است (حداقل ۲ حرف)";
    return;
  }
  const sid = $("f-sid").value.trim();
  if (sid) {
    try {
      const dup = await api("/api/people?search=" + encodeURIComponent(sid));
      if (dup.some((p) => (p.student_id || "") === sid)) {
        $("form-error").textContent = "⚠ این شمارهٔ کارت قبلاً ثبت شده است";
        return;
      }
    } catch { /* continue */ }
  }
  $("form-error").textContent = "";
  const ok = await openCamera();
  if (!ok) return;
  scanState = {
    samples: [], counts: Object.fromEntries(ZONES.map((z) => [z, 0])),
    lastCapture: 0, started: performance.now(), angle: 0, done: false,
  };
  $("btn-scan").disabled = true;
  $("btn-finish").disabled = true;
  status("در حال اسکن چهره");
  startAnalyze();
});

function startAnalyze() {
  if (analyzeTimer) clearInterval(analyzeTimer);
  analyzeTimer = setInterval(analyzeStep, 220);
}

async function analyzeStep() {
  if (!scanState || !cameraOn) return;
  const canvas = $("scan-canvas");
  const ctx = canvas.getContext("2d");
  // animated scan ring
  scanState.angle = (scanState.angle + 5) % 360;
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const ax = canvas.width / 4.2, ay = canvas.height / 3.4;
  ctx.save();
  ctx.strokeStyle = "rgba(45,49,89,0.9)";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.ellipse(cx, cy, ax, ay, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.ellipse(cx, cy, ax, ay, 0, scanState.angle * Math.PI / 180,
              (scanState.angle + 80) * Math.PI / 180);
  ctx.stroke();
  // checkpoint dots
  ZONES.forEach((z, i) => {
    const a = (-90 + i * 360 / 7) * Math.PI / 180;
    const dx = cx + ax * Math.cos(a), dy = cy + ay * Math.sin(a);
    const done = scanState.counts[z] >= ZONE_TARGETS[z];
    ctx.beginPath();
    ctx.arc(dx, dy, done ? 9 : 6, 0, Math.PI * 2);
    ctx.fillStyle = done ? "#34d399" : "#5f6496";
    ctx.fill();
  });
  ctx.restore();

  const missing = missingZones();
  $("scan-hint").textContent = missing.length ? PROMPTS[missing[0]] : "✓ همهٔ زاویه‌ها ثبت شد!";
  $("scan-hint").style.color = missing.length ? "var(--magenta)" : "var(--good)";

  // throttled server analysis
  const now = performance.now();
  if (now - (scanState.lastAnalyze || 0) < 260) return;
  scanState.lastAnalyze = now;
  try {
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.8));
    if (!blob) return;
    const form = new FormData();
    form.append("image", blob, "scan.jpg");
    const res = await fetch("/api/enroll/analyze", { method: "POST", body: form });
    if (!res.ok) return;
    const a = await res.json();
    if (!a.ok) {
      $("scan-reason").textContent = a.reason_text || "";
      return;
    }
    $("scan-reason").textContent = "";
    if (now - scanState.lastCapture < 450) return;
    scanState.lastCapture = now;
    const zone = a.pitch_zone !== "level" ? a.pitch_zone
      : (a.yaw_zone !== "center" ? a.yaw_zone : "center");
    let target = scanState.counts[zone] < ZONE_TARGETS[zone] ? zone : null;
    if (!target && scanState.counts.center < ZONE_TARGETS.center) target = "center";
    scanState.samples.push({ data: blob, pose: target || "overflow" });
    if (target) {
      scanState.counts[target]++;
      const chip = $("chip-" + target);
      chip.classList.add("done");
      chip.textContent = "✅ " + ZONE_LABELS[target];
    }
    const total = scanState.samples.length;
    const targetTotal = Object.values(ZONE_TARGETS).reduce((x, y) => x + y, 0);
    $("scan-progress").style.width = Math.min(100, total / targetTotal * 100) + "%";
    $("scan-count").textContent = `${fa(total)} / ${fa(targetTotal)} نمونه`;
    $("btn-finish").disabled = total < 12;
    if (!missingZones().length || total >= targetTotal ||
        (now - scanState.started > 40000 && total >= 12)) finishScan();
  } catch { /* transient */ }
}

async function finishScan() {
  if (!scanState || scanState.done) return;
  scanState.done = true;
  if (analyzeTimer) clearInterval(analyzeTimer), analyzeTimer = null;
  const samples = scanState.samples;
  if (samples.length < 1) { cancelScan(); return; }
  const person = {
    full_name: $("f-name").value.trim(),
    student_id: $("f-sid").value.trim(),
    department: $("f-dept").value.trim(),
    phone: $("f-phone").value.trim(),
    email: $("f-email").value.trim(),
    notes: $("f-notes").value.trim(),
  };
  // convert captured blobs to base64 in one go
  const payload = { person, samples: [] };
  await Promise.all(samples.map(async (s) => {
    const buf = await s.data.arrayBuffer();
    let bin = "";
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.length; i += 0x8000)
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    payload.samples.push({ data: btoa(bin), pose: s.pose });
  }));
  try {
    const result = await api("/api/enroll", {
      method: "POST", body: JSON.stringify(payload),
    });
    toast(`«${person.full_name}» ثبت شد`);
    status(`«${person.full_name}» ثبت شد`);
    openModal(person.full_name, result.samples, result.quality);
    clearForm();
    resetScanUI();
  } catch (e) {
    toast("خطا در ثبت: " + e.message, false);
    status("خطا در ثبت", false);
  }
}

function cancelScan(silent = false) {
  if (analyzeTimer) clearInterval(analyzeTimer), analyzeTimer = null;
  const was = !!scanState;
  scanState = null;
  $("btn-scan").disabled = false;
  $("btn-finish").disabled = true;
  $("scan-hint").textContent = "آماده برای اسکن";
  $("scan-hint").style.color = "var(--magenta)";
  $("scan-reason").textContent = "";
  $("scan-progress").style.width = "0%";
  $("scan-count").textContent = "۰ / ۲۴ نمونه";
  ZONES.forEach((z) => {
    const chip = $("chip-" + z);
    chip.classList.remove("done");
    chip.textContent = "⬜ " + ZONE_LABELS[z];
  });
  if (!silent && was) { toast("اسکن لغو شد", false); status("اسکن لغو شد", false); }
}

$("btn-cancel-scan").addEventListener("click", () => cancelScan());
$("btn-finish").addEventListener("click", finishScan);
$("btn-clear").addEventListener("click", clearForm);

function clearForm() {
  ["f-name", "f-sid", "f-dept", "f-phone", "f-email"].forEach((id) => $(id).value = "");
  $("f-notes").value = "";
  $("form-error").textContent = "";
}
function resetScanUI() {
  if (scanState) scanState = null;
  cancelScan(true);
}

/* ---------------- modal ---------------- */
function openModal(name, samples, quality) {
  $("modal-icon").textContent = "✅";
  $("modal-title").textContent = "ثبت‌نام موفق!";
  $("modal-avatar").textContent = initials(name);
  $("modal-sub").innerHTML =
    `${esc(name)}<br>${fa(samples)} نمونهٔ چهره ذخیره شد · امتیاز کیفیت: ${fa(Math.round(quality))} از ۱۰۰`;
  $("modal-host").classList.add("show");
}
$("modal-people").addEventListener("click", () => {
  $("modal-host").classList.remove("show");
  showPage("people");
});
$("modal-enroll").addEventListener("click", () => {
  $("modal-host").classList.remove("show");
  showPage("enroll");
});

/* ---------------- people ---------------- */
async function loadPeople() {
  const q = $("search").value.trim();
  try {
    const list = await api("/api/people?search=" + encodeURIComponent(q));
    const grid = $("people-grid");
    if (!list.length) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1">
        <div class="t">هنوز کسی ثبت نشده</div>
        <div class="s">از دکمهٔ «افزودن فرد» شروع کنید</div></div>`;
      return;
    }
    grid.innerHTML = list.map((p) => `
      <div class="card person-card">
        <div class="person-avatar">${esc(initials(p.full_name))}</div>
        <div class="person-info">
          <div class="person-name">${esc(p.full_name)}</div>
          <div class="person-meta">${esc([p.student_id, p.department,
            fa(p.face_count) + " نمونه"].filter(Boolean).join("  ·  "))}</div>
        </div>
        <button class="icon-btn" data-del="${p.id}" title="حذف">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round">${ICONS.trash}</svg>
        </button>
      </div>`).join("");
    grid.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => deletePerson(+b.dataset.del, b)));
  } catch (e) { toast(e.message, false); }
}

let searchTimer = null;
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadPeople, 200);
});
$("btn-add").addEventListener("click", () => showPage("enroll"));

async function deletePerson(id, btn) {
  if (!confirm("این فرد و تمام نمونه‌های چهره‌اش حذف می‌شود. ادامه می‌دهید؟")) return;
  try {
    await api("/api/people/" + id, { method: "DELETE" });
    toast("فرد حذف شد", false);
    loadPeople();
  } catch (e) { toast(e.message, false); }
}

/* ---------------- report ---------------- */
let reportScope = "today";
$("btn-scope").addEventListener("click", () => {
  reportScope = reportScope === "today" ? "all" : "today";
  $("btn-scope").textContent = reportScope === "today" ? "همه" : "امروز";
  loadReport();
});
$("btn-export").addEventListener("click", () => {
  window.location.href = "/api/export?today=" + (reportScope === "today" ? 1 : 0);
});
async function loadReport() {
  try {
    const rows = await api("/api/attendance?scope=" + reportScope);
    $("report-count").textContent = fa(rows.length) + " رکورد";
    const body = $("report-body");
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--muted);padding:30px">
        رکوردی ثبت نشده است</td></tr>`;
      return;
    }
    body.innerHTML = rows.map((r) => `
      <tr>
        <td style="font-variant-numeric:tabular-nums">${esc(r.recorded_at)}</td>
        <td><b>${esc(r.full_name)}</b></td>
        <td>${esc(r.student_id || "—")}</td>
        <td>${esc(r.department || "—")}</td>
      </tr>`).join("");
  } catch (e) { toast(e.message, false); }
}

/* ---------------- settings ---------------- */
async function loadSettings() {
  try {
    const s = await api("/api/settings");
    $("set-threshold").value = s.match_threshold;
    $("thr-val").textContent = Number(s.match_threshold).toFixed(2);
  } catch { /* ignore */ }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    const sel = $("set-camera");
    sel.innerHTML = "";
    cams.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || "دوربین " + fa(i + 1);
      sel.appendChild(opt);
    });
    if (deviceId !== "default" && cams.some((d) => d.deviceId === deviceId))
      sel.value = deviceId;
    sel.onchange = () => {
      deviceId = sel.value;
      localStorage.setItem("camera_device", deviceId);
      toast("دوربین انتخاب شد");
    };
  } catch { /* ignore */ }
}

$("set-threshold").addEventListener("input", async (e) => {
  $("thr-val").textContent = Number(e.target.value).toFixed(2);
  await api("/api/settings", {
    method: "POST", body: JSON.stringify({ match_threshold: e.target.value }),
  }).catch(() => {});
});

$("btn-health").addEventListener("click", async () => {
  const box = $("health-box");
  box.textContent = "در حال بررسی…";
  try {
    const lines = await api("/api/selftest", { method: "POST" });
    box.innerHTML = lines.map((l) => {
      const cls = l.includes("PASS") || l.includes("✓") ? "pass"
        : l.includes("FAIL") ? "fail" : "";
      return `<span class="${cls}">${esc(l)}</span>`;
    }).join("<br>");
  } catch (e) { box.textContent = "خطا: " + e.message; }
});

$("btn-backup").addEventListener("click", () => {
  window.location.href = "/api/backup";
});

/* ---------------- boot ---------------- */
loadStats();
setInterval(loadStats, 15000);
