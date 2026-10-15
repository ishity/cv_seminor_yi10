import { SamModel, AutoProcessor, RawImage, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

env.allowLocalModels = false;

const MODEL_ID = "Xenova/slimsam-77-uniform";
const MAX_SIDE = 1024;
const GRID = 7; // 候補探索の格子点数 (GRID x GRID)

const $ = (id) => document.getElementById(id);
const view = $("view"), overlay = $("overlay");
const vctx = view.getContext("2d"), octx = overlay.getContext("2d");
const statusEl = $("status");

const state = {
  W: 0, H: 0,
  src: null,          // 作業解像度の元画像 canvas
  bg: null,           // 背景 canvas (補完済み or 元画像)
  model: null, processor: null,
  inputs: null, embeddings: null,
  candidates: [],     // {id, mask, bbox, area, score, color, selected, cut, motion}
  hover: null,
  playing: true,
  recording: false,
  t0: performance.now(),
  nextId: 1,
};

const setStatus = (s) => (statusEl.textContent = s);

// ---------- 画像読み込み ----------
const fileInput = $("file"), drop = $("drop");
fileInput.addEventListener("change", () => fileInput.files[0] && loadFile(fileInput.files[0]));
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) loadFile(f); });

async function loadFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const k = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
    const W = Math.round(img.width * k), H = Math.round(img.height * k);
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    c.getContext("2d").drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);
    resetWith(c);
  };
  img.onerror = () => setStatus("Could not load the image.");
  img.src = url;
}

function resetWith(c) {
  state.src = c; state.W = c.width; state.H = c.height;
  state.candidates = []; state.hover = null;
  state.inputs = null; state.embeddings = null;
  for (const cv of [view, overlay]) { cv.width = state.W; cv.height = state.H; }
  state.bg = c;
  $("canvasWrap").classList.add("has-image");
  $("detectBtn").disabled = false;
  $("playBtn").disabled = false;
  $("exportBtn").disabled = false;
  renderList(); drawOverlay();
  setStatus("Press the Suggest objects button.");
}

// ---------- SAM ----------
async function ensureModel() {
  if (state.model) return;
  setStatus("Loading model… (first time only, may take a while)");
  const progress_callback = (p) => {
    if (p.status === "progress") setStatus(`Loading model: ${p.file?.split("/").pop()} ${Math.round(p.progress)}%`);
  };
  state.model = await SamModel.from_pretrained(MODEL_ID, { progress_callback });
  state.processor = await AutoProcessor.from_pretrained(MODEL_ID);
}

async function ensureEmbeddings() {
  if (state.embeddings) return;
  setStatus("Analyzing image…");
  const image = await RawImage.read(state.src.toDataURL("image/png"));
  state.inputs = await state.processor(image);
  state.embeddings = await state.model.get_image_embeddings(state.inputs);
}

// 正規化座標 (0-1) の点群から最良マスクを1つ得る
async function segmentAt(points, labels) {
  const [rh, rw] = state.inputs.reshaped_input_sizes[0];
  const pts = points.flatMap(([x, y]) => [x * rw, y * rh]);
  const input_points = new Tensor("float32", pts, [1, 1, points.length, 2]);
  const input_labels = new Tensor("int64", new BigInt64Array(labels.map(BigInt)), [1, 1, labels.length]);
  const out = await state.model({ ...state.embeddings, input_points, input_labels });
  const masks = await state.processor.post_process_masks(out.pred_masks, state.inputs.original_sizes, state.inputs.reshaped_input_sizes);
  const m = masks[0]; // [1,3,H,W]
  const scores = out.iou_scores.data;
  let best = 0;
  for (let i = 1; i < 3; i++) if (scores[i] > scores[best]) best = i;
  const [, , h, w] = m.dims;
  const mask = new Uint8Array(w * h);
  const off = best * w * h;
  for (let i = 0; i < w * h; i++) mask[i] = m.data[off + i] ? 1 : 0;
  return { mask, w, h, score: scores[best] };
}

function maskStats(mask, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (mask[y * w + x]) { area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  return { area, bbox: area ? [x0, y0, x1 + 1, y1 + 1] : null };
}

function overlapRatio(a, b) {
  let inter = 0, ua = 0, ub = 0;
  for (let i = 0; i < a.length; i++) { ua += a[i]; ub += b[i]; inter += a[i] & b[i]; }
  return inter / Math.max(1, Math.min(ua, ub)) * (inter / Math.max(1, ua + ub - inter) > 0.5 ? 1 : 0.5);
}

const PALETTE = ["#ff4d4f", "#36cfc9", "#ffa940", "#9254de", "#73d13d", "#40a9ff", "#f759ab", "#fadb14"];

function addCandidate(mask, score) {
  const { area, bbox } = maskStats(mask, state.W, state.H);
  if (!bbox) return null;
  const c = { id: state.nextId++, mask, area, bbox, score, selected: false, color: PALETTE[state.candidates.length % PALETTE.length], cut: null, motion: { type: "float", speed: 1, amp: 50 } };
  state.candidates.push(c);
  return c;
}

$("detectBtn").addEventListener("click", async () => {
  const btn = $("detectBtn");
  btn.disabled = true;
  try {
    await ensureModel();
    await ensureEmbeddings();
    const found = [];
    const total = GRID * GRID, imgArea = state.W * state.H;
    for (let gy = 0; gy < GRID; gy++) for (let gx = 0; gx < GRID; gx++) {
      setStatus(`Searching for candidates… ${gy * GRID + gx + 1}/${total}`);
      const r = await segmentAt([[(gx + 0.5) / GRID, (gy + 0.5) / GRID]], [1]);
      await new Promise((res) => setTimeout(res)); // UI更新
      if (r.score < 0.8) continue;
      const st = maskStats(r.mask, r.w, r.h);
      if (st.area < imgArea * 0.01 || st.area > imgArea * 0.6) continue;
      if (found.some((f) => overlapRatio(f.mask, r.mask) > 0.8)) continue;
      found.push(r);
    }
    state.candidates = [];
    found.sort((a, b) => b.score - a.score).forEach((r) => addCandidate(r.mask, r.score));
    setStatus(found.length ? `Found ${found.length} candidates. Click to select.` : "No candidates found. Click directly on an object in the image.");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
  }
  btn.disabled = false;
  renderList(); drawOverlay();
});

// ---------- 選択UI ----------
function pickAt(x, y) {
  let hit = null;
  for (const c of state.candidates) {
    if (c.mask[y * state.W + x] && (!hit || c.area < hit.area)) hit = c;
  }
  return hit;
}

function evPos(ev) {
  const r = overlay.getBoundingClientRect();
  return [Math.floor((ev.clientX - r.left) / r.width * state.W), Math.floor((ev.clientY - r.top) / r.height * state.H)];
}

overlay.addEventListener("mousemove", (ev) => {
  if (!state.src) return;
  const [x, y] = evPos(ev);
  const h = pickAt(x, y);
  if (h !== state.hover) { state.hover = h; drawOverlay(); }
});
overlay.addEventListener("mouseleave", () => { state.hover = null; drawOverlay(); });

overlay.addEventListener("click", async (ev) => {
  if (!state.src || state.recording) return;
  const [x, y] = evPos(ev);
  const hit = pickAt(x, y);
  if (hit) { toggle(hit); return; }
  // 候補にない場所 → その点でSAMを実行して候補追加
  try {
    await ensureModel(); await ensureEmbeddings();
    setStatus("Extracting selection…");
    const r = await segmentAt([[x / state.W, y / state.H]], [1]);
    const c = addCandidate(r.mask, r.score);
    if (c) { toggle(c); setStatus("Added."); } else setStatus("Could not extract an object there.");
  } catch (e) { setStatus("Error: " + e.message); }
});

function toggle(c) {
  c.selected = !c.selected;
  if (c.selected && !c.cut) c.cut = makeCutout(c);
  if (state.candidates.some((k) => k.selected)) rebuildBackground();
  else state.bg = state.src;
  renderList(); drawOverlay();
}

function drawOverlay() {
  octx.clearRect(0, 0, state.W, state.H);
  for (const c of state.candidates) {
    if (c.selected) continue; // 選択済みは動くのでオーバーレイ不要
    paintMask(octx, c, c === state.hover ? 120 : 50);
  }
}
const anySelected = () => state.candidates.some((c) => c.selected);

function paintMask(ctx, c, alpha) {
  const id = ctx.createImageData(state.W, state.H);
  const [r, g, b] = hexToRgb(c.color);
  for (let i = 0; i < c.mask.length; i++) if (c.mask[i]) { const o = i * 4; id.data[o] = r; id.data[o + 1] = g; id.data[o + 2] = b; id.data[o + 3] = alpha; }
  const t = document.createElement("canvas"); t.width = state.W; t.height = state.H;
  t.getContext("2d").putImageData(id, 0, 0);
  ctx.drawImage(t, 0, 0);
}
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }

// ---------- 切り抜き・背景補完 ----------
function makeCutout(c) {
  const [x0, y0, x1, y1] = c.bbox, w = x1 - x0, h = y1 - y0;
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const cx = cv.getContext("2d");
  const sd = state.src.getContext("2d").getImageData(x0, y0, w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!c.mask[(y + y0) * state.W + (x + x0)]) sd.data[(y * w + x) * 4 + 3] = 0;
  }
  cx.putImageData(sd, 0, 0);
  return cv;
}

// 選択物体の領域を周囲の色で拡散補完 (低解像度で実施して拡大)
function rebuildBackground() {
  const sel = state.candidates.filter((c) => c.selected);
  if (!$("fillBg").checked) { state.bg = state.src; return; }
  const W = state.W, H = state.H, s = Math.min(1, 160 / Math.max(W, H));
  const w = Math.max(2, Math.round(W * s)), h = Math.max(2, Math.round(H * s));
  const small = document.createElement("canvas"); small.width = w; small.height = h;
  const sc = small.getContext("2d"); sc.drawImage(state.src, 0, 0, w, h);
  const px = sc.getImageData(0, 0, w, h);
  const hole = new Uint8Array(w * h);
  const grow = 3 / s; // 元解像度で数px膨張させて縁の残りを防ぐ
  for (const c of sel) {
    const [x0, y0, x1, y1] = c.bbox;
    for (let y = Math.max(0, Math.floor((y0 - grow) * s)); y < Math.min(h, Math.ceil((y1 + grow) * s)); y++)
      for (let x = Math.max(0, Math.floor((x0 - grow) * s)); x < Math.min(w, Math.ceil((x1 + grow) * s)); x++) {
        const ox = Math.min(W - 1, Math.floor(x / s)), oy = Math.min(H - 1, Math.floor(y / s));
        if (hitNear(c, ox, oy, grow)) hole[y * w + x] = 1;
      }
  }
  const d = px.data;
  for (let i = 0; i < w * h; i++) if (hole[i]) { d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 128; }
  for (let it = 0; it < 300; it++) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!hole[i]) continue;
      let r = 0, g = 0, b = 0, n = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = (ny * w + nx) * 4; r += d[j]; g += d[j + 1]; b += d[j + 2]; n++;
      }
      d[i * 4] = r / n; d[i * 4 + 1] = g / n; d[i * 4 + 2] = b / n;
    }
  }
  sc.putImageData(px, 0, 0);
  // 穴部分のみ拡大したぼかし画像で置換
  const out = document.createElement("canvas"); out.width = W; out.height = H;
  const oc = out.getContext("2d"); oc.drawImage(state.src, 0, 0);
  const patch = document.createElement("canvas"); patch.width = W; patch.height = H;
  const pc = patch.getContext("2d"); pc.imageSmoothingQuality = "high"; pc.drawImage(small, 0, 0, W, H);
  const holeBig = document.createElement("canvas"); holeBig.width = W; holeBig.height = H;
  const hc = holeBig.getContext("2d"); const hd = hc.createImageData(W, H);
  for (const c of sel) {
    const [x0, y0, x1, y1] = c.bbox;
    for (let y = Math.max(0, y0 - 4); y < Math.min(H, y1 + 4); y++) for (let x = Math.max(0, x0 - 4); x < Math.min(W, x1 + 4); x++)
      if (hitNear(c, x, y, 3)) hd.data[(y * W + x) * 4 + 3] = 255;
  }
  hc.putImageData(hd, 0, 0);
  pc.globalCompositeOperation = "destination-in"; pc.drawImage(holeBig, 0, 0);
  oc.drawImage(patch, 0, 0);
  state.bg = out;
}
function hitNear(c, x, y, r) {
  r = Math.round(r);
  for (let dy = -r; dy <= r; dy += Math.max(1, r)) for (let dx = -r; dx <= r; dx += Math.max(1, r)) {
    const nx = x + dx, ny = y + dy;
    if (nx >= 0 && ny >= 0 && nx < state.W && ny < state.H && c.mask[ny * state.W + nx]) return true;
  }
  return false;
}
$("fillBg").addEventListener("change", () => { if (anySelected()) rebuildBackground(); });

// ---------- モーション ----------
const MOTIONS = {
  none:   "None",
  float:  "Float (up/down)",
  sway:   "Sway (pivot at base)",
  spin:   "Spin",
  pulse:  "Pulse (scale)",
  bounce: "Bounce",
  shake:  "Shake",
  orbit:  "Orbit (circle)",
  slide:  "Slide (left/right)",
  blink:  "Blink",
};

// p = 2π * speed * (t / duration) : ループ内で speed 回繰り返す
function motionAt(m, phase, size) {
  const a = m.amp / 100, d = a * 0.25 * size, p = phase * m.speed;
  const r = { dx: 0, dy: 0, rot: 0, sx: 1, sy: 1, alpha: 1, pivotBottom: false };
  switch (m.type) {
    case "float":  r.dy = -Math.sin(p) * d; break;
    case "sway":   r.rot = Math.sin(p) * a * 0.5; r.pivotBottom = true; break;
    case "spin":   r.rot = p; break;
    case "pulse":  r.sx = r.sy = 1 + Math.sin(p) * a * 0.4; break;
    case "bounce": { const b = Math.abs(Math.sin(p / 2)); r.dy = -b * d * 1.5; r.sy = 1 - (1 - b) * a * 0.25; r.sx = 1 + (1 - b) * a * 0.15; r.pivotBottom = true; break; }
    case "shake":  r.dx = Math.sin(p * 8) * d * 0.12; r.dy = Math.cos(p * 11) * d * 0.08; break;
    case "orbit":  r.dx = Math.cos(p) * d; r.dy = Math.sin(p) * d; break;
    case "slide":  r.dx = Math.sin(p) * d * 1.5; break;
    case "blink":  r.alpha = 0.55 + 0.45 * Math.cos(p); break;
  }
  return r;
}

function renderFrame(tSec) {
  const dur = Number($("duration").value);
  const phase = 2 * Math.PI * ((tSec % dur) / dur);
  vctx.clearRect(0, 0, state.W, state.H);
  vctx.drawImage(state.bg, 0, 0);
  const size = Math.min(state.W, state.H);
  for (const c of state.candidates) {
    if (!c.selected) continue;
    const [x0, y0, x1, y1] = c.bbox, w = x1 - x0, h = y1 - y0;
    const m = motionAt(c.motion, phase, size);
    const px = x0 + w / 2, py = m.pivotBottom ? y1 : y0 + h / 2;
    vctx.save();
    vctx.globalAlpha = m.alpha;
    vctx.translate(px + m.dx, py + m.dy);
    vctx.rotate(m.rot);
    vctx.scale(m.sx, m.sy);
    vctx.drawImage(c.cut, x0 - px, y0 - py);
    vctx.restore();
  }
}

// ---------- 選択リスト ----------
function renderList() {
  const ul = $("selList"); ul.innerHTML = "";
  for (const c of state.candidates.filter((k) => k.selected)) {
    const li = document.createElement("li"); li.className = "sel-item";
    const th = document.createElement("canvas");
    th.width = c.cut.width; th.height = c.cut.height; th.getContext("2d").drawImage(c.cut, 0, 0);
    const ctl = document.createElement("div"); ctl.className = "ctl";
    const sel = document.createElement("select");
    for (const [k, v] of Object.entries(MOTIONS)) { const o = new Option(v, k); o.selected = k === c.motion.type; sel.add(o); }
    sel.onchange = () => (c.motion.type = sel.value);
    ctl.append(sel, slider("Speed", 1, 8, c.motion.speed, (v) => (c.motion.speed = v), "×"), slider("Amount", 0, 100, c.motion.amp, (v) => (c.motion.amp = v), ""));
    const rm = document.createElement("button"); rm.className = "rm"; rm.textContent = "Deselect"; rm.onclick = () => toggle(c);
    ctl.append(rm);
    li.append(th, ctl); ul.append(li);
  }
}
function slider(label, min, max, val, on, unit) {
  const l = document.createElement("label"); const s = document.createElement("span"); s.textContent = label;
  const i = document.createElement("input"); i.type = "range"; i.min = min; i.max = max; i.value = val;
  const o = document.createElement("output"); o.textContent = val + unit;
  i.oninput = () => { on(Number(i.value)); o.textContent = i.value + unit; };
  l.append(s, i, o); return l;
}

$("duration").addEventListener("input", () => ($("durOut").textContent = $("duration").value + " s"));

// ---------- 再生 ----------
$("playBtn").addEventListener("click", () => {
  state.playing = !state.playing;
  $("playBtn").textContent = state.playing ? "Pause" : "Play";
  if (state.playing) state.t0 = performance.now() - (state.pausedAt || 0);
  else state.pausedAt = performance.now() - state.t0;
});

function loop() {
  if (state.src && !state.recording) {
    if (state.playing) renderFrame((performance.now() - state.t0) / 1000);
    else renderFrame((state.pausedAt || 0) / 1000);
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ---------- 動画書き出し ----------
$("exportBtn").addEventListener("click", async () => {
  if (!state.src) return;
  if (!anySelected()) { setStatus("Select at least one object."); return; }
  if (!window.MediaRecorder) { setStatus("This browser does not support video export."); return; }
  const types = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mime = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  const dur = Number($("duration").value);
  const stream = view.captureStream(30);
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const done = new Promise((res) => (rec.onstop = res));
  state.recording = true; $("exportBtn").disabled = true;
  octx.clearRect(0, 0, state.W, state.H);
  rec.start();
  const t0 = performance.now();
  await new Promise((res) => {
    const tick = () => {
      const t = (performance.now() - t0) / 1000;
      if (t >= dur) return res();
      renderFrame(t);
      setStatus(`Exporting… ${t.toFixed(1)} / ${dur} s`);
      requestAnimationFrame(tick);
    };
    tick();
  });
  rec.stop(); await done;
  state.recording = false; $("exportBtn").disabled = false;
  const ext = rec.mimeType.includes("mp4") ? "mp4" : "webm";
  const blob = new Blob(chunks, { type: rec.mimeType });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `motion.${ext}`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  setStatus(`Saved (motion.${ext})`);
  drawOverlay();
});
