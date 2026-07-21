'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  models: [],
  selected: new Set(['yolov11']),
  file: null,
  goldOffset: 0,
  goldLimit: 24,
  goldTotal: 0,
};

const f = (v, n = 4) => (v === null || v === undefined ? '—' : Number(v).toFixed(n));

function showLoader(text) {
  $('#loader-text').textContent = text || 'Đang chạy…';
  $('#loader').classList.remove('hidden');
}
const hideLoader = () => $('#loader').classList.add('hidden');

function errorBox(container, msg) {
  container.innerHTML = `<div class="err">${msg}</div>`;
}

// ---------------------------------------------------------------- tabs
$$('.tab').forEach((t) => t.addEventListener('click', () => {
  $$('.tab').forEach((x) => x.classList.remove('active'));
  $$('.tabpane').forEach((x) => x.classList.remove('active'));
  t.classList.add('active');
  $(`#pane-${t.dataset.tab}`).classList.add('active');
  if (t.dataset.tab === 'gold' && !$('#gold-grid').children.length) loadGoldList();
  if (t.dataset.tab === 'report') loadReport();
}));

// ---------------------------------------------------------------- sliders
$('#conf').addEventListener('input', (e) => { $('#conf-val').textContent = (+e.target.value).toFixed(2); });
$('#iou').addEventListener('input', (e) => { $('#iou-val').textContent = (+e.target.value).toFixed(2); });
const conf = () => parseFloat($('#conf').value);
const nmsIou = () => parseFloat($('#iou').value);
const selectedModels = () => Array.from(state.selected).join(',');

// ---------------------------------------------------------------- init
async function init() {
  const info = await (await fetch('/api/models')).json();
  state.models = info.models;

  $('#env-badges').innerHTML = [
    `<span class="badge ${info.device === 'cuda' ? 'on' : 'off'}">Device <strong>${info.device.toUpperCase()}</strong></span>`,
    `<span class="badge ${info.gold_dataset_available ? 'on' : 'off'}">gold_dataset <strong>${
      info.gold_dataset_available ? info.gold_dataset_size.toLocaleString('vi') + ' ảnh test' : 'chưa có'}</strong></span>`,
    `<span class="badge">conf mặc định <strong>${info.defaults.conf}</strong></span>`,
    `<span class="badge">NMS IoU <strong>${info.defaults.nms_iou}</strong></span>`,
  ].join('');

  $('#model-picker').innerHTML = info.models.map((m) => `
    <div class="chip ${state.selected.has(m.name) ? 'active' : ''} ${m.available ? '' : 'disabled'}"
         data-model="${m.name}" style="${state.selected.has(m.name) ? `color:${m.color}` : ''}"
         title="${m.available ? m.weights : 'Không tìm thấy file weights'}">
      <span class="dot" style="background:${m.color}"></span>
      <span>${m.display_name}</span>
      ${m.size_mb ? `<span class="hint mono" style="margin:0">${m.size_mb}MB</span>` : ''}
    </div>`).join('');

  $$('#model-picker .chip').forEach((c) => c.addEventListener('click', () => {
    if (c.classList.contains('disabled')) return;
    const name = c.dataset.model;
    const color = state.models.find((m) => m.name === name).color;
    if (state.selected.has(name)) { state.selected.delete(name); c.classList.remove('active'); c.style.color = ''; }
    else { state.selected.add(name); c.classList.add('active'); c.style.color = color; }
    $('#detect-btn').disabled = !state.file || state.selected.size === 0;
  }));

  if (!info.gold_dataset_available) {
    $('#pane-gold').querySelector('.panel').insertAdjacentHTML('afterbegin',
      '<div class="err">Chưa có gold_dataset. Giải nén <code>gold_dataset_detector_3models.zip</code> vào <code>demo_app/data/gold_dataset/</code>.</div>');
  }
}

// ---------------------------------------------------------------- upload
const drop = $('#drop');
drop.addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', (e) => e.target.files[0] && setFile(e.target.files[0]));
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

function setFile(file) {
  state.file = file;
  const r = new FileReader();
  r.onload = (e) => {
    $('#preview').src = e.target.result;
    drop.classList.add('has-image');
    const img = new Image();
    img.onload = () => {
      $('#upload-meta').textContent =
        `${file.name} · ${img.width}×${img.height}px · ${(file.size / 1024).toFixed(0)} KB`;
    };
    img.src = e.target.result;
  };
  r.readAsDataURL(file);
  $('#detect-btn').disabled = state.selected.size === 0;
}

$('#clear-btn').addEventListener('click', () => {
  state.file = null;
  drop.classList.remove('has-image');
  $('#file-input').value = '';
  $('#upload-meta').textContent = '';
  $('#upload-results').innerHTML = '';
  $('#detect-btn').disabled = true;
});

$('#detect-btn').addEventListener('click', async () => {
  if (!state.file) return;
  const fd = new FormData();
  fd.append('file', state.file);
  fd.append('models', selectedModels());
  fd.append('conf', conf());
  fd.append('nms_iou', nmsIou());
  showLoader(`Đang chạy ${state.selected.size} model…`);
  try {
    const res = await fetch('/api/detect', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Lỗi server');
    renderResults($('#upload-results'), data);
  } catch (e) {
    errorBox($('#upload-results'), 'Lỗi: ' + e.message);
  } finally { hideLoader(); }
});

// ---------------------------------------------------------------- gold
async function loadGoldList() {
  const level = $('#gold-level').value;
  const url = `/api/gold/list?limit=${state.goldLimit}&offset=${state.goldOffset}` +
              (level ? `&level=${level}` : '');
  const res = await fetch(url);
  if (!res.ok) return;
  const data = await res.json();
  state.goldTotal = data.total;
  $('#gold-page').textContent =
    `${state.goldOffset + 1}–${Math.min(state.goldOffset + state.goldLimit, data.total)} / ${data.total.toLocaleString('vi')}`;
  $('#gold-grid').innerHTML = data.items.map((it) => `
    <div class="thumb" data-id="${it.image_id}" title="${it.file_name}">
      <img src="${it.thumb}" loading="lazy" alt="">
      <div class="meta"><span>${it.n_gt} vật</span><span class="lv ${it.level}">${it.level}</span></div>
    </div>`).join('');
  $$('#gold-grid .thumb').forEach((t) =>
    t.addEventListener('click', () => runGold(parseInt(t.dataset.id, 10))));
}

$('#gold-level').addEventListener('change', () => { state.goldOffset = 0; loadGoldList(); });
$('#gold-prev').addEventListener('click', () => {
  state.goldOffset = Math.max(0, state.goldOffset - state.goldLimit); loadGoldList();
});
$('#gold-next').addEventListener('click', () => {
  if (state.goldOffset + state.goldLimit < state.goldTotal) state.goldOffset += state.goldLimit;
  loadGoldList();
});
$('#gold-go').addEventListener('click', () => {
  const id = parseInt($('#gold-id').value, 10);
  if (!isNaN(id)) runGold(id);
});

async function runGold(imageId) {
  if (state.selected.size === 0) { alert('Chọn ít nhất 1 model.'); return; }
  const fd = new FormData();
  fd.append('image_id', imageId);
  fd.append('models', selectedModels());
  fd.append('conf', conf());
  fd.append('nms_iou', nmsIou());
  showLoader('Đang chạy trên ảnh test…');
  try {
    const res = await fetch('/api/detect_gold', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Lỗi server');
    renderResults($('#gold-results'), data);
    $('#gold-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    errorBox($('#gold-results'), 'Lỗi: ' + e.message);
  } finally { hideLoader(); }
}

// ---------------------------------------------------------------- render
function metric(name, value, cls = '') {
  return `<div class="metric ${cls}"><div class="v">${value}</div><div class="n">${name}</div></div>`;
}

function renderResults(container, data) {
  const isGold = data.source === 'gold_dataset';
  const head = isGold
    ? `<div class="panel">
         <h3 class="sec">Ảnh test <code>${data.file_name}</code> · image_id ${data.image_id}</h3>
         <p class="hint">Ground-truth: <strong>${data.n_gt}</strong> sản phẩm · độ khó
            <strong>${data.level}</strong>${data.level_is_proxy ? ' (suy ra từ số instance/ảnh)' : ''}
            · ${data.image_size.width}×${data.image_size.height}px · conf=${data.settings.conf}
            · NMS IoU=${data.settings.nms_iou} · device=${data.settings.device}</p>
         <img class="annot" src="${data.gt_image}" alt="ground truth">
         <p class="hint">Ground-truth (xám) — box do người gán nhãn.</p>
       </div>`
    : `<div class="panel">
         <p class="hint">Ảnh <code>${data.file_name}</code> · ${data.image_size.width}×${data.image_size.height}px
            · conf=${data.settings.conf} · NMS IoU=${data.settings.nms_iou}
            · device=${data.settings.device} · tổng ${data.total_time_ms} ms</p>
       </div>`;

  const cards = data.results.map((r) => {
    const color = (state.models.find((m) => m.name === r.model) || {}).color || '#888';
    const gtMetrics = isGold ? [
      metric('TP', r.TP, 'tp'), metric('FP', r.FP, 'fp'), metric('FN', r.FN, 'fn'),
      metric('Precision', f(r.precision, 3)), metric('Recall', f(r.recall, 3)),
      metric('F1', f(r.f1, 3)),
    ].join('') : '';
    return `
    <div class="card">
      <div class="card-head">
        <div class="card-title"><span class="dot" style="background:${color}"></span>${r.display_name}</div>
        <span class="hint mono" style="margin:0">${r.num_params_m}M params · ${r.model_size_mb}MB</span>
      </div>
      <div class="card-body">
        <img class="annot" src="${r.annotated_image}" alt="${r.display_name}">
        <div class="metrics">
          ${metric('Số box', r.num_detections)}
          ${metric('Latency (ms)', r.latency_ms)}
          ${metric('FPS', r.fps ?? '—')}
          ${metric('Conf TB', f(r.avg_confidence, 3))}
          ${gtMetrics}
        </div>
        <p class="hint">Input: ${r.input_resolution} · device ${r.device}</p>
        ${r.crops.length ? `<h3 class="sec">Sản phẩm đã cắt (${r.crops.length})</h3>
          <div class="crops">${r.crops.map((c) => `
            <div class="crop"><img src="${c.image}" alt=""><span>#${c.index} ${c.confidence.toFixed(2)}</span></div>`).join('')}
          </div>` : '<p class="hint">Không phát hiện sản phẩm nào ở ngưỡng conf hiện tại.</p>'}
        <details class="raw"><summary>Xem toạ độ box thô [x1, y1, x2, y2, score]</summary>
          <pre>${JSON.stringify(r.boxes, null, 1)}</pre></details>
      </div>
    </div>`;
  }).join('');

  const legend = isGold
    ? `<p class="hint"><span class="k tp">xanh lá = TP</span> <span class="k fp">đỏ = FP</span>
       <span class="k fn">vàng = FN (GT bị bỏ sót)</span> — ghép theo IoU ≥ 0.5.</p>` : '';

  container.innerHTML = head + legend + `<div class="result-grid">${cards}</div>`;
}

// ---------------------------------------------------------------- report
let reportLoaded = false;
async function loadReport() {
  if (reportLoaded) return;
  const root = $('#report-root');
  const data = await (await fetch('/api/report?tag=main')).json();
  if (!data.available) {
    root.innerHTML = `<div class="err">Chưa có kết quả thực nghiệm.</div>
      <p class="hint">Chạy lần lượt:</p>
      <pre class="mono" style="background:#0b0e13;padding:14px;border-radius:10px;border:1px solid var(--line)">
python -m experiments.run_eval
python -m experiments.run_slices
python -m experiments.run_qualitative
python -m experiments.run_error_analysis
python -m experiments.make_report</pre>`;
    return;
  }
  reportLoaded = true;
  const env = data.environment;
  const ms = Object.keys(data.models);
  const dn = (m) => data.models[m].display_name;

  const accRows = ms.map((m) => {
    const a = data.models[m].accuracy, o = data.models[m].operational;
    return `<tr><td>${dn(m)}</td><td>${f(a.mAP50)}</td><td>${f(a.mAP50_95)}</td><td>${f(a.mAP75)}</td>
      <td>${f(o.precision)}</td><td>${f(o.recall)}</td><td>${f(o.f1)}</td><td>${f(a.AP_large)}</td>
      <td>${f(a.AR100)}</td><td>${o.TP}</td><td>${o.FP}</td><td>${o.FN}</td></tr>`;
  }).join('');

  const sysRows = ms.map((m) => {
    const s = data.models[m].system, l = s.latency;
    return `<tr><td>${dn(m)}</td><td>${l.p50_ms}</td><td>${l.p95_ms}</td><td>${l.p99_ms}</td>
      <td>${l.fps}</td><td>${s.model_size_mb}</td><td>${s.num_params_m}</td>
      <td>${s.vram_peak_infer_gb ?? '—'}</td><td>${s.ram_rss_running_gb}</td></tr>`;
  }).join('');

  let sliceHtml = '';
  if (data.slices) {
    for (const [key, title] of [['by_level', 'mAP@0.50:0.95 theo độ khó'],
                                ['by_density', 'mAP@0.50:0.95 theo mật độ sản phẩm/ảnh']]) {
      const groups = Object.keys(data.slices.models[ms[0]][key]);
      sliceHtml += `<h3 class="sec">${title}</h3><div class="table-scroll"><table class="data">
        <thead><tr><th>Model</th>${groups.map((g) => `<th>${g}</th>`).join('')}</tr></thead><tbody>
        ${ms.map((m) => `<tr><td>${dn(m)}</td>${groups.map((g) =>
          `<td>${f(data.slices.models[m][key][g].mAP50_95)}</td>`).join('')}</tr>`).join('')}
        </tbody></table></div>`;
    }
    if (data.slices.level_is_proxy) {
      sliceHtml += '<p class="hint">⚠️ Nhãn <code>level</code> được suy ra từ số instance/ảnh theo quy ước RPC (easy 3–10, medium 11–15, hard 16–20) vì file COCO của gold_dataset không còn field gốc.</p>';
    }
  }

  root.innerHTML = `
    <h3 class="sec">Môi trường đo</h3>
    <p class="hint">${env.cpu} · ${env.ram_total_gb} GB RAM ·
      ${env.gpu ? `${env.gpu.name} ${env.gpu.vram_total_gb} GB (driver ${env.gpu.driver})` : 'không GPU'} ·
      torch ${env.packages.torch} · ultralytics ${env.packages.ultralytics} ·
      device <strong>${env.device_used}</strong> · ${env.dataset.n_images_evaluated.toLocaleString('vi')} ảnh test ·
      batch=1, warm-up ${env.measurement.warmup_images} ảnh, seed ${env.measurement.seed}</p>

    <h3 class="sec">Bảng 1 — Accuracy</h3>
    <div class="table-scroll"><table class="data"><thead><tr>
      <th>Model</th><th>mAP50</th><th>mAP50-95</th><th>mAP75</th><th>P</th><th>R</th><th>F1</th>
      <th>AP_L</th><th>AR100</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
      <tbody>${accRows}</tbody></table></div>

    <h3 class="sec">Bảng 2 — System (inference, local)</h3>
    <div class="table-scroll"><table class="data"><thead><tr>
      <th>Model</th><th>p50 (ms)</th><th>p95</th><th>p99</th><th>FPS</th><th>Size (MB)</th>
      <th>Params (M)</th><th>VRAM (GB)</th><th>RAM RSS (GB)</th></tr></thead>
      <tbody>${sysRows}</tbody></table></div>

    ${sliceHtml}

    ${data.charts.length ? `<h3 class="sec">Biểu đồ</h3>
      <div class="chart-grid">${data.charts.map((c) =>
        `<a href="${c}" target="_blank"><img src="${c}" alt=""></a>`).join('')}</div>` : ''}

    ${data.qualitative.length ? `<h3 class="sec">Ảnh định tính (GT | 3 model)</h3><div class="qual">
      ${data.qualitative.map((q) => `<figure><a href="${q.url}" target="_blank">
        <img src="${q.url}" alt=""></a><figcaption><strong>${q.criterion}</strong> — ${q.description}
        <br>${q.file_name} · ${q.n_gt} sản phẩm · level ${q.level}</figcaption></figure>`).join('')}
      </div>` : ''}

    ${data.report_md ? `<p class="hint">Báo cáo đầy đủ:
      <a href="${data.report_md}" target="_blank" style="color:var(--accent)">BAO_CAO_THUC_NGHIEM.md</a></p>` : ''}`;
}

init();
