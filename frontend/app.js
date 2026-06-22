/* ═══════════════════════════════════════════════════════
   RAIL-SENSE — Frontend Application Logic (app.js)
   Multi-Component LHB Bogie AI Diagnostic System v2.0
═══════════════════════════════════════════════════════ */
'use strict';

const COMPONENTS = {
  coupler:    { label: 'CBC Coupler',    icon: 'fa-link',              sub: 'H-Type Tightlock' },
  axle_box:   { label: 'Axle Box',       icon: 'fa-circle-dot',        sub: 'Roller Bearing Assembly' },
  brake_disk: { label: 'Brake Disk',     icon: 'fa-circle-half-stroke',sub: 'Ventilated Disk' },
  damper:     { label: 'Damper',         icon: 'fa-arrows-up-down',    sub: 'Primary / Secondary' },
  spring:     { label: 'Coil Spring',    icon: 'fa-wave-square',       sub: 'Helical Suspension' },
  wheel:      { label: 'Wheel',          icon: 'fa-circle',            sub: 'Wheel Set / Tread' },
};

const state = {
  activeTab: 'dashboard', selectedComponent: 'coupler',
  selectedFile: null, lastResult: null, lastInspectionId: null,
  datasetCache: [], datasetCompFilter: 'all', datasetLblFilter: 'all',
  summaryChart: null, trendChart: null, modelStatus: {},
};

document.addEventListener('DOMContentLoaded', () => {
  try { startClock(); } catch(e) { console.warn('clock:', e); }
  try { loadStats(); } catch(e) { console.warn('stats:', e); }
  try { loadRecentInspections(); } catch(e) { console.warn('recent:', e); }
  try { loadDataset(); } catch(e) { console.warn('dataset:', e); }
  try { loadHistory(); } catch(e) { console.warn('history:', e); }
  try { setupDragDrop(); } catch(e) { console.warn('dragdrop:', e); }
  try { renderTrainingGrid(); } catch(e) { console.warn('training:', e); }
  try { switchTab('dashboard'); } catch(e) { console.warn('switchtab:', e); }

  // Dismiss loading screen after 4 seconds — failsafe
  function dismissLoader() {
    const loader = document.getElementById('loading-screen');
    if (!loader) return;
    loader.style.transition = 'opacity 0.8s ease';
    loader.style.opacity = '0';
    setTimeout(() => { loader.style.display = 'none'; }, 850);
  }
  setTimeout(dismissLoader, 4000);
  // Extra failsafe: if still showing after 6s, force hide
  setTimeout(() => {
    const loader = document.getElementById('loading-screen');
    if (loader) loader.style.display = 'none';
  }, 6000);
});

function startClock() {
  const tick = () => {
    document.getElementById('time-display').textContent =
      new Date().toLocaleTimeString('en-IN', { hour12: false }) + ' IST';
  };
  tick(); setInterval(tick, 1000);
}

const TAB_TITLES = {
  dashboard: 'Command Dashboard',
  inspector: 'Live Inspection Terminal',
  dataset:   'Dataset Explorer',
  logs:      'Inspection History',
  training:  'Model Training',
};

function switchTab(tab) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const panel = document.getElementById('tab-' + tab);
  const nav   = document.getElementById('nav-' + tab);
  if (panel) panel.classList.add('active');
  if (nav)   nav.classList.add('active');
  document.getElementById('page-title').textContent = TAB_TITLES[tab] || tab;
  state.activeTab = tab;
  if (tab === 'logs')     loadHistory();
  if (tab === 'training') refreshTrainingStatus();
}

function toggleSidebar() { document.getElementById('sidebar').classList.toggle('open'); }

function collapseSidebar() {
  const sidebar = document.getElementById('sidebar');
  const mainContent = document.querySelector('.main-content');
  sidebar.classList.toggle('collapsed');
  if(mainContent) mainContent.classList.toggle('sidebar-collapsed');
}

function selectComponent(comp, btn) {
  state.selectedComponent = comp;
  document.querySelectorAll('.comp-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const cfg   = COMPONENTS[comp];
  const label = cfg ? cfg.label : comp;
  const icon  = cfg ? cfg.icon  : 'fa-cog';
  const sub   = cfg ? cfg.sub   : '';

  // --- Update upload card header dynamically ---
  document.getElementById('upload-label').textContent      = label;
  document.getElementById('upload-header-sub').textContent = sub + ' · Bogie Component';
  document.getElementById('upload-header-fa').className    = 'fa-solid ' + icon;
  document.getElementById('drop-comp-hint').textContent    = label;

  // If a file is already loaded, keep "Image Loaded" state but update comp tag
  if (state.selectedFile) {
    document.getElementById('preview-comp-name').textContent = label;
    const pcTagIcon = document.querySelector('#preview-comp-tag i');
    if (pcTagIcon) pcTagIcon.className = 'fa-solid ' + icon;
    setUploadTag('loaded', 'fa-circle-check', 'Image Loaded');
    document.getElementById('upload-ready-badge').style.display = 'inline-flex';
  } else {
    document.getElementById('upload-ready-badge').style.display = 'none';
    setUploadTag('waiting', 'fa-hourglass-start', 'Awaiting Image');
  }

  // Reset results panel
  document.getElementById('scanning-state').style.display = 'flex';
  document.getElementById('result-content').style.display = 'none';
  document.getElementById('scanning-state').querySelector('.scanner-text').textContent = 'Awaiting image upload...';
}

function setUploadTag(cls, icon, text) {
  const el = document.getElementById('upload-status-tag');
  if (!el) return;
  el.innerHTML = `<span class="upload-tag ${cls}"><i class="fa-solid ${icon}"></i> ${text}</span>`;
}

function formatFileSize(bytes) {
  if (bytes < 1024)         return bytes + ' B';
  if (bytes < 1048576)      return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(2) + ' MB';
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) { const txt = await res.text(); throw new Error('API ' + path + ' -> ' + res.status + ': ' + txt); }
  return res.json();
}

async function loadStats() {
  try {
    // Fetch stats and model status IN PARALLEL for speed
    const [s, modelRes] = await Promise.all([
      apiFetch('/api/stats'),
      apiFetch('/api/models'),
    ]);

    // Populate dashboard stat counters
    document.getElementById('stat-total').textContent   = s.total   ?? 0;
    document.getElementById('stat-fit').textContent     = s.fit     ?? 0;
    document.getElementById('stat-unfit').textContent   = s.unfit   ?? 0;
    document.getElementById('stat-monitor').textContent = s.monitor ?? 0;
    renderSummaryChart(s.fit ?? 0, s.unfit ?? 0, s.monitor ?? 0);

    // Store model status BEFORE rendering the grid — fixes NOT TRAINED display bug
    state.modelStatus = modelRes.model_status || {};

    // Now render grid with correct trained/untrained state
    renderComponentStatusGrid(s.dataset_counts || {});

    // Update the interactive AI status badge in the topbar
    const trained = Object.values(state.modelStatus).filter(m => m.exists).length;
    const total   = Object.keys(state.modelStatus).length;
    const countEl = document.getElementById('ai-status-count');
    const dotEl   = document.getElementById('ai-pulse-dot');
    if (countEl) countEl.textContent = trained + '/' + total;
    if (dotEl)   dotEl.style.background = trained === total ? 'var(--green)' : trained > 0 ? 'var(--amber)' : 'var(--red)';

    // Populate the hover dropdown rows
    const rowsEl = document.getElementById('ai-dropdown-rows');
    if (rowsEl) {
      rowsEl.innerHTML = Object.entries(state.modelStatus).map(([key, val]) => {
        const cfg = COMPONENTS[key] || {};
        const ok  = val.exists;
        return `<div class="ai-dropdown-row">
          <i class="fa-solid ${cfg.icon || 'fa-cog'} ai-dr-icon"></i>
          <span class="ai-dr-label">${val.label || key}</span>
          <span class="ai-dr-badge ${ok ? 'ready' : 'not-ready'}">${ok ? 'READY' : 'NOT TRAINED'}</span>
        </div>`;
      }).join('');
    }
  } catch (e) { console.warn('Stats load failed:', e); }
}

function renderComponentStatusGrid(datasetCounts) {
  const grid = document.getElementById('component-status-grid');
  if (!grid) return;
  grid.innerHTML = Object.entries(COMPONENTS).map(([key, cfg]) => {
    const counts  = datasetCounts[key] || { normal: 0, defect: 0, total: 0 };
    const trained = state.modelStatus[key]?.exists;   // correctly populated now
    const sc      = trained ? 'trained' : 'untrained';
    return '<div class="comp-status-cell" onclick="switchTab(\'inspector\');selectComponent(\'' + key + '\', document.querySelector(\'[data-comp=' + key + ']\'))">' +
      '<div class="comp-status-icon ' + sc + '"><i class="fa-solid ' + cfg.icon + '"></i></div>' +
      '<div class="comp-status-name">' + cfg.label + '</div>' +
      '<div class="comp-status-count">' + counts.normal + 'N - ' + counts.defect + 'D - ' + counts.total + ' imgs</div>' +
      '<div class="comp-status-badge ' + sc + '">' + (trained ? 'MODEL READY' : 'NOT TRAINED') + '</div>' +
      '</div>';
  }).join('');
}

function renderSummaryChart(fit, unfit, monitor) {
  const ctx = document.getElementById('summaryChart');
  if (!ctx) return;
  if (state.summaryChart) state.summaryChart.destroy();
  state.summaryChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels: ['FIT', 'UNFIT', 'MONITOR'],
      datasets: [{ data: [fit, unfit, monitor],
        backgroundColor: ['hsl(142,68%,45%)','hsl(346,80%,55%)','hsl(40,95%,52%)'],
        borderWidth: 0, hoverOffset: 6 }] },
    options: { cutout: '72%',
      plugins: { legend: { labels: { color: '#334155', font: { family: 'Outfit' } } } },
      responsive: true, maintainAspectRatio: false },
  });
}

async function loadRecentInspections() {
  try {
    const rows = await apiFetch('/api/history?limit=6');
    const el   = document.getElementById('recent-list');
    if (!rows.length) {
      el.innerHTML = '<div class="empty-state"><i class="fa-solid fa-inbox"></i><p>No inspections yet</p></div>';
      renderTrendChart([]); return;
    }
    el.innerHTML = rows.map(r => {
      const comp = COMPONENTS[r.component_type];
      return '<div class="recent-row">' +
        '<span class="recent-status ' + r.status + '">' + r.status + '</span>' +
        '<span class="recent-meta"><i class="fa-solid ' + (comp ? comp.icon : 'fa-question') + '" style="color:var(--blue);margin-right:4px;"></i>' +
        (comp ? comp.label : r.component_type) + ' &nbsp;|&nbsp; Coach ' + (r.coach_number || '-') + ' &nbsp;|&nbsp; ' + (r.zone || 'Unknown Zone') + '</span>' +
        '<span class="recent-time">' + formatDate(r.created_at) + '</span></div>';
    }).join('');
    renderTrendChart(rows.reverse());
  } catch (e) { console.warn('Recent load failed:', e); }
}

function renderTrendChart(rows) {
  const ctx = document.getElementById('trendChart');
  if (!ctx) return;
  if (state.trendChart) state.trendChart.destroy();
  state.trendChart = new Chart(ctx, {
    type: 'line',
    data: { labels: rows.map(r => formatDate(r.created_at, true)),
      datasets: [
        { label: 'Defect Score', data: rows.map(r => r.defect_score ?? 0),
          borderColor: 'hsl(346,80%,55%)', backgroundColor: 'hsla(346,80%,55%,0.12)',
          fill: true, tension: 0.4, pointRadius: 4 },
        { label: 'Confidence', data: rows.map(r => r.confidence ?? 0),
          borderColor: 'hsl(215,90%,55%)', backgroundColor: 'hsla(215,90%,55%,0.12)',
          fill: true, tension: 0.4, pointRadius: 4 },
      ] },
    options: {
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#cbd5e1' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#cbd5e1' }, min: 0, max: 100 },
      },
      plugins: { legend: { labels: { color: '#334155', font: { family: 'Outfit', size: 12 } } } },
      responsive: true, maintainAspectRatio: false,
    },
  });
}

function setupDragDrop() {
  const zone = document.getElementById('drop-zone');
  if (!zone) return;

  // Standard local file drag-over
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    const isDatasetDrag = e.dataTransfer.types.includes('text/x-dataset-url');
    zone.classList.add(isDatasetDrag ? 'dataset-drag-over' : 'drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over', 'dataset-drag-over');
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over', 'dataset-drag-over');

    // Check if this is a dataset image being dragged
    const datasetUrl  = e.dataTransfer.getData('text/x-dataset-url');
    const datasetName = e.dataTransfer.getData('text/x-dataset-name');
    const datasetComp = e.dataTransfer.getData('text/x-dataset-comp');

    if (datasetUrl) {
      // Fetch the dataset image and load it
      analyzeDatasetImage(datasetUrl, datasetName, datasetComp);
      return;
    }

    // Otherwise handle local file
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setFile(file);
  });

  // Sidebar nav-item: allow dragging a dataset image and hovering over
  // "Live Inspector" nav item to auto-switch tab after 400ms
  let navHoverTimer = null;
  const navInspector = document.getElementById('nav-inspector');
  if (navInspector) {
    navInspector.addEventListener('dragover', e => {
      e.preventDefault();
      navInspector.classList.add('drag-hover');
      if (!navHoverTimer) {
        navHoverTimer = setTimeout(() => {
          switchTab('inspector');
        }, 400);
      }
    });
    navInspector.addEventListener('dragleave', () => {
      navInspector.classList.remove('drag-hover');
      if (navHoverTimer) { clearTimeout(navHoverTimer); navHoverTimer = null; }
    });
    navInspector.addEventListener('drop', e => {
      e.preventDefault();
      navInspector.classList.remove('drag-hover');
      if (navHoverTimer) { clearTimeout(navHoverTimer); navHoverTimer = null; }
      switchTab('inspector');

      const datasetUrl  = e.dataTransfer.getData('text/x-dataset-url');
      const datasetName = e.dataTransfer.getData('text/x-dataset-name');
      const datasetComp = e.dataTransfer.getData('text/x-dataset-comp');
      if (datasetUrl) {
        setTimeout(() => analyzeDatasetImage(datasetUrl, datasetName, datasetComp), 350);
      }
    });
  }
}

function handleFileSelect(event) { const file = event.target.files[0]; if (file) setFile(file); }

function setFile(file) {
  state.selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('preview-wrap').style.display = 'flex';
    document.getElementById('drop-zone').style.display    = 'none';
  };
  reader.readAsDataURL(file);

  // Populate file info
  document.getElementById('preview-filename').textContent = file.name;
  document.getElementById('preview-filesize').textContent = formatFileSize(file.size);

  // Populate component tag in preview (will be updated by auto-detect)
  const cfg   = COMPONENTS[state.selectedComponent];
  const label = cfg ? cfg.label : state.selectedComponent;
  const icon  = cfg ? cfg.icon  : 'fa-cog';
  document.getElementById('preview-comp-name').textContent = label;
  const pcTagIcon = document.querySelector('#preview-comp-tag i');
  if (pcTagIcon) pcTagIcon.className = 'fa-solid ' + icon;

  // Update header status
  setUploadTag('loaded', 'fa-circle-check', 'Image Loaded');
  document.getElementById('upload-ready-badge').style.display = 'inline-flex';
  document.getElementById('badge-inspector').textContent = '1';

  // Auto-detect which component this image belongs to
  autoDetectComponent(file);
}

function clearPreview() {
  state.selectedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('preview-wrap').style.display = 'none';
  document.getElementById('drop-zone').style.display    = '';
  document.getElementById('badge-inspector').textContent = '';
  document.getElementById('upload-ready-badge').style.display = 'none';
  setUploadTag('waiting', 'fa-hourglass-start', 'Awaiting Image');
}

async function runInspection() {
  if (!state.selectedFile) { alert('Please upload a component image first.'); return; }
  const btn = document.getElementById('run-btn');
  const cfg = COMPONENTS[state.selectedComponent];
  const label = cfg ? cfg.label : state.selectedComponent;

  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analysing…';

  // Update card status to scanning
  setUploadTag('scanning', 'fa-circle-notch fa-spin', 'Scanning ' + label + '…');

  document.getElementById('result-content').style.display = 'none';
  document.getElementById('scanning-state').style.display = 'flex';
  document.getElementById('scanning-state').querySelector('.scanner-text').textContent =
    'Running AI Inspection — ' + label + '…';
  document.getElementById('result-card').classList.add('scanning-active');

  try {
    const fd = new FormData();
    fd.append('file',           state.selectedFile);
    fd.append('component_type', state.selectedComponent);
    fd.append('coach_number',   document.getElementById('inp-coach-number').value);
    fd.append('coach_type',     document.getElementById('inp-coach-type').value);
    fd.append('depot',          document.getElementById('inp-depot').value);
    fd.append('zone',           document.getElementById('inp-zone').value);
    fd.append('inspector_name', document.getElementById('inp-inspector').value);
    fd.append('notes',          document.getElementById('inp-notes').value);
    fd.append('use_ai',         'true');
    const result = await apiFetch('/api/analyze', { method: 'POST', body: fd });
    state.lastResult = result; state.lastInspectionId = result.id;
    renderResult(result); loadStats(); loadRecentInspections();
    // Restore tag to done
    setUploadTag('loaded', 'fa-circle-check', 'Analysis Complete');
  } catch (e) {
    alert('Inspection failed: ' + e.message); console.error(e);
    setUploadTag('loaded', 'fa-circle-check', 'Image Loaded');
  }
  finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Run AI Inspection';
    document.getElementById('result-card').classList.remove('scanning-active');
  }
}

function renderResult(r) {
  document.getElementById('scanning-state').style.display = 'none';
  document.getElementById('result-content').style.display = 'flex';
  const status = r.status || 'UNKNOWN';
  const comp   = COMPONENTS[r.component_type] || {};
  const banner = document.getElementById('status-banner');
  banner.className = 'status-banner ' + status;
  const iconM = { FIT:'fa-circle-check', UNFIT:'fa-triangle-exclamation', MONITOR:'fa-eye' };
  document.getElementById('status-icon').innerHTML = '<i class="fa-solid ' + (iconM[status] || 'fa-circle-question') + '"></i>';

  // Extended label map — includes wheel-specific decisions
  const labelM = {
    FIT:     'FIT FOR RUN',
    UNFIT:   'UNFIT — SEND TO WORKSHOP',
    MONITOR: 'MONITOR CLOSELY',
  };
  const finalDecision = r.final_decision || '';
  // For wheel: surface the actual decision text (e.g. 'Condemn Wheelset') directly
  const wheelDecisions = ['Condemn Wheelset','Remove From Service','Immediate Reprofiling Required','Schedule Maintenance'];
  const isWheelDecision = wheelDecisions.some(d => finalDecision.includes(d));
  document.getElementById('status-label').textContent = isWheelDecision ? finalDecision : (labelM[status] || status);
  document.getElementById('status-sub').textContent = isWheelDecision ? ('AI Wheel Inspection — ' + (r.confidence || 0) + '% confidence') : (finalDecision || ((r.confidence || 0) + '% confidence'));
  drawConfidenceRing(r.confidence || 0, status);
  document.getElementById('confidence-value').textContent = (r.confidence ?? 0) + '%';

  const badge = document.getElementById('component-result-badge');
  badge.innerHTML = '<i class="fa-solid ' + (comp.icon || 'fa-cog') + '"></i> Inspecting: <strong>' + (r.component_label || comp.label || r.component_type) + '</strong>';

  const bboxWrap = document.getElementById('bbox-wrap');
  if (r.image_url) {
    bboxWrap.style.display = 'block';
    const img = document.getElementById('result-img');
    if (r.bbox) {
      img.onload = () => drawBoundingBox(r.bbox, r.defect_score, r.status);
    } else {
      img.onload = () => {
        const canvas = document.getElementById('bbox-canvas');
        canvas.width = img.offsetWidth; canvas.height = img.offsetHeight;
        canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
      };
      document.getElementById('bbox-legend').textContent = 'No defect region detected';
      document.getElementById('bbox-legend').style.color = 'var(--green)';
    }
    img.src = r.image_url;
  } else { bboxWrap.style.display = 'none'; }

  // --- Metric Chips: dynamically color-coded by threshold ---
  const ds = r.defect_score ?? 0;
  const defectEl = document.getElementById('m-defect');
  defectEl.textContent = ds + '%';
  defectEl.className = 'metric-value ' + (ds <= 20 ? 'green' : ds <= 40 ? 'blue' : ds <= 60 ? 'amber' : 'red');
  // Annotate active band on defect score range label
  const defectBands = [['0–20','GOOD','green'],['21–40','MINOR DEFECT','blue'],['41–60','WARNING','amber'],['61–80','HIGH RISK','red'],['81–100','CRITICAL','red']];
  const activeBand = defectBands.find(([r2])=>{const [lo,hi]=r2.split('–').map(Number);return ds>=lo&&ds<=hi;});
  const defectRangeEl = document.getElementById('m-defect-range');
  if (defectRangeEl && activeBand) defectRangeEl.innerHTML = `<strong style="color:var(--${activeBand[2]})">${activeBand[1]}: ${ds}%</strong> &nbsp;|&nbsp; 0–20 GOOD · 21–40 MINOR · 41–60 WARN · 61–80 HIGH · 81–100 CRIT`;

  const rl = r.rust_level ?? 0;
  const rustEl = document.getElementById('m-rust');
  rustEl.textContent = rl + '%';
  rustEl.className = 'metric-value ' + (rl <= 5 ? 'green' : rl <= 15 ? 'amber' : 'red');

  const ol = r.oil_level ?? 0;
  const oilEl = document.getElementById('m-oil');
  oilEl.textContent = ol + '%';
  oilEl.className = 'metric-value ' + (ol <= 5 ? 'green' : ol <= 10 ? 'amber' : 'red');

  const ed = r.edge_density ?? 0;
  const edgeEl = document.getElementById('m-edge');
  edgeEl.textContent = ed + '%';
  edgeEl.className = 'metric-value ' + (ed <= 10 ? 'green' : ed <= 18 ? 'amber' : 'red');

  const alignEl = document.getElementById('m-align');
  alignEl.textContent = r.alignment_ok ? 'NORMAL' : 'DEVIATION';
  alignEl.className   = 'metric-value ' + (r.alignment_ok ? 'green' : 'red');

  const checkSection = document.getElementById('checklist-section');
  const checklist = r.ai_checklist;
  const labels = r.checklist_labels || {};
  const isWheel = (r.component_type === 'wheel');

  if (checklist && Object.keys(checklist).length) {
    checkSection.style.display = 'block';
    const im2 = {
      OK:'fa-circle-check', WARNING:'fa-circle-exclamation',
      CRITICAL:'fa-circle-xmark', UNKNOWN:'fa-circle-question', NOT_DETECTED:'fa-circle-minus'
    };
    document.getElementById('checklist-items').innerHTML = Object.entries(checklist).map(([k,v]) => {
      const s = v.status || 'UNKNOWN';
      const isND = (s === 'NOT_DETECTED' || s === 'UNKNOWN');

      // For wheel: format pipe-delimited detail into readable multi-line text
      let detailHtml = v.detail || '-';
      if (isWheel && detailHtml.includes(' | ')) {
        detailHtml = detailHtml
          .split(' | ')
          .map(seg => '<span style="display:block">' + seg + '</span>')
          .join('');
      }

      // Dim NOT_DETECTED rows
      const rowStyle = isND ? 'opacity:0.5;' : '';
      return '<div class="checklist-item" style="' + rowStyle + '">' +
        '<i class="fa-solid ' + (im2[s]||'fa-circle-question') + ' check-icon ' + s + '"></i>' +
        '<span class="check-name">' + (labels[k] || k.replace(/_/g,' ').toUpperCase()) + '</span>' +
        '<span class="check-detail">' + detailHtml + '</span>' +
        '<span class="check-badge ' + s + '">' + s.replace('_',' ') + '</span></div>';
    }).join('');
    document.getElementById('ai-diagnosis').textContent = r.ai_diagnosis || '';
    const riskBadge = document.getElementById('risk-badge');
    const risk = r.risk_assessment || 'UNKNOWN';
    riskBadge.textContent = 'RISK: ' + risk;
    riskBadge.className   = 'risk-badge ' + risk;
    document.getElementById('action-text').textContent = r.ai_action || '';
  } else { checkSection.style.display = 'none'; }

  const dbar = document.getElementById('decision-bar');
  // Extended color map covering wheel decisions
  const cm = {
    FIT:     ['var(--green-glow)','var(--green-dim)','var(--green)'],
    UNFIT:   ['var(--red-glow)',  'var(--red-dim)',  'var(--red)'],
    MONITOR: ['var(--amber-glow)','var(--amber-dim)','var(--amber)'],
  };
  const [bg,border,color] = cm[status] || ['var(--bg-700)','var(--border)','var(--text-100)'];
  dbar.style.background = bg; dbar.style.borderColor = border; dbar.style.color = color;
  document.getElementById('decision-text').textContent = r.final_decision || (status + ' - Workshop Code: ' + (r.workshop_code || 'N/A'));
  document.getElementById('export-btn').style.display = 'flex';
}

function drawBoundingBox(bbox, defectScore, status) {
  const img = document.getElementById('result-img');
  const canvas = document.getElementById('bbox-canvas');
  const legend = document.getElementById('bbox-legend');
  canvas.width  = img.offsetWidth  || img.naturalWidth;
  canvas.height = img.offsetHeight || img.naturalHeight;
  const sx = canvas.width / img.naturalWidth, sy = canvas.height / img.naturalHeight;
  const x1=bbox.xmin*sx, y1=bbox.ymin*sy, x2=bbox.xmax*sx, y2=bbox.ymax*sy;
  const w=x2-x1, h=y2-y1;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const color = status==='UNFIT'?'#e8364f':'#f0a500';
  const glow  = status==='UNFIT'?'rgba(232,54,79,0.35)':'rgba(240,165,0,0.35)';
  ctx.shadowColor=color; ctx.shadowBlur=18; ctx.fillStyle=glow; ctx.fillRect(x1,y1,w,h);
  ctx.strokeStyle=color; ctx.lineWidth=3; ctx.shadowBlur=0; ctx.strokeRect(x1,y1,w,h);
  const cl=Math.min(w,h)*0.2; ctx.lineWidth=4; ctx.strokeStyle='#fff';
  ctx.beginPath(); ctx.moveTo(x1,y1+cl); ctx.lineTo(x1,y1); ctx.lineTo(x1+cl,y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2-cl,y1); ctx.lineTo(x2,y1); ctx.lineTo(x2,y1+cl); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1,y2-cl); ctx.lineTo(x1,y2); ctx.lineTo(x1+cl,y2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2-cl,y2); ctx.lineTo(x2,y2); ctx.lineTo(x2,y2-cl); ctx.stroke();
  const lbl='DEFECT  '+defectScore+'% ';
  ctx.font='bold 13px JetBrains Mono, monospace';
  const tw=ctx.measureText(lbl).width, lx=Math.max(x1,2), ly=Math.max(y1-28,2);
  ctx.fillStyle=color; ctx.beginPath(); ctx.roundRect(lx,ly,tw+12,22,4); ctx.fill();
  ctx.fillStyle='#fff'; ctx.fillText(lbl,lx+6,ly+15);
  legend.innerHTML='<i class="fa-solid fa-vector-square"></i> Defect localised via Grad-CAM - Score: <strong>'+defectScore+'%</strong>';
  legend.style.color=color;
}

function drawConfidenceRing(value, status) {
  const canvas=document.getElementById('confidence-ring'); if(!canvas)return;
  const ctx=canvas.getContext('2d'), cx=40, cy=40, r=32;
  const color={FIT:'#3db56e',UNFIT:'#e8364f',MONITOR:'#f0a500'}[status]||'#4d91f5';
  ctx.clearRect(0,0,80,80);
  ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.strokeStyle='#1e2538'; ctx.lineWidth=7; ctx.stroke();
  const angle=(value/100)*Math.PI*2-Math.PI/2;
  ctx.beginPath(); ctx.arc(cx,cy,r,-Math.PI/2,angle);
  ctx.strokeStyle=color; ctx.lineWidth=7; ctx.lineCap='round'; ctx.shadowColor=color; ctx.shadowBlur=10; ctx.stroke();
}

async function loadDataset() {
  try {
    const images = await apiFetch('/api/dataset');
    state.datasetCache = images; renderDataset();
  } catch(e) {
    document.getElementById('dataset-grid').innerHTML='<div class="empty-state"><i class="fa-solid fa-triangle-exclamation"></i><p>'+e.message+'</p></div>';
  }
}

function renderDataset() {
  const grid=document.getElementById('dataset-grid');
  let filtered=state.datasetCache;
  if(state.datasetCompFilter!=='all') filtered=filtered.filter(i=>i.component===state.datasetCompFilter);
  if(state.datasetLblFilter!=='all')  filtered=filtered.filter(i=>i.label===state.datasetLblFilter);
  if(!filtered.length){grid.innerHTML='<div class="empty-state"><i class="fa-solid fa-images"></i><p>No images found</p></div>';return;}
  grid.innerHTML=filtered.map(img=>{
    const safeUrl  = img.url.replace(/'/g, '%27');
    const safeName = img.filename.replace(/'/g, '');
    const safeComp = img.component || 'coupler';
    const lightboxLabel = img.filename + ' - ' + (img.component||'').toUpperCase() + ' / ' + img.label.toUpperCase();
    return '<div class="dataset-item ' + img.label + '"' +
      ' draggable="true"' +
      ' ondragstart="handleDatasetDragStart(event,\'' + safeUrl + '\',\'' + safeName + '\',\'' + safeComp + '\')"' +
      ' onclick="openLightbox(\'' + safeUrl + '\',\'' + lightboxLabel.replace(/'/g,'') + '\')">'+
      '<img src="' + img.url + '" alt="' + img.filename + '" loading="lazy" />' +
      '<div class="dataset-item-label">' +
      '<span style="color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:80px;font-size:9px">' + img.filename + '</span>' +
      '<span class="label-badge ' + img.label + '">' + img.label.toUpperCase() + '</span></div>' +
      '<div class="dataset-analyze-overlay" onclick="event.stopPropagation();analyzeDatasetImage(\'' + safeUrl + '\',\'' + safeName + '\',\'' + safeComp + '\')">' +
      '<button class="dataset-analyze-btn"><i class="fa-solid fa-bolt"></i> Analyze in Inspector</button>' +
      '<span class="dataset-drag-hint"><i class="fa-solid fa-hand-pointer"></i> or drag to inspector</span>' +
      '</div>' +
      '</div>';
  }).join('');
}

// Called when dragging a dataset image
function handleDatasetDragStart(event, url, filename, component) {
  event.dataTransfer.effectAllowed = 'copy';
  event.dataTransfer.setData('text/x-dataset-url',  url);
  event.dataTransfer.setData('text/x-dataset-name', filename);
  event.dataTransfer.setData('text/x-dataset-comp', component);
  // Also set plain text so the inspector drop zone gets it
  event.dataTransfer.setData('text/plain', url);
}

// Fetch dataset image, convert to File, switch to Inspector and run auto-detect
async function analyzeDatasetImage(url, filename, compKey) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    const ext  = filename.split('.').pop() || 'jpg';
    const mime = blob.type || ('image/' + ext);
    const file = new File([blob], filename, { type: mime });
    switchTab('inspector');
    setTimeout(() => { setFile(file); }, 300);
  } catch(e) {
    alert('Failed to load dataset image: ' + e.message);
  }
}

function filterDatasetComp(comp, btn) {
  document.querySelectorAll('#comp-filter-chips .chip').forEach(c=>c.classList.remove('active'));
  if(btn)btn.classList.add('active'); state.datasetCompFilter=comp; renderDataset();
}
function filterDatasetLabel(label, btn) {
  const parent=btn?btn.parentElement:null;
  if(parent)parent.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  if(btn)btn.classList.add('active'); state.datasetLblFilter=label; renderDataset();
}
function openLightbox(url,label){document.getElementById('lightbox-img').src=url;document.getElementById('lightbox-label').textContent=label;document.getElementById('lightbox').classList.add('open');}
function closeLightbox(){document.getElementById('lightbox').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLightbox();});

async function loadHistory() {
  try {
    const cf=document.getElementById('log-comp-filter')?.value||'';
    const url='/api/history?limit=100'+(cf?'&component='+cf:'');
    const rows=await apiFetch(url);
    const tbody=document.getElementById('history-tbody');
    if(!rows.length){tbody.innerHTML='<tr><td colspan="9" class="empty-row"><i class="fa-solid fa-inbox"></i> No records found</td></tr>';return;}
    tbody.innerHTML=rows.map(r=>{
      const comp=COMPONENTS[r.component_type];
      return '<tr>'+
        '<td><span style="font-family:JetBrains Mono,monospace;color:var(--blue)">#'+r.id+'</span></td>'+
        '<td style="font-size:12px;font-family:JetBrains Mono,monospace">'+formatDate(r.created_at)+'</td>'+
        '<td><span style="display:flex;align-items:center;gap:6px;font-size:12px"><i class="fa-solid '+(comp?comp.icon:'fa-cog')+'" style="color:var(--blue)"></i>'+(comp?comp.label:(r.component_type||'-'))+'</span></td>'+
        '<td>'+(r.coach_number||'-')+'</td>'+
        '<td style="font-size:12px">'+(r.zone||'-')+'</td>'+
        '<td><span class="recent-status '+r.status+'">'+r.status+'</span></td>'+
        '<td><span style="font-family:JetBrains Mono,monospace">'+(r.confidence??'-')+'%</span></td>'+
        '<td style="font-size:11px;color:var(--text-500)">'+(r.ai_model?r.ai_model.split('/')[1]?.split(':')[0]||r.ai_model:'Local only')+'</td>'+
        '<td><button class="btn btn-ghost btn-sm" onclick="viewInspection('+r.id+')"><i class="fa-solid fa-eye"></i> View</button></td>'+
        '</tr>';
    }).join('');
  } catch(e){console.warn('History load failed:',e);}
}

async function viewInspection(id) {
  try {
    const r=await apiFetch('/api/inspection/'+id);
    state.lastResult=r; state.lastInspectionId=id;
    switchTab('inspector'); setTimeout(()=>renderResult(r),100);
  } catch(e){alert('Failed to load inspection #'+id+': '+e.message);}
}

function renderTrainingGrid() {
  const grid=document.getElementById('train-component-grid'); if(!grid)return;
  grid.innerHTML=Object.entries(COMPONENTS).map(([key,cfg])=>
    '<div class="train-comp-card" id="train-card-'+key+'">' +
    '<div class="train-comp-header">' +
    '<div class="train-comp-name"><i class="fa-solid '+cfg.icon+'"></i> '+cfg.label+'</div>' +
    '<div class="train-comp-badge untrained" id="train-badge-'+key+'">NOT TRAINED</div>' +
    '</div>' +
    '<div class="train-comp-info" id="train-info-'+key+'">Loading...</div>' +
    '<button class="train-comp-btn" id="train-btn-'+key+'" onclick="startTraining(\''+key+'\')">' +
    '<i class="fa-solid fa-play"></i> Train Model</button>' +
    '<div class="train-comp-status" id="train-status-'+key+'"></div>' +
    '</div>'
  ).join('');
}

async function refreshTrainingStatus() {
  try {
    const [modelRes,statsRes]=await Promise.all([apiFetch('/api/models'),apiFetch('/api/stats')]);
    const ms=modelRes.model_status||{}, dc=statsRes.dataset_counts||{};
    Object.keys(COMPONENTS).forEach(key=>{
      const card=document.getElementById('train-card-'+key);
      const badge=document.getElementById('train-badge-'+key);
      const info=document.getElementById('train-info-'+key);
      if(!card)return;
      const trained=ms[key]?.exists;
      const counts=dc[key]||{normal:0,defect:0,total:0};
      card.className='train-comp-card'+(trained?' trained':'');
      badge.className='train-comp-badge '+(trained?'trained':'untrained');
      badge.textContent=trained?'MODEL READY':'NOT TRAINED';
      info.textContent='Dataset: '+counts.normal+' normal - '+counts.defect+' defect - '+counts.total+' total';
    });
  } catch(e){console.warn('Training status refresh failed:',e);}
}

async function startTraining(component) {
  const btn=document.getElementById('train-btn-'+component);
  const status=document.getElementById('train-status-'+component);
  btn.disabled=true; btn.innerHTML='<i class="fa-solid fa-circle-notch fa-spin"></i> Starting...'; status.textContent='';
  try {
    await apiFetch('/api/train?component='+component,{method:'POST'});
    document.getElementById('train-log-wrap').style.display='block';
    document.getElementById('train-log-comp').textContent=COMPONENTS[component]?.label||component;
    document.getElementById('train-log').textContent='';
    document.getElementById('train-progress-bar').style.width='0%';
    status.textContent='Training in progress...';
    pollTrainingStatus(component);
  } catch(e){
    alert('Training failed to start: '+e.message);
    btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-play"></i> Train Model';
  }
}

function pollTrainingStatus(component) {
  const pid=setInterval(async()=>{
    try {
      const s=await apiFetch('/api/train/status?component='+component);
      const logEl=document.getElementById('train-log');
      logEl.textContent=(s.log||[]).join('\n'); logEl.scrollTop=logEl.scrollHeight;
      const match=(s.log?.slice(-1)[0]||'').match(/Epoch (\d+)\/(\d+)/);
      if(match) document.getElementById('train-progress-bar').style.width=((parseInt(match[1])/parseInt(match[2]))*100)+'%';
      const btn=document.getElementById('train-btn-'+component);
      const status=document.getElementById('train-status-'+component);
      if(s.done){
        clearInterval(pid); status.textContent='Training complete! Model saved.';
        document.getElementById('train-progress-bar').style.width='100%';
        btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-play"></i> Re-Train Model';
        refreshTrainingStatus(); loadStats();
      } else if(s.error){
        clearInterval(pid); status.textContent=s.error;
        btn.disabled=false; btn.innerHTML='<i class="fa-solid fa-rotate"></i> Retry Training';
      } else if(s.running){ btn.innerHTML='<i class="fa-solid fa-circle-notch fa-spin"></i> Training...'; }
    } catch(e){console.warn('Poll error:',e);}
  },2500);
}

async function loadModelInfo() {
  try {
    const m=await apiFetch('/api/models');
    const box=document.getElementById('model-info-box');
    const ms=m.model_status||{};
    box.innerHTML=Object.entries(ms).map(([key,val])=>{
      const comp=COMPONENTS[key];
      return '<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid var(--border)">' +
        '<i class="fa-solid '+(comp?comp.icon:'fa-cog')+'" style="color:var(--blue);width:14px;text-align:center"></i>' +
        '<span style="flex:1;font-size:12px">'+val.label+'</span>' +
        '<span style="font-size:11px;color:'+(val.exists?'var(--green)':'var(--amber)')+'">'+
        (val.exists?'Ready':'Not trained')+'</span></div>';
    }).join('')||'<div>No model info</div>';
  } catch(e){console.warn('Model info failed:',e);}
}




function exportReport() {
  const r = state.lastResult; if (!r) return;
  const now = new Date().toLocaleString('en-IN', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false });
  const comp = COMPONENTS[r.component_type] || {};

  // Pull live form values as fallback if result fields are empty
  const coachNum  = r.coach_number   || document.getElementById('inp-coach-number')?.value || '—';
  const coachType = r.coach_type     || document.getElementById('inp-coach-type')?.value    || 'LHB';
  const depot     = r.depot          || document.getElementById('inp-depot')?.value         || '—';
  const zone      = r.zone           || document.getElementById('inp-zone')?.value          || '—';
  const inspector = r.inspector_name || document.getElementById('inp-inspector')?.value     || '—';
  const notes     = r.notes          || document.getElementById('inp-notes')?.value         || '—';
  const wkCode    = r.workshop_code  || (r.status === 'UNFIT' ? 'SEND TO WORKSHOP' : r.status === 'MONITOR' ? 'PERIODIC WATCH' : 'CLEARED');

  const statusColors = { FIT:'#155724:#d4edda', UNFIT:'#721c24:#f8d7da', MONITOR:'#856404:#fff3cd' };
  const [fgC, bgC] = (statusColors[r.status] || '#111:#eee').split(':');
  const statusLabel = r.status === 'FIT' ? 'FIT FOR RUN' : r.status === 'UNFIT' ? 'UNFIT — SEND TO WORKSHOP' : 'MONITOR CLOSELY';

  const checklistHTML = (r.ai_checklist && Object.keys(r.ai_checklist).length)
    ? '<h2>Diagnostic Checklist</h2>' +
      '<table><tr><th style="width:28%">Check Point</th><th style="width:12%">Status</th><th>Inspector\'s Observation</th></tr>' +
      Object.entries(r.ai_checklist).map(([k, v]) => {
        const sc = v.status === 'OK' ? '#155724:#d4edda' : v.status === 'WARNING' ? '#856404:#fff3cd' : '#721c24:#f8d7da';
        const [stFg, stBg] = sc.split(':');
        return `<tr><td><strong>${(r.checklist_labels?.[k] || k).toUpperCase().replace(/_/g,' ')}</strong></td>` +
               `<td><span style="background:${stBg};color:${stFg};padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px">${v.status}</span></td>` +
               `<td style="font-size:12px;line-height:1.5">${v.detail || '—'}</td></tr>`;
      }).join('') + '</table>'
    : '';

  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>RAIL-SENSE — ${comp.label || r.component_type} Inspection Report</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;padding:32px 40px;max-width:900px;margin:0 auto;color:#1a1a2e;font-size:13px;line-height:1.6}
  .header-bar{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:4px solid #003893;padding-bottom:14px;margin-bottom:18px}
  .report-title{font-size:22px;font-weight:800;color:#003893;letter-spacing:.5px}
  .report-sub{font-size:11px;color:#555;margin-top:4px}
  .status-badge{padding:8px 20px;border-radius:6px;font-weight:800;font-size:15px;background:${bgC};color:${fgC};text-align:center}
  .report-id{font-size:11px;color:#888;text-align:center;margin-top:5px}
  h2{color:#003893;font-size:14px;font-weight:700;margin:22px 0 8px;letter-spacing:.3px;text-transform:uppercase;border-left:4px solid #003893;padding-left:10px}
  table{width:100%;border-collapse:collapse;margin-top:6px}
  td,th{border:1px solid #ccc;padding:8px 12px;font-size:12.5px;vertical-align:top}
  th{background:#003893;color:#fff;font-weight:600;text-align:left}
  tr:nth-child(even){background:#f7f9fc}
  .meta-label{color:#555;font-weight:600;width:18%}
  .highlight-row td{background:#eef2ff;font-weight:600}
  .sig-row{display:flex;gap:40px;margin-top:50px}
  .sig-box{flex:1;border-top:2px solid #333;padding-top:8px;text-align:center;font-size:11px;color:#555}
  .footer{margin-top:36px;border-top:2px solid #ddd;padding-top:14px;font-size:11px;color:#888;display:flex;justify-content:space-between}
  .rdso-banner{background:#003893;color:#fff;padding:6px 16px;border-radius:4px;font-size:11px;font-weight:600;margin-bottom:16px;display:inline-block}
  @media print{body{padding:14px 20px}.sig-row{margin-top:30px}}
</style></head><body>
<div class="rdso-banner">🇮🇳 Indian Railways · RDSO Compliant · AI-Assisted Mechanical Inspection System</div>
<div class="header-bar">
  <div>
    <div class="report-title">RAIL-SENSE — ${comp.label || r.component_type} Inspection Report</div>
    <div class="report-sub">LHB Fiat Bogie Diagnostic Suite · NDLS Depot · v1.0</div>
  </div>
  <div>
    <div class="status-badge">${statusLabel}</div>
    <div class="report-id">Report #${r.id || 'N/A'} &nbsp;|&nbsp; ${now}</div>
  </div>
</div>

<h2>Coach Metadata</h2>
<table>
  <tr><th class="meta-label">Parameter</th><th>Value</th><th class="meta-label">Parameter</th><th>Value</th></tr>
  <tr><td class="meta-label">Component</td><td><strong>${comp.label || r.component_type}</strong> (${comp.sub || ''})</td><td class="meta-label">Coach Type</td><td><strong>${coachType}</strong></td></tr>
  <tr><td class="meta-label">Coach Number</td><td><strong>${coachNum}</strong></td><td class="meta-label">Railway Zone</td><td>${zone}</td></tr>
  <tr><td class="meta-label">Depot / Workshop</td><td><strong>${depot}</strong></td><td class="meta-label">Date &amp; Time</td><td>${now}</td></tr>
  <tr><td class="meta-label">Inspector Name</td><td><strong>${inspector}</strong></td><td class="meta-label">Workshop Code</td><td><strong>${wkCode}</strong></td></tr>
  <tr><td class="meta-label">Inspector Notes</td><td colspan="3" style="font-style:italic;color:#444">${notes}</td></tr>
</table>

<h2>AI Analytical Results &amp; Reference Guide</h2>
<table>
  <tr><th style="width:22%">Parameter</th><th style="width:13%">Detected Value</th><th style="width:12%">Status</th><th>Reference Range &amp; Technician Guidance</th></tr>
  <tr class="highlight-row"><td><strong>Overall Status</strong></td><td><span style="background:${bgC};color:${fgC};padding:2px 10px;border-radius:4px;font-weight:bold">${r.status}</span></td><td>—</td><td>FIT = Component serviceable · MONITOR = Watch closely · UNFIT = Remove from service</td></tr>
  <tr><td>AI Confidence</td><td><strong>${r.confidence}%</strong></td><td>${r.confidence >= 75 ? '✅ HIGH' : r.confidence >= 50 ? '⚠ MEDIUM' : '❌ LOW'}</td><td>&gt;75% High confidence · 50–75% Medium · &lt;50% Low — recommend manual verification</td></tr>
  <tr><td>Defect Score</td><td><strong>${r.defect_score}%</strong></td><td style="background:${r.defect_score<=20?'#d4edda':r.defect_score<=40?'#d1ecf1':r.defect_score<=60?'#fff3cd':'#f8d7da'};font-weight:bold">${r.defect_score<=20?'GOOD':r.defect_score<=40?'MINOR DEFECT':r.defect_score<=60?'WARNING':r.defect_score<=80?'HIGH RISK':'CRITICAL'}</td><td>0–20: GOOD | 21–40: Minor Defect | 41–60: Warning | 61–80: High Risk | 81–100: CRITICAL — Immediate action</td></tr>
  <tr><td>Rust / Corrosion</td><td>${r.rust_level}%</td><td>${r.rust_level<=5?'✅ Normal':r.rust_level<=15?'⚠ Elevated':'❌ High'}</td><td>0–5% Normal · 5–15% Elevated (schedule descaling) · &gt;15% High corrosion — inspect rim/web/hub integrity</td></tr>
  <tr><td>Oil / Fluid Leakage</td><td>${r.oil_level ?? 0}%</td><td>${(r.oil_level??0)<=5?'✅ Clean':(r.oil_level??0)<=10?'⚠ Monitor':'❌ Risk'}</td><td>0–5% Clean · 5–10% Monitor seal condition · &gt;10% Oil leakage — inspect axle bearing seal immediately</td></tr>
  <tr><td>Edge Anomaly Index</td><td>${r.edge_density}%</td><td>${r.edge_density<=10?'✅ Normal':r.edge_density<=18?'⚠ Irregular':'❌ Critical'}</td><td>0–10% Normal surface · 10–18% Irregular (possible wear/flat) · &gt;18% Critical — check for wheel flat, crack, or tread defect</td></tr>
  <tr><td>Alignment Status</td><td>${r.alignment_ok ? 'NORMAL' : '⚠ DEVIATION'}</td><td>${r.alignment_ok?'✅ OK':'❌ Deviated'}</td><td>NORMAL = Within RDSO geometric tolerance · DEVIATION = Inspect wheel diameter difference and flange geometry</td></tr>
  <tr><td>Risk Assessment</td><td><strong>${r.risk_assessment || '—'}</strong></td><td>—</td><td>LOW: Routine service · MEDIUM: Expedite inspection · HIGH: Withdraw at next depot · CRITICAL: Do not run — remove from service</td></tr>
</table>

${checklistHTML}

${r.ai_diagnosis ? `<h2>AI Diagnosis</h2><p style="font-size:13px;line-height:1.7;padding:10px;background:#f7f9fc;border-left:4px solid #003893;border-radius:0 4px 4px 0">${r.ai_diagnosis}</p>` : ''}

${r.ai_action ? `<h2>Immediate Action Required</h2><p style="font-size:13px;color:#721c24;font-weight:700;background:#f8d7da;padding:10px 14px;border-radius:4px;border-left:4px solid #721c24">${r.ai_action}</p>` : ''}

<h2>Final Decision</h2>
<p style="font-size:15px;font-weight:800;color:${fgC};background:${bgC};padding:10px 16px;border-radius:6px;display:inline-block">${r.final_decision || statusLabel}</p>

<div class="sig-row">
  <div class="sig-box">Inspector Signature<br><br><em>${inspector}</em></div>
  <div class="sig-box">Supervisor / SSE (Mech)</div>
  <div class="sig-box">TXR / Station Master</div>
</div>

<div class="footer">
  <span>RAIL-SENSE LHB Bogie AI Diagnostic System &nbsp;|&nbsp; ${now}</span>
  <span>AI-assisted — must be validated by RDSO-certified engineer</span>
</div>
<script>window.onload = () => window.print();<\/script>
</body></html>`);
  win.document.close();
}


function formatDate(iso, short=false) {
  if(!iso)return '-';
  const d=new Date(iso); if(isNaN(d))return iso;
  if(short)return d.toLocaleDateString('en-IN',{month:'short',day:'numeric'});
  return d.toLocaleString('en-IN',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false});
}

function showToast(msg, type) {
  const t=document.createElement('div');
  t.textContent=msg;
  const bg = type==='error' ? 'hsl(346,80%,50%)' : type==='blue' ? '#003893' : 'hsl(142,68%,45%)';
  t.style.cssText='position:fixed;bottom:24px;right:24px;z-index:9999;background:'+bg+';color:#fff;padding:10px 20px;border-radius:8px;font-family:Outfit,sans-serif;font-size:14px;font-weight:600;box-shadow:0 4px 20px rgba(0,0,0,0.4);animation:slideDown 0.3s ease;';
  document.body.appendChild(t); setTimeout(()=>t.remove(),2800);
}

/* ═══════════════════════════════════════════════════════
   AUTO COMPONENT DETECTION
   Uploads the image to /api/detect-component which runs
   ALL 6 CNN models and returns the highest-confidence match.
   The result is used to auto-select the component in the UI.
═══════════════════════════════════════════════════════ */
async function autoDetectComponent(file) {
  const banner   = document.getElementById('auto-detect-banner');
  const titleEl  = document.getElementById('auto-detect-title');
  const subEl    = document.getElementById('auto-detect-sub');
  const scoresEl = document.getElementById('auto-detect-scores');
  const iconEl   = document.getElementById('auto-detect-icon-fa');

  if (!banner) return;

  // Show the banner in loading state
  banner.style.display = 'block';
  titleEl.textContent  = 'Detecting component…';
  subEl.textContent    = 'Running all 6 CNN models in parallel';
  scoresEl.innerHTML   = '';
  if (iconEl) iconEl.className = 'fa-solid fa-circle-notch fa-spin';

  try {
    const fd = new FormData();
    fd.append('file', file);
    const result = await apiFetch('/api/detect-component', { method: 'POST', body: fd });

    const detected = result.detected_component;
    const cfg      = COMPONENTS[detected];

    // Auto-select the detected component in the UI
    const compBtn = document.querySelector('[data-comp="' + detected + '"]');
    selectComponent(detected, compBtn);

    // Update banner icon
    if (iconEl) iconEl.className = 'fa-solid ' + (cfg ? cfg.icon : 'fa-check-circle');

    // Update texts
    titleEl.textContent = '✓ Detected: ' + (result.detected_label || (cfg ? cfg.label : detected));
    subEl.textContent   = 'Confidence: ' + result.confidence + '% · Component auto-selected';

    // Show score pills for all 6 components, sorted by confidence
    const allScores = result.all_scores || {};
    scoresEl.innerHTML = Object.entries(allScores)
      .sort((a, b) => b[1].confidence - a[1].confidence)
      .map(([key, s]) => {
        const shortLabel = (s.label || key).split(' ').slice(0, 2).join(' ');
        const isWinner   = key === detected;
        return '<span class="detect-score-pill' + (isWinner ? ' winner' : '') + '">' +
               shortLabel + ' ' + s.confidence + '%</span>';
      })
      .join('');

    // Show a toast and auto-dismiss banner after 7s
    showToast('Component auto-detected: ' + (cfg ? cfg.label : detected), 'blue');
    setTimeout(() => {
      if (banner) banner.style.animation = 'slideDown 0.3s ease reverse';
      setTimeout(() => { if (banner) banner.style.display = 'none'; }, 300);
    }, 7000);

    return detected;

  } catch(e) {
    // Detection failed — user can still pick manually
    if (iconEl) iconEl.className = 'fa-solid fa-triangle-exclamation';
    titleEl.textContent = 'Auto-detect unavailable — select component manually';
    subEl.textContent   = e.message.includes('API') ? 'No trained models found for detection' : e.message;
    setTimeout(() => { if (banner) banner.style.display = 'none'; }, 5000);
    return null;
  }
}
