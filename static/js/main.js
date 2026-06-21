/* ParkingIQ — Frontend Logic */

// ─── API Configuration ────────────────────────────────────────────────
// If hosting the frontend on Netlify and backend on Render, you can either:
// 1. Keep API_BASE = "" and use Netlify redirects/rewrites (see netlify.toml)
// 2. Or set API_BASE to your Render backend URL (e.g., "https://parkingiq.onrender.com")
const API_BASE = "";

// ─── State ───────────────────────────────────────────────────────────
let currentJobId = null;
let pollTimer    = null;
let charts       = {};

const CHART_COLORS = ["#ff6b35","#4ecdc4","#45b7d1","#96ceb4","#ffeaa7","#fd79a8","#a29bfe","#55efc4","#fab1a0","#74b9ff"];
const TIER_COLORS  = { URGENT:"#f54b4b", PRIORITY:"#f5c33a", MODERATE:"#4488ff", ROUTINE:"#22c77a" };

// ─── Upload wiring ────────────────────────────────────────────────────
const dropZone  = document.getElementById("dropZone");
const csvInput  = document.getElementById("csvInput");

csvInput.addEventListener("change", e => { if (e.target.files[0]) startUpload(e.target.files[0]); });
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragging");
  if (e.dataTransfer.files[0]) startUpload(e.dataTransfer.files[0]);
});

// ─── Upload & poll ────────────────────────────────────────────────────
async function startUpload(file) {
  if (!file.name.endsWith(".csv")) { alert("Please upload a CSV file."); return; }

  setNavStatus("running", "Uploading…");
  showProgress(0, "Uploading file…");

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch(API_BASE + "/api/upload", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) { showError(data.error); return; }
    currentJobId = data.job_id;
    pollTimer = setInterval(() => pollStatus(currentJobId), 800);
  } catch(e) {
    showError("Upload failed: " + e.message);
  }
}

async function pollStatus(jobId) {
  try {
    const resp = await fetch(API_BASE + `/api/status/${jobId}`);
    const data = await resp.json();

    showProgress(data.progress, data.message);

    if (data.status === "done") {
      clearInterval(pollTimer);
      setNavStatus("done", "Analysis complete");
      renderAll(data.result, jobId);
    } else if (data.status === "error") {
      clearInterval(pollTimer);
      setNavStatus("error", "Error");
      showError(data.message);
    }
  } catch(e) {
    console.error("Poll error", e);
  }
}

// ─── Progress UI ──────────────────────────────────────────────────────
function showProgress(pct, msg) {
  document.getElementById("progressWrap").style.display = "block";
  document.getElementById("progressBar").style.width    = pct + "%";
  document.getElementById("progressMsg").textContent    = msg;
  document.getElementById("progressPct").textContent    = pct + "%";
  dropZone.style.display = pct >= 100 ? "none" : "none"; // keep hidden once started
  if (pct === 0) dropZone.style.display = "none";
}

function showError(msg) {
  document.getElementById("progressMsg").textContent = "⚠️ " + msg;
  document.getElementById("progressMsg").style.color = "#f54b4b";
}

function setNavStatus(state, msg) {
  const el  = document.getElementById("navStatus");
  const dot = el.querySelector(".status-dot");
  dot.className = "status-dot " + state;
  el.childNodes[1].textContent = " " + msg;
}

// ─── Render all results ───────────────────────────────────────────────
function renderAll(result, jobId) {
  renderMetrics(result.metrics);
  renderCharts(result);
  renderPriorityTable(result.junctions);
  storeHeatmapSlots(result.heatmap_slots, jobId);
  renderHeatmap(jobId);
  renderDownloads(result.files, jobId);

  // Show all sections
  ["dashboard","heatmap","priority","downloads"].forEach(id => {
    document.getElementById(id).style.display = "block";
  });

  // Smooth scroll to dashboard
  setTimeout(() => document.getElementById("dashboard").scrollIntoView({ behavior: "smooth" }), 200);
}

// ─── Metrics ──────────────────────────────────────────────────────────
function renderMetrics(m) {
  const defs = [
    { label:"Total Violations",   value: fmtNum(m.total_violations),    sub:"records analyzed",       accent: true  },
    { label:"Avg Delay/Violation", value: m.avg_delay_min + " min",      sub:"per illegal vehicle"                  },
    { label:"Total Delay Burden",  value: fmtNum(m.total_delay_hours) + " hrs", sub:"aggregate traffic lost"        },
    { label:"Avg Speed Reduction", value: m.avg_speed_reduction + "%",   sub:`max ${m.max_speed_reduction}%`        },
    { label:"Junctions Scored",    value: m.junctions_scored,            sub:"EPI computed"                         },
    { label:"URGENT Junctions",    value: m.urgent_junctions,            sub:"EPI > 75",               accent: true  },
    { label:"Critical Violations", value: fmtNum(m.critical_count),      sub:`+ ${fmtNum(m.high_count)} HIGH`       },
    { label:"Priority Junctions",  value: m.priority_junctions,          sub:"EPI 50–75"                            },
  ];
  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = defs.map(d => `
    <div class="metric-card ${d.accent ? 'accent' : ''}">
      <div class="metric-label">${d.label}</div>
      <div class="metric-value ${d.accent ? 'accent' : ''}">${d.value}</div>
      <div class="metric-sub">${d.sub}</div>
    </div>`).join("");
}

// ─── Charts ───────────────────────────────────────────────────────────
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

function chartDefaults(ctx) {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color:"#8898b8", font:{size:10} }, grid: { color:"rgba(255,255,255,0.04)" } },
      y: { ticks: { color:"#8898b8", font:{size:10} }, grid: { color:"rgba(255,255,255,0.04)" } },
    }
  };
}

function renderCharts(r) {
  // Hourly
  destroyChart("hourlyChart");
  const hours = Array.from({length:24},(_,i)=>i<12?(i===0?"12a":i+"a"):(i===12?"12p":(i-12)+"p"));
  const hColors = r.hourly.map(v => v > 65 ? "#ff4444" : v > 45 ? "#ff8800" : "#4ecdc4");
  charts.hourlyChart = new Chart(document.getElementById("hourlyChart"), {
    type: "bar",
    data: { labels: hours, datasets: [{ data: r.hourly, backgroundColor: hColors, borderRadius: 3, borderSkipped: false }] },
    options: { ...chartDefaults(), plugins: { legend:{display:false}, tooltip:{ callbacks:{ label: ctx => `Impact: ${ctx.raw.toFixed(1)}` }}}}
  });

  // Violation donut
  destroyChart("violChart");
  charts.violChart = new Chart(document.getElementById("violChart"), {
    type: "doughnut",
    data: { labels: r.viol_data.labels, datasets: [{ data: r.viol_data.values, backgroundColor: CHART_COLORS, borderWidth: 0 }] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:"bottom", labels:{ color:"#8898b8", font:{size:10}, padding:10, boxWidth:10 }}}}
  });

  // Vehicle bar
  destroyChart("vehChart");
  charts.vehChart = new Chart(document.getElementById("vehChart"), {
    type: "bar",
    data: { labels: r.veh_data.labels, datasets: [{ data: r.veh_data.values, backgroundColor: "#4488ff", borderRadius: 4, borderSkipped: false }] },
    options: { ...chartDefaults(), plugins:{ legend:{display:false} }, scales:{ x:{ ticks:{color:"#8898b8",font:{size:9},maxRotation:30}, grid:{color:"rgba(255,255,255,0.04)"} }, y:{ ticks:{color:"#8898b8",font:{size:10}}, grid:{color:"rgba(255,255,255,0.04)"} }}}
  });

  // Risk tier donut
  destroyChart("tierChart");
  const tierClr = r.tier_data.labels.map(l => ({"CRITICAL":"#f54b4b","HIGH":"#ff8800","MEDIUM":"#4488ff","LOW":"#22c77a"})[l] || "#888");
  charts.tierChart = new Chart(document.getElementById("tierChart"), {
    type: "doughnut",
    data: { labels: r.tier_data.labels, datasets: [{ data: r.tier_data.values, backgroundColor: tierClr, borderWidth: 0 }] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:"bottom", labels:{ color:"#8898b8", font:{size:10}, padding:8, boxWidth:10 }}}}
  });

  // Monthly bar
  destroyChart("monthChart");
  charts.monthChart = new Chart(document.getElementById("monthChart"), {
    type: "bar",
    data: { labels: r.monthly_data.labels, datasets: [{ data: r.monthly_data.values, backgroundColor: "#22c77a", borderRadius: 4, borderSkipped: false }] },
    options: { ...chartDefaults(), plugins:{ legend:{display:false} }, scales:{ x:{ ticks:{color:"#8898b8",font:{size:10},maxRotation:35,autoSkip:false}, grid:{color:"rgba(255,255,255,0.04)"} }, y:{ ticks:{color:"#8898b8",font:{size:10}}, grid:{color:"rgba(255,255,255,0.04)"} }}}
  });
}

// ─── Priority table ───────────────────────────────────────────────────
function renderPriorityTable(junctions) {
  const tbody = document.getElementById("priorityBody");
  tbody.innerHTML = junctions.map(j => {
    const tier = String(j.priority_tier || "ROUTINE");
    const epiW = Math.round((j.enforcement_priority_index || 0));
    return `<tr>
      <td>${j.rank}</td>
      <td>${j.junction_name}</td>
      <td>
        <div class="epi-bar-wrap">
          <div class="epi-bar-bg"><div class="epi-bar-fill" style="width:${epiW}%"></div></div>
          <span class="epi-val">${(j.enforcement_priority_index||0).toFixed(1)}</span>
        </div>
      </td>
      <td><span class="tier-badge tier-${tier}">${tier}</span></td>
      <td>${fmtNum(j.total_violations)}</td>
      <td>${(j.avg_delay_min||0).toFixed(3)} min</td>
      <td>${(j.daily_hours_lost||0).toFixed(1)} hrs</td>
      <td>${(j.avg_speed_red||0).toFixed(1)}%</td>
      <td>${fmtNum(j.critical_count||0)}</td>
    </tr>`;
  }).join("");
}

// ─── Heatmap iframe + timeline slider ────────────────────────────────
let _heatmapSlots = {};
let _heatmapJobId = null;

const SLOT_RANGES = {
  all:       "All 24 hours",
  night:     "10 PM – 6 AM",
  morning:   "6 AM – 12 PM",
  afternoon: "12 PM – 5 PM",
  evening:   "5 PM – 10 PM",
};

function storeHeatmapSlots(slots, jobId) {
  _heatmapSlots = slots || {};
  _heatmapJobId = jobId;
}

function renderHeatmap(jobId) {
  _heatmapJobId = jobId;
  const frame = document.getElementById("mapFrame");
  frame.src   = API_BASE + `/api/download/${jobId}/congestion_heatmap.html`;
}

function setSlot(slot, btn) {
  // Update active button
  document.querySelectorAll(".smode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  // Update range label
  const lbl = document.getElementById("slotRangeLabel");
  if (lbl) lbl.textContent = SLOT_RANGES[slot] || "";

  // Swap iframe src to the pre-generated slot heatmap
  const frame    = document.getElementById("mapFrame");
  const filename = _heatmapSlots[slot] || "congestion_heatmap.html";
  frame.src      = API_BASE + `/api/download/${_heatmapJobId}/${filename}`;
}

// ─── Downloads ────────────────────────────────────────────────────────
function renderDownloads(files, jobId) {
  const defs = [
    { key:"violation_scores",   icon:"📊", name:"Violation Impact Scores",  desc:"Per-row: impact score, delay minutes, speed reduction %, risk tier" },
    { key:"junction_priority",  icon:"📍", name:"Junction Priority Index",   desc:"EPI scores, delay burden, tier classification for all junctions" },
    { key:"enforcement_report", icon:"📋", name:"Enforcement Report",        desc:"Full text report with hotspot analysis and patrol recommendations" },
    { key:"charts",             icon:"📈", name:"Analysis Charts (PNG)",     desc:"10-panel static visualization: trends, geo map, heatmaps" },
    { key:"heatmap",            icon:"🗺️", name:"Interactive Heatmap (HTML)", desc:"Folium heatmap with junction markers — open in any browser" },
  ];
  const grid = document.getElementById("downloadsGrid");
  grid.innerHTML = defs.map(d => `
    <div class="dl-card">
      <div class="dl-icon">${d.icon}</div>
      <div class="dl-name">${d.name}</div>
      <div class="dl-desc">${d.desc}</div>
      <a class="dl-btn" href="${API_BASE}/api/download/${jobId}/${files[d.key]}" download>
        ↓ Download
      </a>
    </div>`).join("");
}

// ─── Utils ────────────────────────────────────────────────────────────
function fmtNum(n) { return (n||0).toLocaleString(); }
