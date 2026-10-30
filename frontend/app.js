/**
 * app.js — Medical Diagnostic Assistant  (Stage 2)
 * Lab PDF parsing, X-ray inventory, findings summary, dataset stats.
 * No inference. No personal data stored. No external requests.
 */

"use strict";

const API = "";   // same origin — Flask serves frontend

/* ─────────────────────────────────────────────────────────────
   Helpers
───────────────────────────────────────────────────────────── */

function $(id) { return document.getElementById(id); }
function fmt(n) {
  if (n == null) return "—";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(2) + " MB";
}
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

async function parseJsonResponse(res) {
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    const text = await res.text().catch(() => "");
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      throw new Error(`Server is warming up or temporarily busy (HTTP ${res.status}). Please wait 10-15 seconds and try again.`);
    }
    const cleanSnippet = text.replace(/<[^>]*>?/gm, "").trim().slice(0, 120);
    throw new Error(`Server returned HTTP ${res.status}${cleanSnippet ? `: ${cleanSnippet}` : ""}`);
  }
  return await res.json();
}

/* ─────────────────────────────────────────────────────────────
   Application State
───────────────────────────────────────────────────────────── */

const state = {
  labFile: null,    // File object
  labResult: null,    // parsed API response
  xrayFile: null,    // File object
  xrayResult: null,    // API response
  notes: "",
  symptoms: [],      // Array of { id, text, is_red_flag }
  analyzedSymptoms: null,  // /api/symptoms/analyze response
  voiceScript: null,    // /api/summary/voice-script response
  voiceLang: "en-US", // "en-US" or "hi-IN" for voice assistant
  voiceTypingLang: "en-US", // "en-US" or "hi-IN" for speech recognition
  isSpeaking: false,
  isListening: false,
  speechRate: 1.0,
  datasetStats: null,    // /api/lab/dataset-stats response
  inventoryResult: null,    // /api/xray/inventory response
};

/* ─────────────────────────────────────────────────────────────
   Continuous Multi-Section Scroll & Sidebar Navigation
───────────────────────────────────────────────────────────── */

const SECTION_IDS = ["labreport", "xray", "notes", "findings"];

const NAV_TITLES = {
  labreport: "Lab Report Analysis",
  xray: "Chest X-Ray Model",
  notes: "Symptom Notes",
  findings: "Findings Summary",
};

let currentActiveSection = "labreport";
let isUserScrollingProgrammatically = false;
let programmaticScrollTimer = null;
let scrollTicking = false;

/**
 * Highlight the specified section in the sidebar & update mobile title.
 */
function highlightNav(id) {
  if (!id) return;
  currentActiveSection = id;

  document.querySelectorAll(".nav-item").forEach(btn => {
    const active = btn.dataset.panel === id;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-current", active ? "page" : "false");
  });

  document.querySelectorAll(".panel").forEach(p => {
    p.classList.toggle("active-section", p.id === "panel-" + id);
  });

  const mt = $("mobileTitle");
  if (mt) mt.textContent = NAV_TITLES[id] ?? "";

  // Auto-sync findings summary if user scrolls into section 4
  if (id === "findings") {
    syncFindingsPanel();
    const grid = $("datasetStatusGrid");
    if (grid && !grid.innerHTML.trim()) loadDatasetStatus();
  }
}

/**
 * Smoothly scroll to a specific section by panel ID.
 */
function scrollToSection(id, smooth = true) {
  const panel = $("panel-" + id);
  if (!panel) return;

  isUserScrollingProgrammatically = true;
  highlightNav(id);

  const ribbonHeight = 44;
  const isMobile = window.innerWidth <= 900;
  const topbarHeight = isMobile ? 52 : 0;
  const offset = ribbonHeight + topbarHeight + 16;

  const rect = panel.getBoundingClientRect();
  const targetTop = window.pageYOffset + rect.top - offset;

  window.scrollTo({
    top: Math.max(0, targetTop),
    behavior: smooth ? "smooth" : "auto",
  });

  closeMobileSidebar();

  clearTimeout(programmaticScrollTimer);
  programmaticScrollTimer = setTimeout(() => {
    isUserScrollingProgrammatically = false;
  }, 750);
}

/**
 * Backward compatibility alias: any internal button or code calling activatePanel(id)
 * will smoothly scroll to that section.
 */
function activatePanel(id) {
  scrollToSection(id, true);
}

/**
 * Scroll spy: Determine which section is currently in view and auto-highlight sidebar.
 */
function updateActiveSectionOnScroll() {
  if (isUserScrollingProgrammatically) return;

  const ribbonHeight = 44;
  const isMobile = window.innerWidth <= 900;
  const topbarHeight = isMobile ? 52 : 0;
  const offset = ribbonHeight + topbarHeight + 90; // threshold from top of viewport

  const scrollBottom = window.innerHeight + window.pageYOffset;
  const docHeight = document.documentElement.scrollHeight || document.body.scrollHeight;

  // If user scrolled close to the bottom of the page, highlight findings summary
  if (scrollBottom >= docHeight - 90) {
    highlightNav("findings");
    return;
  }

  let activeId = SECTION_IDS[0];
  for (const id of SECTION_IDS) {
    const el = $("panel-" + id);
    if (!el) continue;
    const top = el.getBoundingClientRect().top;
    if (top <= offset) {
      activeId = id;
    } else {
      break;
    }
  }

  highlightNav(activeId);
}

// Throttled scroll listener
window.addEventListener("scroll", () => {
  if (!scrollTicking) {
    window.requestAnimationFrame(() => {
      updateActiveSectionOnScroll();
      scrollTicking = false;
    });
    scrollTicking = true;
  }
}, { passive: true });

// Attach click & keyboard listeners to sidebar navigation items
document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", e => {
    e.preventDefault();
    scrollToSection(btn.dataset.panel);
  });
  btn.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      scrollToSection(btn.dataset.panel);
    }
  });
});

/* Cross-panel nav buttons */
function navBtn(id, panel) {
  const el = $(id);
  if (el) {
    el.addEventListener("click", e => {
      e.preventDefault();
      scrollToSection(panel);
    });
  }
}
navBtn("goToLabReport", "labreport");
navBtn("goToXray", "xray");
navBtn("goToNotes", "notes");
navBtn("goToFindings", "findings");
navBtn("goToFindingsFromNotes", "findings");
navBtn("backToTopBtn", "labreport");


/* ─────────────────────────────────────────────────────────────
   Mobile Sidebar
───────────────────────────────────────────────────────────── */

const sidebar = $("sidebar");
const overlay = $("mobileOverlay");
const hamburger = $("hamburger");

function openMobileSidebar() {
  sidebar.classList.add("open");
  overlay.classList.add("visible");
  overlay.setAttribute("aria-hidden", "false");
  hamburger.setAttribute("aria-expanded", "true");
}
function closeMobileSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("visible");
  overlay.setAttribute("aria-hidden", "true");
  hamburger.setAttribute("aria-expanded", "false");
}
if (hamburger) hamburger.addEventListener("click", openMobileSidebar);
if (overlay) overlay.addEventListener("click", closeMobileSidebar);

/* ─────────────────────────────────────────────────────────────
   Desktop & Mobile Sidebar Collapse / Expand Toggle
───────────────────────────────────────────────────────────── */

function initSidebarCollapse() {
  const toggleBtn = $("sidebarToggleBtn");
  const sidebar = $("sidebar");
  if (!toggleBtn || !sidebar) return;

  // Restore saved collapse preference (default: expanded)
  const isCollapsed = localStorage.getItem("sidebar_collapsed") === "true";
  if (isCollapsed && window.innerWidth > 900) {
    document.body.classList.add("sidebar-collapsed");
    sidebar.classList.add("collapsed");
    toggleBtn.setAttribute("aria-expanded", "false");
    toggleBtn.setAttribute("title", "Expand sidebar (Ctrl+B)");
    toggleBtn.setAttribute("aria-label", "Expand sidebar");
  } else {
    toggleBtn.setAttribute("aria-expanded", "true");
    toggleBtn.setAttribute("title", "Collapse sidebar (Ctrl+B)");
    toggleBtn.setAttribute("aria-label", "Collapse sidebar");
  }

  function toggleSidebar() {
    // On small mobile viewports, toggle drawer
    if (window.innerWidth <= 900) {
      if (sidebar.classList.contains("open")) {
        closeMobileSidebar();
      } else {
        openMobileSidebar();
      }
      return;
    }

    // On desktop viewports: toggle collapsed state
    const willCollapse = !document.body.classList.contains("sidebar-collapsed");
    document.body.classList.toggle("sidebar-collapsed", willCollapse);
    sidebar.classList.toggle("collapsed", willCollapse);
    localStorage.setItem("sidebar_collapsed", willCollapse ? "true" : "false");

    if (willCollapse) {
      toggleBtn.setAttribute("aria-expanded", "false");
      toggleBtn.setAttribute("title", "Expand sidebar (Ctrl+B)");
      toggleBtn.setAttribute("aria-label", "Expand sidebar");
    } else {
      toggleBtn.setAttribute("aria-expanded", "true");
      toggleBtn.setAttribute("title", "Collapse sidebar (Ctrl+B)");
      toggleBtn.setAttribute("aria-label", "Collapse sidebar");
    }
  }

  toggleBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleSidebar();
  });

  // Keyboard shortcut: Ctrl + B or Cmd + B toggles sidebar
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
      const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
      if (activeTag === "input" || activeTag === "textarea") return;
      e.preventDefault();
      toggleSidebar();
    }
  });
}

/* ─────────────────────────────────────────────────────────────
   API Health Check
───────────────────────────────────────────────────────────── */

async function checkHealth() {
  const dot = $("sidebarDot");
  const text = $("sidebarStatusText");
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    if (data.status === "ok") {
      dot.className = "status-dot ok";
      text.textContent = data.model_available ? "API online · Model ready" : "API online · Model offline";
    } else {
      throw new Error("not ok");
    }
  } catch {
    dot.className = "status-dot err";
    text.textContent = "API offline";
  }
}
checkHealth();

/* ─────────────────────────────────────────────────────────────
   Symptom notes character counter
───────────────────────────────────────────────────────────── */

const notesEl = $("symptomNotes");
const charCount = $("charCount");
const clearNotesBtn = $("clearNotesBtn");

if (notesEl) {
  notesEl.addEventListener("input", () => {
    state.notes = notesEl.value;
    if (charCount) charCount.textContent = notesEl.value.length;
  });
}

if (clearNotesBtn) {
  clearNotesBtn.addEventListener("click", () => {
    if (notesEl) notesEl.value = "";
    state.notes = "";
    if (charCount) charCount.textContent = "0";
  });
}


/* ─────────────────────────────────────────────────────────────
   File Drop-Zone Factory
───────────────────────────────────────────────────────────── */

function makeDropZone({
  zoneId, innerId, previewId, fileInputId,
  clearBtnId, analyseOrValidateId,
  allowedExts, maxBytes, onFile
}) {
  const zone = $(zoneId);
  const inner = $(innerId);
  const preview = $(previewId);
  const fileInput = $(fileInputId);
  const clearBtn = $(clearBtnId);
  const actBtn = $(analyseOrValidateId);

  if (!zone || !fileInput) return;

  function showPreview(file) {
    inner.classList.add("hidden");
    preview.classList.remove("hidden");
    onFile(file);
  }
  function clearFile() {
    fileInput.value = "";
    inner.classList.remove("hidden");
    preview.classList.add("hidden");
    onFile(null);
  }

  function handleFile(file) {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
      alert(`Unsupported file type ".${ext}". Accepted: ${allowedExts.join(", ")}`);
      return;
    }
    if (file.size > maxBytes) {
      alert(`File too large (${fmt(file.size)}). Maximum: ${fmt(maxBytes)}`);
      return;
    }
    showPreview(file);
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  ["dragover", "dragenter"].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.add("drag-over");
  }));
  ["dragleave", "drop"].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.remove("drag-over");
  }));
  zone.addEventListener("drop", ev => {
    const f = ev.dataTransfer?.files?.[0];
    if (f) handleFile(f);
  });
  zone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });

  if (clearBtn) clearBtn.addEventListener("click", clearFile);
  return { showPreview, clearFile, handleFile };
}

/* ─────────────────────────────────────────────────────────────
   Lab Report Upload & Parsing
───────────────────────────────────────────────────────────── */

makeDropZone({
  zoneId: "labDropZone", innerId: "labDropInner", previewId: "labPreview",
  fileInputId: "labFileInput", clearBtnId: "labClearBtn",
  analyseOrValidateId: "labAnalyseBtn",
  allowedExts: ["pdf", "png", "jpg", "jpeg"],
  maxBytes: 20 * 1024 * 1024,
  onFile(file) {
    state.labFile = file;
    if (!file) { resetLabResults(); return; }
    $("labPreviewIcon").textContent = file.name.endsWith(".pdf") ? "📄" : "🖼️";
    $("labPreviewName").textContent = file.name;
    $("labPreviewSize").textContent = fmt(file.size);
  }
});

const labAnalyseBtn = $("labAnalyseBtn");
if (labAnalyseBtn) labAnalyseBtn.addEventListener("click", analyseLabReport);

const labRetryBtn = $("labRetryBtn");
if (labRetryBtn) {
  labRetryBtn.addEventListener("click", () => {
    $("labErrorBox")?.classList.add("hidden");
    const fileInput = $("labFileInput");
    if (fileInput) fileInput.value = "";
    $("labDropInner")?.classList.remove("hidden");
    $("labPreview")?.classList.add("hidden");
    state.labFile = null;
    resetLabResults();
  });
}

async function analyseLabReport() {
  if (!state.labFile) return;
  const statusEl = $("labStatus");
  const progressEl = $("labProgress");
  const textEl = $("labStatusText");
  const errBox = $("labErrorBox");

  if (errBox) errBox.classList.add("hidden");
  statusEl.classList.remove("hidden");
  progressEl.style.width = "15%";
  progressEl.style.background = "";
  textEl.textContent = "Initializing On-Device Wasm Engine…";

  try {
    let data = null;

    // 1. Primary: 100% Client-Side On-Device Parser (Browser Memory / Wasm / Web Workers)
    if (typeof window.parseLabDocumentOnDevice === "function") {
      data = await window.parseLabDocumentOnDevice(state.labFile, ({ percent, message }) => {
        progressEl.style.width = `${Math.min(100, Math.max(10, percent))}%`;
        textEl.textContent = message;
      });
    }

    // 2. Secondary Fallback: If client parser encountered critical failure or returned no data
    if (!data || data.error) {
      const clientErrMsg = data?.error;
      textEl.textContent = "Connecting to fallback endpoint…";
      progressEl.style.width = "60%";

      const fd = new FormData();
      fd.append("file", state.labFile);

      const res = await fetch(`${API}/api/lab/parse`, { method: "POST", body: fd });
      const serverData = await parseJsonResponse(res);

      if (res.status >= 400 || serverData.error) {
        const msg = serverData.error || clientErrMsg || "Parsing failed. Please verify the document format.";
        if (errBox) {
          errBox.classList.remove("hidden");
          $("labErrorTitle").textContent = res.status === 413 ? "File Too Large (Max 20 MB)" : res.status === 415 ? "Unsupported Media Type" : "PDF Parsing Error";
          $("labErrorMsg").textContent = msg;
        }
        throw new Error(msg);
      }
      data = serverData;
    }

    progressEl.style.width = "100%";
    textEl.textContent = data.client_side ? "Extracted ✓ (100% On-Device)" : "Extracted ✓";

    state.labResult = data;
    renderLabResults(data);
    syncFindingsPanel();

    setTimeout(() => statusEl.classList.add("hidden"), 1200);

    // Show nav row and auto-navigate if present
    if ($("labNavRow")) $("labNavRow").style.display = "flex";
  } catch (err) {
    progressEl.style.width = "100%";
    textEl.textContent = "Error: " + (err.message ?? "Parsing failed");
    $("labProgress").style.background = "var(--red)";
  }
}

function resetLabResults() {
  state.labResult = null;
  syncFindingsPanel();
  $("labErrorBox")?.classList.add("hidden");
  $("labIdle").classList.remove("hidden");
  $("labResults").classList.add("hidden");
  $("labNavRow") && ($("labNavRow").style.display = "none");
}

function renderLabResults(data) {
  const idle = $("labIdle");
  const results = $("labResults");
  idle.classList.add("hidden");
  results.classList.remove("hidden");

  // Error from server
  if (data.error && !data.raw_text) {
    results.innerHTML = `
      <div class="error-state">
        <div class="error-icon">⚠️</div>
        <h3 class="error-title">Parsing Error</h3>
        <pre class="error-body">${esc(data.error)}</pre>
      </div>`;
    return;
  }

  // Re-render inner content (keep structure by resetting via innerHTML of subpieces)
  results.innerHTML = `
    <div class="ocr-notice hidden" id="ocrNotice">
      <span>🔬</span><span id="ocrNoticeText"></span>
    </div>
    <div class="warnings-block hidden" id="labWarnings"></div>
    <details class="raw-text-accordion" id="rawTextAccordion">
      <summary class="raw-text-toggle">📃 Show Extracted Raw Text</summary>
      <pre class="raw-text-body" id="rawTextBody"></pre>
    </details>
    <div class="table-section">
      <h2 class="section-title">Structured Results Table</h2>
      <p class="section-desc">
        Review these rows against your original report before acting on any value.
        <em>Values and decimal points are auto-calibrated using clinical bounds. You can click any cell to edit or correct values directly.</em>
      </p>
      <div class="table-wrapper" role="region" aria-label="Lab results table" tabindex="0">
        <table class="lab-table" id="labTable">
          <thead>
            <tr>
              <th>Test Name</th><th>Value</th><th>Unit</th>
              <th>Reference Range</th><th>Flag</th><th>Plain Explanation</th>
            </tr>
          </thead>
          <tbody id="labTableBody"></tbody>
        </table>
      </div>
    </div>
  `;

  // OCR & Privacy notice
  if (data.client_side) {
    $("ocrNotice").classList.remove("hidden");
    $("ocrNoticeText").innerHTML = data.ocr_used
      ? "<strong>🔒 Client-Side Wasm OCR:</strong> Text was recognized locally in browser memory (WebAssembly). Your document was never uploaded to any cloud server."
      : "<strong>🔒 Client-Side Vector Extraction:</strong> Document text and table layout were parsed locally in browser memory (PDF.js). No files were sent over the network.";
  } else if (data.ocr_used) {
    $("ocrNotice").classList.remove("hidden");
    $("ocrNoticeText").textContent =
      "OCR was used to extract text from this image. Please verify all values against the original document.";
  }

  // Warnings
  if (data.warnings?.length || data.error) {
    const wb = $("labWarnings");
    wb.classList.remove("hidden");
    const items = [...(data.warnings ?? [])];
    if (data.error) items.unshift("⚠️ " + data.error);
    wb.innerHTML = `<ul>${items.map(w => `<li>${esc(w)}</li>`).join("")}</ul>`;
  }

  // Raw text
  if (data.raw_text) {
    $("rawTextBody").textContent = data.raw_text;
  } else {
    $("rawTextAccordion").style.display = "none";
  }

  // Table
  const tbody = $("labTableBody");
  if (!data.rows?.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="uncertain" style="text-align:center;padding:1.5rem">
      No structured rows were detected. Please review the raw text above.</td></tr>`;
  } else {
    tbody.innerHTML = data.rows.map((row, idx) => {
      const flag = (row.flag || row.ref_flag || "").toUpperCase();
      const flagHtml = flag
        ? `<span class="flag-badge flag-${flag === "HIGH" ? "high" : flag === "LOW" ? "low" : "normal"}">${esc(flag)}</span>`
        : `<span class="flag-none">—</span>`;
      const unitHtml = isUncertain(row.unit) ? `<span class="uncertain">—</span>` : esc(row.unit);
      const rangeHtml = isUncertain(row.ref_range)
        ? `<span class="uncertain">—</span>`
        : esc(row.ref_range);
      const shortExp = formatShortExplanation(row);
      row.plain_explanation = shortExp;
      return `<tr data-idx="${idx}">
        <td contenteditable="true" class="editable-cell" data-field="test_name" title="Click to edit test name"><strong>${esc(row.test_name)}</strong></td>
        <td contenteditable="true" class="editable-cell" data-field="value" title="Click to edit value or decimal"><code>${esc(row.value)}</code></td>
        <td contenteditable="true" class="editable-cell" data-field="unit" title="Click to edit unit">${unitHtml}</td>
        <td contenteditable="true" class="editable-cell" data-field="ref_range" title="Click to edit reference range">${rangeHtml}</td>
        <td class="flag-cell">${flagHtml}</td>
        <td class="explanation-cell">${esc(shortExp)}</td>
      </tr>`;
    }).join("");

    // Enable live in-table editing so user can immediately correct any OCR typo or decimal
    tbody.onblur = (e) => {
      const target = e.target.closest(".editable-cell");
      if (!target) return;
      const tr = target.closest("tr");
      const idx = parseInt(tr?.dataset?.idx, 10);
      const field = target.dataset.field;
      if (isNaN(idx) || !data.rows[idx] || !field) return;

      const newText = target.innerText.trim();
      data.rows[idx][field] = newText;

      // When value, range or test name changes, re-evaluate flag and explanation
      if (field === "value" || field === "ref_range" || field === "test_name") {
        const row = data.rows[idx];
        const fuzzy = window.fuzzyMatchTestName ? window.fuzzyMatchTestName(row.test_name) : null;
        const testKey = fuzzy ? fuzzy.key : row.test_name.toLowerCase();

        if (window.repairValueDecimal && field === "value") {
          row.value = window.repairValueDecimal(testKey, row.value);
          const codeEl = target.querySelector("code");
          if (codeEl) codeEl.textContent = row.value;
        }

        const parsedVal = parseFloat(row.value);
        const dashMatch = (row.ref_range || "").match(/([0-9]+(?:\.[0-9]+)?)\s*[\-–—to]+\s*([0-9]+(?:\.[0-9]+)?)/i);
        if (dashMatch && !isNaN(parsedVal)) {
          const low = parseFloat(dashMatch[1]);
          const high = parseFloat(dashMatch[2]);
          row.flag = parsedVal < low ? "LOW" : parsedVal > high ? "HIGH" : "NORMAL";
        }
        const flagCell = tr.querySelector(".flag-cell");
        if (flagCell) {
          const f = (row.flag || "").toUpperCase();
          flagCell.innerHTML = f
            ? `<span class="flag-badge flag-${f === "HIGH" ? "high" : f === "LOW" ? "low" : "normal"}">${esc(f)}</span>`
            : `<span class="flag-none">—</span>`;
        }
        const expCell = tr.querySelector(".explanation-cell");
        if (expCell) {
          row.plain_explanation = formatShortExplanation(row);
          expCell.textContent = row.plain_explanation;
        }
        syncFindingsPanel();
      }
    };
  }

  // Ref disclaimer shown as footnote
  if (data.ref_disclaimer) {
    const note = document.createElement("p");
    note.style.cssText = "font-size:.75rem;color:var(--text-faint);margin-top:.75rem;";
    note.textContent = data.ref_disclaimer;
    $("labTableBody").closest(".table-section").appendChild(note);
  }
}

function isUncertain(s) {
  return !s || s === "Please check the original report." || s === "Not specified" || s === "—";
}

function formatShortExplanation(row) {
  const flag = (row.flag || row.ref_flag || "").toUpperCase();
  const ref = row.ref_range;
  const isRefMissing = isUncertain(ref);

  if (isRefMissing) {
    if (flag === "HIGH") return "High: Elevated level";
    if (flag === "LOW") return "Low: Below standard level";
    return "Normal";
  }

  if (flag === "HIGH") return "High: Above expected range";
  if (flag === "LOW") return "Low: Below expected range";
  if (flag === "NORMAL") return "Normal: Within expected range";
  return "Normal";
}

/* ─────────────────────────────────────────────────────────────
   X-Ray Upload
───────────────────────────────────────────────────────────── */

makeDropZone({
  zoneId: "xrayDropZone", innerId: "xrayDropInner", previewId: "xrayPreview",
  fileInputId: "xrayFileInput", clearBtnId: "xrayClearBtn",
  analyseOrValidateId: "xrayAnalyseBtn",
  allowedExts: ["png", "jpg", "jpeg"],
  maxBytes: 10 * 1024 * 1024,
  onFile(file) {
    state.xrayFile = file;
    if (!file) { state.xrayResult = null; return; }
    $("xrayPreviewName").textContent = file.name;
    $("xrayPreviewSize").textContent = fmt(file.size);
    // Thumbnail
    const thumb = $("xrayThumb");
    if (thumb) {
      const url = URL.createObjectURL(file);
      thumb.src = url;
      thumb.onload = () => URL.revokeObjectURL(url);
    }
  }
});

const xrayAnalyseBtn = $("xrayAnalyseBtn");
if (xrayAnalyseBtn) xrayAnalyseBtn.addEventListener("click", validateXray);

const xrayRetryBtn = $("xrayRetryBtn");
if (xrayRetryBtn) {
  xrayRetryBtn.addEventListener("click", () => {
    $("xrayErrorBox")?.classList.add("hidden");
    const fileInput = $("xrayFileInput");
    if (fileInput) fileInput.value = "";
    $("xrayDropInner")?.classList.remove("hidden");
    $("xrayPreview")?.classList.add("hidden");
    state.xrayFile = null;
    state.xrayResult = null;
    const thumb = $("xrayThumb");
    if (thumb) thumb.src = "";
  });
}

async function validateXray() {
  if (!state.xrayFile) return;
  const statusEl = $("xrayStatus");
  const progressEl = $("xrayProgress");
  const textEl = $("xrayStatusText");
  const errBox = $("xrayErrorBox");

  if (errBox) errBox.classList.add("hidden");
  statusEl.classList.remove("hidden");
  progressEl.style.width = "40%";
  progressEl.style.background = "var(--accent)";
  textEl.textContent = "Uploading & evaluating baseline in RAM…";

  const fd = new FormData();
  fd.append("file", state.xrayFile);

  try {
    const res = await fetch(`${API}/api/xray/upload`, { method: "POST", body: fd });
    const data = await parseJsonResponse(res);
    if (res.status >= 400 || data.error) {
      const msg = data.error || "Analysis failed";
      state.xrayResult = null;
      const outputEl = $("screeningOutput");
      if (outputEl) outputEl.classList.add("hidden");
      syncFindingsPanel();

      if (errBox) {
        errBox.classList.remove("hidden");
        const isNonCxr = msg.toLowerCase().includes("chest x-ray") || msg.toLowerCase().includes("inappropriate") || msg.toLowerCase().includes("radiograph");
        $("xrayErrorTitle").textContent = res.status === 413
          ? "Image Too Large (Max 10 MB)"
          : res.status === 415
            ? "Unsupported Image Format"
            : isNonCxr
              ? "Inappropriate Image: Chest X-Ray Required"
              : "X-Ray Processing Error";
        $("xrayErrorMsg").textContent = msg;
      }
      throw new Error(msg);
    }
    state.xrayResult = data;
    progressEl.style.width = "100%";

    const pred = data.prediction || data;
    const score = pred.experimental_score;
    const threshold = pred.threshold ?? 0.50;
    const pClass = pred.predicted_class || pred.predicted_label || "—";
    textEl.textContent = `Class: ${pClass} · Score: ${score.toFixed(4)} (Threshold: ${threshold.toFixed(2)})`;

    const outputEl = $("screeningOutput");
    if (outputEl) {
      outputEl.classList.remove("hidden");
      const thumb = $("xrayThumb");
      if (thumb && thumb.src) $("screeningImg").src = thumb.src;
      $("screeningImgCaption").textContent = `Uploaded: ${state.xrayFile.name} (In-Memory)`;
      renderScreeningPrediction(pred);
      syncFindingsPanel();
    }
    setTimeout(() => statusEl.classList.add("hidden"), 1500);
  } catch (err) {
    progressEl.style.width = "100%";
    textEl.textContent = "Error: " + (err.message ?? "Request failed");
    $("xrayProgress").style.background = "var(--red)";
    if (errBox) {
      errBox.classList.remove("hidden");
      $("xrayErrorTitle").textContent = "X-Ray Screening Error";
      $("xrayErrorMsg").textContent = err.message || "Failed to communicate with screening server.";
    }
  }
}


/* ─────────────────────────────────────────────────────────────
   X-Ray Inventory
───────────────────────────────────────────────────────────── */

const runInventoryBtn = $("runInventoryBtn");
if (runInventoryBtn) runInventoryBtn.addEventListener("click", runInventory);

async function runInventory() {
  const statusTxt = $("invStatusText");
  const resultsEl = $("invResults");
  const errorEl = $("invError");
  const successEl = $("invSuccess");

  statusTxt.innerHTML = '<span class="spinner"></span> Scanning…';
  runInventoryBtn.disabled = true;

  try {
    const res = await fetch(`${API}/api/xray/inventory`);
    const data = await parseJsonResponse(res);
    state.inventoryResult = data;

    resultsEl.style.display = "block";
    statusTxt.textContent = "";

    if (data.error) {
      errorEl.classList.remove("hidden");
      successEl.classList.add("hidden");
      $("invErrorBody").textContent = data.error;
    } else {
      errorEl.classList.add("hidden");
      successEl.classList.remove("hidden");
      renderInventory(data);
    }
  } catch (err) {
    statusTxt.textContent = "Error: " + err.message;
  } finally {
    runInventoryBtn.disabled = false;
  }
}

function renderInventory(data) {
  $("invNormalCount").textContent = data.normal_count ?? "—";
  $("invTbCount").textContent = data.tb_count ?? "—";
  $("invTotalCount").textContent = data.total_count ?? "—";
  $("invDbPath").textContent = data.db_path ?? "—";
  $("invLabelSource").textContent = data.label_source ?? "—";
  $("invUnreadable").textContent = (data.unreadable ?? []).length;
  $("invSummary").textContent = data.summary ?? "—";

  // Balance bar
  const total = (data.normal_count ?? 0) + (data.tb_count ?? 0);
  const normPct = total ? (data.normal_count / total * 100) : 0;
  const tbPct = total ? (data.tb_count / total * 100) : 0;
  $("balanceNormal").style.width = normPct.toFixed(1) + "%";
  $("balanceTb").style.width = tbPct.toFixed(1) + "%";

  // Warnings
  const warnEl = $("invWarnings");
  if (data.warnings?.length) {
    warnEl.classList.remove("hidden");
    warnEl.innerHTML = `<ul>${data.warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul>`;
  } else {
    warnEl.classList.add("hidden");
  }

  // Show uploaded X-ray result if available
  const xrayEl = $("xrayUploadResult");
  const xrayCard = $("xrayResultCard");
  if (state.xrayResult) {
    xrayEl.classList.remove("hidden");
    const r = state.xrayResult;
    xrayCard.innerHTML = r.error
      ? `<span style="color:var(--red)">⚠️ ${esc(r.error)}</span>`
      : `<strong>${esc(r.filename)}</strong> · ${fmt(r.size_bytes)} · Status: ${esc(r.model_status)}<br>
         <span style="color:var(--text-muted);font-size:.8rem">${esc(r.message ?? "")}</span>`;
  } else {
    xrayEl.classList.add("hidden");
  }
}

/* ─────────────────────────────────────────────────────────────
   Dataset Stats (population context)
───────────────────────────────────────────────────────────── */

async function fetchDatasetStats() {
  try {
    const res = await fetch(`${API}/api/lab/dataset-stats`);
    const data = await parseJsonResponse(res);
    state.datasetStats = data;
    renderPopContext(data);
  } catch {
    /* silently ignore — pop context is optional */
  }
}

function renderPopContext(data) {
  const ctx = $("popContext");
  const grid = $("popChartGrid");
  if (!ctx || !grid || !data?.stats) return;

  const stats = data.stats;
  const keys = Object.keys(stats);
  if (!keys.length) return;

  // Find global max for bar scaling
  const maxMean = Math.max(...keys.map(k => stats[k].mean ?? 0));

  grid.innerHTML = keys.map(col => {
    const s = stats[col];
    const pct = maxMean ? ((s.mean / maxMean) * 100).toFixed(1) : 0;
    const hPct = (s.healthy_mean != null && maxMean)
      ? ((s.healthy_mean / maxMean) * 100).toFixed(1) : null;
    return `<div class="pop-stat-card">
      <div class="pop-stat-name">${esc(col.replace(/_/g, " "))}</div>
      <div class="pop-stat-val">${s.mean != null ? s.mean.toFixed(1) : "—"}</div>
      <div class="pop-stat-sub">
        Healthy avg: ${s.healthy_mean != null ? s.healthy_mean.toFixed(1) : "—"} &nbsp;·&nbsp;
        n=${s.count}
      </div>
      <div class="pop-stat-bar">
        <div class="pop-stat-bar-fill" style="width:${pct}%"></div>
      </div>
    </div>`;
  }).join("");

  ctx.classList.remove("hidden");
}

/* ─────────────────────────────────────────────────────────────
   Findings Summary Panel
───────────────────────────────────────────────────────────── */

function syncFindingsPanel() {
  // 1. Lab card: Report-Extracted Facts (syncs both Section 1 side card and Section 4 summary card)
  const labBodies = [document.getElementById("findingsLabBody"), document.getElementById("sideLabFactsBody")].filter(Boolean);
  if (labBodies.length) {
    if (state.labResult && state.labResult.rows?.length) {
      const rows = state.labResult.rows;
      const flagged = rows.filter(row => {
        const f = (row.flag || "").toUpperCase();
        return f === "HIGH" || f === "LOW";
      });

      const rowsHtml = rows.map(row => {
        const flag = (row.flag || "").toUpperCase();
        const flagBadge = flag === "HIGH"
          ? `<span class="flag-badge flag-high">HIGH</span>`
          : flag === "LOW"
            ? `<span class="flag-badge flag-low">LOW</span>`
            : `<span class="flag-none">Normal</span>`;
        return `
          <tr style="border-bottom: 1px solid var(--border);">
            <td style="padding: .4rem .5rem; font-weight: 600;">${esc(row.test_name)}</td>
            <td style="padding: .4rem .5rem;"><code>${esc(row.value)} ${esc(row.unit || '')}</code></td>
            <td style="padding: .4rem .5rem; color: var(--text-muted); font-size: .8rem;">${isUncertain(row.ref_range) ? '—' : esc(row.ref_range)}</td>
            <td style="padding: .4rem .5rem;">${flagBadge}</td>
          </tr>
        `;
      }).join("");

      const content = `
        <div style="margin-bottom: .6rem; display: flex; justify-content: space-between; font-size: .8rem; color: var(--text-muted);">
          <span>Total tests extracted: <strong>${rows.length}</strong></span>
          <span>Out-of-range: <strong style="color: ${flagged.length ? 'var(--amber)' : 'var(--green)'};">${flagged.length}</strong></span>
        </div>
        <div class="table-wrapper" style="max-height: 220px; overflow-y: auto; border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: .6rem;">
          <table style="width: 100%; border-collapse: collapse; font-size: .82rem;">
            <thead>
              <tr style="background: var(--bg-card-hi); text-align: left; border-bottom: 1px solid var(--border);">
                <th style="padding: .4rem .5rem;">Test Name</th>
                <th style="padding: .4rem .5rem;">Value</th>
                <th style="padding: .4rem .5rem;">Report Reference Range</th>
                <th style="padding: .4rem .5rem;">Flag</th>
              </tr>
            </thead>
            <tbody>
              ${rowsHtml}
            </tbody>
          </table>
        </div>
        <div style="font-size: .74rem; color: var(--text-faint); line-height: 1.4;">
          * Extracted directly from uploaded report. Reference ranges and flags are strictly based on the report itself. Never diagnose a condition based on out-of-range values alone. Consult a qualified physician.
        </div>
      `;
      labBodies.forEach(b => { b.innerHTML = content; });
    } else {
      const summaryLabBody = document.getElementById("findingsLabBody");
      if (summaryLabBody) {
        summaryLabBody.innerHTML = `
          <div class="findings-empty-wrap">
            <div class="findings-empty-icon">🧪</div>
            <p class="findings-empty">No lab report analysed yet. Upload a report in Lab Report section to extract structured values.</p>
            <button type="button" class="empty-action-btn" id="summaryAddLabBtn">+ Upload Lab Report</button>
          </div>
        `;
        $("summaryAddLabBtn")?.addEventListener("click", () => activatePanel("labreport"));
      }
      const sideLabBody = document.getElementById("sideLabFactsBody");
      if (sideLabBody) {
        sideLabBody.innerHTML = `
          <div class="findings-empty-wrap">
            <div class="findings-empty-icon">🧪</div>
            <p class="findings-empty-title" style="font-weight:600; margin-bottom:.35rem;">No Lab Report Uploaded Yet</p>
            <p class="findings-empty" style="font-size:.82rem;">Upload a PDF or image on the left and click <strong>Analyse Report</strong> to view structured extracted values here.</p>
          </div>
        `;
      }
    }
  }

  // 2. X-ray card: Model Output (syncs both Section 2 side card and Section 4 summary card)
  const xrayBodies = [document.getElementById("findingsXrayBody"), document.getElementById("sideXrayOutputBody")].filter(Boolean);
  if (xrayBodies.length) {
    if (state.xrayResult && !state.xrayResult.error) {
      const r = state.xrayResult;
      const pred = r.prediction || r;
      const score = pred.experimental_score != null ? pred.experimental_score : 0;
      const threshold = pred.threshold != null ? pred.threshold : 0.50;
      const pClass = pred.predicted_class || pred.predicted_label || "—";
      const isElevated = score >= threshold;

      const content = `
        <div class="findings-row">
          <span class="findings-key">Model Predicted Class</span>
          <span class="findings-val" style="font-weight: 700; color: ${isElevated ? 'var(--red)' : 'var(--green)'}; font-size: .92rem;">
            ${esc(pClass)}
          </span>
        </div>
        <div class="findings-row">
          <span class="findings-key">Experimental Screening Score</span>
          <span class="findings-val"><code>${score.toFixed(4)}</code></span>
        </div>
        <div class="findings-row">
          <span class="findings-key">Decision Threshold</span>
          <span class="findings-val"><code>${threshold.toFixed(2)}</code></span>
        </div>
        <div class="findings-row">
          <span class="findings-key">Model Status</span>
          <span class="findings-val" style="color: var(--accent);">Pretrained ResNet18 Baseline</span>
        </div>
        <div style="margin-top: .75rem; padding: .75rem; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); border-radius: var(--radius-md); font-size: .78rem; color: var(--text-muted); line-height: 1.45;">
          <strong>⚠️ Experimental Screening Score Only — Not a Medical Diagnosis.</strong><br>
          This score reflects visual resemblance to documented dataset labels only. It does NOT diagnose or rule out disease. All interpretations must be performed by a licensed radiologist or physician.
        </div>
      `;
      xrayBodies.forEach(b => { b.innerHTML = content; });
    } else if (state.xrayResult?.error) {
      xrayBodies.forEach(b => {
        b.innerHTML = `<p style="color: var(--red);">${esc(state.xrayResult.error)}</p>`;
      });
    } else {
      const summaryXrayBody = document.getElementById("findingsXrayBody");
      if (summaryXrayBody) {
        summaryXrayBody.innerHTML = `
          <div class="findings-empty-wrap">
            <div class="findings-empty-icon">🫁</div>
            <p class="findings-empty">No chest X-ray uploaded or evaluated yet. Test the baseline model in Chest X-Ray section.</p>
            <button type="button" class="empty-action-btn" id="summaryAddXrayBtn">+ Upload Chest X-Ray</button>
          </div>
        `;
        $("summaryAddXrayBtn")?.addEventListener("click", () => activatePanel("xray"));
      }
      const sideXrayBody = document.getElementById("sideXrayOutputBody");
      if (sideXrayBody) {
        sideXrayBody.innerHTML = `
          <div class="findings-empty-wrap">
            <div class="findings-empty-icon">🫁</div>
            <p class="findings-empty-title" style="font-weight:600; margin-bottom:.35rem;">No Chest X-Ray Evaluated Yet</p>
            <p class="findings-empty" style="font-size:.82rem;">Upload an image on the left or select a verified sample below to view baseline model outputs here.</p>
          </div>
        `;
      }
    }
  }

  // 3. Symptoms & Notes Card: User Input & Clinical Domains
  const notesBody = $("findingsNotesBody");
  if (notesBody) {
    const hasSymptoms = state.symptoms && state.symptoms.length > 0;
    const hasNotes = state.notes && state.notes.trim().length > 0;

    if (hasSymptoms || hasNotes) {
      let redFlagsHtml = "";
      const analysis = state.analyzedSymptoms;
      const rfList = (analysis && analysis.red_flags) ? analysis.red_flags : state.symptoms.filter(s => s.is_red_flag).map(s => s.text);

      if (rfList && rfList.length > 0) {
        redFlagsHtml = `
          <div style="margin-bottom: .75rem; padding: .65rem .85rem; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); border-radius: var(--radius-sm);">
            <div style="font-size: .8rem; font-weight: 700; color: #fca5a5; display: flex; align-items: center; gap: .4rem;">
              <span>🚨</span> Red-Flag Warning Symptoms Recorded:
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .4rem;">
              ${rfList.map(rf => `<span style="font-size:.72rem; padding:.15rem .45rem; background:rgba(239,68,68,0.3); color:#fff; border-radius:var(--radius-sm); font-weight:600;">${esc(rf)}</span>`).join("")}
            </div>
            <p style="margin:.4rem 0 0 0; font-size:.74rem; color:#fee2e2;">
              Seek prompt clinical evaluation from a healthcare professional or emergency care if symptoms are severe.
            </p>
          </div>
        `;
      }

      let symptomsListHtml = "";
      if (hasSymptoms) {
        symptomsListHtml = `
          <div style="margin-bottom: .6rem;">
            <div style="font-size: .78rem; font-weight: 600; color: var(--text-muted); margin-bottom: .35rem;">
              Discrete Symptoms Recorded (${state.symptoms.length}):
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: .4rem;">
              ${state.symptoms.map(s => `
                <span style="font-size: .76rem; padding: .25rem .6rem; background: var(--bg-card-hi); border: 1px solid ${s.is_red_flag ? 'rgba(239,68,68,0.5)' : 'var(--border)'}; border-radius: var(--radius-pill); color: var(--text); display: inline-flex; align-items: center; gap: .3rem;">
                  ${s.is_red_flag ? '<strong style="color:var(--red)">⚠️</strong>' : '<span>•</span>'}
                  ${esc(s.text)}
                  ${s.duration ? `<small style="color:var(--text-faint);">(${esc(s.duration)})</small>` : ''}
                </span>
              `).join("")}
            </div>
          </div>
        `;
      }

      let domainsHtml = "";
      if (analysis && analysis.domains && analysis.domains.length > 0) {
        domainsHtml = `
          <div style="margin-bottom: .6rem; padding: .6rem .75rem; background: rgba(129,140,248,0.06); border: 1px solid rgba(129,140,248,0.2); border-radius: var(--radius-sm);">
            <div style="font-size: .76rem; font-weight: 700; color: var(--indigo); margin-bottom: .3rem;">
              Clinical Physiological Domains Breakdown:
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: .4rem;">
              ${analysis.domains.map(d => `
                <span style="font-size: .72rem; padding: .2rem .5rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text);">
                  <strong>${esc(d.name.split('/')[0].trim())}</strong>: ${d.symptoms.length}
                </span>
              `).join("")}
            </div>
          </div>
        `;
      }

      let notesHtml = "";
      if (hasNotes) {
        notesHtml = `
          <div style="font-size: .76rem; font-weight: 600; color: var(--text-muted); margin-bottom: .3rem;">User Personal Notes:</div>
          <div style="padding: .65rem; background: var(--bg-card-hi); border: 1px solid var(--border); border-radius: var(--radius-sm); font-size: .82rem; color: var(--text); white-space: pre-wrap; max-height: 120px; overflow-y: auto; line-height: 1.45;">
            ${esc(state.notes)}
          </div>
        `;
      }

      notesBody.innerHTML = `
        ${redFlagsHtml}
        ${symptomsListHtml}
        ${domainsHtml}
        ${notesHtml}
        <div style="margin-top: .6rem; font-size: .72rem; color: var(--text-faint); line-height: 1.4;">
          * Recorded via local browser voice typing and user input. Categorized to organize observations for physician discussion only. Never self-diagnose or alter medical treatments.
        </div>
      `;
    } else {
      notesBody.innerHTML = `
        <div class="findings-empty-wrap">
          <div class="findings-empty-icon">📝</div>
          <p class="findings-empty">No symptoms recorded yet. Use Voice Typing or enter symptoms in Section 3.</p>
          <button type="button" class="empty-action-btn" id="summaryAddNotesBtn">+ Record Symptoms with Voice</button>
        </div>
      `;
      $("summaryAddNotesBtn")?.addEventListener("click", () => activatePanel("notes"));
    }
  }

  // Refresh Voice Assistant script whenever findings panel is rendered
  fetchVoiceScript();

  // Synchronize Health Risk Meter
  updateRiskMeter();

  // Auto-load dataset status if not yet loaded
  const grid = $("datasetStatusGrid");
  if (grid && !grid.innerHTML.trim()) loadDatasetStatus();
}

/* ─────────────────────────────────────────────────────────────
   Health Risk Meter Component (Dynamic Multi-Factor Assessment)
───────────────────────────────────────────────────────────── */

function updateRiskMeter() {
  const container = document.getElementById("riskMeterContainer");
  if (!container) return;

  const needle = document.getElementById("gaugeNeedleGroup");
  const scoreNum = document.getElementById("riskScoreNum");
  const statusPill = document.getElementById("riskStatusPill");
  const adviceTitle = document.getElementById("riskAdviceTitle");
  const adviceSub = document.getElementById("riskAdviceSub");
  const adviceMsgBox = document.getElementById("riskAdviceMessageBox");
  const adviceIcon = document.getElementById("riskAdviceIcon");
  const adviceCard = document.getElementById("riskAdviceCard");
  const factorsWrap = document.getElementById("riskFactorsWrap");
  const factorsList = document.getElementById("riskFactorsList");

  // Gather current diagnostic facts
  const hasLab = state.labResult && Array.isArray(state.labResult.rows) && state.labResult.rows.length > 0;
  const labRows = hasLab ? state.labResult.rows : [];
  const flaggedLabRows = labRows.filter(r => {
    const f = (r.flag || "").toUpperCase();
    return f === "HIGH" || f === "LOW";
  });

  const hasXray = state.xrayResult && !state.xrayResult.error && (state.xrayResult.prediction || state.xrayResult.experimental_score != null);
  const xrayPred = hasXray ? (state.xrayResult.prediction || state.xrayResult) : null;
  const xrayScore = xrayPred && xrayPred.experimental_score != null ? xrayPred.experimental_score : 0;
  const xrayThreshold = xrayPred && xrayPred.threshold != null ? xrayPred.threshold : 0.50;
  const xrayClass = xrayPred && (xrayPred.predicted_class || xrayPred.predicted_label || "");
  const isXrayElevated = hasXray && (xrayScore >= xrayThreshold || (xrayClass && xrayClass.toLowerCase().includes("tuberculosis")));

  const hasSymptoms = Array.isArray(state.symptoms) && state.symptoms.length > 0;
  const symptomsList = hasSymptoms ? state.symptoms : [];
  const redFlags = [];
  if (state.analyzedSymptoms?.red_flags?.length) {
    redFlags.push(...state.analyzedSymptoms.red_flags);
  }
  symptomsList.forEach(s => {
    if (s.is_red_flag && !redFlags.includes(s.text)) {
      redFlags.push(s.text);
    }
  });

  const hasAnyData = hasLab || hasXray || hasSymptoms || (state.notes && state.notes.trim().length > 0);

  // If no data uploaded/recorded yet
  if (!hasAnyData) {
    if (needle) needle.setAttribute("transform", "rotate(-90, 120, 120)");
    if (scoreNum) scoreNum.textContent = "0%";
    if (statusPill) {
      statusPill.className = "risk-status-pill neutral";
      statusPill.textContent = "Awaiting Data";
    }
    if (adviceTitle) adviceTitle.textContent = "Awaiting Assessment Data";
    if (adviceSub) adviceSub.textContent = "Upload a lab report, chest X-ray, or record symptoms to calculate unified health risk.";
    if (adviceIcon) adviceIcon.textContent = "🩺";
    if (adviceMsgBox) {
      adviceMsgBox.className = "risk-advice-message-box neutral";
      adviceMsgBox.innerHTML = `
        <div class="risk-main-action">Awaiting diagnostic inputs</div>
        <p class="risk-sub-action">Upload your medical documents or enter symptom notes above to evaluate potential health risks.</p>
      `;
    }
    if (factorsWrap) factorsWrap.style.display = "none";
    if (adviceCard) adviceCard.className = "risk-advice-card neutral";
    return;
  }

  // Calculate risk points & factors
  const factors = [];
  let calculatedScore = 12; // baseline normal health score

  // 1. Lab factors
  if (hasLab) {
    if (flaggedLabRows.length === 0) {
      factors.push({
        status: "normal",
        text: `All ${labRows.length} extracted lab tests are within standard reference intervals.`
      });
    } else {
      const pointPerFlag = 18;
      calculatedScore += Math.min(48, flaggedLabRows.length * pointPerFlag);
      const flaggedNames = flaggedLabRows.map(r => `${r.test_name} (${r.flag})`).join(", ");
      factors.push({
        status: "elevated",
        text: `${flaggedLabRows.length} out-of-range lab result${flaggedLabRows.length > 1 ? 's' : ''}: ${flaggedNames}`
      });
    }
  }

  // 2. Chest X-Ray factors
  if (hasXray) {
    if (isXrayElevated) {
      calculatedScore += Math.min(45, Math.round(xrayScore * 48));
      factors.push({
        status: "elevated",
        text: `Chest X-ray screening score is elevated (${xrayScore.toFixed(2)} ≥ ${xrayThreshold.toFixed(2)}) indicating potential pulmonary abnormality.`
      });
    } else {
      factors.push({
        status: "normal",
        text: `Chest X-ray screening baseline score (${xrayScore.toFixed(2)}) is below decision threshold (${xrayThreshold.toFixed(2)}).`
      });
    }
  }

  // 3. Symptoms & Red flags
  if (redFlags.length > 0) {
    calculatedScore += Math.min(50, redFlags.length * 30);
    factors.push({
      status: "elevated",
      text: `Red-flag symptom${redFlags.length > 1 ? 's' : ''} recorded: ${redFlags.join(", ")}`
    });
  } else if (hasSymptoms) {
    calculatedScore += Math.min(18, symptomsList.length * 5);
    factors.push({
      status: "normal",
      text: `${symptomsList.length} general symptom${symptomsList.length > 1 ? 's' : ''} recorded with no acute red flags.`
    });
  }

  // Cap score between 8% and 98%
  calculatedScore = Math.max(8, Math.min(98, Math.round(calculatedScore)));

  // Determine if Risk is Higher vs Normal
  const isHigherRisk = calculatedScore >= 35 || flaggedLabRows.length > 0 || isXrayElevated || redFlags.length > 0;

  // Compute needle angle: 0% -> -90 deg, 100% -> +90 deg
  const needleAngle = -90 + (calculatedScore / 100) * 180;
  if (needle) {
    needle.setAttribute("transform", `rotate(${needleAngle}, 120, 120)`);
  }
  if (scoreNum) {
    scoreNum.textContent = `${calculatedScore}%`;
  }

  // Render Risk Factors list
  if (factorsWrap && factorsList) {
    factorsWrap.style.display = "block";
    factorsList.innerHTML = factors.map(f => `
      <li class="risk-factor-item ${f.status}">
        <span class="risk-factor-dot ${f.status}"></span>
        <span>${esc(f.text)}</span>
      </li>
    `).join("");
  }

  // Render Messaging strictly according to user instructions:
  // "if it is higher tell the person to consult a doctor if it is normal tell no worries and dont add any desclaimer ."
  if (isHigherRisk) {
    if (statusPill) {
      statusPill.className = "risk-status-pill elevated";
      statusPill.textContent = calculatedScore >= 65 ? "High Risk" : "Elevated Risk";
    }
    if (adviceCard) adviceCard.className = "risk-advice-card elevated";
    if (adviceIcon) adviceIcon.textContent = "⚠️";
    if (adviceTitle) adviceTitle.textContent = "Elevated Health Risk Detected";
    if (adviceSub) adviceSub.textContent = "One or more screening factors indicate potential health issues.";

    if (adviceMsgBox) {
      adviceMsgBox.className = "risk-advice-message-box elevated";
      adviceMsgBox.innerHTML = `
        <div class="risk-main-action urgent">
          🚨 Please consult a doctor or healthcare specialist
        </div>
        <p class="risk-sub-action">
          Your diagnostic screening highlights elevated indicators. Please consult a qualified doctor or physician promptly for comprehensive in-person medical examination and diagnosis.
        </p>
      `;
    }
  } else {
    // Normal Risk
    if (statusPill) {
      statusPill.className = "risk-status-pill normal";
      statusPill.textContent = "Normal Risk";
    }
    if (adviceCard) adviceCard.className = "risk-advice-card normal";
    if (adviceIcon) adviceIcon.textContent = "✨";
    if (adviceTitle) adviceTitle.textContent = "Normal Health Risk";
    if (adviceSub) adviceSub.textContent = "All analyzed indicators fall within expected normal baseline ranges.";

    if (adviceMsgBox) {
      adviceMsgBox.className = "risk-advice-message-box normal";
      adviceMsgBox.innerHTML = `
        <div class="risk-main-action normal">
          ✨ No worries!
        </div>
        <p class="risk-sub-action">
          All analyzed screening parameters are within normal reference ranges. Your lab tests, chest screening baseline score, and symptom observations show no acute abnormalities.
        </p>
      `;
    }
  }
}

/* ─────────────────────────────────────────────────────────────
   Dataset Status (Demo Panel)
───────────────────────────────────────────────────────────── */

const refreshDatasetBtn = $("refreshDatasetBtn");
if (refreshDatasetBtn) refreshDatasetBtn.addEventListener("click", loadDatasetStatus);

async function loadDatasetStatus() {
  const grid = $("datasetStatusGrid");
  if (!grid) return;
  grid.innerHTML = `<div style="color:var(--text-faint);font-size:.85rem"><span class="spinner"></span> Loading…</div>`;
  try {
    const res = await fetch(`${API}/api/dataset/status`);
    const data = await parseJsonResponse(res);
    renderDatasetStatus(data);
  } catch (err) {
    grid.innerHTML = `<span style="color:var(--red)">${esc(err.message)}</span>`;
  }
}

function renderDatasetStatus(data) {
  const grid = $("datasetStatusGrid");
  if (!grid) return;

  const xr = data.xray_database ?? {};
  const bl = data.blood_test_csv ?? {};
  const la = data.laboratory_csv ?? {};
  const rf = data.reference_pdf ?? {};

  grid.innerHTML = `
    <div class="ds-item ${xr.db_exists ? "ds-ok" : "ds-err"}">
      <div class="ds-item-name">X-Ray DB</div>
      <div class="ds-item-val">${xr.db_exists ? "Found ✓" : "Missing ✗"}</div>
      <div class="ds-item-sub">${xr.total_count ?? 0} images total</div>
    </div>
    <div class="ds-item ${bl.exists ? "ds-ok" : "ds-err"}">
      <div class="ds-item-name">Blood Test CSV</div>
      <div class="ds-item-val">${bl.exists ? "Found ✓" : "Missing ✗"}</div>
      <div class="ds-item-sub">${bl.rows ?? 0} rows</div>
    </div>
    <div class="ds-item ${la.exists ? "ds-ok" : "ds-err"}">
      <div class="ds-item-name">Lab Data CSV</div>
      <div class="ds-item-val">${la.exists ? "Found ✓" : "Missing ✗"}</div>
      <div class="ds-item-sub">${la.rows ?? 0} rows</div>
    </div>
    <div class="ds-item ${rf.exists ? "ds-ok" : "ds-err"}">
      <div class="ds-item-name">Reference PDF</div>
      <div class="ds-item-val">${rf.exists ? "Found ✓" : "Missing ✗"}</div>
      <div class="ds-item-sub">Blood Test Ranges</div>
    </div>
  `;

  // Show xray breakdown if available
  if (xr.db_exists) {
    const extra = document.createElement("div");
    extra.className = "ds-item ds-ok";
    extra.innerHTML = `
      <div class="ds-item-name">Normal / TB</div>
      <div class="ds-item-val">${xr.normal_count ?? "?"} / ${xr.tb_count ?? "?"}</div>
      <div class="ds-item-sub">label source: ${xr.label_source ?? "—"}</div>`;
    grid.appendChild(extra);
  }
}

// Auto-load dataset status when findings panel is shown
document.querySelector('[data-panel="findings"]')?.addEventListener("click", () => {
  const grid = $("datasetStatusGrid");
  if (grid && !grid.innerHTML.trim()) loadDatasetStatus();
});


/* ─────────────────────────────────────────────────────────────
   Chest X-Ray Pretrained Baseline Model & Interactive Demo
───────────────────────────────────────────────────────────── */

async function loadXrayModelInfo() {
  try {
    const res = await fetch(`${API}/api/xray/model-info`);
    const data = await parseJsonResponse(res);
    state.modelInfo = data;
    renderModelInfo(data);
  } catch (err) {
    console.warn("Failed to load X-ray model info:", err);
  }
}

function renderModelInfo(data) {
  if (!data) return;
  const metrics = data.test_metrics;

  if (metrics) {
    if ($("metricSensitivity")) $("metricSensitivity").textContent = (metrics.sensitivity * 100).toFixed(1) + "%";
    if ($("metricSpecificity")) $("metricSpecificity").textContent = (metrics.specificity * 100).toFixed(1) + "%";
    if ($("metricPrecision")) $("metricPrecision").textContent = (metrics.precision * 100).toFixed(1) + "%";
    if ($("metricF1")) $("metricF1").textContent = (metrics.f1 * 100).toFixed(1) + "%";
    if ($("metricRocAuc")) $("metricRocAuc").textContent = metrics.roc_auc != null ? metrics.roc_auc.toFixed(4) : "N/A";
    if ($("metricAccuracy")) $("metricAccuracy").textContent = (metrics.accuracy * 100).toFixed(1) + "%";

    const cm = metrics.confusion_matrix;
    if (cm) {
      if ($("cmTN")) $("cmTN").textContent = cm.tn;
      if ($("cmFP")) $("cmFP").textContent = cm.fp;
      if ($("cmFN")) $("cmFN").textContent = cm.fn;
      if ($("cmTP")) $("cmTP").textContent = cm.tp;
    }
  }

  const badge = $("modelStatusBadge");
  if (badge) {
    if (data.model_loaded && metrics) {
      badge.textContent = `Baseline Ready · Held-Out Test (n=${metrics.sample_count})`;
      badge.className = "badge badge-green";
    } else if (data.model_loaded) {
      badge.textContent = "Baseline Weights Available";
      badge.className = "badge badge-accent";
    } else {
      badge.textContent = "Training in Progress…";
      badge.className = "badge badge-amber";
    }
  }
}

function initScreeningDemo() {
  const sampleBtns = document.querySelectorAll(".sample-btn");
  sampleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      sampleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const path = btn.getAttribute("data-path");
      const name = btn.getAttribute("data-name");
      runScreeningSample(path, name);
    });
  });
}

async function runScreeningSample(imagePath, displayName) {
  const outputEl = $("screeningOutput");
  if (!outputEl) return;

  outputEl.classList.remove("hidden");
  $("screeningImg").src = `${API}/api/xray/image?path=${encodeURIComponent(imagePath)}`;
  $("screeningImgCaption").textContent = displayName ?? imagePath;

  $("screeningBadge").textContent = "Analyzing…";
  $("screeningBadge").className = "badge badge-amber";
  $("screeningScoreVal").textContent = "…";
  $("screeningScoreDesc").textContent = "Running baseline model on image…";
  $("probBarFill").style.width = "0%";
  $("probNormalVal").textContent = "—";
  $("probTbVal").textContent = "—";

  try {
    const res = await fetch(`${API}/api/xray/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: imagePath }),
    });
    const data = await parseJsonResponse(res);
    if (data.error) {
      $("screeningBadge").textContent = "Error";
      $("screeningBadge").className = "badge badge-red";
      $("screeningScoreDesc").textContent = data.error;
      return;
    }
    renderScreeningPrediction(data);
  } catch (err) {
    $("screeningBadge").textContent = "Request Error";
    $("screeningBadge").className = "badge badge-red";
    $("screeningScoreDesc").textContent = err.message ?? "Failed to connect";
  }
}

function renderScreeningPrediction(pred) {
  const score = pred.experimental_score;
  const isElevated = score >= 0.50;

  $("screeningScoreVal").textContent = score.toFixed(4);
  $("screeningScoreDesc").textContent = `${pred.predicted_label} (${pred.status_category})`;

  const badge = $("screeningBadge");
  if (badge) {
    badge.textContent = isElevated ? "Elevated Screening Score" : "Low Screening Score";
    badge.className = isElevated ? "badge badge-red" : "badge badge-green";
  }

  const normalPct = (pred.prob_normal * 100).toFixed(1);
  const tbPct = (pred.prob_tb * 100).toFixed(1);
  $("probNormalVal").textContent = `${normalPct}%`;
  $("probTbVal").textContent = `${tbPct}%`;

  const fill = $("probBarFill");
  if (fill) {
    fill.style.width = `${(score * 100).toFixed(1)}%`;
  }
}

// Auto-load model info when X-ray panel is opened
document.querySelector('[data-panel="xray"]')?.addEventListener("click", () => {
  loadXrayModelInfo();
});

/* ─────────────────────────────────────────────────────────────
   Voice Typing & Symptom Recording (Section 3)
───────────────────────────────────────────────────────────── */

const RED_FLAGS_PATTERN = /(shortness of breath|breathlessness|difficulty breathing|dyspnea|chest pain|chest tightness|angina|coughing blood|blood in sputum|blood in cough|hemoptysis|high fever|fever > 103|unconscious|fainting|syncope|सांस लेने में तकलीफ|सांस फूलना|छाती में दर्द|सीने में दर्द|खांसी में खून|खून आना|तेज बुखार)/i;

function isRedFlagSymptom(text) {
  return RED_FLAGS_PATTERN.test(text || "");
}

let speechRecognizer = null;

function initVoiceTyping() {
  const micBtn = $("voiceTypeMicBtn");
  const micLabel = $("micBtnLabel");
  const micIcon = $("micBtnIcon");
  const langEnBtn = $("voiceTypeLangEn");
  const langHiBtn = $("voiceTypeLangHi");
  const badge = $("listeningBadge");
  const listeningText = $("listeningText");
  const simulateBtn = $("simulateVoiceBtn");

  // Language toggles for voice typing
  if (langEnBtn && langHiBtn) {
    langEnBtn.addEventListener("click", () => {
      state.voiceTypingLang = "en-US";
      langEnBtn.classList.add("active");
      langHiBtn.classList.remove("active");
      if (micLabel && !state.isListening) micLabel.textContent = "Start Voice Typing";
      if ($("symptomNotes")) $("symptomNotes").placeholder = "Type your symptoms here or click the microphone in the corner to speak naturally...";
      if (speechRecognizer) speechRecognizer.lang = "en-US";
    });

    langHiBtn.addEventListener("click", () => {
      state.voiceTypingLang = "hi-IN";
      langHiBtn.classList.add("active");
      langEnBtn.classList.remove("active");
      if (micLabel && !state.isListening) micLabel.textContent = "बोलकर लिखें";
      if ($("symptomNotes")) $("symptomNotes").placeholder = "यहाँ लक्षण टाइप करें या कोने में माइक बटन दबाकर बोलें...";
      if (speechRecognizer) speechRecognizer.lang = "hi-IN";
    });
  }

  // Check Web SpeechRecognition support
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (micBtn) {
    if (!SpeechRecognition) {
      micBtn.title = "Web Speech API not supported in this browser. Use Try Sample or manual entry.";
    }

    micBtn.addEventListener("click", () => {
      if (!SpeechRecognition) {
        alert("Web Speech Recognition is not supported by your current browser. You can use 'Try Sample Voice Input' or enter symptoms directly.");
        return;
      }

      if (state.isListening) {
        if (speechRecognizer) speechRecognizer.stop();
        return;
      }

      try {
        speechRecognizer = new SpeechRecognition();
        speechRecognizer.lang = state.voiceTypingLang;
        speechRecognizer.continuous = false;
        speechRecognizer.interimResults = true;

        speechRecognizer.onstart = () => {
          state.isListening = true;
          micBtn.classList.add("recording");
          if (micIcon) micIcon.textContent = "🔴";
          if (micLabel) micLabel.textContent = state.voiceTypingLang === "hi-IN" ? "सुन रहे हैं… (रोकने के लिए दबाएं)" : "Stop Voice Typing";
          if (badge) badge.classList.remove("hidden");
          if (listeningText) listeningText.textContent = state.voiceTypingLang === "hi-IN" ? "सुन रहे हैं… कृपया अपने लक्षण स्पष्ट बोलें" : "Listening… speak your symptoms now";
        };

        speechRecognizer.onresult = (e) => {
          let interimTranscript = "";
          let finalTranscript = "";

          for (let i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
              finalTranscript += e.results[i][0].transcript;
            } else {
              interimTranscript += e.results[i][0].transcript;
            }
          }

          if (interimTranscript && listeningText) {
            listeningText.textContent = `"${interimTranscript}"`;
          }

          if (finalTranscript) {
            appendVoiceText(finalTranscript);
          }
        };

        speechRecognizer.onerror = (e) => {
          console.warn("Speech recognition notice/error:", e.error);
          if (listeningText) listeningText.textContent = `Mic notice: ${e.error}. Try speaking again.`;
          setTimeout(resetMicUI, 2000);
        };

        speechRecognizer.onend = () => {
          resetMicUI();
        };

        speechRecognizer.start();
      } catch (err) {
        console.warn("Speech recognition start failed:", err);
        resetMicUI();
      }
    });
  }

  function resetMicUI() {
    state.isListening = false;
    if (micBtn) micBtn.classList.remove("recording");
    if (micIcon) micIcon.textContent = "🎙️";
    if (micLabel) micLabel.textContent = state.voiceTypingLang === "hi-IN" ? "बोलकर लिखें" : "Start Voice Typing";
    if (badge) badge.classList.add("hidden");
  }

  // Voice simulation sample button for instant one-click demonstration
  if (simulateBtn) {
    simulateBtn.addEventListener("click", () => {
      const sampleText = state.voiceTypingLang === "hi-IN"
        ? "मुझे 3 हफ्तों से लगातार खांसी है और सांस लेने में तकलीफ के साथ तेज बुखार है"
        : "I have had a persistent dry cough for 3 weeks and high fever with shortness of breath and fatigue";
      appendVoiceText(sampleText);
    });
  }
}

function appendVoiceText(text) {
  const notesEl = $("symptomNotes");
  if (!notesEl) return;
  const current = notesEl.value.trim();
  const updated = current ? `${current}. ${text.trim()}` : text.trim();
  notesEl.value = updated;
  state.notes = updated;

  const countEl = $("charCount");
  if (countEl) countEl.textContent = updated.length;

  // Automatically separate into symptom items
  separateSymptomsFromNotes();
}

function initSymptomRecording() {
  const notesEl = $("symptomNotes");
  const countEl = $("charCount");
  const clearBtn = $("clearNotesBtn");
  const separateBtn = $("separateSymptomsBtn");
  const addBtn = $("addSymptomBtn");
  const inputEl = $("customSymptomInput");
  const clearAllBtn = $("clearAllSymptomsBtn");
  const analyzeBtn = $("analyzeSymptomsBtn");

  // Sync textarea changes
  if (notesEl) {
    notesEl.addEventListener("input", (e) => {
      state.notes = e.target.value;
      if (countEl) countEl.textContent = e.target.value.length;
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (notesEl) notesEl.value = "";
      state.notes = "";
      state.symptoms = [];
      state.analyzedSymptoms = null;
      if (countEl) countEl.textContent = "0";
      const card = $("symptomAnalysisCard");
      if (card) card.classList.add("hidden");
      syncFindingsPanel();
    });
  }

  if (separateBtn) {
    separateBtn.addEventListener("click", () => {
      separateSymptomsFromNotes();
    });
  }

  // Quick symptom pills
  document.querySelectorAll(".quick-symptom-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const symptomName = pill.dataset.symptom;
      if (symptomName) {
        addSymptom(symptomName);
        if (notesEl) {
          const current = notesEl.value.trim();
          if (!current.toLowerCase().includes(symptomName.toLowerCase())) {
            notesEl.value = current ? `${current}, ${symptomName}` : symptomName;
            state.notes = notesEl.value;
            if (countEl) countEl.textContent = notesEl.value.length;
          }
        }
      }
    });
  });

  // Manual input
  if (addBtn && inputEl) {
    addBtn.addEventListener("click", () => {
      const text = inputEl.value.trim();
      if (text) {
        addSymptom(text);
        inputEl.value = "";
      }
    });

    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const text = inputEl.value.trim();
        if (text) {
          addSymptom(text);
          inputEl.value = "";
        }
      }
    });
  }

  if (clearAllBtn) {
    clearAllBtn.addEventListener("click", () => {
      state.symptoms = [];
      state.analyzedSymptoms = null;
      renderSymptomCards();
      const card = $("symptomAnalysisCard");
      if (card) card.classList.add("hidden");
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeSymptoms);
  }
}

function addSymptom(text, duration = "") {
  if (!text || !text.trim()) return;
  const clean = text.trim();

  // Avoid exact duplicates
  const exists = state.symptoms.some(s => s.text.toLowerCase() === clean.toLowerCase());
  if (exists) return;

  const isRf = isRedFlagSymptom(clean);
  state.symptoms.push({
    id: "sym_" + Date.now() + "_" + Math.floor(Math.random() * 1000),
    text: clean,
    duration: duration,
    is_red_flag: isRf,
  });

  renderSymptomCards();
}

function removeSymptom(id) {
  state.symptoms = state.symptoms.filter(s => s.id !== id);
  renderSymptomCards();
}

function separateSymptomsFromNotes() {
  const text = state.notes || $("symptomNotes")?.value || "";
  if (!text.trim()) return;

  // Split clauses on commas, periods, semicolons, or conjunctions
  const clauses = text
    .split(/[,;\n\r\.]|\band\b|\bwith\b|\balso\b|\baur\b|\bतथा\b|\bऔर\b|\bके साथ\b/i)
    .map(c => c.trim())
    .filter(c => c.length > 2);

  let added = 0;
  clauses.forEach(clause => {
    let clean = clause
      .replace(/^(i have had a|i have had|i have a|i feel|patient reports|suffering from|मुझे|है|था)/i, "")
      .trim();
    if (clean.length > 2) {
      addSymptom(clean);
      added++;
    }
  });

  const statusEl = $("analysisStatusText");
  if (statusEl && added > 0) {
    statusEl.textContent = `✓ ${state.symptoms.length} symptoms recognized. Click 'Analyse Recorded Symptoms' to evaluate.`;
    setTimeout(() => { if (statusEl && statusEl.textContent.startsWith("✓")) statusEl.textContent = ""; }, 4000);
  }
}

function renderSymptomCards() {
  const listEl = $("symptomCardsList");
  const countBadge = $("symptomCountBadge");

  if (countBadge) {
    countBadge.textContent = `${state.symptoms.length} recorded`;
  }

  if (listEl) {
    if (!state.symptoms.length) {
      listEl.innerHTML = `
        <div class="symptom-empty-state" id="symptomEmptyState">
          <span class="symptom-empty-icon">🩺</span>
          <p>No symptoms recorded yet. Speak via the mic above, click quick-add pills, or type symptoms.</p>
        </div>
      `;
    } else {
      listEl.innerHTML = state.symptoms.map(s => {
        const isRf = s.is_red_flag;
        return `
          <div class="symptom-card ${isRf ? 'is-red-flag' : ''}">
            <div class="symptom-card-main">
              <div class="symptom-card-title">${esc(s.text)}</div>
              <div class="symptom-card-tags">
                ${isRf ? '<span class="symptom-card-rf-badge">⚠️ Red-Flag</span>' : ''}
                ${s.duration ? `<span class="symptom-card-duration">⏱️ ${esc(s.duration)}</span>` : ''}
              </div>
            </div>
            <button type="button" class="symptom-card-remove" data-id="${s.id}" title="Remove symptom">✕</button>
          </div>
        `;
      }).join("");

      listEl.querySelectorAll(".symptom-card-remove").forEach(btn => {
        btn.addEventListener("click", () => {
          const id = btn.getAttribute("data-id");
          removeSymptom(id);
        });
      });
    }
  }

  syncFindingsPanel();
}

async function analyzeSymptoms() {
  const statusEl = $("analysisStatusText");
  const cardEl = $("symptomAnalysisCard");
  const btn = $("analyzeSymptomsBtn");

  if (!state.symptoms.length && !state.notes.trim()) {
    if (statusEl) statusEl.textContent = "Please record at least one symptom or note first.";
    return;
  }

  if (statusEl) statusEl.innerHTML = `<span class="spinner"></span> Analyzing physiological domains…`;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(`${API}/api/symptoms/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        notes: state.notes,
        symptoms: state.symptoms
      })
    });
    const data = await parseJsonResponse(res);
    state.analyzedSymptoms = data;

    if (cardEl) {
      cardEl.classList.remove("hidden");
      renderAnalysisResults(data);
    }
    if (statusEl) statusEl.textContent = "Analysis complete ✓";
    setTimeout(() => { if (statusEl) statusEl.textContent = ""; }, 3000);
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderAnalysisResults(data) {
  // Metrics summary
  const summaryEl = $("analysisMetricsSummary");
  if (summaryEl) {
    summaryEl.innerHTML = `
      <div class="analysis-metric-pill">
        <span class="amp-val">${data.symptoms_count ?? 0}</span>
        <span class="amp-lbl">Symptoms</span>
      </div>
      <div class="analysis-metric-pill">
        <span class="amp-val">${(data.domains ?? []).length}</span>
        <span class="amp-lbl">Domains</span>
      </div>
      <div class="analysis-metric-pill" style="${data.has_red_flags ? 'border-color: rgba(239,68,68,0.5);' : ''}">
        <span class="amp-val" style="${data.has_red_flags ? 'color: var(--red);' : 'color: var(--green);'}">${(data.red_flags ?? []).length}</span>
        <span class="amp-lbl">Red Flags</span>
      </div>
    `;
  }

  // Red-Flag Warning Banner
  const rfBanner = $("redFlagBanner");
  const rfTags = $("redFlagTagsList");
  if (rfBanner) {
    if (data.has_red_flags && data.red_flags?.length) {
      rfBanner.classList.remove("hidden");
      if (rfTags) {
        rfTags.innerHTML = data.red_flags.map(rf => `<span class="rf-tag">🚨 ${esc(rf)}</span>`).join("");
      }
    } else {
      rfBanner.classList.add("hidden");
    }
  }

  // Physiological Domains Grid
  const domainsGrid = $("domainsGrid");
  if (domainsGrid && data.domains) {
    domainsGrid.innerHTML = data.domains.map(d => {
      return `
        <div class="domain-card">
          <div class="domain-header">
            <div class="domain-title-wrap">
              <span class="domain-icon">${esc(d.icon || '🩺')}</span>
              <h4 class="domain-title">${esc(d.name)}</h4>
            </div>
            <span class="domain-count-badge">${d.symptoms.length}</span>
          </div>
          <ul class="domain-items-list">
            ${d.symptoms.map(item => `
              <li class="domain-item ${item.is_red_flag ? 'is-rf' : ''}">
                <span>${esc(item.text)}</span>
                ${item.duration ? `<small style="color:var(--text-faint);">${esc(item.duration)}</small>` : ''}
              </li>
            `).join("")}
          </ul>
        </div>
      `;
    }).join("");
  }

  // Doctor discussion points
  const pointsList = $("doctorPointsList");
  if (pointsList && data.discussion_points) {
    pointsList.innerHTML = data.discussion_points.map(pt => `<li>${esc(pt)}</li>`).join("");
  }
}

/* ═══════════════════════════════════════════════════════════════
   Shared Human-Like Voice Synthesis Engine (Natural TTS)
   ═══════════════════════════════════════════════════════════════ */
let cachedVoices = [];

function refreshVoices() {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    const list = window.speechSynthesis.getVoices();
    if (list && list.length) {
      cachedVoices = list;
    }
  }
}

if (typeof window !== "undefined" && "speechSynthesis" in window) {
  refreshVoices();
  window.speechSynthesis.onvoiceschanged = () => {
    refreshVoices();
  };
}

/**
 * Normalizes text to produce smooth, conversational human speech.
 * Expands clinical acronyms/units, strips emojis/markdown, formats numbers & pauses.
 */
function cleanTextForSpeech(raw) {
  if (!raw || typeof raw !== "string") return "";
  let s = raw;

  // 1. Strip markdown buttons, links, images
  s = s.replace(/\[\[button:.*?\|(.*?)\]\]/gi, "");
  s = s.replace(/!\[([^\]]*)\]\([^)]*\)/g, "");
  s = s.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");

  // 2. Strip HTML tags
  s = s.replace(/<[^>]+>/g, " ");

  // 3. Remove markdown headers, bold, italics, strikethrough, backticks
  s = s.replace(/^#{1,6}\s+/gm, "");
  s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  s = s.replace(/\*([^*]+)\*/g, "$1");
  s = s.replace(/__([^_]+)__/g, "$1");
  s = s.replace(/_([^_]+)_/g, "$1");
  s = s.replace(/~~([^~]+)~~/g, "$1");
  s = s.replace(/`([^`]+)`/g, "$1");
  s = s.replace(/^---+$/gm, " ");

  // 4. Strip emojis and pictographs so speech synthesizer doesn't speak emoji names
  s = s.replace(/\p{Extended_Pictographic}/gu, " ");

  // 5. Expand clinical units & medical abbreviations to conversational words
  s = s.replace(/\bResNet18\b/gi, "ResNet 18");
  s = s.replace(/\bg\/dL\b/gi, "grams per deciliter");
  s = s.replace(/\bmg\/dL\b/gi, "milligrams per deciliter");
  s = s.replace(/\bcells\/mcL\b/gi, "cells per microliter");
  s = s.replace(/\bcells\/cumm\b/gi, "cells per cubic millimeter");
  s = s.replace(/\bmm\/hr\b/gi, "millimeters per hour");
  s = s.replace(/\bWBC\b/g, "W-B-C");
  s = s.replace(/\bRBC\b/g, "R-B-C");
  s = s.replace(/\bESR\b/g, "E-S-R");
  s = s.replace(/\bTB\b/g, "T-B");
  s = s.replace(/\bAI\b/g, "A-I");
  s = s.replace(/\be\.g\.,?/gi, "for example,");
  s = s.replace(/\bi\.e\.,?/gi, "that is,");
  s = s.replace(/\bapprox\./gi, "approximately");

  // 6. Round clinical score numbers with 3+ decimals to 2 decimals for natural speech
  s = s.replace(/(\d+)\.(\d{3,})/g, (m, whole, dec) => `${whole}.${dec.slice(0, 2)}`);

  // 7. Human prosody formatting:
  // Convert bullet points to periods for breathing pauses
  s = s.replace(/^[•\-\*]\s+/gm, ". ");
  // Turn colons into gentle commas
  s = s.replace(/:\s+/g, ", ");
  // Collapse repeated punctuation & whitespace
  s = s.replace(/\.{2,}/g, ".");
  s = s.replace(/\s+/g, " ");

  return s.trim();
}

/**
 * Intelligent voice selector that prioritizes modern Neural / Natural / Google voices
 * and strictly penalizes / excludes robotic legacy desktop voices (David, Mark, Hazel, eSpeak).
 */
function getBestHumanVoice(targetLang = "en-IN") {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  refreshVoices();
  const voices = cachedVoices.length ? cachedVoices : window.speechSynthesis.getVoices();
  if (!voices || !voices.length) return null;

  const target = (targetLang || "en-IN").toLowerCase().replace("_", "-");
  const isTargetHindi = target.startsWith("hi");
  const isTargetIndian = target.includes("in");

  function scoreVoice(v) {
    let score = 0;
    const name = (v.name || "").toLowerCase();
    const lang = (v.lang || "").toLowerCase().replace("_", "-");

    // 1. Severe penalty for legacy robotic desktop synthesizers (Windows SAPI5 David, Mark, Hazel, eSpeak)
    const isRobotic = /desktop|david|espeak|festival|mark|hazel|george|susan|sam\b|mike\b/i.test(name);
    if (isRobotic) {
      score -= 500;
    }

    // 2. High priority for Neural / Natural / Online modern cloud voices
    if (/natural|online|neural|studio|premium|wavenet|enhanced|deepmind/i.test(name)) {
      score += 350;
    }

    // 3. High priority for Google Cloud / Chrome voices
    if (/google/i.test(name)) {
      score += 250;
    }

    // 4. Apple Siri / Enhanced voices
    if (/siri|samantha|veena|rishi|karen/i.test(name)) {
      score += 200;
    }

    // 5. Language matching
    if (isTargetHindi) {
      if (lang.startsWith("hi") || /hindi|swara|madhur|kalpana|hemant/i.test(name)) {
        score += 450;
      }
    } else if (isTargetIndian) {
      if (lang === "en-in" || lang.startsWith("en-in") || /india|neerja|prabhat|heera|ravi|kavya/i.test(name)) {
        score += 450;
      } else if (lang.startsWith("en")) {
        score += 60;
      }
    } else {
      // General English (en-US / en-GB)
      if (lang === target) {
        score += 250;
      } else if (lang.startsWith(target.split("-")[0])) {
        score += 150;
      }
    }

    // 6. Prefer warm, articulate female medical assistant voices
    if (/neerja|swara|jenny|aria|sonia|zira|samantha|female|kavya|kalpana|heera/i.test(name)) {
      score += 50;
    }

    // 7. Between legacy desktop voices on Windows, Zira is much softer and less robotic than David
    if (/zira/i.test(name)) {
      score += 80;
    }

    return score;
  }

  const sorted = [...voices].sort((a, b) => scoreVoice(b) - scoreVoice(a));
  return sorted[0] || null;
}

/* ─────────────────────────────────────────────────────────────
   Voice Assistant Model (Section 4 — English & Hindi Audio)
───────────────────────────────────────────────────────────── */

let activeUtterance = null;

function initVoiceAssistant() {
  const playBtn = $("vaPlayBtn");
  const pauseBtn = $("vaPauseBtn");
  const stopBtn = $("vaStopBtn");
  const langEnBtn = $("vaLangEn");
  const langHiBtn = $("vaLangHi");
  const speedPills = document.querySelectorAll(".speed-pill");

  // Language buttons
  if (langEnBtn && langHiBtn) {
    langEnBtn.addEventListener("click", () => {
      state.voiceLang = "en-US";
      langEnBtn.classList.add("active");
      langHiBtn.classList.remove("active");
      $("vaTranscriptLangLabel").textContent = "English";
      updateVoiceTranscriptUI();
      if (state.isSpeaking) {
        stopVoiceAssistant();
        speakSummaryReport();
      }
    });

    langHiBtn.addEventListener("click", () => {
      state.voiceLang = "hi-IN";
      langHiBtn.classList.add("active");
      langEnBtn.classList.remove("active");
      $("vaTranscriptLangLabel").textContent = "हिंदी";
      updateVoiceTranscriptUI();
      if (state.isSpeaking) {
        stopVoiceAssistant();
        speakSummaryReport();
      }
    });
  }

  // Speed selector
  speedPills.forEach(pill => {
    pill.addEventListener("click", () => {
      speedPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.speechRate = parseFloat(pill.dataset.speed) || 1.0;
      if (state.isSpeaking) {
        stopVoiceAssistant();
        speakSummaryReport();
      }
    });
  });

  // Play button
  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (window.speechSynthesis && window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        setSpeakingState(true);
      } else if (state.isSpeaking) {
        // Already speaking
      } else {
        speakSummaryReport();
      }
    });
  }

  // Pause button
  if (pauseBtn) {
    pauseBtn.addEventListener("click", () => {
      if (window.speechSynthesis && window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
        window.speechSynthesis.pause();
        setPausedState();
      }
    });
  }

  // Stop button
  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      stopVoiceAssistant();
    });
  }
}

async function fetchVoiceScript() {
  try {
    const xrayPred = state.xrayResult ? (state.xrayResult.prediction || state.xrayResult) : null;
    const labRows = state.labResult ? (state.labResult.rows || []) : [];

    const res = await fetch(`${API}/api/summary/voice-script`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lab_rows: labRows,
        xray_result: xrayPred,
        symptoms: state.symptoms,
        notes: state.notes
      })
    });
    const data = await parseJsonResponse(res);
    state.voiceScript = data;
    updateVoiceTranscriptUI();
  } catch (err) {
    console.warn("Failed to generate voice script:", err);
  }
}

function updateVoiceTranscriptUI() {
  const contentEl = $("vaTranscriptContent");
  if (!contentEl) return;

  if (state.voiceScript) {
    const text = state.voiceLang === "hi-IN" ? state.voiceScript.hindi_script : state.voiceScript.english_script;
    contentEl.textContent = text || "No script generated.";
  } else {
    contentEl.innerHTML = `<em>Click "Read Report Aloud" or switch languages to preview and hear the personalized diagnostic walkthrough script…</em>`;
  }
}

async function speakSummaryReport() {
  if (!window.speechSynthesis) {
    alert("Speech Synthesis (text-to-speech) is not supported in this browser.");
    return;
  }

  // Ensure script is ready
  if (!state.voiceScript) {
    setLoadingVoiceState();
    await fetchVoiceScript();
  }

  if (!state.voiceScript) {
    alert("Unable to generate audio script. Please review your session findings.");
    resetVoiceUI();
    return;
  }

  const rawScript = state.voiceLang === "hi-IN"
    ? state.voiceScript.hindi_script
    : state.voiceScript.english_script;

  if (!rawScript || !rawScript.trim()) {
    alert("No speech text available for this language.");
    resetVoiceUI();
    return;
  }

  const spokenText = cleanTextForSpeech(rawScript);
  if (!spokenText) {
    resetVoiceUI();
    return;
  }

  window.speechSynthesis.cancel();

  // Split into natural sentences for human breathing cadence & intonation contours
  const sentences = spokenText
    .split(/(?<=[.?!।])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);

  if (!sentences.length) {
    resetVoiceUI();
    return;
  }

  const selectedVoice = getBestHumanVoice(state.voiceLang);
  const baseRate = Math.max(0.8, Math.min(1.2, (state.speechRate || 1.0) * 0.95));

  setSpeakingState(true);

  // Queue sentence utterances sequentially with natural human prosody
  sentences.forEach((sentence, idx) => {
    const utterance = new SpeechSynthesisUtterance(sentence);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    } else {
      utterance.lang = state.voiceLang;
    }

    // Pure natural human acoustics:
    // Pitch 1.0 preserves natural formant harmonics (pitch > 1.0 creates metallic vocoder distortion)
    utterance.pitch = 1.0;
    utterance.rate = baseRate;
    utterance.volume = 1.0;

    if (idx === 0) {
      activeUtterance = utterance;
      utterance.onstart = () => {
        setSpeakingState(true);
      };
    }

    utterance.onpause = () => {
      setPausedState();
    };

    utterance.onresume = () => {
      setSpeakingState(true);
    };

    if (idx === sentences.length - 1) {
      utterance.onend = () => {
        resetVoiceUI();
      };
    }

    utterance.onerror = (e) => {
      if (e.error !== "canceled" && e.error !== "interrupted") {
        console.warn("Speech synthesis error on sentence:", e);
      }
      if (idx === sentences.length - 1) {
        resetVoiceUI();
      }
    };

    window.speechSynthesis.speak(utterance);
  });
}

function setSpeakingState(speaking) {
  state.isSpeaking = speaking;
  const playLabel = $("vaPlayLabel");
  const playIcon = $("vaPlayIcon");
  const pauseBtn = $("vaPauseBtn");
  const stopBtn = $("vaStopBtn");
  const dot = $("vaStatusDot");
  const text = $("vaStatusText");
  const viz = $("vaVisualizer");

  if (speaking) {
    if (playLabel) playLabel.textContent = state.voiceLang === "hi-IN" ? "बोल रहे हैं…" : "Speaking…";
    if (playIcon) playIcon.textContent = "🔊";
    if (pauseBtn) pauseBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = false;
    if (dot) { dot.className = "va-status-dot speaking"; }
    if (text) text.textContent = state.voiceLang === "hi-IN" ? "हिंदी में बोल रहे हैं…" : "Speaking in English…";
    if (viz) viz.classList.add("playing");
  }
}

function setPausedState() {
  const dot = $("vaStatusDot");
  const text = $("vaStatusText");
  const viz = $("vaVisualizer");
  const playLabel = $("vaPlayLabel");
  const playIcon = $("vaPlayIcon");

  if (dot) { dot.className = "va-status-dot paused"; }
  if (text) text.textContent = "Playback paused";
  if (playLabel) playLabel.textContent = state.voiceLang === "hi-IN" ? "पुनः शुरू करें" : "Resume";
  if (playIcon) playIcon.textContent = "▶️";
  if (viz) viz.classList.remove("playing");
}

function setLoadingVoiceState() {
  const text = $("vaStatusText");
  if (text) text.textContent = "Generating audio script…";
}

function stopVoiceAssistant() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  resetVoiceUI();
}

function resetVoiceUI() {
  state.isSpeaking = false;
  activeUtterance = null;
  const playLabel = $("vaPlayLabel");
  const playIcon = $("vaPlayIcon");
  const pauseBtn = $("vaPauseBtn");
  const stopBtn = $("vaStopBtn");
  const dot = $("vaStatusDot");
  const text = $("vaStatusText");
  const viz = $("vaVisualizer");

  if (playLabel) playLabel.textContent = state.voiceLang === "hi-IN" ? "रिपोर्ट सुनें" : "Read Report Aloud";
  if (playIcon) playIcon.textContent = "▶️";
  if (pauseBtn) pauseBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;
  if (dot) { dot.className = "va-status-dot"; }
  if (text) text.textContent = "Ready to speak";
  if (viz) viz.classList.remove("playing");
}

/* ═══════════════════════════════════════════════════════════════
   Settings & Theme Management (VS Code Popup & Dialog)
   ═══════════════════════════════════════════════════════════════ */
function initSettingsAndTheme() {
  const settingsBtn = $("settingsBtn");
  const settingsPopup = $("settingsPopup");
  const themeMenuTrigger = $("themeMenuTrigger");
  const themeSubmenu = $("themeSubmenu");
  const themeDarkBtn = $("themeDarkBtn");
  const themeLightBtn = $("themeLightBtn");
  const openFutureScopeBtn = $("openFutureScopeBtn");
  const openDatasetBtn = $("openDatasetBtn");

  const backdrop = $("settingsModalBackdrop");
  const modalClose = $("settingsModalClose");
  const tabBtns = document.querySelectorAll(".modal-tab-btn");
  const tabPanes = document.querySelectorAll(".modal-tab-pane");

  const themeCardDark = $("themeCardDark");
  const themeCardLight = $("themeCardLight");

  // 1. Theme State & Application
  function applyTheme(theme) {
    const activeTheme = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", activeTheme);
    try {
      localStorage.setItem("app_theme", activeTheme);
    } catch (_) { }

    // Update submenu checks
    if (themeDarkBtn) themeDarkBtn.classList.toggle("active", activeTheme === "dark");
    if (themeLightBtn) themeLightBtn.classList.toggle("active", activeTheme === "light");

    // Update modal cards
    if (themeCardDark) themeCardDark.classList.toggle("active", activeTheme === "dark");
    if (themeCardLight) themeCardLight.classList.toggle("active", activeTheme === "light");
  }

  // Load saved theme or default to light
  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("app_theme") || "light";
  } catch (_) { }
  applyTheme(savedTheme);

  // 2. Settings Popup Toggle
  function toggleSettingsPopup(show) {
    if (!settingsPopup) return;
    const isHidden = settingsPopup.hasAttribute("hidden");
    const shouldShow = typeof show === "boolean" ? show : isHidden;

    if (shouldShow) {
      settingsPopup.removeAttribute("hidden");
      if (settingsBtn) {
        settingsBtn.classList.add("active");
        settingsBtn.setAttribute("aria-expanded", "true");
      }
    } else {
      settingsPopup.setAttribute("hidden", "");
      if (themeSubmenu) themeSubmenu.classList.remove("show");
      if (settingsBtn) {
        settingsBtn.classList.remove("active");
        settingsBtn.setAttribute("aria-expanded", "false");
      }
    }
  }

  if (settingsBtn) {
    settingsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSettingsPopup();
    });
  }

  // Mobile/touch support for submenu toggle
  if (themeMenuTrigger) {
    themeMenuTrigger.addEventListener("click", (e) => {
      if (e.target.closest(".submenu-item")) return;
      if (themeSubmenu) {
        themeSubmenu.classList.toggle("show");
      }
    });
  }

  // Theme change triggers
  if (themeDarkBtn) {
    themeDarkBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      applyTheme("dark");
      toggleSettingsPopup(false);
    });
  }
  if (themeLightBtn) {
    themeLightBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      applyTheme("light");
      toggleSettingsPopup(false);
    });
  }
  if (themeCardDark) {
    themeCardDark.addEventListener("click", () => applyTheme("dark"));
  }
  if (themeCardLight) {
    themeCardLight.addEventListener("click", () => applyTheme("light"));
  }

  // 3. Settings Modal Tabs & Open/Close
  function switchModalTab(targetTab) {
    tabBtns.forEach((btn) => {
      const match = btn.dataset.targetTab === targetTab;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-selected", match ? "true" : "false");
    });

    tabPanes.forEach((pane) => {
      const paneId = pane.id.toLowerCase();
      const match =
        (targetTab === "future-scope" && paneId.includes("future")) ||
        (targetTab === "dataset" && paneId.includes("dataset")) ||
        (targetTab === "theme" && paneId.includes("theme"));
      pane.classList.toggle("active", match);
    });

    if (targetTab === "dataset") {
      loadDatasetStatus();
    }
  }

  function openSettingsModal(tabName = "future-scope") {
    toggleSettingsPopup(false);
    if (backdrop) {
      backdrop.removeAttribute("hidden");
      switchModalTab(tabName);
    }
  }

  function closeSettingsModal() {
    if (backdrop) {
      backdrop.setAttribute("hidden", "");
    }
  }

  if (openFutureScopeBtn) {
    openFutureScopeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openSettingsModal("future-scope");
    });
  }

  if (openDatasetBtn) {
    openDatasetBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openSettingsModal("dataset");
    });
  }

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      switchModalTab(btn.dataset.targetTab);
    });
  });

  if (modalClose) {
    modalClose.addEventListener("click", closeSettingsModal);
  }

  if (backdrop) {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) {
        closeSettingsModal();
      }
    });
  }

  // Close popup or modal on Escape key, close popup when clicking outside
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (backdrop && !backdrop.hasAttribute("hidden")) {
        closeSettingsModal();
      } else {
        toggleSettingsPopup(false);
      }
    }
  });

  document.addEventListener("click", (e) => {
    if (settingsPopup && !settingsPopup.hasAttribute("hidden")) {
      if (!settingsPopup.contains(e.target) && !settingsBtn.contains(e.target)) {
        toggleSettingsPopup(false);
      }
    }
  });
}

/* ═══════════════════════════════════════════════════════════════
   Smart Hybrid Medical Chatbot (Session Analyzer + Medical Q&A)
   ═══════════════════════════════════════════════════════════════ */
function getActiveSessionData() {
  const labRows = state.labResult?.rows || state.labResult?.lab_rows || state.labResult?.parsed_rows || [];
  const xray = state.xrayResult?.prediction || state.xrayResult || null;
  const symptoms = state.analyzedSymptoms?.separated_symptoms || state.symptoms || [];
  const redFlags = state.analyzedSymptoms?.red_flags || [];
  const notes = state.notes || $("notesInput")?.value || "";

  return {
    lab_report: {
      lab_rows: labRows,
      filename: state.labFile?.name || "",
      count: labRows.length,
    },
    xray: xray,
    symptoms: {
      notes: notes,
      separated_symptoms: symptoms,
      red_flags: redFlags,
    },
  };
}

function initChatbot() {
  const toggleBtn = $("chatToggleBtn");
  const chatWindow = $("chatWindow");
  const closeBtn = $("chatCloseBtn");
  const clearBtn = $("chatClearBtn");
  const voiceToggleBtn = $("chatVoiceToggleBtn");
  const chatForm = $("chatForm");
  const chatInput = $("chatInput");
  const sendBtn = $("chatSendBtn");
  const messagesContainer = $("chatMessages");
  const engineBadge = $("chatEngineBadge");
  const engineText = $("chatEngineText");
  const unreadDot = $("chatUnreadDot");

  if (!toggleBtn || !chatWindow || !chatForm || !messagesContainer) return;

  let chatHistory = [];
  let isSending = false;
  let autoVoiceEnabled = false; // Auto-voice output OFF by default in setting
  let currentUtterance = null;
  let activeSpeakingBtn = null;

  // Voice toggle button state helper
  function updateVoiceToggleBtnState() {
    if (!voiceToggleBtn) return;
    if (autoVoiceEnabled) {
      voiceToggleBtn.textContent = "🔊";
      voiceToggleBtn.className = "chat-action-btn voice-active";
      voiceToggleBtn.title = "Auto-Voice Output: ON (Natural Human Voice)";
      voiceToggleBtn.setAttribute("aria-label", "Auto-Voice Output: ON");
    } else {
      voiceToggleBtn.textContent = "🔇";
      voiceToggleBtn.className = "chat-action-btn";
      voiceToggleBtn.title = "Auto-Voice Output: OFF (Muted - Click to Enable)";
      voiceToggleBtn.setAttribute("aria-label", "Auto-Voice Output: OFF");
    }
  }

  // Restore saved preference if user explicitly changed it, otherwise default is false (OFF)
  try {
    const saved = localStorage.getItem("vida_auto_voice");
    if (saved !== null) {
      autoVoiceEnabled = saved === "true";
    }
  } catch (_) { }
  updateVoiceToggleBtnState();

  // Speech Synthesis Controller
  function stopVidaSpeech() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    currentUtterance = null;
    if (voiceToggleBtn) {
      voiceToggleBtn.classList.remove("speaking");
    }
    if (activeSpeakingBtn) {
      activeSpeakingBtn.classList.remove("speaking");
      activeSpeakingBtn.innerHTML = `<span class="listen-icon">🔊</span><span class="listen-label">Listen</span>`;
      activeSpeakingBtn = null;
    }
  }

  function speakVidaText(rawText, listenBtn = null) {
    if (!("speechSynthesis" in window)) return;
    stopVidaSpeech();

    const spokenText = cleanTextForSpeech(rawText);
    if (!spokenText) return;

    // Detect language: if response contains Devanagari Hindi characters, use Hindi voice
    const isHindi = /[\u0900-\u097F]/.test(spokenText);
    const targetLang = isHindi ? "hi-IN" : "en-IN";
    const humanVoice = getBestHumanVoice(targetLang);

    const utterance = new SpeechSynthesisUtterance(spokenText);
    if (humanVoice) {
      utterance.voice = humanVoice;
      utterance.lang = humanVoice.lang;
    } else {
      utterance.lang = targetLang;
    }

    // Pure natural human acoustics:
    // Pitch 1.0 (crucial: avoids metallic vocoder rasp and robotic distortion)
    utterance.pitch = 1.0;
    // Calm, articulate, empathetic conversational pacing
    utterance.rate = 0.95;
    utterance.volume = 1.0;

    utterance.onstart = () => {
      if (voiceToggleBtn && autoVoiceEnabled) {
        voiceToggleBtn.classList.add("speaking");
      }
      if (listenBtn) {
        activeSpeakingBtn = listenBtn;
        listenBtn.classList.add("speaking");
        listenBtn.innerHTML = `<span class="listen-icon">⏹️</span><span class="listen-label">Stop</span>`;
      }
    };

    utterance.onend = () => {
      stopVidaSpeech();
    };

    utterance.onerror = (e) => {
      if (e.error !== "canceled" && e.error !== "interrupted") {
        console.warn("Speech synthesis error:", e);
      }
      stopVidaSpeech();
    };

    currentUtterance = utterance;
    window.speechSynthesis.speak(utterance);
  }

  // Voice Toggle Button in Header
  if (voiceToggleBtn) {
    voiceToggleBtn.addEventListener("click", () => {
      if (currentUtterance && window.speechSynthesis.speaking) {
        // If speaking right now, clicking stops current speech
        stopVidaSpeech();
        return;
      }
      autoVoiceEnabled = !autoVoiceEnabled;
      try {
        localStorage.setItem("vida_auto_voice", autoVoiceEnabled ? "true" : "false");
      } catch (_) { }
      updateVoiceToggleBtnState();
      if (!autoVoiceEnabled) {
        stopVidaSpeech();
      }
    });
  }

  // 5. Markdown formatting
  function renderMarkdown(raw) {
    if (!raw) return "";
    let s = esc(raw);
    s = s.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    s = s.replace(/^## (.*$)/gim, "<h3>$1</h3>");
    s = s.replace(/^# (.*$)/gim, "<h3>$1</h3>");
    s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*(.*?)\*/g, "<em>$1</em>");
    s = s.replace(/`(.*?)`/g, "<code>$1</code>");
    s = s.replace(/^---$/gim, "<hr>");
    s = s.replace(/(?:^|\n)[-•] (.*)/g, "\n<li>$1</li>");
    s = s.replace(/((?:<li>.*<\/li>\s*)+)/g, "<ul>$1</ul>");
    s = s.replace(/\n\n+/g, "</p><p>");
    s = s.replace(/\n/g, "<br>");
    s = s.replace(
      /\[\[button:(.*?)\|(.*?)\]\]/g,
      '<div class="chat-inline-actions"><button type="button" class="chip-btn chat-inline-btn" data-query="$1">$2</button></div>'
    );
    return `<p>${s}</p>`;
  }

  function appendMessage(role, content, shouldAutoSpeak = false) {
    const msgEl = document.createElement("div");
    msgEl.className = `chat-msg ${role}`;

    if (role === "user") {
      msgEl.textContent = content;
    } else {
      msgEl.innerHTML = renderMarkdown(content);

      // Add individual message listen toolbar
      const footer = document.createElement("div");
      footer.className = "chat-msg-footer";

      const listenBtn = document.createElement("button");
      listenBtn.type = "button";
      listenBtn.className = "chat-listen-btn";
      listenBtn.title = "Listen to Vida read aloud in Indian accent";
      listenBtn.innerHTML = `<span class="listen-icon">🔊</span><span class="listen-label">Listen</span>`;

      listenBtn.addEventListener("click", () => {
        if (listenBtn.classList.contains("speaking")) {
          stopVidaSpeech();
        } else {
          speakVidaText(content, listenBtn);
        }
      });

      footer.appendChild(listenBtn);
      msgEl.appendChild(footer);

      // Auto-read aloud if enabled so user doesn't have to read
      if (shouldAutoSpeak && autoVoiceEnabled) {
        setTimeout(() => speakVidaText(content, listenBtn), 150);
      }
    }

    messagesContainer.appendChild(msgEl);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return msgEl;
  }

  function showTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "chat-msg assistant typing-indicator";
    indicator.id = "chatTypingIndicator";
    indicator.innerHTML = `
      <div class="chat-typing-dots">
        <span></span><span></span><span></span>
      </div>
    `;
    messagesContainer.appendChild(indicator);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = $("chatTypingIndicator");
    if (el) el.remove();
  }

  function resetChat() {
    stopVidaSpeech();
    chatHistory = [];
    messagesContainer.innerHTML = "";

    // Initial greeting
    appendMessage(
      "assistant",
      "👋 Hello! I am **Vida**, your Medical Screening & Diagnostic Assistant.\n\n" +
      "*How can I assist you today?*",
      false
    );

    // Re-attach suggestion chips
    const suggestions = document.createElement("div");
    suggestions.className = "chat-suggestions";
    suggestions.id = "chatSuggestions";
    suggestions.innerHTML = `
      <span class="suggestions-label">💡 Suggested Questions:</span>
      <div class="suggestions-grid">
        <button type="button" class="chip-btn" data-query="Explain my lab results">🧪 Explain my lab results</button>
        <button type="button" class="chip-btn" data-query="What does my X-ray score mean?">🫁 What does my X-ray score mean?</button>
        <button type="button" class="chip-btn" data-query="Review my symptoms">📝 Review my symptoms</button>
        <button type="button" class="chip-btn" data-query="What is Tuberculosis?">🔬 What is Tuberculosis?</button>
        <button type="button" class="chip-btn" data-query="When to see a doctor?">🚨 When to see a doctor?</button>
      </div>
    `;
    messagesContainer.appendChild(suggestions);

    suggestions.querySelectorAll(".chip-btn").forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.dataset.query;
        if (query && !isSending) {
          sendMessage(query);
        }
      });
    });
  }

  // Toggle open/close
  function toggleChat(show) {
    const isHidden = chatWindow.hasAttribute("hidden");
    const open = typeof show === "boolean" ? show : isHidden;

    if (open) {
      chatWindow.removeAttribute("hidden");
      toggleBtn.classList.add("active");
      if (unreadDot) unreadDot.style.display = "none";
      if (!messagesContainer.querySelector(".chat-msg")) {
        resetChat();
      }
      setTimeout(() => chatInput?.focus(), 150);
    } else {
      chatWindow.setAttribute("hidden", "");
      toggleBtn.classList.remove("active");
      stopVidaSpeech(); // Stop speaking when closed
    }
  }

  toggleBtn.addEventListener("click", () => toggleChat());
  if (closeBtn) closeBtn.addEventListener("click", () => toggleChat(false));
  if (clearBtn) clearBtn.addEventListener("click", () => resetChat());

  // Send message handler
  async function sendMessage(text) {
    const query = (text || chatInput?.value || "").trim();
    if (!query || isSending) return;

    if (chatInput) chatInput.value = "";
    isSending = true;
    if (sendBtn) sendBtn.disabled = true;

    // Stop previous speech if any
    stopVidaSpeech();

    // Remove old suggestions on first message
    const oldSuggestions = $("chatSuggestions");
    if (oldSuggestions) oldSuggestions.remove();

    appendMessage("user", query);
    showTypingIndicator();

    const sessionData = getActiveSessionData();

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: query,
          history: chatHistory.slice(-6),
          session_data: sessionData,
        }),
      });

      const data = await parseJsonResponse(res);
      removeTypingIndicator();

      if (data && data.reply) {
        // Automatically speak response aloud in Indian accent
        appendMessage("assistant", data.reply, true);
        chatHistory.push({ role: "user", content: query });
        chatHistory.push({ role: "assistant", content: data.reply });

        // Update engine badge
        if (engineText && engineBadge) {
          if (data.mode === "cloud_llm") {
            engineText.textContent = "Gemini AI";
            engineBadge.className = "chat-engine-badge cloud";
            engineBadge.title = `Active Model: ${data.model || "Gemini 1.5 Flash"}`;
          } else {
            engineText.textContent = "Clinical Engine";
            engineBadge.className = "chat-engine-badge";
            engineBadge.title = "Built-in Clinical Reasoning Engine (Offline Ready)";
          }
        }
      } else {
        appendMessage(
          "assistant",
          "⚠️ I encountered an unexpected response. Please try asking again or rephrase your question."
        );
      }
    } catch (err) {
      removeTypingIndicator();
      appendMessage(
        "assistant",
        `⚠️ Network or server error: ${err.message}. Please verify the local server connection.`
      );
    } finally {
      isSending = false;
      if (sendBtn) sendBtn.disabled = false;
      if (chatInput) chatInput.focus();
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });

  // Delegate clicks on suggestion chips
  messagesContainer.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip-btn");
    if (chip && chip.dataset.query && !isSending) {
      sendMessage(chip.dataset.query);
    }
  });

  // Initialize first greeting
  resetChat();
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initSidebarCollapse();
  initSettingsAndTheme();
  initScreeningDemo();
  loadXrayModelInfo();
  initVoiceTyping();
  initSymptomRecording();
  initVoiceAssistant();
  initChatbot();
  syncFindingsPanel();
  loadDatasetStatus();
  updateActiveSectionOnScroll();
});
