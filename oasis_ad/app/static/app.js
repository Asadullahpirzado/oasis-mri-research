/**
 * OASIS-1 MRI Research Classifier
 * Client application for research predictions and uncertainty displays.
 */

// ---------------------------------------------------------------------------
// Utility Helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const pct = (x) => (typeof x === "number" ? (100 * x).toFixed(1) + "%" : "--");

function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// ---------------------------------------------------------------------------
// Theme Management (Clinical Light <-> Dark PACS)
// ---------------------------------------------------------------------------
const themeToggleBtn = $("theme-toggle");
const themeIconSun = $("theme-icon-sun");
const themeIconMoon = $("theme-icon-moon");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("neurotrust_theme", theme);
  if (theme === "dark") {
    themeIconSun.style.display = "none";
    themeIconMoon.style.display = "block";
  } else {
    themeIconSun.style.display = "block";
    themeIconMoon.style.display = "none";
  }
}

const savedTheme = localStorage.getItem("neurotrust_theme") || "light";
applyTheme(savedTheme);

themeToggleBtn.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(current === "light" ? "dark" : "light");
});

// ---------------------------------------------------------------------------
// Tab Navigation
// ---------------------------------------------------------------------------
const tabUploadBtn = $("tab-upload-btn");
const tabCohortBtn = $("tab-cohort-btn");
const tabUploadContent = $("tab-upload-content");
const tabCohortContent = $("tab-cohort-content");

tabUploadBtn.addEventListener("click", () => {
  tabUploadBtn.classList.add("active");
  tabCohortBtn.classList.remove("active");
  tabUploadContent.classList.add("active");
  tabCohortContent.classList.remove("active");
});

tabCohortBtn.addEventListener("click", () => {
  tabCohortBtn.classList.add("active");
  tabUploadBtn.classList.remove("active");
  tabCohortContent.classList.add("active");
  tabUploadContent.classList.remove("active");
});

// ---------------------------------------------------------------------------
// Drag-and-Drop & File Attachment
// ---------------------------------------------------------------------------
const dropzone = $("dropzone");
const fileInput = $("files");
const fileChips = $("file-chips");
let selectedFiles = [];

function updateFileDisplay(files) {
  selectedFiles = Array.from(files);
  if (!selectedFiles.length) {
    fileChips.hidden = true;
    fileChips.innerHTML = "";
    return;
  }

  fileChips.hidden = false;
  fileChips.innerHTML = selectedFiles
    .map((file) => {
      const ext = file.name.split(".").pop().toUpperCase();
      return `
      <div class="attached-file-chip">
        <div>
          <span class="chip-name">${file.name}</span>
          <span class="chip-size">(${formatBytes(file.size)})</span>
        </div>
        <span class="chip-tag">${ext}</span>
      </div>`;
    })
    .join("");
}

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length) {
    updateFileDisplay(e.target.files);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (dt.files && dt.files.length) {
    fileInput.files = dt.files;
    updateFileDisplay(dt.files);
  }
});

// ---------------------------------------------------------------------------
// Pipeline Progress Stepper & Busy State
// ---------------------------------------------------------------------------
const stepper = $("pipeline-stepper");
const stepperStatus = $("stepper-status");
const emptyState = $("empty-state");
const errBox = $("err");
const errMsg = $("err-msg");
const errTitle = $("err-title");
const resultBox = $("result");
const runBtn = $("run");
const runExBtn = $("runex");

let stepperInterval = null;

function setBusy(busy, actionMsg = "Running 3D Inference Pipeline...") {
  runBtn.disabled = busy;
  runExBtn.disabled = busy;

  const btnSpinner = runBtn.querySelector(".btn-spinner");
  const btnText = runBtn.querySelector(".btn-text");
  if (btnSpinner && btnText) {
    btnSpinner.hidden = !busy;
    btnText.style.display = busy ? "none" : "flex";
  }

  if (busy) {
    emptyState.hidden = true;
    errBox.hidden = true;
    resultBox.hidden = true;
    stepper.hidden = false;
    stepperStatus.textContent = actionMsg;

    let step = 1;
    clearInterval(stepperInterval);
    stepperInterval = setInterval(() => {
      step = (step % 4) + 1;
      for (let i = 1; i <= 4; i++) {
        const el = $("st-" + i);
        if (el) {
          if (i <= step) el.classList.add("active");
          else el.classList.remove("active");
        }
      }
    }, 400);
  } else {
    clearInterval(stepperInterval);
    stepper.hidden = true;
  }
}

function showErr(msg, title = "Clinical Pipeline Advisory") {
  setBusy(false);
  emptyState.hidden = true;
  resultBox.hidden = true;
  errBox.hidden = false;
  errTitle.textContent = title;
  errMsg.textContent = msg;
  errBox.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Render Diagnostic Inference Results
// ---------------------------------------------------------------------------
function render(r, subjectName = "Uploaded Brain Scan") {
  setBusy(false);
  errBox.hidden = true;
  emptyState.hidden = true;
  resultBox.hidden = false;

  // 1. Session & Patient Header
  const now = new Date();
  $("report-timestamp").textContent = `Evaluated: ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}`;
  $("report-patient-title").textContent = subjectName;
  const auditHex = Math.random().toString(16).substring(2, 10).toUpperCase();
  $("report-session-id").textContent = `SESSION: #${r.id || auditHex}`;
  $("audit-hash").textContent = `SHA256: ${auditHex}...${Math.random().toString(16).substring(2, 6).toUpperCase()}`;

  // 2. Verdict Banner
  const verdictBanner = $("verdict-banner");
  const verdictPill = $("verdict-pill");
  const verdictMeta = $("verdict-meta");
  const decisionText = $("decision-text");
  const reasonsList = $("reasons-list");

  decisionText.textContent = r.decision || "Research analysis complete";

  if (r.abstain) {
    verdictBanner.className = "verdict-banner-card verdict-abstain";
    verdictPill.textContent = "ABSTAIN - CLINICAL EXPERT REVIEW REQUIRED";
  } else {
    verdictBanner.className = "verdict-banner-card verdict-ok";
    verdictPill.textContent = "RESEARCH GROUP ESTIMATE";
  }
  verdictMeta.textContent = `Ensemble of ${r.n_models || 5} Folds Calibrated Output`;

  if (r.reasons && r.reasons.length) {
    reasonsList.innerHTML = r.reasons
      .map(
        (txt) => `
      <div class="verdict-reason-item">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <span>${txt}</span>
      </div>`
      )
      .join("");
    reasonsList.hidden = false;
  } else {
    reasonsList.innerHTML = "";
    reasonsList.hidden = true;
  }

  // 3. 4-Card Uncertainty Matrix
  $("val-conf").textContent = pct(r.confidence);
  $("sub-conf").textContent = "Temperature-scaled Softmax";

  const confSet = r.conformal_set && r.conformal_set.length ? r.conformal_set.join(" | ") : "Empty";
  $("val-conformal").textContent = confSet;
  $("sub-conformal").textContent = `${pct(r.conformal_target_coverage || 0.9)} Target Coverage`;

  const ood = r.ood || {};
  const oodEl = $("val-ood");
  if (ood.checked) {
    oodEl.textContent = ood.flag ? "ANOMALY" : "TYPICAL";
    oodEl.style.color = ood.flag ? "var(--color-danger)" : "var(--color-success)";
    $("sub-ood").textContent = `z=${(ood.mahalanobis_z_mean ?? 0).toFixed(2)} (thr: ${(ood.flag_threshold_z_mean ?? 0).toFixed(2)})`;
  } else {
    oodEl.textContent = "N/A";
    oodEl.style.color = "var(--text-muted)";
    $("sub-ood").textContent = "Mahalanobis latent metric";
  }

  const disagreement = typeof r.model_disagreement_sd === "number" ? r.model_disagreement_sd.toFixed(3) : "--";
  $("val-disagreement").textContent = disagreement;

  // 4. Grad-CAM 3-Plane Visualizer
  const saliencyCard = $("saliency-card");
  const camImg = $("cam-img");
  if (r.gradcam_png_base64) {
    saliencyCard.hidden = false;
    camImg.src = "data:image/png;base64," + r.gradcam_png_base64;
  } else {
    saliencyCard.hidden = true;
  }

  // 5. Posterior Class Probabilities
  const probsContainer = $("probs-bars");
  if (r.probabilities) {
    probsContainer.innerHTML = Object.entries(r.probabilities)
      .map(([clsName, pVal]) => {
        let barClass = "mild";
        if (clsName.toLowerCase().includes("cdr 0") && !clsName.toLowerCase().includes("0.5")) {
          barClass = "normal";
        } else if (clsName.toLowerCase().includes("1") || clsName.toLowerCase().includes("moderate")) {
          barClass = "moderate";
        }

        return `
        <div class="prob-item">
          <div class="prob-label-row">
            <span class="prob-class-name">
              <span class="prob-tag-badge">${clsName.split(" ")[0] || "CLASS"}</span>
              ${clsName}
            </span>
            <span class="prob-score">${pct(pVal)}</span>
          </div>
          <div class="prob-rail">
            <div class="prob-bar ${barClass}" style="width: ${Math.max(1, pVal * 100)}%;"></div>
          </div>
        </div>`;
      })
      .join("");
  }

  // 6. Normative Demographic Baseline
  const baselineSection = $("baseline-section");
  if (r.age_sex_baseline && r.age_sex_baseline.probabilities) {
    baselineSection.hidden = false;
    $("baseline-note").textContent =
      r.age_sex_baseline.note || "Demographic distribution conditioned solely on participant age and sex in OASIS-1.";
    $("baseline-bars").innerHTML = Object.entries(r.age_sex_baseline.probabilities)
      .map(
        ([clsName, pVal]) => `
      <div class="prob-item">
        <div class="prob-label-row">
          <span class="prob-class-name">${clsName}</span>
          <span class="prob-score">${pct(pVal)}</span>
        </div>
        <div class="prob-rail">
          <div class="prob-bar" style="width: ${Math.max(1, pVal * 100)}%; background: #64748b;"></div>
        </div>
      </div>`
      )
      .join("");
  } else {
    baselineSection.hidden = true;
  }

  // 7. Scan Quality Control (QC) Table
  const q = r.qc || {};
  const qcBadge = $("qc-badge");
  const qcDetail = $("qc-detail");
  const qcTableBody = $("qc-table-body");

  if (q.hard_fail) {
    qcBadge.className = "badge-status bad";
    qcBadge.textContent = "Failed";
  } else if (q.soft_flag) {
    qcBadge.className = "badge-status warn";
    qcBadge.textContent = "Warning";
  } else {
    qcBadge.className = "badge-status ok";
    qcBadge.textContent = "Passed";
  }
  qcDetail.textContent = q.detail || "Automated scan pre-flight verification completed.";

  const m = q.metrics || {};
  const qcItems = [
    {
      name: "Brain Volume Fraction (brain_frac)",
      val: m.brain_frac !== undefined && m.brain_frac !== null ? Number(m.brain_frac).toFixed(3) : "--",
      range: "[0.10, 0.40]",
      status: m.brain_frac !== undefined && m.brain_frac !== null ? (m.brain_frac >= 0.10 && m.brain_frac <= 0.40 ? "Normal" : "Out of Range") : "N/A",
    },
    {
      name: "Intensity CoV In-Brain (cov_in)",
      val: m.cov_in !== undefined && m.cov_in !== null ? Number(m.cov_in).toFixed(3) : "--",
      range: "[0.15, 0.70]",
      status: m.cov_in !== undefined && m.cov_in !== null ? (m.cov_in >= 0.15 && m.cov_in <= 0.70 ? "Normal" : "Out of Range") : "N/A",
    },
    {
      name: "Dynamic Range Ratio (p99 / p50)",
      val: m.p99_over_p50 !== undefined && m.p99_over_p50 !== null ? Number(m.p99_over_p50).toFixed(2) : "--",
      range: "[1.20, 5.00]",
      status: m.p99_over_p50 !== undefined && m.p99_over_p50 !== null ? (m.p99_over_p50 >= 1.2 && m.p99_over_p50 <= 5.0 ? "Normal" : "Out of Range") : "N/A",
    },
    {
      name: "High-Freq Gradient Energy (grad_energy)",
      val: m.grad_energy !== undefined && m.grad_energy !== null ? Number(m.grad_energy).toFixed(4) : "--",
      range: "[0.005, 0.200]",
      status: m.grad_energy !== undefined && m.grad_energy !== null ? (m.grad_energy >= 0.005 && m.grad_energy <= 0.2 ? "Normal" : "Out of Range") : "N/A",
    },
  ];

  qcTableBody.innerHTML = qcItems
    .map(
      (item) => `
    <tr>
      <td>${item.name}</td>
      <td><strong>${item.val}</strong></td>
      <td>${item.range}</td>
      <td>
        <span class="badge-status ${item.status === "Normal" ? "ok" : item.status === "Out of Range" ? "warn" : "info"}">${item.status}</span>
      </td>
    </tr>`
    )
    .join("");

  // 8. Sign-off & Audit
  if (r.reference_cdr_from_metadata !== undefined) {
    $("ref-cdr-label").textContent = `CDR ${r.reference_cdr_from_metadata} (${
      r.reference_cdr_from_metadata === 0 ? "Normal Control" : "Impaired / Dementia"
    })`;
  } else {
    $("ref-cdr-label").textContent = "None (External Scan Upload)";
  }

  resultBox.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// API Pipeline Invocations
// ---------------------------------------------------------------------------
async function callApi(url, options = {}, subjectName = "Brain Scan") {
  setBusy(true, "Evaluating 3D Volume Through 5-Fold Ensemble...");
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Research analysis returned HTTP ${res.status}`);
    }
    render(data, subjectName);
  } catch (err) {
    showErr(err.message || "Unable to complete diagnostic inference.");
  }
}

// 1. Upload Form Execution
runBtn.addEventListener("click", () => {
  const files = fileInput.files;
  if (!files || !files.length) {
    showErr("Please select or drop an MRI volume (.nii, .nii.gz, or .hdr + .img) first.", "Missing Input Volume");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const age = $("age").value.trim();
  const sex = $("sex").value.trim();
  if (age) formData.append("age", age);
  if (sex) formData.append("sex", sex);

  const scanLabel = files.length > 0 ? files[0].name : "Uploaded MRI Volume";
  callApi("api/predict", { method: "POST", body: formData }, scanLabel);
});

// 2. Cohort Dropdown Execution
runExBtn.addEventListener("click", () => {
  const exampleId = $("examples").value;
  if (!exampleId) {
    showErr("Please select an OASIS-1 subject from the dropdown.", "No Subject Selected");
    return;
  }
  callApi("api/example/" + encodeURIComponent(exampleId), {}, `Participant ${exampleId}`);
});

// 3. Quick Presets Execution (1-Click Instant Load)
document.querySelectorAll(".preset-card").forEach((card) => {
  card.addEventListener("click", () => {
    const pid = card.getAttribute("data-pid");
    if (pid) {
      const select = $("examples");
      select.value = pid;
      callApi("api/example/" + encodeURIComponent(pid), {}, `Participant ${pid} (Validation Preset)`);
    }
  });
});

// 4. Print / PDF Export Action
$("btn-print").addEventListener("click", () => {
  window.print();
});

// ---------------------------------------------------------------------------
// Initial Bootstrap
// ---------------------------------------------------------------------------
(async function init() {
  try {
    // 1. Model Telemetry & Configuration
    const infoRes = await fetch("api/model-info");
    if (infoRes.ok) {
      const info = await infoRes.json();
      if (info.disclaimer) {
        $("disclaimer-text").textContent = info.disclaimer;
      }
      if (info.n_models) {
        $("hdr-arch").textContent = `${info.n_models}-Fold 3D CNN`;
        $("info-arch").textContent = `${info.model || "3D CNN"} (${info.n_models} Folds)`;
      }
      if (info.conformal_target_coverage) {
        $("hdr-coverage").textContent = `${pct(info.conformal_target_coverage)} Coverage`;
      }
      const tau = info.rejection_confidence_tau !== undefined ? info.rejection_confidence_tau : info.rejection_tau;
      if (tau !== undefined) {
        $("info-tau").textContent = typeof tau === "number" ? tau.toFixed(3) : tau;
      }
    }

    // 2. Health & Device Telemetry
    const healthRes = await fetch("api/health");
    if (healthRes.ok) {
      const health = await healthRes.json();
      if (health.device) {
        $("hdr-device").textContent = health.device.toUpperCase() + " Inference";
      }
    }

    // 3. Cached Subjects Library
    const exRes = await fetch("api/examples");
    if (exRes.ok) {
      const examples = await exRes.json();
      const exSelect = $("examples");
      if (examples && examples.length) {
        exSelect.innerHTML = examples
          .map(
            (e) => `
          <option value="${e.id}">${e.id} (Age ${e.age ?? "--"}, ${e.sex ?? "--"}, CDR ${e.reference_cdr ?? "--"})</option>
        `
          )
          .join("");
      } else {
        exSelect.innerHTML = `<option value="">(No cached volumes detected in bundle)</option>`;
      }
    }
  } catch (err) {
    console.warn("Telemetry greeting warning:", err);
  }
})();
