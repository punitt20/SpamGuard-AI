/**
 * Model Info page.
 *
 * Fetches model metadata from /api/model and renders task cards, detailed
 * model information, and candidate comparison tables.
 */
(function () {
  "use strict";

  async function load() {
    try {
      const resp = await fetch("/api/model");
      if (!resp.ok) throw new Error("Failed to load model info");
      const data = await resp.json();

      renderTaskCards(data.tasks || {});
      renderComparison(data.tasks || {});
      renderDetails(data.tasks || {});
    } catch (err) {
      console.error(err);
      document.getElementById("taskCards").innerHTML =
        '<div class="alert alert-danger">Failed to load model information.</div>';
    }
  }

  function renderTaskCards(tasks) {
    const container = document.getElementById("taskCards");
    container.innerHTML = "";

    for (const [task, info] of Object.entries(tasks)) {
      const available = info.available === true;
      const badgeCls = available ? "badge-available" : "badge-unavailable";
      const badgeText = available ? "Available" : "Not trained";

      const card = document.createElement("div");
      card.className = "card mb-3";
      card.innerHTML = `
        <div class="card-body">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <div>
              <div class="fw-bold">${info.task === "spam" ? "SMS Spam" : "Email Phishing"}</div>
              <div class="text-muted small">${info.task_label || ""}</div>
            </div>
            <span class="badge ${badgeCls}">${badgeText}</span>
          </div>
          ${available ? `
            <div class="row g-3 mt-3">
              <div class="col-6">
                <div class="text-muted small">Model:</div>
                <div class="fw-semibold">${escapeHtml(info.model_name || "—")}</div>
              </div>
              <div class="col-6">
                <div class="text-muted small">Version:</div>
                <div class="fw-semibold">${escapeHtml(info.model_version || "—")}</div>
              </div>
              <div class="col-6">
                <div class="text-muted small">Trained:</div>
                <div class="fw-semibold">${escapeHtml(info.trained_at || "—")}</div>
              </div>
              <div class="col-6">
                <div class="text-muted small">Dataset:</div>
                <div class="fw-semibold text-truncate" style="max-width: 140px;">${escapeHtml(info.dataset_source || "—")}</div>
              </div>
            </div>
          ` : `
            <div class="mt-3 text-muted small">
              The ${task} model has not been trained yet. Run the training pipeline to enable it.
            </div>
          `}
        </div>
      `;
      container.appendChild(card);
    }
  }

  function renderDetails(tasks) {
    const active = Object.values(tasks).find((task) => task.available && task.task === "spam") || Object.values(tasks).find((task) => task.available);
    const performance = document.getElementById("modelPerformance");
    const training = document.getElementById("trainingDetails");
    const notes = document.getElementById("modelNotes");
    if (!active) {
      performance.innerHTML = '<div class="col-12 text-muted">No trained model metrics are available.</div>';
      training.innerHTML = notes.innerHTML = '<p class="text-muted mb-0">Train a model to see its evaluation details.</p>';
      return;
    }
    const metadata = active.metadata || {};
    const metrics = metadata.metrics || {};
    const dataset = metadata.dataset || {};
    const cards = [
      ["Accuracy", percent(metrics.accuracy)], ["Spam precision", percent(metrics.precision)],
      ["Spam recall", percent(metrics.recall)], ["Spam F1", percent(metrics.f1_score)],
      ["ROC-AUC", format(metrics.roc_auc)], ["Decision cutoff", format(metadata.classification_threshold)],
    ];
    performance.innerHTML = cards.map(([label, value]) => `<div class="col-6 col-md-4 col-xl-2"><div class="metric-card h-100"><div class="text-muted small">${label}</div><div class="metric-value">${value}</div></div></div>`).join("");

    const matrix = metadata.confusion_matrix && metadata.confusion_matrix.matrix;
    training.innerHTML = `<dl class="detail-list mb-0">
      <div><dt>Dataset</dt><dd>${escapeHtml(dataset.name || "—")}</dd></div>
      <div><dt>Source</dt><dd>${dataset.source ? `<a href="${escapeAttr(dataset.source)}" target="_blank" rel="noopener noreferrer">Open dataset source</a>` : "—"}</dd></div>
      <div><dt>Clean examples</dt><dd>${count(dataset.clean_rows)}</dd></div>
      <div><dt>Train / test</dt><dd>${count(dataset.train_rows)} / ${count(dataset.test_rows)}</dd></div>
      <div><dt>Feature method</dt><dd>${escapeHtml(metadata.features && metadata.features.method || "—")}</dd></div>
      <div><dt>Vocabulary limit</dt><dd>${count(metadata.features && metadata.features.max_features)}</dd></div>
      <div><dt>N-grams</dt><dd>${escapeHtml(metadata.features && metadata.features.ngram_range ? metadata.features.ngram_range.join("–") : "—")}</dd></div>
    </dl>`;
    notes.innerHTML = `<dl class="detail-list mb-3">
      <div><dt>Model selection</dt><dd>${escapeHtml(metadata.selection_criterion || "—")}</dd></div>
      <div><dt>Selection rationale</dt><dd>${escapeHtml(metadata.selection_rationale || "—")}</dd></div>
      <div><dt>Spam TN / FP</dt><dd>${matrix ? `${count(matrix[0][0])} / ${count(matrix[0][1])}` : "—"}</dd></div>
      <div><dt>Spam FN / TP</dt><dd>${matrix ? `${count(matrix[1][0])} / ${count(matrix[1][1])}` : "—"}</dd></div>
      <div><dt>Inference time</dt><dd>${metrics.inference_seconds == null ? "—" : `${Number(metrics.inference_seconds).toFixed(4)} s`}</dd></div>
    </dl><p class="small text-muted mb-0">${escapeHtml(metadata.preprocessing && metadata.preprocessing.note || "Evaluation reflects the saved held-out dataset split; real-world performance may differ.")}</p>`;
  }

  function renderComparison(tasks) {
    const container = document.getElementById("comparisonContainer");
    container.innerHTML = "";

    let hasAny = false;
    for (const [task, info] of Object.entries(tasks)) {
      if (!info.available) continue;
      const metadata = info.metadata || {};
      const comparison = metadata.comparison || [];
      if (!comparison || comparison.length === 0) continue;

      hasAny = true;

      const section = document.createElement("div");
      section.className = "mb-4";
      section.innerHTML = `
        <h6 class="text-muted small text-uppercase fw-bold letter-spacing-wide mb-3">
          ${task === "spam" ? "SMS Spam" : "Email Phishing"} — Candidate Comparison
        </h6>
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>ROC-AUC</th>
              <th>Cutoff</th>
              <th>Selected</th>
              <th>Probability</th>
            </tr>
          </thead>
          <tbody>
            ${comparison
              .map(
                (row) => `
                <tr>
                  <td>${escapeHtml(row.model_name || "—")}</td>
                  <td>${formatNumber(row.accuracy)}</td>
                  <td>${formatNumber(row.precision)}</td>
                  <td>${formatNumber(row.recall)}</td>
                  <td>${formatNumber(row.f1_score)}</td>
                  <td>${formatNumber(row.roc_auc)}</td>
                  <td>${formatNumber(row.decision_threshold)}</td>
                  <td>${row.is_selected ? '<span class="badge badge-available">Yes</span>' : '<span class="badge badge-unavailable">No</span>'}</td>
                  <td>${row.probability_supported ? '<span class="badge badge-available">Yes</span>' : '<span class="badge badge-unavailable">No</span>'}</td>
                </tr>
              `
              )
              .join("")}
          </tbody>
        </table>
      `;
      container.appendChild(section);
    }

    if (!hasAny) {
      container.innerHTML = '<div class="text-muted small">No candidate comparison data available.</div>';
    }
  }

  function formatNumber(v) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(4);
  }

  function format(v) { return v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(3); }
  function percent(v) { return v == null ? "—" : `${(Number(v) * 100).toFixed(1)}%`; }
  function count(v) { return v == null ? "—" : Number(v).toLocaleString(); }
  function escapeAttr(value) { return escapeHtml(value).replace(/"/g, "&quot;"); }

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
  }

  load();
})();
