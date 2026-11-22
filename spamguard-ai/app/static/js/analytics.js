/** Render training evaluation statistics. No live message history is collected. */
(function () {
  "use strict";

  async function load() {
    try {
      const response = await fetch("/api/analytics?task=all");
      if (!response.ok) throw new Error("Evaluation data could not be loaded. Please refresh the page.");
      const data = await response.json();
      render(data);
      renderComparison(data.model_comparison || {});
    } catch (error) {
      document.getElementById("analyticsStats").innerHTML = `<div class="col-12"><div class="alert alert-danger">${escapeHtml(error.message)}</div></div>`;
    }
  }

  function render(data) {
    const model = (data.active_models || []).find((item) => item.task === "spam") || (data.active_models || [])[0];
    if (!model) return;
    const metrics = model.metrics || {};
    const dataset = model.dataset || {};
    const stats = [
      ["Test accuracy", percent(metrics.accuracy), "Correct predictions across the held-out set"],
      ["Spam precision", percent(metrics.precision), "How often spam predictions were correct"],
      ["Spam recall", percent(metrics.recall), "Share of spam messages detected"],
      ["Spam F1", percent(metrics.f1_score), "Balance of precision and recall"],
    ];
    document.getElementById("analyticsStats").innerHTML = stats.map(([title, value, note]) => `
      <div class="col-6 col-xl-3"><article class="card metric-card h-100"><div class="card-body">
        <div class="text-muted small">${title}</div><div class="metric-value">${value}</div><div class="small text-muted">${note}</div>
      </div></article></div>`).join("");

    document.getElementById("evaluationSnapshot").innerHTML = `
      <div class="d-flex flex-wrap gap-2 mb-3">
        <span class="badge badge-available">${escapeHtml(model.model_name || "Active model")}</span>
        <span class="badge bg-secondary">v${escapeHtml(model.model_version || "—")}</span>
        <span class="badge bg-secondary">Spam cutoff ${format(model.classification_threshold)}</span>
      </div>
      <p class="small text-muted mb-2">${escapeHtml(data.note || "Saved held-out evaluation metrics.")}</p>
      <p class="small mb-0"><strong>Model selection:</strong> ${escapeHtml(model.selection_criterion || "Selection details are not available.")}</p>`;

    const train = dataset.train_rows;
    const test = dataset.test_rows;
    document.getElementById("datasetProfile").innerHTML = `
      <div class="small text-muted mb-2">${escapeHtml(dataset.name || model.dataset_source || "Training dataset")}</div>
      <dl class="detail-list mb-0">
        <div><dt>Clean examples</dt><dd>${count(dataset.clean_rows)}</dd></div>
        <div><dt>Training split</dt><dd>${count(train)}</dd></div>
        <div><dt>Held-out test</dt><dd>${count(test)}</dd></div>
        <div><dt>Spam in test</dt><dd>${count(dataset.test_class_counts && dataset.test_class_counts.SPAM)}</dd></div>
        <div><dt>Ham in test</dt><dd>${count(dataset.test_class_counts && dataset.test_class_counts.HAM)}</dd></div>
        <div><dt>Features</dt><dd>${escapeHtml(model.features && model.features.method || "—")}</dd></div>
      </dl>`;

    const matrix = model.confusion_matrix && model.confusion_matrix.matrix;
    document.getElementById("confusionMatrix").innerHTML = `
      <h3 class="h6">Test-set outcomes</h3>
      ${matrix && matrix.length === 2 ? `<div class="confusion-grid" role="table" aria-label="Confusion matrix">
        <span></span><span class="text-muted small">Predicted ham</span><span class="text-muted small">Predicted spam</span>
        <span class="text-muted small">Actual ham</span><strong>${count(matrix[0][0])}</strong><strong class="error-cell">${count(matrix[0][1])} false alarms</strong>
        <span class="text-muted small">Actual spam</span><strong class="error-cell">${count(matrix[1][0])} missed</strong><strong>${count(matrix[1][1])}</strong>
      </div>` : '<div class="small text-muted">Confusion matrix is not available.</div>'}
      <div class="small text-muted mt-2">${count(metrics.tn)} true negatives · ${count(metrics.fp)} false positives · ${count(metrics.fn)} false negatives · ${count(metrics.tp)} true positives</div>`;
  }

  function renderComparison(comparison) {
    const container = document.getElementById("analyticsComparison");
    const entries = Object.entries(comparison).filter(([, rows]) => rows && rows.length);
    if (!entries.length) {
      container.innerHTML = '<div class="text-muted small">No candidate comparison data available.</div>';
      return;
    }
    container.innerHTML = entries.map(([task, rows]) => `
      <section class="mb-3"><h3 class="text-muted small text-uppercase fw-bold mb-2">${task === "spam" ? "SMS Spam" : "Email Phishing"} candidates</h3>
      <div class="table-responsive"><table class="comparison-table"><thead><tr><th>Model</th><th>Test F1</th><th>CV F1</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th><th>Cutoff</th><th>Selected</th></tr></thead>
      <tbody>${rows.map((row) => `<tr><td>${escapeHtml(row.model_name || "—")}</td><td>${format(row.f1_score)}</td><td>${format(row.cv_f1_selected_threshold)}</td><td>${format(row.precision)}</td><td>${format(row.recall)}</td><td>${format(row.roc_auc)}</td><td>${format(row.decision_threshold)}</td><td>${row.is_selected ? '<span class="badge badge-available">Yes</span>' : '<span class="badge badge-unavailable">No</span>'}</td></tr>`).join("")}</tbody></table></div></section>`).join("");
  }

  function format(value) { return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3); }
  function percent(value) { return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`; }
  function count(value) { return value == null ? "—" : Number(value).toLocaleString(); }
  function escapeHtml(value) { const node = document.createElement("span"); node.textContent = value == null ? "" : String(value); return node.innerHTML; }
  load();
})();
