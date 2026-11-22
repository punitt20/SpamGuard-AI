/**
 * Single-page SMS spam analyzer.
 *
 * Sends one message to POST /api/predict and renders the classifier label and
 * spam probability. Shows which tokens pushed the prediction toward
 * positive or negative class.
 */
(function () {
  "use strict";

  const root = document.getElementById("analyzerRoot");
  if (!root) return;

  const maxLength = Number(root.dataset.maxLength || 20000);
  const els = {
    input: document.getElementById("messageInput"),
    form: document.getElementById("analyzerForm"),
    counter: document.getElementById("charCounter"),
    analyze: document.getElementById("analyzeBtn"),
    clear: document.getElementById("clearBtn"),
    samples: document.getElementById("sampleChips"),
    error: document.getElementById("analyzerError"),
    errorText: document.getElementById("analyzerErrorText"),
    loading: document.getElementById("analysisLoading"),
    result: document.getElementById("resultPanel"),
    badge: document.getElementById("resultBadge"),
    verdict: document.getElementById("verdictLabel"),
    sub: document.getElementById("verdictSub"),
    probability: document.getElementById("resultProbability"),
    meter: document.getElementById("probabilityMeter"),
    meta: document.getElementById("resultMeta"),
    signalsCard: document.getElementById("signalsCard"),
    signalsPositive: document.getElementById("signalsPositive"),
    signalsNegative: document.getElementById("signalsNegative"),
  };

  const SAMPLES = {
    spam: [
      {
        label: "Prize offer",
        text: "Congratulations! You have won a FREE cash prize of £500. Claim it now at http://bit.ly/demo",
      },
      {
        label: "Account warning",
        text: "URGENT: Your account will be suspended. Reply immediately with your PIN to verify your identity.",
      },
      {
        label: "Delivery fee",
        text: "Your parcel is waiting. Pay a small delivery fee now to release it: http://delivery.example.test/track",
      },
      {
        label: "Everyday chat",
        text: "Hi, are we still meeting near the station at 6 pm today?",
      },
      {
        label: "Work reminder",
        text: "Reminder: our project check-in is at 10 tomorrow. Please bring the latest draft.",
      },
    ],
  };

  function updateCounter() {
    const length = els.input.value.length;
    els.counter.textContent = `${length.toLocaleString()} / ${maxLength.toLocaleString()} characters`;
    els.counter.classList.toggle("is-near", length > maxLength * 0.9);
  }

  function showError(message) {
    els.errorText.textContent = message;
    els.error.classList.remove("d-none");
  }

  function hideError() {
    els.error.classList.add("d-none");
    els.errorText.textContent = "";
  }

  function setBusy(busy) {
    els.analyze.disabled = busy;
    els.clear.disabled = busy;
    els.loading.classList.toggle("d-none", !busy);
    els.analyze.innerHTML = busy
      ? '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing…'
      : '<i class="bi bi-shield-check me-1"></i> Analyze message';
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderResult(data) {
    const probability = Number(data.model_probability);
    const hasProbability = Number.isFinite(probability);
    const label = String(data.classification || "UNKNOWN").toUpperCase();

    // Badge + verdict
    els.badge.textContent = label;
    els.verdict.textContent = label;
    els.sub.textContent =
      label === "SPAM" || label === "PHISHING"
        ? `The classifier predicts this message is ${label.toLowerCase()}.`
        : `The classifier predicts this message is ${label.toLowerCase()}.`;

    // Probability + meter
    els.probability.textContent = hasProbability
      ? `${(probability * 100).toFixed(1)}%`
      : "—";
    els.meter.style.width = hasProbability
      ? `${Math.max(0, Math.min(100, probability * 100))}%`
      : "0%";
    els.meter.parentElement.setAttribute(
      "aria-valuenow",
      hasProbability ? String(Math.round(Math.max(0, Math.min(100, probability * 100)))) : "0"
    );

    // Styling hooks
    els.result.dataset.classification = label;
    els.badge.className = "badge result-badge";
    els.verdict.className = "verdict-label";

    const cutoff = Number(data.decision_threshold);
    const cutoffText = Number.isFinite(cutoff)
      ? ` Spam cutoff: ${(cutoff * 100).toFixed(0)}%.`
      : "";
    els.meta.textContent = data.model_name
      ? `Model: ${data.model_name}.${cutoffText} Probability is a model score, not a guarantee.`
      : `Probability is the model's spam score.${cutoffText}`;

    // Show why this classification
    const signals = data.top_signals || { available: false, positive: [], negative: [] };
    if (signals.available && (signals.positive.length || signals.negative.length)) {
      els.signalsPositive.innerHTML = signals.positive
        .map(
          (s) =>
            `<div class="signal-item d-flex justify-content-between align-items-center mb-1">
              <span class="text-truncate" style="max-width: 70%;">${escapeHtml(s.term)}</span>
              <span class="badge bg-danger text-white small">${s.score > 0 ? "+" : ""}${s.score}</span>
            </div>`
        )
        .join("");
      els.signalsNegative.innerHTML = signals.negative
        .map(
          (s) =>
            `<div class="signal-item d-flex justify-content-between align-items-center mb-1">
              <span class="text-truncate" style="max-width: 70%;">${escapeHtml(s.term)}</span>
              <span class="badge bg-success text-white small">${s.score}</span>
            </div>`
        )
        .join("");
      els.signalsCard.classList.remove("d-none");
    } else {
      els.signalsCard.classList.add("d-none");
    }

    els.result.classList.remove("d-none");
  }

  async function analyze() {
    const message = els.input.value.trim();
    const task = "spam";
    hideError();

    if (!message) {
      showError("Please enter a message to analyze.");
      els.input.focus();
      return;
    }

    setBusy(true);
    els.result.classList.add("d-none");
    delete els.result.dataset.classification;
    els.signalsCard.classList.add("d-none");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, task }),
      });

      let payload = {};
      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }

      if (!response.ok) {
        throw new Error(
          payload.error || `The server returned an error (${response.status}).`
        );
      }
      renderResult(payload);
    } catch (error) {
      showError(error.message || "The message could not be analyzed.");
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    els.input.value = "";
    updateCounter();
    hideError();
    els.result.classList.add("d-none");
    delete els.result.dataset.classification;
    els.signalsCard.classList.add("d-none");
    els.input.focus();
  }

  els.input.addEventListener("input", updateCounter);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      analyze();
    }
  });
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    analyze();
  });
  els.clear.addEventListener("click", clear);
  els.samples.querySelectorAll("[data-demo]").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = SAMPLES.spam[Number(button.dataset.demo)];
      if (!sample) return;
      els.input.value = sample.text;
      updateCounter();
      hideError();
      els.input.focus();
    });
  });
  updateCounter();
})();
