"""SMS spam prediction service.

The ML classification is authoritative. The top_signals field shows which
tokens in the message pushed the prediction toward positive or negative class,
based on the fitted classifier's feature weights. These are model associations,
not causal explanations.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.model_registry import ModelBundle, ModelRegistry, ModelUnavailableError

LOGGER = logging.getLogger(__name__)

# Human-readable label names per task.
TASK_LABELS = {
    "spam": {"positive": "SPAM", "negative": "HAM", "task_label": "SMS / text message spam"},
}

TASK_INPUT_DOMAIN = {
    "spam": "SMS and short text messages",
}


class PredictionInputError(ValueError):
    """Invalid user input (empty message, wrong type, too long, bad task)."""


class PredictionService:
    def __init__(self, registry: ModelRegistry, max_message_length: int = 20_000):
        self.registry = registry
        self.max_message_length = max_message_length

    # ------------------------------------------------------------------
    def analyze(self, raw_message: Any, task: str = "spam") -> dict:
        if task not in TASK_LABELS:
            raise PredictionInputError(
                f"Unknown task '{task}'. Supported tasks: {', '.join(TASK_LABELS)}."
            )
        if not isinstance(raw_message, str):
            raise PredictionInputError("The 'message' field must be a string.")
        message = raw_message.replace("\x00", "").strip()
        if not message:
            raise PredictionInputError("Please provide a non-empty message to analyse.")
        if len(message) > self.max_message_length:
            raise PredictionInputError(
                f"Message is too long. Maximum length is {self.max_message_length:,} characters."
            )

        bundle = self.registry.get(task)
        label, probability = self._predict(bundle, message)
        labels = TASK_LABELS[task]
        positive_class = 1
        classification = labels["positive"] if label == positive_class else labels["negative"]
        top_signals = self._top_signals(bundle, message, positive_class=positive_class)

        return {
            "message": message,
            "task": task,
            "task_label": labels["task_label"],
            "input_domain": TASK_INPUT_DOMAIN[task],
            "classification": classification,
            "model_probability": round(probability, 4),
            "decision_threshold": bundle.metadata.get("classification_threshold", 0.5),
            "model_name": bundle.metadata.get("model_name", "Classifier"),
            "model_version": bundle.metadata.get("model_version", "unknown"),
            "top_signals": top_signals,
        }

    # ------------------------------------------------------------------
    def _predict(self, bundle: ModelBundle, message: str) -> tuple[int, float]:
        """Return the predicted class and positive-class probability."""
        try:
            probabilities = bundle.pipeline.predict_proba([message])[0]
            classes = list(bundle.pipeline.classes_)
            positive_index = classes.index(1)
            probability = round(float(probabilities[positive_index]), 4)
            threshold = float(bundle.metadata.get("classification_threshold", 0.5))
            label = int(probability >= threshold)
        except Exception as exc:  # pragma: no cover - defensive model integration path
            LOGGER.exception("Prediction failed for task %s", bundle.task)
            raise ModelUnavailableError(
                "The model could not process this message. Check the saved model artefact."
            ) from exc
        return label, probability

    # ------------------------------------------------------------------
    def _top_signals(
        self, bundle: ModelBundle, message: str, positive_class: int = 1
    ) -> dict[str, Any]:
        """Return top tokens pushing toward positive or negative class.

        Uses TF-IDF feature values with either Naive Bayes log-probability
        differences or linear model coefficients to score present features.
        """
        try:
            pipeline = bundle.pipeline
            vectorizer = pipeline.named_steps.get("tfidf")
            classifier = pipeline.named_steps.get("clf")
            if vectorizer is None or classifier is None:
                return {"available": False, "positive": [], "negative": []}

            # A calibrated linear model wraps each base estimator; its margin
            # coefficients still provide useful directional feature weights.
            if not (hasattr(classifier, "feature_log_prob_") or hasattr(classifier, "coef_")):
                calibrated = getattr(classifier, "calibrated_classifiers_", [])
                if calibrated:
                    classifier = getattr(calibrated[0], "estimator", classifier)

            has_log_prob = hasattr(classifier, "feature_log_prob_")
            has_coefficients = hasattr(classifier, "coef_")
            if not (has_log_prob or has_coefficients):
                return {"available": False, "positive": [], "negative": []}

            classes = list(getattr(classifier, "classes_", [0, 1]))
            if len(classes) != 2:
                return {"available": False, "positive": [], "negative": []}

            pos_index = classes.index(positive_class)
            neg_index = classes.index(0 if positive_class == 1 else 1)
            if has_log_prob:
                log_probs = classifier.feature_log_prob_
            else:
                coefficients = classifier.coef_
                if coefficients.shape[0] != 1:
                    return {"available": False, "positive": [], "negative": []}
                coefficient_sign = 1.0 if pos_index == 1 else -1.0

            # Transform the message and get which feature indices are present.
            matrix = vectorizer.transform([message])
            feature_names = vectorizer.get_feature_names_out()

            coo = matrix.tocoo()
            token_scores: dict[str, float] = {}
            for row, col, feature_value in zip(coo.row, coo.col, coo.data):
                if row != 0:
                    continue
                token = str(feature_names[col])
                # Difference in log-probability: how much more/less positive-like.
                if has_log_prob:
                    score = float(
                        (log_probs[pos_index, col] - log_probs[neg_index, col])
                        * feature_value
                    )
                else:
                    score = float(coefficient_sign * coefficients[0, col] * feature_value)
                # Keep the max score if a token appears multiple times.
                if token not in token_scores or score > token_scores[token]:
                    token_scores[token] = score

            positive = sorted(
                ((t, s) for t, s in token_scores.items() if s > 0),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
            negative = sorted(
                ((t, s) for t, s in token_scores.items() if s < 0),
                key=lambda x: x[1],
            )[:8]

            pos_label = TASK_LABELS.get(bundle.task, {}).get("positive", "positive")
            neg_label = TASK_LABELS.get(bundle.task, {}).get("negative", "negative")

            return {
                "available": True,
                "positive": [
                    {"term": t, "score": round(s, 3), "direction": f"supports {pos_label}"}
                    for t, s in positive
                ],
                "negative": [
                    {"term": t, "score": round(s, 3), "direction": f"supports {neg_label}"}
                    for t, s in negative
                ],
            }
        except Exception:  # pragma: no cover - explanation must never break prediction
            LOGGER.exception("Top signals computation failed")
            return {"available": False, "positive": [], "negative": []}
